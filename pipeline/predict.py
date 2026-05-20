#!/usr/bin/env python3
"""Challenge inference entrypoint.

Required interface:
  python predict.py --input_dir <PATH_TO_INPUT_RAWS> --output_dir <PATH_TO_SAVE_IMAGES>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _REPO_ROOT / "model"
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from raw_pipeline import (
    PseudoISPConfig,
    load_raw_npy,
    make_pseudo_rgb_from_raw4,
    parse_levels,
    parse_white_balance,
    reorder_raw4_to_canonical,
)
from submission_models import DemosaicCNN
from train_utils import load_checkpoint_state, torch_load_compat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True, help="Directory containing input RAW4 .npy files.")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory to save output images.")

    # Optional overrides. Organizers can ignore these.
    p.add_argument("--checkpoint", type=Path, default=None, help="Optional explicit checkpoint path.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--raw4_order", type=str, default="R,Gr,Gb,B")
    p.add_argument("--black_levels", type=str, default="0,0,0,0")
    p.add_argument("--white_levels", type=str, default="1023,1023,1023,1023")
    p.add_argument("--white_balance", type=str, default="1.0,1.0,1.0,1.0")
    p.add_argument("--demosaic_method", type=str, default="demosaicnet", choices=["demosaicnet", "opencv", "bilinear", "opencv_ea"])
    p.add_argument("--ffdnet_noise_sigma", type=float, default=0.01)
    p.add_argument("--ffdnet_weights", type=str, default=None, help="Default resolves to model/third_party/ffdnet_pytorch/models/net_rgb.pth")
    p.add_argument("--ffdnet_device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--input_gamma", type=float, default=2.2)
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _resolve_ffdnet_weights(cli_value: str | None) -> str:
    if cli_value:
        return str(Path(cli_value))
    return str(_REPO_ROOT / "model" / "third_party" / "ffdnet_pytorch" / "models" / "net_rgb.pth")


def _resolve_checkpoint(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value

    env_ckpt = os.getenv("NTIRE_ISP_CHECKPOINT", "").strip()

    candidates: list[Path] = []
    if env_ckpt:
        candidates.append(Path(env_ckpt))
    candidates.extend(
        [
            _REPO_ROOT / "checkpoints" / "last.pt",
            _REPO_ROOT / "checkpoints" / "best.pt",
            _REPO_ROOT / "checkpoints" / "model.pt",
            _REPO_ROOT / "code" / "checkpoints" / "last.pt",
            _REPO_ROOT / "code" / "checkpoints" / "best.pt",
            _REPO_ROOT / "code" / "checkpoints" / "model.pt",
        ]
    )

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "Checkpoint not found. Place checkpoint at checkpoints/last.pt "
        "or set NTIRE_ISP_CHECKPOINT environment variable."
    )


def _load_model(checkpoint_path: Path, device: torch.device) -> DemosaicCNN:
    raw_ckpt = torch_load_compat(checkpoint_path, map_location="cpu")
    ckpt_args = raw_ckpt.get("args", {}) if isinstance(raw_ckpt, dict) else {}

    # Submission model is fixed to demosaiccnn + nonlinear target.
    model_type = str(ckpt_args.get("model_type", "demosaiccnn"))
    if model_type != "demosaiccnn":
        raise ValueError(f"This submission code supports model_type='demosaiccnn' only, got: {model_type}")
    target_domain = str(ckpt_args.get("target_domain", "nonlinear"))
    if target_domain != "nonlinear":
        raise ValueError(f"This submission code supports target_domain='nonlinear' only, got: {target_domain}")

    kernel_size = int(ckpt_args.get("kernel_size", 3))
    hidden = int(ckpt_args.get("hidden", 16))
    use_lut_head = bool(ckpt_args.get("use_lut_head", False))
    lut_size = int(ckpt_args.get("lut_size", 33))

    model = DemosaicCNN(
        kernel_size=kernel_size,
        hidden=hidden,
        use_lut_head=use_lut_head,
        lut_size=lut_size,
    )
    state = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model


def _write_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.clip(rgb, 0.0, 1.0)
    u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"--input_dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(args.input_dir.glob("*.npy"))
    if not input_files:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")

    device = _resolve_device(args.device)
    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    model = _load_model(checkpoint_path, device=device)

    black_levels = parse_levels(args.black_levels, "black_levels")
    white_levels = parse_levels(args.white_levels, "white_levels")
    white_balance = parse_white_balance(args.white_balance)
    ffdnet_weights = _resolve_ffdnet_weights(args.ffdnet_weights)

    # Fixed preprocessing used by this submission: demosaicnet + FFDNet + gamma(2.2), no tone mapping.
    pre_cfg = PseudoISPConfig(
        white_balance=white_balance,
        black_levels=black_levels,
        white_levels=white_levels,
        demosaic_method=args.demosaic_method,
        denoise_method="ffdnet",
        ffdnet_noise_sigma=float(args.ffdnet_noise_sigma),
        ffdnet_weights=ffdnet_weights,
        ffdnet_device=str(args.ffdnet_device),
        gamma=float(args.input_gamma),
        tone_mapping="none",
        ccm_mode="none",
    )

    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Inputs: {len(input_files)} files")

    for in_file in input_files:
        raw4 = load_raw_npy(str(in_file))
        raw4 = reorder_raw4_to_canonical(raw4, args.raw4_order)
        _, source_rgb = make_pseudo_rgb_from_raw4(raw4, pre_cfg)
        source_t = torch.from_numpy(source_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(source_t).clamp(0.0, 1.0)

        pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_file = args.output_dir / f"{in_file.stem}.png"
        _write_png(out_file, pred_np)
        print(f"[OK] {in_file.name} -> {out_file.name}")

    print("[DONE] Inference complete.")


if __name__ == "__main__":
    main()
