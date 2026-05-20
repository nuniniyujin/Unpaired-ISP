"""Inference for train_cnn_new checkpoints.

Pipeline:
  RAW4 npy -> demosaic/ffdnet/gamma pre-process -> CNN (train_cnn_new arch) -> image save
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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
from train_utils import load_checkpoint_state, torch_load_compat


def gamma_compress_torch(image: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return torch.clamp(image, min=0.0, max=1.0).pow(1.0 / max(gamma, 1e-6))


def smoothstep_torch(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    return image * image * (3.0 - 2.0 * image)


class LUT3DHead(nn.Module):
    """Differentiable 3D LUT color head with trilinear interpolation."""

    def __init__(self, lut_size: int = 33):
        super().__init__()
        if int(lut_size) <= 1:
            raise ValueError("lut_size must be > 1")
        self.lut_size = int(lut_size)
        coords = torch.linspace(0.0, 1.0, self.lut_size, dtype=torch.float32)
        rr, gg, bb = torch.meshgrid(coords, coords, coords, indexing="ij")
        identity = torch.stack([rr, gg, bb], dim=0).unsqueeze(0)  # [1,3,L,L,L]
        self.register_buffer("identity_lut", identity)
        self.lut = nn.Parameter(identity.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"LUT3DHead expects [B,3,H,W], got {tuple(x.shape)}")
        rgb = x.clamp(0.0, 1.0).permute(0, 2, 3, 1)
        grid = torch.stack(
            [
                rgb[..., 2] * 2.0 - 1.0,  # x -> B axis
                rgb[..., 1] * 2.0 - 1.0,  # y -> G axis
                rgb[..., 0] * 2.0 - 1.0,  # z -> R axis
            ],
            dim=-1,
        ).unsqueeze(1)
        lut = self.lut.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1, -1)
        out = F.grid_sample(
            lut,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return out.squeeze(2)


class DemosaicCNN(nn.Module):
    def __init__(self, kernel_size: int = 3, hidden: int = 16, use_lut_head: bool = False, lut_size: int = 33):
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError("kernel_size must be 1 or 3")
        padding = 0 if kernel_size == 1 else 1
        conv_kwargs = {"padding_mode": "reflect"} if padding > 0 else {}
        self.conv1 = nn.Conv2d(3, hidden, kernel_size=kernel_size, padding=padding, **conv_kwargs)
        self.conv2 = nn.Conv2d(hidden, 3, kernel_size=kernel_size, padding=padding, **conv_kwargs)
        self.ccm_head = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=True)
        with torch.no_grad():
            self.ccm_head.weight.zero_()
            self.ccm_head.bias.zero_()
            self.ccm_head.weight.copy_(torch.eye(3, dtype=self.ccm_head.weight.dtype).view(3, 3, 1, 1))
        self.lut_head = LUT3DHead(lut_size=lut_size) if use_lut_head else None

    def _demosaic(self, raw4: torch.Tensor) -> torch.Tensor:
        r = F.interpolate(raw4[:, 0:1], scale_factor=2, mode="bicubic", align_corners=False)
        g1 = F.interpolate(raw4[:, 1:2], scale_factor=2, mode="bicubic", align_corners=False)
        g2 = F.interpolate(raw4[:, 2:3], scale_factor=2, mode="bicubic", align_corners=False)
        b = F.interpolate(raw4[:, 3:4], scale_factor=2, mode="bicubic", align_corners=False)
        g = 0.5 * (g1 + g2)
        return torch.cat([r, g, b], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input with shape [B,C,H,W], got shape={tuple(x.shape)}")
        if x.shape[1] == 4:
            base = self._demosaic(x)
        elif x.shape[1] == 3:
            base = x
        else:
            raise ValueError(f"Expected 3 or 4 channels, got {x.shape[1]}")
        h = F.relu(self.conv1(base), inplace=True)
        delta = self.conv2(h)
        out = base + delta
        out = self.ccm_head(out)
        if self.lut_head is not None:
            out = self.lut_head(out)
        return torch.clamp(out, 0.0, 1.0)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError("kernel_size must be 1 or 3")
        padding = 0 if kernel_size == 1 else 1
        conv_kwargs = {"padding_mode": "reflect"} if padding > 0 else {}
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, **conv_kwargs)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, **conv_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return x + y

class BiggerDemosaicCNN(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        hidden: int = 64,
        num_res_blocks: int = 4,
        use_lut_head: bool = False,
        lut_size: int = 33,
    ):
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError("kernel_size must be 1 or 3")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")
        padding = 0 if kernel_size == 1 else 1
        conv_kwargs = {"padding_mode": "reflect"} if padding > 0 else {}
        self.stem = nn.Conv2d(3, hidden, kernel_size=kernel_size, padding=padding, **conv_kwargs)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, kernel_size=kernel_size) for _ in range(num_res_blocks)])
        self.head = nn.Conv2d(hidden, 3, kernel_size=kernel_size, padding=padding, **conv_kwargs)
        self.lut_head = LUT3DHead(lut_size=lut_size) if use_lut_head else None

    def _demosaic(self, raw4: torch.Tensor) -> torch.Tensor:
        r = F.interpolate(raw4[:, 0:1], scale_factor=2, mode="bicubic", align_corners=False)
        g1 = F.interpolate(raw4[:, 1:2], scale_factor=2, mode="bicubic", align_corners=False)
        g2 = F.interpolate(raw4[:, 2:3], scale_factor=2, mode="bicubic", align_corners=False)
        b = F.interpolate(raw4[:, 3:4], scale_factor=2, mode="bicubic", align_corners=False)
        g = 0.5 * (g1 + g2)
        return torch.cat([r, g, b], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input with shape [B,C,H,W], got shape={tuple(x.shape)}")
        if x.shape[1] == 4:
            base = self._demosaic(x)
        elif x.shape[1] == 3:
            base = x
        else:
            raise ValueError(f"Expected 3 or 4 channels, got {x.shape[1]}")
        feat = F.relu(self.stem(base), inplace=True)
        for block in self.blocks:
            feat = block(feat)
        delta = self.head(F.relu(feat, inplace=True))
        out = base + delta
        if self.lut_head is not None:
            out = self.lut_head(out)
        return torch.clamp(out, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_path", type=Path, required=True, help="Path to .npy file or directory.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out_path", type=Path, required=True)
    p.add_argument("--output_ext", type=str, default="jpg", choices=["jpg", "jpeg", "png"])
    p.add_argument("--jpeg_quality", type=int, default=100)
    p.add_argument("--make_flat_zip", action="store_true")
    p.add_argument("--zip_path", type=Path, default=None)
    p.add_argument("--zip_compression", type=str, default="store", choices=["store", "deflate"])

    p.add_argument("--raw4_order", type=str, default="R,Gr,Gb,B")
    p.add_argument("--black_levels", type=str, default="0,0,0,0")
    p.add_argument("--white_levels", type=str, default="1023,1023,1023,1023")
    p.add_argument("--white_balance", type=str, default="1.0,1.0,1.0,1.0")
    p.add_argument("--demosaic_method", type=str, default="demosaicnet", choices=["demosaicnet", "opencv", "bilinear", "opencv_ea"])
    p.add_argument("--ffdnet_noise_sigma", type=float, default=0.01)
    p.add_argument("--ffdnet_weights", type=str, default="model/third_party/ffdnet_pytorch/models/net_rgb.pth")
    p.add_argument("--ffdnet_device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--input_gamma", type=float, default=2.2)
    p.add_argument("--input_tone_mapping", type=str, default="none", choices=["none", "smoothstep"])

    p.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["demosaiccnn", "bigger_demosaiccnn"],
        help="If omitted, read from checkpoint args.",
    )
    p.add_argument("--kernel_size", type=int, default=None, choices=[1, 3], help="If omitted, read from checkpoint args.")
    p.add_argument("--hidden", type=int, default=None, help="If omitted, read from checkpoint args.")
    p.add_argument("--num_res_blocks", type=int, default=None, choices=[3, 4, 5], help="If omitted, read from checkpoint args.")
    p.add_argument(
        "--use_lut_head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If omitted, read from checkpoint args.",
    )
    p.add_argument("--lut_size", type=int, default=None, help="If omitted, read from checkpoint args.")
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True, help="Strict checkpoint loading.")
    p.add_argument(
        "--output_postprocess",
        type=str,
        default="auto",
        choices=["auto", "none", "gamma", "gamma_smoothstep"],
        help="How to postprocess model output before save.",
    )
    p.add_argument("--output_gamma", type=float, default=2.2, help="Used when output_postprocess uses gamma.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _pick_config(cli_value, ckpt_args: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    if isinstance(ckpt_args, dict) and key in ckpt_args:
        return ckpt_args[key]
    return default


def _resolve_model_config(args: argparse.Namespace, ckpt_args: dict) -> tuple[str, int, int, int, bool, int]:
    model_type = str(_pick_config(args.model_type, ckpt_args, "model_type", "demosaiccnn"))
    kernel_size = int(_pick_config(args.kernel_size, ckpt_args, "kernel_size", 3))
    hidden = int(_pick_config(args.hidden, ckpt_args, "hidden", 16))
    num_res_blocks = int(_pick_config(args.num_res_blocks, ckpt_args, "num_res_blocks", 4))
    use_lut_head = bool(_pick_config(args.use_lut_head, ckpt_args, "use_lut_head", False))
    lut_size = int(_pick_config(args.lut_size, ckpt_args, "lut_size", 33))
    if lut_size <= 1:
        raise ValueError("lut_size must be > 1")
    return model_type, kernel_size, hidden, num_res_blocks, use_lut_head, lut_size


def _resolve_output_path(raw_file: Path, raw_root: Path, out_path: Path, output_ext: str) -> Path:
    if raw_root.is_dir():
        rel = raw_file.relative_to(raw_root)
        out_file = out_path / rel
        return out_file.with_suffix(f".{output_ext}")
    if out_path.suffix:
        return out_path
    return out_path / f"{raw_file.stem}.{output_ext}"


def _write_image(path: Path, rgb: np.ndarray, jpeg_quality: int) -> None:
    rgb = np.clip(rgb, 0.0, 1.0)
    u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        quality = int(max(1, min(100, jpeg_quality)))
        ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _apply_output_postprocess(pred: torch.Tensor, mode: str, gamma: float) -> torch.Tensor:
    if mode == "none":
        return pred.clamp(0.0, 1.0)
    if mode == "gamma":
        return gamma_compress_torch(pred, gamma=gamma)
    if mode == "gamma_smoothstep":
        return smoothstep_torch(gamma_compress_torch(pred, gamma=gamma))
    raise ValueError(f"Unsupported output_postprocess: {mode}")


def main() -> None:
    args = parse_args()

    raw_ckpt = torch_load_compat(args.checkpoint, map_location="cpu")
    ckpt_args = raw_ckpt.get("args", {}) if isinstance(raw_ckpt, dict) else {}
    model_type, kernel_size, hidden, num_res_blocks, use_lut_head, lut_size = _resolve_model_config(args, ckpt_args)

    if model_type == "demosaiccnn":
        model = DemosaicCNN(
            kernel_size=kernel_size,
            hidden=hidden,
            use_lut_head=use_lut_head,
            lut_size=lut_size,
        )
    elif model_type == "bigger_demosaiccnn":
        model = BiggerDemosaicCNN(
            kernel_size=kernel_size,
            hidden=hidden,
            num_res_blocks=num_res_blocks,
            use_lut_head=use_lut_head,
            lut_size=lut_size,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    state = load_checkpoint_state(args.checkpoint)
    incompatible = model.load_state_dict(state, strict=args.strict)
    if not args.strict:
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(f"Loaded checkpoint with strict=False (missing={len(missing)}, unexpected={len(unexpected)})")

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model = model.to(device).eval()
    print(
        f"Loaded model: type={model_type}, kernel_size={kernel_size}, hidden={hidden}, "
        f"num_res_blocks={num_res_blocks}, lut_head={use_lut_head}, lut_size={lut_size}"
    )

    black_levels = parse_levels(args.black_levels, "black_levels")
    white_levels = parse_levels(args.white_levels, "white_levels")
    white_balance = parse_white_balance(args.white_balance)

    if args.raw_path.is_dir():
        raw_files = sorted(args.raw_path.rglob("*.npy"))
        if not raw_files:
            raise FileNotFoundError(f"No .npy files found in {args.raw_path}")
        args.out_path.mkdir(parents=True, exist_ok=True)
    else:
        if args.raw_path.suffix.lower() != ".npy":
            raise ValueError(f"--raw_path must be .npy or directory of .npy files, got: {args.raw_path}")
        raw_files = [args.raw_path]
        if args.out_path.suffix:
            args.out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            args.out_path.mkdir(parents=True, exist_ok=True)

    pre_cfg = PseudoISPConfig(
        white_balance=white_balance,
        black_levels=black_levels,
        white_levels=white_levels,
        demosaic_method=args.demosaic_method,
        denoise_method="ffdnet",
        ffdnet_noise_sigma=float(args.ffdnet_noise_sigma),
        ffdnet_weights=str(args.ffdnet_weights),
        ffdnet_device=str(args.ffdnet_device),
        gamma=float(args.input_gamma),
        tone_mapping=str(args.input_tone_mapping),
        ccm_mode="none",
    )

    post_mode = args.output_postprocess
    if post_mode == "auto":
        target_domain = str(ckpt_args.get("target_domain", "nonlinear")) if isinstance(ckpt_args, dict) else "nonlinear"
        if target_domain == "linear":
            tone_mapping = str(ckpt_args.get("tone_mapping", "smoothstep"))
            post_mode = "gamma_smoothstep" if tone_mapping == "smoothstep" else "gamma"
        else:
            post_mode = "none"
    print(f"Output postprocess: mode={post_mode}, gamma={args.output_gamma}")

    written_files: list[Path] = []
    for raw_file in raw_files:
        raw4 = load_raw_npy(str(raw_file))
        raw4 = reorder_raw4_to_canonical(raw4, args.raw4_order)
        _, source_rgb = make_pseudo_rgb_from_raw4(raw4, pre_cfg)
        source_t = torch.from_numpy(source_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(source_t)
            pred = _apply_output_postprocess(pred, mode=post_mode, gamma=float(args.output_gamma))
            pred = pred.clamp(0.0, 1.0)

        pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_file = _resolve_output_path(raw_file, args.raw_path, args.out_path, args.output_ext)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        _write_image(out_file, pred_np, args.jpeg_quality)
        print(f"Saved: {out_file}")
        written_files.append(out_file)

    if args.make_flat_zip or args.zip_path is not None:
        if not written_files:
            raise RuntimeError("No output files were written; cannot create zip.")
        zip_path = args.zip_path if args.zip_path is not None else (args.out_path.parent / f"{args.out_path.name}.zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        basenames = [p.name for p in written_files]
        if len(set(basenames)) != len(basenames):
            raise ValueError("Cannot create flat zip because output basenames are not unique.")

        compression = zipfile.ZIP_STORED if args.zip_compression == "store" else zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
            for file_path in sorted(written_files):
                zf.write(file_path, arcname=file_path.name)
        print(f"Saved flat zip: {zip_path} ({len(written_files)} files)")


if __name__ == "__main__":
    main()
