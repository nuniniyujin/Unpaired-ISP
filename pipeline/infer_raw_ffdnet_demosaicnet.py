"""Infer RGB images from RAW4 .npy using: demosaicnet -> FFDNet -> gamma(2.2).

Supports:
- Single file or directory input
- JPG/PNG output
- Flat zip packaging (no nested folders)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_path", type=Path, required=True, help="Path to a .npy file or a directory of .npy files.")
    p.add_argument("--out_path", type=Path, required=True, help="Output file or output directory.")
    p.add_argument("--output_ext", type=str, default="jpg", choices=["jpg", "jpeg", "png"])
    p.add_argument("--jpeg_quality", type=int, default=100)
    p.add_argument("--make_flat_zip", action="store_true", help="Create a flat zip after inference.")
    p.add_argument("--zip_path", type=Path, default=None, help="Optional zip output path. Default: <out_path>.zip")
    p.add_argument("--zip_compression", type=str, default="store", choices=["store", "deflate"])

    p.add_argument("--raw4_order", type=str, default="R,Gr,Gb,B")
    p.add_argument("--black_levels", type=str, default="0,0,0,0")
    p.add_argument("--white_levels", type=str, default="1023,1023,1023,1023")
    p.add_argument("--white_balance", type=str, default="1.0,1.0,1.0,1.0")

    p.add_argument("--demosaic_method", type=str, default="demosaicnet", choices=["demosaicnet", "opencv", "bilinear", "opencv_ea"])
    p.add_argument("--ffdnet_noise_sigma", type=float, default=0.01)
    p.add_argument("--ffdnet_weights", type=str, default="model/third_party/ffdnet_pytorch/models/net_rgb.pth")
    p.add_argument("--ffdnet_device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--tone_mapping", type=str, default="none", choices=["none", "smoothstep"])
    return p.parse_args()


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


def _resolve_output_path(raw_file: Path, raw_root: Path, out_path: Path, output_ext: str) -> Path:
    if raw_root.is_dir():
        rel = raw_file.relative_to(raw_root)
        out_file = out_path / rel
        return out_file.with_suffix(f".{output_ext}")

    if out_path.suffix:
        return out_path
    return out_path / f"{raw_file.stem}.{output_ext}"


def main() -> None:
    args = parse_args()

    black_levels = parse_levels(args.black_levels, "black_levels")
    white_levels = parse_levels(args.white_levels, "white_levels")
    white_balance = parse_white_balance(args.white_balance)

    if args.raw_path.is_dir():
        raw_files = sorted(args.raw_path.rglob("*.npy"))
        if not raw_files:
            raise FileNotFoundError(f"No .npy files found in {args.raw_path}")
        args.out_path.mkdir(parents=True, exist_ok=True)
    else:
        raw_files = [args.raw_path]
        if args.out_path.suffix:
            args.out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            args.out_path.mkdir(parents=True, exist_ok=True)

    cfg = PseudoISPConfig(
        white_balance=white_balance,
        black_levels=black_levels,
        white_levels=white_levels,
        demosaic_method=args.demosaic_method,
        denoise_method="ffdnet",
        ffdnet_noise_sigma=float(args.ffdnet_noise_sigma),
        ffdnet_weights=str(args.ffdnet_weights),
        ffdnet_device=str(args.ffdnet_device),
        gamma=float(args.gamma),
        tone_mapping=str(args.tone_mapping),
        ccm_mode="none",
    )

    written_files: list[Path] = []
    for raw_file in raw_files:
        raw4 = load_raw_npy(str(raw_file))
        raw4 = reorder_raw4_to_canonical(raw4, args.raw4_order)
        _, rgb = make_pseudo_rgb_from_raw4(raw4, cfg)

        out_file = _resolve_output_path(raw_file, args.raw_path, args.out_path, args.output_ext)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        _write_image(out_file, rgb, args.jpeg_quality)
        print(f"Saved: {out_file}")
        written_files.append(out_file)

    if args.make_flat_zip or args.zip_path is not None:
        if not written_files:
            raise RuntimeError("No output files were written; cannot create zip.")

        if args.zip_path is not None:
            zip_path = args.zip_path
        else:
            zip_path = args.out_path.parent / f"{args.out_path.name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        basenames = [p.name for p in written_files]
        if len(set(basenames)) != len(basenames):
            raise ValueError(
                "Cannot create flat zip because output basenames are not unique. "
                "Use unique file names or provide nested zip packaging."
            )

        compression = zipfile.ZIP_STORED if args.zip_compression == "store" else zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
            for file_path in sorted(written_files):
                zf.write(file_path, arcname=file_path.name)
        print(f"Saved flat zip: {zip_path} ({len(written_files)} files)")


if __name__ == "__main__":
    main()
