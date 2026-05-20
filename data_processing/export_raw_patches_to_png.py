#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _parse_csv_floats(value: str, n: int, name: str) -> np.ndarray:
    arr = np.array([float(x.strip()) for x in value.split(",")], dtype=np.float32)
    if arr.size != n:
        raise ValueError(f"{name} must have {n} comma-separated values, got {arr.size}: {value}")
    return arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert raw patch npy files (H,W,4) to RGB image patches using demosaic + optional FFDNet denoising. "
            "Outputs one image per input patch with the same stem."
        )
    )
    p.add_argument("--input_dir", type=Path, required=True, help="Directory containing raw patch .npy files")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory to save output image patches")
    p.add_argument("--raw4_order", type=str, default="R,Gr,Gb,B")
    p.add_argument("--black_levels", type=str, default="0,0,0,0")
    p.add_argument("--white_levels", type=str, default="1023,1023,1023,1023")

    p.add_argument(
        "--demosaic_method",
        type=str,
        default="demosaicnet",
        choices=["opencv", "opencv_ea", "demosaicnet", "bilinear"],
    )
    p.add_argument("--denoise_method", type=str, default="ffdnet", choices=["none", "ffdnet"])
    p.add_argument("--ffdnet_noise_sigma", type=float, default=0.01)
    p.add_argument("--ffdnet_weights", type=str, default="model/third_party/ffdnet_pytorch/models/net_rgb.pth")
    p.add_argument("--ffdnet_device", type=str, default="auto", choices=["cpu", "cuda", "auto"])

    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--tone_mapping", type=str, default="none", choices=["none", "smoothstep"])

    p.add_argument("--output_ext", type=str, default="png", choices=["png", "jpg", "jpeg"])
    p.add_argument("--jpeg_quality", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_files", type=int, default=0, help="0 means process all files")
    p.add_argument("--progress_every", type=int, default=500)
    p.add_argument("--strict", action="store_true", help="Raise on first file error")
    return p.parse_args()


def _import_raw_pipeline():
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "model"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from raw_pipeline import (  # type: ignore
        demosaic_rggb,
        ffdnet_denoise_rgb_np,
        normalize_raw4,
        raw4_to_mosaic_rggb,
        reorder_raw4_to_canonical,
        smoothstep_np,
    )

    return {
        "reorder_raw4_to_canonical": reorder_raw4_to_canonical,
        "normalize_raw4": normalize_raw4,
        "raw4_to_mosaic_rggb": raw4_to_mosaic_rggb,
        "demosaic_rggb": demosaic_rggb,
        "ffdnet_denoise_rgb_np": ffdnet_denoise_rgb_np,
        "smoothstep_np": smoothstep_np,
    }


def _save_rgb(rgb: np.ndarray, out_path: Path, output_ext: str, jpeg_quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr8 = cv2.cvtColor((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    ext = output_ext.lower()
    if ext == "png":
        ok = cv2.imwrite(str(out_path), bgr8)
    else:
        ok = cv2.imwrite(str(out_path), bgr8, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    black_levels = _parse_csv_floats(args.black_levels, 4, "black_levels")
    white_levels = _parse_csv_floats(args.white_levels, 4, "white_levels")
    gamma = max(float(args.gamma), 1e-8)

    rp = _import_raw_pipeline()
    reorder_raw4_to_canonical = rp["reorder_raw4_to_canonical"]
    normalize_raw4 = rp["normalize_raw4"]
    raw4_to_mosaic_rggb = rp["raw4_to_mosaic_rggb"]
    demosaic_rggb = rp["demosaic_rggb"]
    ffdnet_denoise_rgb_np = rp["ffdnet_denoise_rgb_np"]
    smoothstep_np = rp["smoothstep_np"]

    files = sorted(args.input_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")
    if args.max_files > 0:
        files = files[: args.max_files]

    done = 0
    skipped = 0
    failed = 0

    for i, npy_path in enumerate(files, start=1):
        out_path = args.output_dir / f"{npy_path.stem}.{args.output_ext.lower()}"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            raw = np.load(npy_path)
            if raw.ndim != 3 or raw.shape[-1] != 4:
                raise ValueError(f"Expected raw4 (H,W,4), got {raw.shape}")

            raw4 = reorder_raw4_to_canonical(raw, args.raw4_order)
            raw4_norm = normalize_raw4(raw4, black_levels, white_levels)
            mosaic = raw4_to_mosaic_rggb(raw4_norm)

            rgb = demosaic_rggb(mosaic, method=args.demosaic_method)
            if args.denoise_method == "ffdnet":
                rgb = ffdnet_denoise_rgb_np(
                    rgb,
                    noise_sigma=float(args.ffdnet_noise_sigma),
                    weights_path=str(args.ffdnet_weights),
                    device=str(args.ffdnet_device),
                )

            rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gamma)
            if args.tone_mapping == "smoothstep":
                rgb = smoothstep_np(rgb)
            rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

            _save_rgb(
                rgb=rgb,
                out_path=out_path,
                output_ext=args.output_ext,
                jpeg_quality=int(args.jpeg_quality),
            )
            done += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {npy_path.name}: {e}")
            if args.strict:
                raise

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(files)):
            print(f"processed {i}/{len(files)} (done={done}, skipped={skipped}, failed={failed})")

    print("Done.")
    print(f"input_dir : {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"done={done}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
