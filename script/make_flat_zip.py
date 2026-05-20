#!/usr/bin/env python3
"""Create a flat zip from image outputs (no subdirectories)."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_zip", type=Path, required=True)
    p.add_argument("--pattern", type=str, default="*.png", help="Glob pattern, e.g. '*.png' or '*.jpg'")
    p.add_argument("--compression", type=str, default="store", choices=["store", "deflate"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")

    files = sorted(args.input_dir.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern={args.pattern} in {args.input_dir}")

    basenames = [p.name for p in files]
    if len(set(basenames)) != len(basenames):
        raise ValueError("Basename collision detected; flat zip requires unique filenames.")

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_STORED if args.compression == "store" else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(args.output_zip, "w", compression=compression) as zf:
        for file_path in files:
            zf.write(file_path, arcname=file_path.name)
    print(f"Saved: {args.output_zip} ({len(files)} files)")


if __name__ == "__main__":
    main()

