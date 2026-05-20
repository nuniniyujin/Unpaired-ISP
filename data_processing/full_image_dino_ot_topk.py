#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full-image DINO+OT matching and export top-k JSON/CSV files."
    )
    p.add_argument("--source_dir", type=Path, required=True, help="Source full-image directory")
    p.add_argument("--target_dir", type=Path, required=True, help="Target full-image directory")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory for top-k files")
    p.add_argument("--topk", type=int, default=10, help="Top-k matches per source image")
    p.add_argument("--viz_count", type=int, default=0, help="Number of visualizations to save")
    p.add_argument("--device", type=str, default="cuda", help="Device for feature extraction")
    p.add_argument("--extract_batch_size", type=int, default=8)
    p.add_argument("--match_chunk_size", type=int, default=1024)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--dino_model_name", type=str, default="dinov2_vits14")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve().with_name("match_unpaired_targets.py")
    cmd = [
        sys.executable,
        str(script),
        "--task",
        "full_dino_ot",
        "--method",
        "dino_ot",
        "--source_dir",
        str(args.source_dir),
        "--target_dir",
        str(args.target_dir),
        "--output_dir",
        str(args.output_dir),
        "--topk",
        str(args.topk),
        "--viz_count",
        str(args.viz_count),
        "--device",
        str(args.device),
        "--extract_batch_size",
        str(args.extract_batch_size),
        "--match_chunk_size",
        str(args.match_chunk_size),
        "--image_size",
        str(args.image_size),
        "--dino_model_name",
        str(args.dino_model_name),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
