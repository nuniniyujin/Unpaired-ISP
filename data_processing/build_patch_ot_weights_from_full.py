#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_prefix(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"^(\d+)", stem)
    if not m:
        raise ValueError(f"Cannot parse numeric prefix from: {name}")
    return m.group(1).zfill(3)


def parse_patch_name(path: Path) -> Tuple[str, int]:
    stem = path.stem
    m = re.match(r"^(\d+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"Expected '<prefix>_<idx>' patch filename, got: {path.name}")
    pref = m.group(1).zfill(3)
    idx = int(m.group(2))
    return pref, idx


def list_patch_files(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    files.sort()
    return files


def load_full_topk_map(csv_path: Path, topk: int) -> Dict[str, List[dict]]:
    by_source: Dict[str, List[dict]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        needed = {"source_path", "target_path", "score", "rank"}
        missing = needed.difference(r.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

        for row in r:
            rank = int(row["rank"])
            if rank > topk:
                continue
            src_pref = parse_prefix(row["source_path"])
            tgt_pref = parse_prefix(row["target_path"])
            score = float(row["score"])
            by_source[src_pref].append(
                {
                    "rank": rank,
                    "target_prefix": tgt_pref,
                    "score": score,
                    "target_path": row["target_path"],
                }
            )

    for k in list(by_source.keys()):
        by_source[k] = sorted(by_source[k], key=lambda x: x["rank"])
    return by_source


def build_target_index(target_patch_dir: Path) -> Dict[str, Dict[int, Path]]:
    index: Dict[str, Dict[int, Path]] = defaultdict(dict)
    bad = 0
    for p in list_patch_files(target_patch_dir):
        try:
            pref, idx = parse_patch_name(p)
        except ValueError:
            bad += 1
            continue
        index[pref][idx] = p
    if bad:
        print(f"[WARN] skipped {bad} non patch-like target files")
    return index


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Propagate full-image OT top-k weights to patch-level pairs by prefix and patch index. "
            "For source patch <P>_<i>, candidate target patches are <Q>_<i> where Q comes from full top-k for P."
        )
    )
    ap.add_argument("--full_topk_csv", type=Path, required=True, help="matches_topk_long.csv from full-image matching")
    ap.add_argument("--source_patch_dir", type=Path, required=True)
    ap.add_argument("--target_patch_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min_score", type=float, default=0.0, help="drop candidates with score < min_score before renorm")
    args = ap.parse_args()

    if not args.full_topk_csv.exists():
        raise FileNotFoundError(args.full_topk_csv)
    if not args.source_patch_dir.exists():
        raise FileNotFoundError(args.source_patch_dir)
    if not args.target_patch_dir.exists():
        raise FileNotFoundError(args.target_patch_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_map = load_full_topk_map(args.full_topk_csv, topk=args.topk)
    tgt_index = build_target_index(args.target_patch_dir)

    src_files = list_patch_files(args.source_patch_dir)
    src_patch_like = []
    bad_src = 0
    for p in src_files:
        try:
            src_patch_like.append((p, *parse_patch_name(p)))
        except ValueError:
            bad_src += 1
    if bad_src:
        print(f"[WARN] skipped {bad_src} non patch-like source files")

    long_rows = []
    top1_rows = []
    by_patch_json = {}

    stats_total = 0
    stats_no_full = 0
    stats_no_patch = 0

    for src_path, src_pref, src_idx in src_patch_like:
        stats_total += 1
        cands = full_map.get(src_pref, [])
        if not cands:
            stats_no_full += 1
            by_patch_json[src_path.name] = {
                "source_prefix": src_pref,
                "source_idx": src_idx,
                "candidates": [],
                "note": "no_full_match_for_prefix",
            }
            continue

        picked = []
        for c in cands:
            if c["score"] < args.min_score:
                continue
            tgt_pref = c["target_prefix"]
            tgt_patch = tgt_index.get(tgt_pref, {}).get(src_idx)
            exists = tgt_patch is not None
            row = {
                "source_patch": src_path.name,
                "source_patch_path": str(src_path),
                "source_prefix": src_pref,
                "source_idx": src_idx,
                "rank": c["rank"],
                "full_target_prefix": tgt_pref,
                "full_score": c["score"],
                "target_patch": tgt_patch.name if exists else "",
                "target_patch_path": str(tgt_patch) if exists else "",
                "exists": int(exists),
            }
            if exists:
                picked.append(row)
            long_rows.append(row)

        if not picked:
            stats_no_patch += 1
            by_patch_json[src_path.name] = {
                "source_prefix": src_pref,
                "source_idx": src_idx,
                "candidates": [],
                "note": "no_target_patch_found_for_same_idx",
            }
            continue

        s = sum(max(0.0, r["full_score"]) for r in picked)
        if s <= 0.0:
            w = 1.0 / len(picked)
            for r in picked:
                r["weight_norm"] = w
        else:
            for r in picked:
                r["weight_norm"] = max(0.0, r["full_score"]) / s

        picked_sorted = sorted(picked, key=lambda x: x["weight_norm"], reverse=True)
        best = picked_sorted[0]
        top1_rows.append(
            {
                "source_patch": best["source_patch"],
                "source_patch_path": best["source_patch_path"],
                "source_prefix": best["source_prefix"],
                "source_idx": best["source_idx"],
                "target_patch": best["target_patch"],
                "target_patch_path": best["target_patch_path"],
                "target_prefix": best["full_target_prefix"],
                "weight_norm": best["weight_norm"],
                "full_score": best["full_score"],
            }
        )

        by_patch_json[src_path.name] = {
            "source_prefix": src_pref,
            "source_idx": src_idx,
            "candidates": [
                {
                    "rank": r["rank"],
                    "target_prefix": r["full_target_prefix"],
                    "target_patch": r["target_patch"],
                    "target_patch_path": r["target_patch_path"],
                    "full_score": r["full_score"],
                    "weight_norm": r["weight_norm"],
                }
                for r in picked_sorted
            ],
        }

    # Write long CSV
    long_csv = args.output_dir / "patch_ot_weights_topk_long.csv"
    with open(long_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "source_patch",
            "source_patch_path",
            "source_prefix",
            "source_idx",
            "rank",
            "full_target_prefix",
            "full_score",
            "target_patch",
            "target_patch_path",
            "exists",
            "weight_norm",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in long_rows:
            rr = dict(r)
            rr.setdefault("weight_norm", "")
            w.writerow(rr)

    # Write top1 CSV
    top1_csv = args.output_dir / "patch_ot_weights_top1.csv"
    with open(top1_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "source_patch",
            "source_patch_path",
            "source_prefix",
            "source_idx",
            "target_patch",
            "target_patch_path",
            "target_prefix",
            "weight_norm",
            "full_score",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(top1_rows)

    # Write JSON
    json_path = args.output_dir / "patch_ot_weights_by_source.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(by_patch_json, f, indent=2)

    summary = {
        "total_source_patches": stats_total,
        "no_full_prefix_match": stats_no_full,
        "no_target_patch_same_idx": stats_no_patch,
        "resolved_patches": stats_total - stats_no_full - stats_no_patch,
        "topk": args.topk,
        "min_score": args.min_score,
        "full_topk_csv": str(args.full_topk_csv),
        "source_patch_dir": str(args.source_patch_dir),
        "target_patch_dir": str(args.target_patch_dir),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"- {long_csv}")
    print(f"- {top1_csv}")
    print(f"- {json_path}")
    print(f"- {args.output_dir / 'summary.json'}")
    print(summary)


if __name__ == "__main__":
    main()
