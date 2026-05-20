#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from losses import load_dinov2_model, semantic_descriptors_dinov2  # noqa: E402
from optimal_transport import sinkhorn_transport_plan  # noqa: E402

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_prefix(stem_or_path: str) -> str:
    stem = Path(stem_or_path).stem
    m = re.match(r"^(\d+)", stem)
    if not m:
        raise ValueError(f"Cannot parse numeric prefix from: {stem_or_path}")
    return m.group(1).zfill(3)


def parse_patch_name(path: Path):
    m = re.match(r"^(\d+)_(\d+)$", path.stem)
    if not m:
        return None
    return m.group(1).zfill(3), int(m.group(2))


def list_patch_files(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    files.sort()
    return files


def load_rgb01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def to_tensor_batch(paths: List[Path], image_size: int) -> torch.Tensor:
    out = []
    for p in paths:
        img = load_rgb01(p)
        if image_size > 0:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        out.append(torch.from_numpy(img).permute(2, 0, 1).float())
    return torch.stack(out, dim=0)


def extract_dino_desc(
    paths: List[Path],
    device: torch.device,
    image_size: int,
    batch_size: int,
    model_name: str,
) -> torch.Tensor:
    model = load_dinov2_model(model_name=model_name).to(device).eval()
    chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="extract_dino_patch"):
            bpaths = paths[i : i + batch_size]
            x = to_tensor_batch(bpaths, image_size=image_size).to(device, non_blocking=True)
            z = semantic_descriptors_dinov2(x, model, resize=image_size if image_size > 0 else 224)
            chunks.append(z.cpu().float())
            del x, z
    return torch.cat(chunks, dim=0)


def parse_rank_prior(rank_prior_csv: str, topk: int) -> Optional[np.ndarray]:
    """
    Parse rank prior weights for full-image top-k entries.
    Returns normalized weights of shape [topk] or None.
    """
    prior_raw = str(rank_prior_csv).strip()
    if not prior_raw:
        if topk == 4:
            return np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        return None

    parts = [p.strip() for p in prior_raw.split(",") if p.strip()]
    if len(parts) != topk:
        raise ValueError(
            f"--full_rank_prior must contain exactly topk_full={topk} values, got {len(parts)}: {parts}"
        )

    try:
        arr = np.asarray([float(x) for x in parts], dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"Invalid --full_rank_prior values: {parts}") from e

    if not np.isfinite(arr).all():
        raise ValueError("--full_rank_prior contains non-finite values")
    if (arr < 0).any():
        raise ValueError("--full_rank_prior must be non-negative")
    s = float(arr.sum())
    if s <= 0:
        raise ValueError("--full_rank_prior sum must be > 0")
    return arr / s


def load_full_topk(
    full_topk_csv: Path,
    topk: int,
    full_weight_mode: str = "score_norm",
    full_rank_prior: Optional[np.ndarray] = None,
) -> Dict[str, List[dict]]:
    per_source = defaultdict(list)
    with open(full_topk_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"source_path", "target_path", "rank", "score"}
        miss = need.difference(r.fieldnames or [])
        if miss:
            raise ValueError(f"Missing columns in {full_topk_csv}: {sorted(miss)}")

        for row in r:
            rank = int(row["rank"])
            if rank > topk:
                continue
            s_pref = parse_prefix(row["source_path"])
            t_pref = parse_prefix(row["target_path"])
            per_source[s_pref].append(
                {
                    "rank": rank,
                    "target_prefix": t_pref,
                    "score": float(row["score"]),
                    "target_path": row["target_path"],
                }
            )

    # sort + renorm per source prefix
    out = {}
    for s_pref, rows in per_source.items():
        rows = sorted(rows, key=lambda x: x["rank"])
        if full_weight_mode == "score_norm":
            v = np.array([max(0.0, r["score"]) for r in rows], dtype=np.float64)
        elif full_weight_mode == "rank_prior":
            if full_rank_prior is None:
                raise ValueError("full_rank_prior is required when full_weight_mode='rank_prior'")
            v = np.array(
                [
                    float(full_rank_prior[int(r["rank"]) - 1]) if 1 <= int(r["rank"]) <= topk else 0.0
                    for r in rows
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"Unsupported full_weight_mode: {full_weight_mode}")

        if v.sum() <= 0:
            w = np.full_like(v, 1.0 / max(len(v), 1))
        else:
            w = v / v.sum()
        for i, wi in enumerate(w.tolist()):
            rows[i]["weight_norm"] = float(wi)
        out[s_pref] = rows
    return out


def build_prefix_index(paths: List[Path]):
    by_prefix = defaultdict(list)
    prefixes = []
    idxs = []
    valid_paths = []

    for i, p in enumerate(paths):
        parsed = parse_patch_name(p)
        if parsed is None:
            continue
        pref, pidx = parsed
        by_prefix[pref].append(len(valid_paths))
        prefixes.append(pref)
        idxs.append(pidx)
        valid_paths.append(p)

    return valid_paths, np.array(prefixes, dtype=object), np.array(idxs, dtype=np.int32), by_prefix


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build all patch-level OT weights from full-image top-k matches. "
            "For each source patch, candidates are all patches from target prefixes selected by full-image top-k."
        )
    )
    ap.add_argument("--full_topk_csv", type=Path, required=True)
    ap.add_argument("--source_patch_dir", type=Path, required=True)
    ap.add_argument("--target_patch_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--topk_full", type=int, default=10)
    ap.add_argument(
        "--full_weight_mode",
        type=str,
        default="score_norm",
        choices=["score_norm", "rank_prior"],
        help=(
            "How full-image top-k is converted to full_weight. "
            "score_norm: normalize clipped scores. "
            "rank_prior: ignore score values and use rank prior."
        ),
    )
    ap.add_argument(
        "--full_rank_prior",
        type=str,
        default="",
        help=(
            "Comma-separated rank prior weights used when --full_weight_mode rank_prior. "
            "Example for top4: '0.4,0.3,0.2,0.1'. "
            "If empty and topk_full=4, default prior 0.4,0.3,0.2,0.1 is used."
        ),
    )

    ap.add_argument("--dino_model_name", type=str, default="dinov2_vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--extract_batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--ot_reg", type=float, default=0.05)
    ap.add_argument("--ot_iters", type=int, default=20)
    ap.add_argument("--ot_temperature", type=float, default=1.0)
    ap.add_argument(
        "--blend_mode",
        type=str,
        default="multiply",
        choices=["multiply", "replace"],
        help="multiply: final = patch_ot * full_weight. replace: final = patch_ot only.",
    )

    ap.add_argument("--save_long_csv", action="store_true", help="Save full candidate table (can be large)")
    ap.add_argument(
        "--no_save_json",
        action="store_true",
        help="Disable JSON exports (full_topk_by_source_prefix.json, patch_ot_top1.json, source_index.json).",
    )
    ap.add_argument("--max_source_prefixes", type=int, default=0, help="debug limit (0=all)")

    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"device={device}")

    full_rank_prior = None
    if args.full_weight_mode == "rank_prior":
        full_rank_prior = parse_rank_prior(args.full_rank_prior, topk=args.topk_full)
        if full_rank_prior is None:
            raise ValueError(
                "--full_weight_mode rank_prior requires --full_rank_prior unless topk_full=4 "
                "(default prior 0.4,0.3,0.2,0.1)."
            )
        print(f"full rank prior={full_rank_prior.tolist()}")

    full_map = load_full_topk(
        args.full_topk_csv,
        topk=args.topk_full,
        full_weight_mode=args.full_weight_mode,
        full_rank_prior=full_rank_prior,
    )
    print(f"loaded full top-k for {len(full_map)} source prefixes")

    src_all = list_patch_files(args.source_patch_dir)
    tgt_all = list_patch_files(args.target_patch_dir)
    src_paths, src_prefix, src_idx, src_by_pref = build_prefix_index(src_all)
    tgt_paths, tgt_prefix, tgt_idx, tgt_by_pref = build_prefix_index(tgt_all)

    if not src_paths:
        raise RuntimeError("No valid source patch files with '<prefix>_<idx>' names found")
    if not tgt_paths:
        raise RuntimeError("No valid target patch files with '<prefix>_<idx>' names found")

    print(f"source patches={len(src_paths)}, target patches={len(tgt_paths)}")

    # Extract patch descriptors once.
    src_desc = extract_dino_desc(
        paths=src_paths,
        device=device,
        image_size=args.image_size,
        batch_size=args.extract_batch_size,
        model_name=args.dino_model_name,
    )
    tgt_desc = extract_dino_desc(
        paths=tgt_paths,
        device=device,
        image_size=args.image_size,
        batch_size=args.extract_batch_size,
        model_name=args.dino_model_name,
    )

    # Prepare source prefixes to process.
    source_prefixes = sorted(src_by_pref.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if args.max_source_prefixes > 0:
        source_prefixes = source_prefixes[: args.max_source_prefixes]

    # Flat sparse storage.
    offsets = [0]
    flat_target_ids: List[int] = []
    flat_full_rank: List[int] = []
    flat_full_score: List[float] = []
    flat_full_weight: List[float] = []
    flat_patch_weight: List[float] = []
    flat_final_weight: List[float] = []

    source_rows = []
    top1_rows = []
    long_rows = []

    eps = 1e-8
    processed_sources = 0
    skipped_no_full = 0
    skipped_no_target_pool = 0

    src_desc_cpu = src_desc.float().cpu()
    tgt_desc_cpu = tgt_desc.float().cpu()

    for pref in tqdm(source_prefixes, desc="prefix_groups"):
        s_ids = src_by_pref[pref]
        if pref not in full_map:
            # no full-image match for this prefix
            for sid in s_ids:
                offsets.append(offsets[-1])
                source_rows.append(
                    {
                        "source_id": sid,
                        "source_patch": src_paths[sid].name,
                        "source_patch_path": str(src_paths[sid]),
                        "source_prefix": pref,
                        "source_idx": int(src_idx[sid]),
                        "num_candidates": 0,
                        "note": "no_full_match",
                    }
                )
                skipped_no_full += 1
            continue

        # Candidate target pool = all patches from each full top-k target prefix.
        cand_prefix_entries = full_map[pref]
        cand_target_ids: List[int] = []
        cand_rank = []
        cand_score = []
        cand_weight = []

        for entry in cand_prefix_entries:
            tp = entry["target_prefix"]
            ids = tgt_by_pref.get(tp, [])
            if not ids:
                continue
            cand_target_ids.extend(ids)
            cand_rank.extend([int(entry["rank"])] * len(ids))
            cand_score.extend([float(entry["score"])] * len(ids))
            cand_weight.extend([float(entry["weight_norm"])] * len(ids))

        if len(cand_target_ids) == 0:
            for sid in s_ids:
                offsets.append(offsets[-1])
                source_rows.append(
                    {
                        "source_id": sid,
                        "source_patch": src_paths[sid].name,
                        "source_patch_path": str(src_paths[sid]),
                        "source_prefix": pref,
                        "source_idx": int(src_idx[sid]),
                        "num_candidates": 0,
                        "note": "no_target_patch_pool",
                    }
                )
                skipped_no_target_pool += 1
            continue

        # Compute group OT weights: [S, C]
        s_desc = src_desc_cpu[s_ids].to(device)
        c_desc = tgt_desc_cpu[cand_target_ids].to(device)
        d = torch.cdist(s_desc, c_desc, p=2).pow(2)  # [S, C]
        d = d / d.mean(dim=1, keepdim=True).clamp_min(eps)
        d = d / max(float(args.ot_temperature), eps)

        plan = sinkhorn_transport_plan(d, reg=float(args.ot_reg), iters=int(args.ot_iters), eps=eps)
        patch_w = plan / plan.sum(dim=1, keepdim=True).clamp_min(eps)  # row-stochastic [S, C]

        fw = torch.tensor(cand_weight, dtype=patch_w.dtype, device=patch_w.device).view(1, -1)
        if args.blend_mode == "multiply":
            final_w = patch_w * fw
            final_w = final_w / final_w.sum(dim=1, keepdim=True).clamp_min(eps)
        else:
            final_w = patch_w

        patch_w = patch_w.detach().cpu().numpy()
        final_w = final_w.detach().cpu().numpy()

        # Save per source patch sparse rows.
        for local_i, sid in enumerate(s_ids):
            processed_sources += 1
            start = offsets[-1]
            n_c = len(cand_target_ids)
            offsets.append(start + n_c)

            # flat arrays
            flat_target_ids.extend(cand_target_ids)
            flat_full_rank.extend(cand_rank)
            flat_full_score.extend(cand_score)
            flat_full_weight.extend(cand_weight)
            flat_patch_weight.extend(patch_w[local_i].tolist())
            flat_final_weight.extend(final_w[local_i].tolist())

            source_rows.append(
                {
                    "source_id": sid,
                    "source_patch": src_paths[sid].name,
                    "source_patch_path": str(src_paths[sid]),
                    "source_prefix": pref,
                    "source_idx": int(src_idx[sid]),
                    "num_candidates": n_c,
                    "note": "ok",
                }
            )

            # top1 for quick use
            best_j = int(np.argmax(final_w[local_i]))
            best_tid = int(cand_target_ids[best_j])
            top1_rows.append(
                {
                    "source_id": sid,
                    "source_patch": src_paths[sid].name,
                    "source_patch_path": str(src_paths[sid]),
                    "source_prefix": pref,
                    "source_idx": int(src_idx[sid]),
                    "target_id": best_tid,
                    "target_patch": tgt_paths[best_tid].name,
                    "target_patch_path": str(tgt_paths[best_tid]),
                    "target_prefix": str(tgt_prefix[best_tid]),
                    "target_idx": int(tgt_idx[best_tid]),
                    "full_rank": int(cand_rank[best_j]),
                    "full_score": float(cand_score[best_j]),
                    "full_weight": float(cand_weight[best_j]),
                    "patch_weight": float(patch_w[local_i, best_j]),
                    "final_weight": float(final_w[local_i, best_j]),
                }
            )

            if args.save_long_csv:
                for j, tid in enumerate(cand_target_ids):
                    long_rows.append(
                        {
                            "source_id": sid,
                            "source_patch": src_paths[sid].name,
                            "source_patch_path": str(src_paths[sid]),
                            "source_prefix": pref,
                            "source_idx": int(src_idx[sid]),
                            "target_id": int(tid),
                            "target_patch": tgt_paths[tid].name,
                            "target_patch_path": str(tgt_paths[tid]),
                            "target_prefix": str(tgt_prefix[tid]),
                            "target_idx": int(tgt_idx[tid]),
                            "full_rank": int(cand_rank[j]),
                            "full_score": float(cand_score[j]),
                            "full_weight": float(cand_weight[j]),
                            "patch_weight": float(patch_w[local_i, j]),
                            "final_weight": float(final_w[local_i, j]),
                        }
                    )

        del s_desc, c_desc, d, plan

    # Save compact sparse NPZ.
    np.savez_compressed(
        args.output_dir / "patch_ot_all_weights_sparse.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        target_ids=np.asarray(flat_target_ids, dtype=np.int64),
        full_rank=np.asarray(flat_full_rank, dtype=np.int16),
        full_score=np.asarray(flat_full_score, dtype=np.float32),
        full_weight=np.asarray(flat_full_weight, dtype=np.float32),
        patch_weight=np.asarray(flat_patch_weight, dtype=np.float32),
        final_weight=np.asarray(flat_final_weight, dtype=np.float32),
        source_paths=np.asarray([str(p) for p in src_paths], dtype=object),
        source_prefix=np.asarray(src_prefix, dtype=object),
        source_idx=np.asarray(src_idx, dtype=np.int32),
        target_paths=np.asarray([str(p) for p in tgt_paths], dtype=object),
        target_prefix=np.asarray(tgt_prefix, dtype=object),
        target_idx=np.asarray(tgt_idx, dtype=np.int32),
    )

    write_csv(
        args.output_dir / "source_index.csv",
        ["source_id", "source_patch", "source_patch_path", "source_prefix", "source_idx", "num_candidates", "note"],
        source_rows,
    )
    write_csv(
        args.output_dir / "patch_ot_top1.csv",
        [
            "source_id", "source_patch", "source_patch_path", "source_prefix", "source_idx",
            "target_id", "target_patch", "target_patch_path", "target_prefix", "target_idx",
            "full_rank", "full_score", "full_weight", "patch_weight", "final_weight",
        ],
        top1_rows,
    )

    if args.save_long_csv:
        write_csv(
            args.output_dir / "patch_ot_all_long.csv",
            [
                "source_id", "source_patch", "source_patch_path", "source_prefix", "source_idx",
                "target_id", "target_patch", "target_patch_path", "target_prefix", "target_idx",
                "full_rank", "full_score", "full_weight", "patch_weight", "final_weight",
            ],
            long_rows,
        )

    summary = {
        "full_topk_csv": str(args.full_topk_csv),
        "source_patch_dir": str(args.source_patch_dir),
        "target_patch_dir": str(args.target_patch_dir),
        "topk_full": int(args.topk_full),
        "full_weight_mode": args.full_weight_mode,
        "full_rank_prior": full_rank_prior.tolist() if full_rank_prior is not None else None,
        "blend_mode": args.blend_mode,
        "dino_model_name": args.dino_model_name,
        "image_size": int(args.image_size),
        "extract_batch_size": int(args.extract_batch_size),
        "ot_reg": float(args.ot_reg),
        "ot_iters": int(args.ot_iters),
        "ot_temperature": float(args.ot_temperature),
        "processed_sources": int(processed_sources),
        "skipped_no_full": int(skipped_no_full),
        "skipped_no_target_pool": int(skipped_no_target_pool),
        "num_source_patches": int(len(src_paths)),
        "num_target_patches": int(len(tgt_paths)),
        "num_sparse_edges": int(len(flat_target_ids)),
        "save_long_csv": bool(args.save_long_csv),
        "save_json": not bool(args.no_save_json),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_save_json:
        with open(args.output_dir / "full_topk_by_source_prefix.json", "w", encoding="utf-8") as f:
            json.dump(full_map, f, indent=2)

        with open(args.output_dir / "patch_ot_top1.json", "w", encoding="utf-8") as f:
            json.dump(top1_rows, f, indent=2)

        source_index_json = []
        for row in source_rows:
            sid = int(row["source_id"])
            source_index_json.append(
                {
                    "source_id": sid,
                    "source_patch": row["source_patch"],
                    "source_patch_path": row["source_patch_path"],
                    "source_prefix": row["source_prefix"],
                    "source_idx": int(row["source_idx"]),
                    "num_candidates": int(row["num_candidates"]),
                    "note": row["note"],
                    "sparse_start": int(offsets[sid]),
                    "sparse_end": int(offsets[sid + 1]),
                }
            )
        with open(args.output_dir / "source_index.json", "w", encoding="utf-8") as f:
            json.dump(source_index_json, f, indent=2)

    print("Done")
    print(f"- {args.output_dir / 'patch_ot_all_weights_sparse.npz'}")
    print(f"- {args.output_dir / 'source_index.csv'}")
    print(f"- {args.output_dir / 'patch_ot_top1.csv'}")
    if args.save_long_csv:
        print(f"- {args.output_dir / 'patch_ot_all_long.csv'}")
    if not args.no_save_json:
        print(f"- {args.output_dir / 'full_topk_by_source_prefix.json'}")
        print(f"- {args.output_dir / 'patch_ot_top1.json'}")
        print(f"- {args.output_dir / 'source_index.json'}")
    print(f"- {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
