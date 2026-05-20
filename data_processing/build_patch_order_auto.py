#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


ImageLike = np.ndarray


_RAW_PIPELINE_MODULE = None
_RAW_PIPELINE_IMPORT_FAILED = False
_WARNED_RAW_IMPORT_FALLBACK = False
_WARNED_FFD_SKIP = False


def _get_raw_pipeline_module():
    global _RAW_PIPELINE_MODULE, _RAW_PIPELINE_IMPORT_FAILED
    if _RAW_PIPELINE_MODULE is not None:
        return _RAW_PIPELINE_MODULE
    if _RAW_PIPELINE_IMPORT_FAILED:
        return None

    repo_root = Path(__file__).resolve().parents[1]

    # Preferred path: import raw_pipeline from repo_root/model
    model_dir = repo_root / "model"
    if model_dir.exists() and str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    try:
        import raw_pipeline as rp  # type: ignore
        _RAW_PIPELINE_MODULE = rp
        return _RAW_PIPELINE_MODULE
    except Exception:
        pass

    # Fallback: load raw_pipeline.py directly from file.
    raw_pipeline_py = model_dir / "raw_pipeline.py"
    if raw_pipeline_py.exists():
        spec = importlib.util.spec_from_file_location("raw_pipeline_local", raw_pipeline_py)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                _RAW_PIPELINE_MODULE = module
                return _RAW_PIPELINE_MODULE
            except Exception:
                pass

    _RAW_PIPELINE_IMPORT_FAILED = True
    _RAW_PIPELINE_MODULE = None
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Auto-build patch order metadata from patch files named '<prefix>_<idx>.<ext>'. "
            "Outputs patch_order_all.json, patch_order/*.json, prefix_count.csv, layout_scores.csv, "
            "low_confidence_prefixes.txt, and optional reconstructions."
        )
    )
    p.add_argument("--patch_root", type=Path, required=True, help="Input patch folder (raw npy or jpeg)")
    p.add_argument("--output_root", type=Path, required=True, help="Output root folder (e.g., raw_auto_val)")
    p.add_argument("--kind", choices=["raw4", "jpeg"], required=True, help="Patch type")

    p.add_argument(
        "--layout_map",
        type=str,
        default="42:6x7,7x6;35:5x7,7x5;14:2x7,7x2",
        help="Count-to-layout candidates. Example: '42:6x7,7x6;35:5x7,7x5;14:2x7,7x2'",
    )
    p.add_argument(
        "--fallback_all_factor_pairs",
        action="store_true",
        help="If count is missing in layout_map, try all (rows,cols) factor pairs.",
    )
    p.add_argument(
        "--low_confidence_threshold",
        type=float,
        default=0.05,
        help="Low-confidence threshold on normalized score margin between best and second layout.",
    )
    # Backward-compatible alias.
    p.add_argument(
        "--low_conf_margin_norm",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--max_prefixes", type=int, default=0, help="Optional limit for quick runs (0 = all)")

    p.add_argument("--save_raw4", action="store_true", help="When kind=raw4, save stitched raw4 npy under output_root/raw4")
    p.add_argument("--save_preview", action="store_true", help="When kind=raw4, save preview png under output_root/preview")
    p.add_argument("--save_jpeg", action="store_true", help="When kind=jpeg, save stitched jpg under output_root/reconstructed")

    p.add_argument(
        "--preview_demosaic",
        choices=["opencv", "opencv_ea", "demosaicnet"],
        default="opencv",
        help="Preview demosaic method for raw4 preview png.",
    )
    p.add_argument(
        "--preview_denoise_method",
        choices=["none", "ffdnet"],
        default="none",
        help="Optional preview denoising after demosaic.",
    )
    p.add_argument("--preview_ffdnet_noise_sigma", type=float, default=0.01)
    p.add_argument(
        "--preview_ffdnet_weights",
        type=str,
        default="model/third_party/ffdnet_pytorch/models/net_rgb.pth",
    )
    p.add_argument("--preview_ffdnet_device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--preview_black_level", type=float, default=0.0)
    p.add_argument("--preview_white_level", type=float, default=1023.0)
    p.add_argument("--preview_gamma", type=float, default=2.2)
    args = p.parse_args()
    if args.low_conf_margin_norm is not None:
        args.low_confidence_threshold = float(args.low_conf_margin_norm)
    return args


def parse_layout_map(spec: str) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, List[Tuple[int, int]]] = {}
    if not spec.strip():
        return out
    for block in spec.split(";"):
        block = block.strip()
        if not block:
            continue
        cnt_s, layouts_s = block.split(":", 1)
        cnt = int(cnt_s.strip())
        cand: List[Tuple[int, int]] = []
        for pair in layouts_s.split(","):
            pair = pair.strip().lower()
            if not pair:
                continue
            r_s, c_s = pair.split("x", 1)
            cand.append((int(r_s), int(c_s)))
        if cand:
            out[cnt] = cand
    return out


def list_patch_files(root: Path, kind: str) -> List[Path]:
    if kind == "raw4":
        files = sorted(root.glob("*.npy"))
    else:
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            files.extend(root.glob(ext))
        files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No patch files found in {root} for kind={kind}")
    return files


def parse_prefix_index(path: Path) -> Tuple[str, int]:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Invalid patch name (missing underscore): {path.name}")
    pref, idx_s = stem.rsplit("_", 1)
    return pref, int(idx_s)


def group_by_prefix(paths: Iterable[Path]) -> Dict[str, List[Tuple[int, Path]]]:
    groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    bad = 0
    for p in paths:
        try:
            pref, idx = parse_prefix_index(p)
            groups[pref].append((idx, p))
        except Exception:
            bad += 1
    if bad > 0:
        print(f"[WARN] skipped files with invalid naming: {bad}")
    for pref in groups:
        groups[pref].sort(key=lambda t: t[0])
    return groups


def all_factor_pairs(n: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for r in range(1, int(math.sqrt(n)) + 1):
        if n % r != 0:
            continue
        c = n // r
        pairs.append((r, c))
        if r != c:
            pairs.append((c, r))
    # Prefer less degenerate shapes first.
    pairs.sort(key=lambda rc: (abs(rc[0] - rc[1]), -min(rc[0], rc[1])))
    return pairs


def load_patch(path: Path, kind: str) -> ImageLike:
    if kind == "raw4":
        arr = np.load(str(path))
        if arr.ndim != 3 or arr.shape[-1] != 4:
            raise ValueError(f"Expected raw4 patch (H,W,4), got {arr.shape}: {path}")
        return arr

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def to_score_plane(tile: ImageLike, kind: str) -> np.ndarray:
    x = tile.astype(np.float32)
    if kind == "raw4":
        # Legacy behavior: seam matching on green proxy (Gr/Gb mean).
        x = (x[..., 1] + x[..., 2]) * 0.5
    else:
        # BGR->gray-ish average (channel order doesn't matter for seam score).
        x = x.mean(axis=2)

    return x


def seam_score_rowmajor(tiles: List[ImageLike], rows: int, cols: int, kind: str) -> float:
    n = len(tiles)
    if rows * cols != n:
        return float("inf")

    planes = [to_score_plane(t, kind) for t in tiles]
    h, w = planes[0].shape
    for p in planes[1:]:
        if p.shape != (h, w):
            return float("inf")

    total = 0.0
    count = 0
    k = max(1, min(4, h, w))

    # Horizontal seams.
    for r in range(rows):
        for c in range(cols - 1):
            i0 = r * cols + c
            i1 = r * cols + (c + 1)
            left = planes[i0][:, -k:]
            right = planes[i1][:, :k]
            total += float(np.mean(np.abs(left - right)))
            count += 1

    # Vertical seams.
    for r in range(rows - 1):
        for c in range(cols):
            i0 = r * cols + c
            i1 = (r + 1) * cols + c
            top = planes[i0][-k:, :]
            bot = planes[i1][:k, :]
            total += float(np.mean(np.abs(top - bot)))
            count += 1

    if count == 0:
        return float("inf")
    return total / count


def choose_layout(
    tiles: List[ImageLike],
    count: int,
    kind: str,
    layout_map: Dict[int, List[Tuple[int, int]]],
    fallback_all_factor_pairs: bool,
) -> Tuple[Tuple[int, int], float, Tuple[int, int], float, List[Tuple[Tuple[int, int], float]]]:
    candidates = list(layout_map.get(count, []))
    if not candidates and fallback_all_factor_pairs:
        candidates = all_factor_pairs(count)
    if not candidates:
        # Final fallback keeps deterministic output.
        candidates = [(1, count)]

    scored: List[Tuple[Tuple[int, int], float]] = []
    for rc in candidates:
        r, c = rc
        sc = seam_score_rowmajor(tiles, r, c, kind)
        scored.append((rc, sc))

    scored.sort(key=lambda x: x[1])
    best_layout, best_score = scored[0]
    if len(scored) > 1:
        second_layout, second_score = scored[1]
    else:
        second_layout, second_score = best_layout, float("inf")
    return best_layout, best_score, second_layout, second_score, scored


def build_order(paths_sorted: List[Path], rows: int, cols: int) -> List[dict]:
    order = []
    for k, p in enumerate(paths_sorted):
        r = k // cols
        c = k % cols
        order.append(
            {
                "k": int(k),
                "row": int(r),
                "col": int(c),
                "patch_index": int(p.stem.rsplit("_", 1)[1]),
                "filename": p.name,
            }
        )
    return order


def stitch_tiles_rowmajor(tiles: List[ImageLike], rows: int, cols: int) -> ImageLike:
    h, w = tiles[0].shape[:2]
    if tiles[0].ndim == 3:
        ch = tiles[0].shape[2]
        out = np.zeros((rows * h, cols * w, ch), dtype=tiles[0].dtype)
        for k, tile in enumerate(tiles):
            r = k // cols
            c = k % cols
            out[r * h : (r + 1) * h, c * w : (c + 1) * w, :] = tile
        return out

    out2 = np.zeros((rows * h, cols * w), dtype=tiles[0].dtype)
    for k, tile in enumerate(tiles):
        r = k // cols
        c = k % cols
        out2[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    return out2


def raw4_to_mosaic(raw4: np.ndarray) -> np.ndarray:
    h, w, _ = raw4.shape
    m = np.empty((h * 2, w * 2), dtype=raw4.dtype)
    m[0::2, 0::2] = raw4[..., 0]  # R
    m[0::2, 1::2] = raw4[..., 1]  # Gr
    m[1::2, 0::2] = raw4[..., 2]  # Gb
    m[1::2, 1::2] = raw4[..., 3]  # B
    return m


def save_raw_preview(
    raw4: np.ndarray,
    out_png: Path,
    black_level: float,
    white_level: float,
    gamma: float,
    demosaic: str,
    denoise_method: str = "none",
    ffdnet_noise_sigma: float = 0.01,
    ffdnet_weights: str = "model/third_party/ffdnet_pytorch/models/net_rgb.pth",
    ffdnet_device: str = "cpu",
) -> None:
    global _WARNED_RAW_IMPORT_FALLBACK, _WARNED_FFD_SKIP
    m = raw4_to_mosaic(raw4).astype(np.float32)
    denom = max(float(white_level - black_level), 1e-6)
    m = np.clip((m - float(black_level)) / denom, 0.0, 1.0)

    if demosaic == "demosaicnet":
        rp = _get_raw_pipeline_module()
        if rp is None:
            if not _WARNED_RAW_IMPORT_FALLBACK:
                print("[WARN] Could not import raw_pipeline. Falling back to OpenCV demosaic.")
                _WARNED_RAW_IMPORT_FALLBACK = True
            m16 = (m * 65535.0).astype(np.uint16)
            rgb = cv2.cvtColor(m16, cv2.COLOR_BAYER_RG2RGB).astype(np.float32) / 65535.0
        else:
            try:
                rgb = rp.demosaic_rggb(m, method="demosaicnet")
            except Exception as e:
                print(f"[WARN] demosaicnet preview failed ({e}). Falling back to OpenCV demosaic.")
                m16 = (m * 65535.0).astype(np.uint16)
                rgb = cv2.cvtColor(m16, cv2.COLOR_BAYER_RG2RGB).astype(np.float32) / 65535.0
    else:
        m16 = (m * 65535.0).astype(np.uint16)
        if demosaic == "opencv_ea" and hasattr(cv2, "COLOR_BayerRGGB2RGB_EA"):
            code = cv2.COLOR_BayerRGGB2RGB_EA
        else:
            code = cv2.COLOR_BAYER_RG2RGB
        rgb = cv2.cvtColor(m16, code).astype(np.float32) / 65535.0

    rgb = np.clip(rgb, 0.0, 1.0)

    if denoise_method == "ffdnet":
        rp = _get_raw_pipeline_module()
        if rp is None:
            if not _WARNED_FFD_SKIP:
                print("[WARN] Could not import raw_pipeline. Skip FFDNet preview denoise.")
                _WARNED_FFD_SKIP = True
        else:
            try:
                rgb = rp.ffdnet_denoise_rgb_np(
                    rgb,
                    noise_sigma=float(ffdnet_noise_sigma),
                    weights_path=str(ffdnet_weights),
                    device=str(ffdnet_device),
                )
                rgb = np.clip(rgb, 0.0, 1.0)
            except Exception as e:
                print(f"[WARN] FFDNet preview denoise failed ({e}). Continue without denoise.")

    rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / max(float(gamma), 1e-6))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    bgr8 = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_png), bgr8)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    files = list_patch_files(args.patch_root, args.kind)
    groups = group_by_prefix(files)

    prefixes = sorted(groups.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if args.max_prefixes > 0:
        prefixes = prefixes[: args.max_prefixes]

    layout_map = parse_layout_map(args.layout_map)

    patch_order_dir = args.output_root / "patch_order"
    patch_order_dir.mkdir(parents=True, exist_ok=True)

    raw4_dir = args.output_root / "raw4"
    preview_dir = args.output_root / "preview"
    recon_dir = args.output_root / "reconstructed"
    if args.save_raw4:
        raw4_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)
    if args.save_jpeg:
        recon_dir.mkdir(parents=True, exist_ok=True)

    prefix_counts: List[Tuple[str, int]] = []
    layout_rows: List[dict] = []
    low_conf: List[str] = []
    all_meta: Dict[str, dict] = {}

    for i, pref in enumerate(prefixes, start=1):
        idx_paths = groups[pref]
        idx_paths = sorted(idx_paths, key=lambda t: t[0])
        count = len(idx_paths)
        prefix_counts.append((pref, count))

        # Require contiguous indices for deterministic row-major assignment.
        idxs = [t[0] for t in idx_paths]
        contiguous = (idxs == list(range(count)))

        try:
            tiles = [load_patch(p, args.kind) for _, p in idx_paths]
        except Exception as e:
            print(f"[FAIL] {pref}: failed to load patches ({e})")
            low_conf.append(pref)
            continue

        best_layout, best_score, second_layout, second_score, scored = choose_layout(
            tiles,
            count,
            args.kind,
            layout_map,
            args.fallback_all_factor_pairs,
        )

        margin = float(second_score - best_score) if np.isfinite(second_score) else float("inf")
        # Legacy confidence used in previous script.
        confidence = margin / max(float(second_score), 1e-8) if np.isfinite(second_score) else float("inf")
        if confidence < float(args.low_confidence_threshold):
            low_conf.append(pref)

        rows, cols = best_layout
        paths_sorted = [p for _, p in idx_paths]
        order = build_order(paths_sorted, rows, cols)

        meta = {
            "prefix": pref,
            "layout": {"rows": int(rows), "cols": int(cols)},
            "alt_layout": {"rows": int(second_layout[0]), "cols": int(second_layout[1])},
            "chosen_score": float(best_score),
            "alt_score": float(second_score),
            "confidence": float(confidence),
            "scan_order": "row_major",
            "num_patches": int(count),
            "order": order,
        }
        all_meta[pref] = meta

        with open(patch_order_dir / f"{pref}.json", "w") as f:
            json.dump(meta, f, indent=2)

        layout_rows.append(
            {
                "prefix": pref,
                "count": count,
                "chosen_layout": f"{rows}x{cols}",
                "chosen_score": float(best_score),
                "alt_layout": f"{second_layout[0]}x{second_layout[1]}",
                "alt_score": float(second_score),
                "margin": float(margin),
                "confidence": float(confidence),
                "contiguous_indices": int(contiguous),
                "candidates": ";".join(f"{rc[0]}x{rc[1]}:{sc:.8f}" for rc, sc in scored),
            }
        )

        if args.kind == "raw4" and (args.save_raw4 or args.save_preview):
            stitched = stitch_tiles_rowmajor(tiles, rows, cols)
            if args.save_raw4:
                np.save(raw4_dir / f"{pref}_raw4_{rows}x{cols}.npy", stitched)
            if args.save_preview:
                save_raw_preview(
                    stitched,
                    preview_dir / f"{pref}_preview_{rows}x{cols}.png",
                    black_level=args.preview_black_level,
                    white_level=args.preview_white_level,
                    gamma=args.preview_gamma,
                    demosaic=args.preview_demosaic,
                    denoise_method=args.preview_denoise_method,
                    ffdnet_noise_sigma=args.preview_ffdnet_noise_sigma,
                    ffdnet_weights=args.preview_ffdnet_weights,
                    ffdnet_device=args.preview_ffdnet_device,
                )
        elif args.kind == "jpeg" and args.save_jpeg:
            stitched = stitch_tiles_rowmajor(tiles, rows, cols)
            cv2.imwrite(str(recon_dir / f"{pref}_reconstructed_{rows}x{cols}.jpg"), stitched)

        if i % 50 == 0 or i == len(prefixes):
            print(f"processed {i}/{len(prefixes)}")

    # Save patch_order_all.json
    with open(args.output_root / "patch_order_all.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    # Save prefix_count.csv
    with open(args.output_root / "prefix_count.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prefix", "count"])
        for pref, cnt in prefix_counts:
            w.writerow([pref, cnt])

    # Save layout_scores.csv
    with open(args.output_root / "layout_scores.csv", "w", newline="") as f:
        fieldnames = [
            "prefix",
            "count",
            "chosen_layout",
            "chosen_score",
            "alt_layout",
            "alt_score",
            "confidence",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in layout_rows:
            w.writerow(
                {
                    "prefix": row["prefix"],
                    "count": row["count"],
                    "chosen_layout": row["chosen_layout"],
                    "chosen_score": row["chosen_score"],
                    "alt_layout": row["alt_layout"],
                    "alt_score": row["alt_score"],
                    "confidence": row["confidence"],
                }
            )

    # Save low_confidence_prefixes.txt
    with open(args.output_root / "low_confidence_prefixes.txt", "w") as f:
        for pref in sorted(set(low_conf), key=lambda x: int(x) if x.isdigit() else x):
            f.write(f"{pref}\n")

    print("Done.")
    print(f"output_root: {args.output_root}")
    print(f"prefixes: {len(prefixes)}")
    print(f"low_confidence: {len(set(low_conf))}")


if __name__ == "__main__":
    main()
