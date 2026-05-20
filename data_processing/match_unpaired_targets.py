#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from losses import (
    VGG19StyleEncoder,
    gram_matrix,
    load_dinov2_model,
    semantic_descriptors_dinov2,
    semantic_gram_luv_descriptors,
)
from optimal_transport import sinkhorn_transport_plan, soft_pseudo_target_ot_FGW


TASK_PRESETS = {
    "patch_dino_ot": {
        "method": "dino_ot",
        "source_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/raws_png_dmnet_ffdnet",
        "target_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/jpegs",
    },
    "full_dino_ot": {
        "method": "dino_ot",
        "source_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/raw_auto_train/preview",
        "target_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/reconstructed_auto/chosen",
    },
    "patch_iwm": {
        "method": "iwm",
        "source_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/raws_png_dmnet_ffdnet",
        "target_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/jpegs",
    },
    "full_iwm": {
        "method": "iwm",
        "source_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/raw_auto_train/preview",
        "target_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/reconstructed_auto/chosen",
    },
    "patch_iwm_ot": {
        "method": "iwm_ot",
        "source_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/raws_png_dmnet_ffdnet",
        "target_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/jpegs",
    },
    "full_iwm_ot": {
        "method": "iwm_ot",
        "source_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/raw_auto_train/preview",
        "target_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/reconstructed_auto/chosen",
    },
    "patch_hybrid_ot": {
        "method": "hybrid_ot",
        "source_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/raws_png_dmnet_ffdnet",
        "target_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/jpegs",
    },
    "full_hybrid_ot": {
        "method": "hybrid_ot",
        "source_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/raw_auto_train/preview",
        "target_dir": "/lustre/fsn1/projects/rech/nkj/ubz77ta/Unpaired_ISP_train/train/reconstructed_auto/chosen",
    },
}

DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _paths_sha1(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _natural_key(path: Path):
    parts = re.split(r"(\d+)", path.as_posix())
    out = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def _list_images(root: Path, exts: set[str]) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=_natural_key)


def _load_rgb01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _to_tensor_batch(paths: list[Path], image_size: int) -> torch.Tensor:
    tensors = []
    for p in paths:
        img = _load_rgb01(p)
        if image_size > 0:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        tensors.append(torch.from_numpy(img).permute(2, 0, 1).float())
    return torch.stack(tensors, dim=0)


def _extract_dino_ot_descriptor_batch(
    batch_paths: list[Path],
    *,
    device: torch.device,
    image_size: int,
    include_hist: bool,
    style_encoder: torch.nn.Module,
    semantic_encoder: torch.nn.Module,
) -> torch.Tensor:
    x = _to_tensor_batch(batch_paths, image_size=image_size).to(device, non_blocking=True)
    if include_hist:
        d = semantic_gram_luv_descriptors(
            x,
            style_encoder=style_encoder,
            semantic_encoder=semantic_encoder,
            bins_l=64,
            bins_uv=32,
            sigma_l=0.02,
            sigma_uv=0.03,
        )
    else:
        # Patch-level matching: keep semantic + Gram only (no luma/chroma histogram).
        semantic_part = [semantic_descriptors_dinov2(x, semantic_encoder)]
        feats = style_encoder(x)
        gram_parts = []
        for f in feats:
            g = gram_matrix(f)
            a, b = torch.triu_indices(g.size(1), g.size(2), device=g.device)
            gram_parts.append(g[:, a, b])
        d = torch.cat([*semantic_part, *gram_parts], dim=1)
    return d


def _extract_dino_ot_descriptors(
    image_paths: list[Path],
    device: torch.device,
    image_size: int,
    batch_size: int,
    dino_model_name: str,
    include_hist: bool,
) -> torch.Tensor:
    style_encoder = VGG19StyleEncoder(pretrained=True).to(device).eval()
    semantic_encoder = load_dinov2_model(model_name=dino_model_name).to(device).eval()

    desc_cpu = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="extract_dino_ot"):
            batch_paths = image_paths[i : i + batch_size]
            d = _extract_dino_ot_descriptor_batch(
                batch_paths,
                device=device,
                image_size=image_size,
                include_hist=include_hist,
                style_encoder=style_encoder,
                semantic_encoder=semantic_encoder,
            )
            desc_cpu.append(d.cpu())
            del d
    return torch.cat(desc_cpu, dim=0).float()


def _extract_dino_ot_descriptors_to_memmap(
    image_paths: list[Path],
    *,
    device: torch.device,
    image_size: int,
    batch_size: int,
    dino_model_name: str,
    include_hist: bool,
    cache_dir: Path,
    cache_prefix: str,
    cache_dtype: str = "float16",
    resume: bool = True,
) -> np.ndarray:
    if not image_paths:
        raise ValueError("image_paths must be non-empty.")
    if cache_dtype not in ("float16", "float32"):
        raise ValueError(f"Unsupported cache_dtype: {cache_dtype}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    desc_path = cache_dir / f"{cache_prefix}_desc.npy"
    meta_path = cache_dir / f"{cache_prefix}_meta.json"
    state_path = cache_dir / f"{cache_prefix}_state.json"
    np_dtype = np.float16 if cache_dtype == "float16" else np.float32
    expected = {
        "n_images": len(image_paths),
        "paths_sha1": _paths_sha1(image_paths),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "dino_model_name": str(dino_model_name),
        "include_hist": bool(include_hist),
        "dtype": str(np.dtype(np_dtype)),
    }

    meta = _read_json(meta_path) if resume else None
    state = _read_json(state_path) if resume else None
    can_resume = (
        resume
        and desc_path.exists()
        and meta is not None
        and state is not None
        and meta.get("n_images") == expected["n_images"]
        and meta.get("paths_sha1") == expected["paths_sha1"]
        and meta.get("image_size") == expected["image_size"]
        and meta.get("dino_model_name") == expected["dino_model_name"]
        and meta.get("include_hist") == expected["include_hist"]
        and meta.get("dtype") == expected["dtype"]
    )

    if can_resume:
        mm = np.load(desc_path, mmap_mode="r+")
        next_index = int(state.get("next_index", 0))
        if mm.shape[0] != expected["n_images"]:
            can_resume = False
        elif next_index >= len(image_paths):
            print(f"[resume] {cache_prefix} descriptors already complete: {desc_path}")
            return np.load(desc_path, mmap_mode="r")

    if not can_resume:
        next_index = 0
        style_encoder = VGG19StyleEncoder(pretrained=True).to(device).eval()
        semantic_encoder = load_dinov2_model(model_name=dino_model_name).to(device).eval()
        with torch.no_grad():
            first_end = min(batch_size, len(image_paths))
            first_batch = image_paths[:first_end]
            first_desc = _extract_dino_ot_descriptor_batch(
                first_batch,
                device=device,
                image_size=image_size,
                include_hist=include_hist,
                style_encoder=style_encoder,
                semantic_encoder=semantic_encoder,
            )
            desc_dim = int(first_desc.shape[1])
            mm = np.lib.format.open_memmap(
                desc_path,
                mode="w+",
                dtype=np_dtype,
                shape=(len(image_paths), desc_dim),
            )
            mm[0:first_end, :] = first_desc.detach().cpu().numpy().astype(np_dtype, copy=False)
            mm.flush()
            _write_json(
                meta_path,
                {
                    **expected,
                    "desc_dim": desc_dim,
                    "cache_prefix": cache_prefix,
                },
            )
            next_index = first_end
            _write_json(
                state_path,
                {
                    "next_index": int(next_index),
                    "n_images": len(image_paths),
                    "cache_prefix": cache_prefix,
                },
            )
            del first_desc
    else:
        mm = np.load(desc_path, mmap_mode="r+")
        style_encoder = VGG19StyleEncoder(pretrained=True).to(device).eval()
        semantic_encoder = load_dinov2_model(model_name=dino_model_name).to(device).eval()
        print(f"[resume] {cache_prefix} descriptor extraction from index {next_index}/{len(image_paths)}")

    if next_index < len(image_paths):
        with torch.no_grad():
            for i in tqdm(range(next_index, len(image_paths), batch_size), desc=f"extract_dino_ot:{cache_prefix}"):
                j = min(i + batch_size, len(image_paths))
                batch_paths = image_paths[i:j]
                d = _extract_dino_ot_descriptor_batch(
                    batch_paths,
                    device=device,
                    image_size=image_size,
                    include_hist=include_hist,
                    style_encoder=style_encoder,
                    semantic_encoder=semantic_encoder,
                )
                mm[i:j, :] = d.detach().cpu().numpy().astype(np_dtype, copy=False)
                mm.flush()
                _write_json(
                    state_path,
                    {
                        "next_index": int(j),
                        "n_images": len(image_paths),
                        "cache_prefix": cache_prefix,
                    },
                )
                del d

    _write_json(
        state_path,
        {
            "next_index": len(image_paths),
            "n_images": len(image_paths),
            "cache_prefix": cache_prefix,
            "status": "done",
        },
    )
    return np.load(desc_path, mmap_mode="r")


def _load_iwm_encoder(
    code_iwm_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    if not code_iwm_root.exists():
        raise FileNotFoundError(f"code_iwm_root not found: {code_iwm_root}")
    if str(code_iwm_root) not in sys.path:
        sys.path.insert(0, str(code_iwm_root))
    src_root = code_iwm_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import src.deit as deit  # pylint: disable=import-error

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["meta"]["model_name"]
    crop_size = int(cfg["data"]["crop_size"])
    patch_size = int(cfg["mask"]["patch_size"])

    model = deit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        use_projector=False,
        use_xformers=False,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("target_encoder", ckpt)
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model = model.to(device).eval()
    return model, crop_size


def _extract_iwm_embeddings(
    image_paths: list[Path],
    device: torch.device,
    batch_size: int,
    code_iwm_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    image_size_override: int,
) -> torch.Tensor:
    model, crop_size = _load_iwm_encoder(
        code_iwm_root=code_iwm_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    image_size = image_size_override if image_size_override > 0 else crop_size

    desc_cpu = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="extract_iwm"):
            batch_paths = image_paths[i : i + batch_size]
            x = _to_tensor_batch(batch_paths, image_size=image_size).to(device, non_blocking=True)
            x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
            z = model(x)  # [B, N, D]
            z = z.mean(dim=1)  # [B, D]
            desc_cpu.append(z.cpu())
            del x, z
    return torch.cat(desc_cpu, dim=0).float()


def _cosine_topk(
    source_desc: torch.Tensor,
    target_desc: torch.Tensor,
    topk: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk = min(max(1, int(topk)), target_desc.shape[0])

    target = F.normalize(target_desc.to(device), dim=1)
    idx_out = []
    val_out = []
    with torch.no_grad():
        for i in tqdm(range(0, source_desc.shape[0], chunk_size), desc="match_cosine"):
            src = F.normalize(source_desc[i : i + chunk_size].to(device), dim=1)
            sim = src @ target.t()
            vals, idxs = torch.topk(sim, k=topk, dim=1, largest=True, sorted=True)
            idx_out.append(idxs.cpu())
            val_out.append(vals.cpu())
            del src, sim, vals, idxs
    return torch.cat(idx_out, dim=0), torch.cat(val_out, dim=0)


def _dino_ot_topk(
    source_desc: torch.Tensor,
    target_desc: torch.Tensor,
    topk: int,
    chunk_size: int,
    ot_target_pool: int,
    device: torch.device,
    ot_alpha: float,
    ot_reg: float,
    ot_temperature: float,
    ot_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk = min(max(1, int(topk)), target_desc.shape[0])
    n_src = source_desc.shape[0]
    n_tgt = target_desc.shape[0]

    src_cpu = source_desc.cpu()
    tgt_cpu = target_desc.cpu()

    idx_all = torch.empty((n_src, topk), dtype=torch.long)
    val_all = torch.empty((n_src, topk), dtype=torch.float32)
    warned_feature_ot = False

    def _feature_ot_weights(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.detach().float().flatten(1).contiguous()
        target = target.detach().float().flatten(1).contiguous()
        d_feat = torch.cdist(pred, target, p=2).pow(2)
        eps = 1e-8
        cost = d_feat / d_feat.mean().clamp_min(eps)
        cost = cost / max(ot_temperature, eps)
        plan = sinkhorn_transport_plan(cost, reg=ot_reg, iters=ot_iters, eps=eps)
        return plan / plan.sum(dim=1, keepdim=True).clamp_min(eps)

    with torch.no_grad():
        tgt_norm = F.normalize(tgt_cpu.to(device), dim=1)

        for i in tqdm(range(0, n_src, chunk_size), desc="match_dino_ot"):
            j = min(i + chunk_size, n_src)
            src_chunk = src_cpu[i:j]
            c = src_chunk.shape[0]

            if c == 0:
                continue

            if ot_target_pool <= 0 or ot_target_pool >= n_tgt:
                cand_idx = torch.arange(n_tgt, dtype=torch.long)
            else:
                src_norm = F.normalize(src_chunk.to(device), dim=1)
                sim = src_norm @ tgt_norm.t()
                cand_k = min(max(ot_target_pool, topk), n_tgt)
                _, cand_idx_dev = torch.topk(sim.mean(dim=0), k=cand_k, dim=0, largest=True, sorted=False)
                cand_idx = cand_idx_dev.cpu()
                del src_norm, sim, cand_idx_dev

            target_sub = tgt_cpu[cand_idx]
            use_fgw = (src_chunk.shape[0] == target_sub.shape[0]) and (src_chunk.shape[0] > 1)

            if use_fgw:
                w = soft_pseudo_target_ot_FGW(
                    src_chunk.to(device),
                    target_sub.to(device),
                    alpha=ot_alpha,
                    reg=ot_reg,
                    temperature=ot_temperature,
                    sinkhorn_iters=ot_iters,
                )
            else:
                if not warned_feature_ot:
                    print(
                        "[INFO] Using feature-OT fallback for unequal source/target candidate sizes "
                        f"(source_chunk={src_chunk.shape[0]}, target_pool={target_sub.shape[0]})."
                    )
                    warned_feature_ot = True
                w = _feature_ot_weights(src_chunk.to(device), target_sub.to(device))

            k_keep = min(topk, w.shape[1])
            vals, idx_local = torch.topk(w, k=k_keep, dim=1, largest=True, sorted=True)
            idx_global = cand_idx[idx_local.cpu()]
            idx_all[i:j, :k_keep] = idx_global
            val_all[i:j, :k_keep] = vals.cpu()

            if k_keep < topk:
                idx_all[i:j, k_keep:] = idx_global[:, -1:]
                val_all[i:j, k_keep:] = vals[:, -1:].cpu()

            del target_sub, w, vals, idx_local, idx_global

    return idx_all, val_all


def _dino_ot_topk_streaming_memmap(
    source_desc_mmap: np.ndarray,
    target_desc_mmap: np.ndarray,
    *,
    source_paths: list[Path],
    target_paths: list[Path],
    topk: int,
    chunk_size: int,
    ot_target_pool: int,
    target_scan_chunk_size: int,
    device: torch.device,
    ot_alpha: float,
    ot_reg: float,
    ot_temperature: float,
    ot_iters: int,
    cache_dir: Path,
    resume: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_src = int(source_desc_mmap.shape[0])
    n_tgt = int(target_desc_mmap.shape[0])
    if n_src != len(source_paths):
        raise ValueError(f"source descriptor count mismatch: {n_src} vs {len(source_paths)} paths")
    if n_tgt != len(target_paths):
        raise ValueError(f"target descriptor count mismatch: {n_tgt} vs {len(target_paths)} paths")
    if n_tgt <= 0:
        raise ValueError("target_desc_mmap is empty")
    if target_scan_chunk_size <= 0:
        raise ValueError("target_scan_chunk_size must be > 0")

    topk = min(max(1, int(topk)), n_tgt)
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx_path = cache_dir / "match_topk_idx.npy"
    score_path = cache_dir / "match_topk_score.npy"
    meta_path = cache_dir / "match_meta.json"
    state_path = cache_dir / "match_state.json"

    match_meta_expected = {
        "n_source": n_src,
        "n_target": n_tgt,
        "topk": int(topk),
        "chunk_size": int(chunk_size),
        "ot_target_pool": int(ot_target_pool),
        "target_scan_chunk_size": int(target_scan_chunk_size),
        "ot_alpha": float(ot_alpha),
        "ot_reg": float(ot_reg),
        "ot_temperature": float(ot_temperature),
        "ot_iters": int(ot_iters),
        "source_paths_sha1": _paths_sha1(source_paths),
        "target_paths_sha1": _paths_sha1(target_paths),
    }

    meta = _read_json(meta_path) if resume else None
    state = _read_json(state_path) if resume else None
    can_resume = (
        resume
        and idx_path.exists()
        and score_path.exists()
        and meta is not None
        and state is not None
        and all(meta.get(k) == v for k, v in match_meta_expected.items())
    )

    if can_resume:
        idx_mm = np.load(idx_path, mmap_mode="r+")
        score_mm = np.load(score_path, mmap_mode="r+")
        if idx_mm.shape != (n_src, topk) or score_mm.shape != (n_src, topk):
            can_resume = False
        else:
            start_src = int(state.get("next_source_index", 0))
            if start_src >= n_src:
                print(f"[resume] OT matching already complete: {idx_path}")
                return (
                    torch.from_numpy(np.asarray(idx_mm, dtype=np.int64)),
                    torch.from_numpy(np.asarray(score_mm, dtype=np.float32)),
                )
            print(f"[resume] OT matching from source index {start_src}/{n_src}")

    if not can_resume:
        idx_mm = np.lib.format.open_memmap(idx_path, mode="w+", dtype=np.int64, shape=(n_src, topk))
        score_mm = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(n_src, topk))
        idx_mm[:] = 0
        score_mm[:] = 0.0
        idx_mm.flush()
        score_mm.flush()
        start_src = 0
        _write_json(meta_path, match_meta_expected)
        _write_json(
            state_path,
            {
                "next_source_index": 0,
                "n_source": n_src,
                "status": "running",
            },
        )

    warned_feature_ot = False

    def _feature_ot_weights(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.detach().float().flatten(1).contiguous()
        target = target.detach().float().flatten(1).contiguous()
        d_feat = torch.cdist(pred, target, p=2).pow(2)
        eps = 1e-8
        cost = d_feat / d_feat.mean().clamp_min(eps)
        cost = cost / max(ot_temperature, eps)
        plan = sinkhorn_transport_plan(cost, reg=ot_reg, iters=ot_iters, eps=eps)
        return plan / plan.sum(dim=1, keepdim=True).clamp_min(eps)

    with torch.no_grad():
        for i in tqdm(range(start_src, n_src, chunk_size), desc="match_dino_ot_stream"):
            j = min(i + chunk_size, n_src)
            src_np = np.asarray(source_desc_mmap[i:j], dtype=np.float32)
            src_chunk = torch.from_numpy(src_np)
            c = src_chunk.shape[0]
            if c == 0:
                continue

            if ot_target_pool <= 0 or ot_target_pool >= n_tgt:
                cand_idx_np = np.arange(n_tgt, dtype=np.int64)
            else:
                src_norm = F.normalize(src_chunk.to(device), dim=1)
                score_all = np.empty(n_tgt, dtype=np.float32)
                for t0 in range(0, n_tgt, target_scan_chunk_size):
                    t1 = min(t0 + target_scan_chunk_size, n_tgt)
                    tgt_np = np.asarray(target_desc_mmap[t0:t1], dtype=np.float32)
                    tgt_chunk = torch.from_numpy(tgt_np).to(device)
                    tgt_norm = F.normalize(tgt_chunk, dim=1)
                    sim_mean = (src_norm @ tgt_norm.t()).mean(dim=0)
                    score_all[t0:t1] = sim_mean.detach().cpu().numpy()
                    del tgt_chunk, tgt_norm, sim_mean
                cand_k = min(max(ot_target_pool, topk), n_tgt)
                cand_idx_np = np.argpartition(score_all, -cand_k)[-cand_k:]
                cand_idx_np = cand_idx_np[np.argsort(score_all[cand_idx_np])[::-1]]
                del src_norm, score_all

            target_sub_np = np.asarray(target_desc_mmap[cand_idx_np], dtype=np.float32)
            target_sub = torch.from_numpy(target_sub_np)
            use_fgw = (src_chunk.shape[0] == target_sub.shape[0]) and (src_chunk.shape[0] > 1)

            if use_fgw:
                w = soft_pseudo_target_ot_FGW(
                    src_chunk.to(device),
                    target_sub.to(device),
                    alpha=ot_alpha,
                    reg=ot_reg,
                    temperature=ot_temperature,
                    sinkhorn_iters=ot_iters,
                )
            else:
                if not warned_feature_ot:
                    print(
                        "[INFO] Using feature-OT fallback for unequal source/target candidate sizes "
                        f"(source_chunk={src_chunk.shape[0]}, target_pool={target_sub.shape[0]})."
                    )
                    warned_feature_ot = True
                w = _feature_ot_weights(src_chunk.to(device), target_sub.to(device))

            cand_idx_t = torch.from_numpy(cand_idx_np.astype(np.int64, copy=False))
            k_keep = min(topk, w.shape[1])
            vals, idx_local = torch.topk(w, k=k_keep, dim=1, largest=True, sorted=True)
            idx_global = cand_idx_t[idx_local.cpu()]
            idx_np = idx_global.detach().cpu().numpy().astype(np.int64, copy=False)
            val_np = vals.detach().cpu().numpy().astype(np.float32, copy=False)

            idx_mm[i:j, :k_keep] = idx_np
            score_mm[i:j, :k_keep] = val_np
            if k_keep < topk:
                idx_mm[i:j, k_keep:] = idx_np[:, -1:]
                score_mm[i:j, k_keep:] = val_np[:, -1:]
            idx_mm.flush()
            score_mm.flush()

            _write_json(
                state_path,
                {
                    "next_source_index": int(j),
                    "n_source": n_src,
                    "status": "running",
                },
            )
            del src_chunk, target_sub, w, vals, idx_local, idx_global

    _write_json(
        state_path,
        {
            "next_source_index": n_src,
            "n_source": n_src,
            "status": "done",
        },
    )
    return (
        torch.from_numpy(np.asarray(idx_mm, dtype=np.int64)),
        torch.from_numpy(np.asarray(score_mm, dtype=np.float32)),
    )


def _write_csv_outputs(
    source_paths: list[Path],
    target_paths: list[Path],
    topk_idx: torch.Tensor,
    topk_score: torch.Tensor,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    long_csv = out_dir / "matches_topk_long.csv"
    with open(long_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_index", "source_path", "rank", "target_index", "target_path", "score"])
        for s_idx, s_path in enumerate(source_paths):
            for r in range(topk_idx.shape[1]):
                t_idx = int(topk_idx[s_idx, r].item())
                score = float(topk_score[s_idx, r].item())
                w.writerow([s_idx, str(s_path), r + 1, t_idx, str(target_paths[t_idx]), score])

    top1_csv = out_dir / "matches_top1.csv"
    with open(top1_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_index", "source_path", "target_index", "target_path", "score"])
        for s_idx, s_path in enumerate(source_paths):
            t_idx = int(topk_idx[s_idx, 0].item())
            score = float(topk_score[s_idx, 0].item())
            w.writerow([s_idx, str(s_path), t_idx, str(target_paths[t_idx]), score])

    wide_csv = out_dir / "matches_topk_wide.csv"
    with open(wide_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["source_index", "source_path"]
        for r in range(topk_idx.shape[1]):
            header += [f"target_index_r{r+1}", f"target_path_r{r+1}", f"score_r{r+1}"]
        w.writerow(header)
        for s_idx, s_path in enumerate(source_paths):
            row = [s_idx, str(s_path)]
            for r in range(topk_idx.shape[1]):
                t_idx = int(topk_idx[s_idx, r].item())
                score = float(topk_score[s_idx, r].item())
                row += [t_idx, str(target_paths[t_idx]), score]
            w.writerow(row)


def _fit_hwc(img: np.ndarray, out_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == out_h:
        return img
    out_w = max(1, int(round(w * (out_h / max(h, 1)))))
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _resize_to_hw(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == out_h and w == out_w:
        return img
    return cv2.resize(img, (max(1, out_w), max(1, out_h)), interpolation=cv2.INTER_AREA)


def _add_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _save_visualizations(
    source_paths: list[Path],
    target_paths: list[Path],
    topk_idx: torch.Tensor,
    out_dir: Path,
    viz_count: int,
    viz_label_mode: str = "full",
):
    if viz_count <= 0:
        return
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    n = min(viz_count, len(source_paths))
    for i in tqdm(range(n), desc="save_viz"):
        src = cv2.imread(str(source_paths[i]), cv2.IMREAD_COLOR)
        if src is None:
            continue
        src = _fit_hwc(src, 320)
        src_h, src_w = src.shape[:2]
        if viz_label_mode == "short":
            src_label = "source"
        else:
            src_label = f"source: {source_paths[i].name}"
        src = _add_label(src, src_label)
        panels = [src]

        for r in range(min(3, topk_idx.shape[1])):
            t_path = target_paths[int(topk_idx[i, r].item())]
            tgt = cv2.imread(str(t_path), cv2.IMREAD_COLOR)
            if tgt is None:
                continue
            tgt = _resize_to_hw(tgt, src_h, src_w)
            if viz_label_mode == "short":
                tgt_label = f"top{r+1}"
            else:
                tgt_label = f"top{r+1}: {t_path.name}"
            tgt = _add_label(tgt, tgt_label)
            panels.append(tgt)

        if not panels:
            continue
        strip = np.concatenate(panels, axis=1)
        out_path = viz_dir / f"{i:05d}_{source_paths[i].stem}.jpg"
        cv2.imwrite(str(out_path), strip)


def _parse_exts(value: str) -> set[str]:
    exts = set()
    for x in value.split(","):
        x = x.strip().lower()
        if not x:
            continue
        if not x.startswith("."):
            x = f".{x}"
        exts.add(x)
    return exts if exts else DEFAULT_EXTS


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Match unpaired source/target images for analysis (DINO+OT or IWM embedding).",
    )
    p.add_argument("--task", type=str, default="patch_dino_ot", choices=sorted(TASK_PRESETS.keys()))
    p.add_argument("--source_dir", type=Path, default=None, help="Override source directory from task preset.")
    p.add_argument("--target_dir", type=Path, default=None, help="Override target directory from task preset.")
    p.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["dino_ot", "iwm", "iwm_ot", "hybrid_ot"],
        help="Override matching method.",
    )
    p.add_argument("--output_dir", type=Path, default=None)

    p.add_argument("--source_exts", type=str, default=".jpg,.jpeg,.png,.bmp,.tif,.tiff")
    p.add_argument("--target_exts", type=str, default=".jpg,.jpeg,.png,.bmp,.tif,.tiff")
    p.add_argument("--max_sources", type=int, default=0, help="0 means all.")
    p.add_argument("--max_targets", type=int, default=0, help="0 means all.")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--viz_count", type=int, default=50)
    p.add_argument(
        "--viz_label_mode",
        type=str,
        default="full",
        choices=["full", "short"],
        help="Viz label style. full=include filename, short=source/top1/top2/top3 only.",
    )

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--extract_batch_size", type=int, default=32)
    p.add_argument("--match_chunk_size", type=int, default=64)
    p.add_argument("--target_scan_chunk_size", type=int, default=256, help="Target chunk size for streaming candidate scan in dino_ot.")
    p.add_argument("--image_size", type=int, default=224, help="<=0 uses model default for IWM.")
    p.add_argument(
        "--cache_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Descriptor cache precision for dino_ot resume mode.",
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Cache/checkpoint directory for dino_ot extraction+matching. Default: <output_dir>/.resume_cache",
    )
    p.add_argument(
        "--disable_resume",
        action="store_true",
        help="Disable cache resume. Recompute descriptors/matching from scratch.",
    )

    p.add_argument("--dino_model_name", type=str, default="dinov2_vits14")
    p.add_argument("--ot_target_pool", type=int, default=1024, help="Candidate target pool before OT when targets are large.")
    p.add_argument("--ot_alpha", type=float, default=0.5)
    p.add_argument("--ot_reg", type=float, default=0.05)
    p.add_argument("--ot_temperature", type=float, default=1.0)
    p.add_argument("--ot_iters", type=int, default=20)
    p.add_argument("--hybrid_iwm_weight", type=float, default=1.0)
    p.add_argument("--hybrid_dino_weight", type=float, default=1.0)

    p.add_argument("--iwm_root", type=Path, default=REPO_ROOT / "code_IWM")
    p.add_argument("--iwm_config", type=Path, default=REPO_ROOT / "code_IWM" / "configs" / "pretrain" / "default_equi.yaml")
    p.add_argument("--iwm_checkpoint", type=Path, default=REPO_ROOT / "code_IWM" / "checkpoint" / "iwm.pth.tar")
    return p


def _resolve_task(args: argparse.Namespace) -> dict:
    preset = TASK_PRESETS[args.task]
    method = args.method if args.method is not None else preset["method"]
    source_dir = args.source_dir if args.source_dir is not None else Path(preset["source_dir"])
    target_dir = args.target_dir if args.target_dir is not None else Path(preset["target_dir"])
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (REPO_ROOT / "results" / "pairing" / args.task)
    )
    return {
        "method": method,
        "source_dir": Path(source_dir),
        "target_dir": Path(target_dir),
        "output_dir": Path(output_dir),
    }


def _subset(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0:
        return paths
    return paths[: min(limit, len(paths))]


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    task_cfg = _resolve_task(args)

    source_dir: Path = task_cfg["source_dir"]
    target_dir: Path = task_cfg["target_dir"]
    output_dir: Path = task_cfg["output_dir"]
    method = task_cfg["method"]

    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"target_dir not found: {target_dir}")

    source_exts = _parse_exts(args.source_exts)
    target_exts = _parse_exts(args.target_exts)

    source_paths = _subset(_list_images(source_dir, source_exts), args.max_sources)
    target_paths = _subset(_list_images(target_dir, target_exts), args.max_targets)
    if not source_paths:
        raise FileNotFoundError(f"No source images found in: {source_dir}")
    if not target_paths:
        raise FileNotFoundError(f"No target images found in: {target_dir}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    print(f"task={args.task}")
    print(f"method={method}")
    print(f"source_dir={source_dir}")
    print(f"target_dir={target_dir}")
    print(f"n_source={len(source_paths)}")
    print(f"n_target={len(target_paths)}")
    print(f"device={device}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "dino_ot":
        include_hist_dino = not str(args.task).startswith("patch_")
        print(f"dino_descriptor_include_hist={include_hist_dino}")
        cache_dir = args.cache_dir if args.cache_dir is not None else (output_dir / ".resume_cache")
        resume_enabled = not args.disable_resume
        print(f"resume_cache_dir={cache_dir} (enabled={resume_enabled})")
        source_desc_mm = _extract_dino_ot_descriptors_to_memmap(
            image_paths=source_paths,
            device=device,
            image_size=args.image_size,
            batch_size=args.extract_batch_size,
            dino_model_name=args.dino_model_name,
            include_hist=include_hist_dino,
            cache_dir=cache_dir,
            cache_prefix="source",
            cache_dtype=args.cache_dtype,
            resume=resume_enabled,
        )
        target_desc_mm = _extract_dino_ot_descriptors_to_memmap(
            image_paths=target_paths,
            device=device,
            image_size=args.image_size,
            batch_size=args.extract_batch_size,
            dino_model_name=args.dino_model_name,
            include_hist=include_hist_dino,
            cache_dir=cache_dir,
            cache_prefix="target",
            cache_dtype=args.cache_dtype,
            resume=resume_enabled,
        )
        topk_idx, topk_score = _dino_ot_topk_streaming_memmap(
            source_desc_mmap=source_desc_mm,
            target_desc_mmap=target_desc_mm,
            source_paths=source_paths,
            target_paths=target_paths,
            topk=args.topk,
            chunk_size=args.match_chunk_size,
            ot_target_pool=args.ot_target_pool,
            target_scan_chunk_size=args.target_scan_chunk_size,
            device=device,
            ot_alpha=args.ot_alpha,
            ot_reg=args.ot_reg,
            ot_temperature=args.ot_temperature,
            ot_iters=args.ot_iters,
            cache_dir=cache_dir,
            resume=resume_enabled,
        )
    elif method in ("iwm", "iwm_ot", "hybrid_ot"):
        if not args.iwm_checkpoint.exists():
            raise FileNotFoundError(
                f"IWM checkpoint not found: {args.iwm_checkpoint}. "
                "Set --iwm_checkpoint to your iwm.pth.tar path."
            )
        source_desc = _extract_iwm_embeddings(
            image_paths=source_paths,
            device=device,
            batch_size=args.extract_batch_size,
            code_iwm_root=args.iwm_root,
            config_path=args.iwm_config,
            checkpoint_path=args.iwm_checkpoint,
            image_size_override=args.image_size,
        )
        target_desc = _extract_iwm_embeddings(
            image_paths=target_paths,
            device=device,
            batch_size=args.extract_batch_size,
            code_iwm_root=args.iwm_root,
            config_path=args.iwm_config,
            checkpoint_path=args.iwm_checkpoint,
            image_size_override=args.image_size,
        )

        if method == "hybrid_ot":
            include_hist_dino = not str(args.task).startswith("patch_")
            print(f"dino_descriptor_include_hist={include_hist_dino}")
            dino_source_desc = _extract_dino_ot_descriptors(
                image_paths=source_paths,
                device=device,
                image_size=args.image_size,
                batch_size=args.extract_batch_size,
                dino_model_name=args.dino_model_name,
                include_hist=include_hist_dino,
            )
            dino_target_desc = _extract_dino_ot_descriptors(
                image_paths=target_paths,
                device=device,
                image_size=args.image_size,
                batch_size=args.extract_batch_size,
                dino_model_name=args.dino_model_name,
                include_hist=include_hist_dino,
            )

            # Normalize each branch before fusion to avoid one descriptor dominating the other.
            iwm_w = float(args.hybrid_iwm_weight)
            dino_w = float(args.hybrid_dino_weight)
            source_desc = torch.cat(
                [
                    iwm_w * F.normalize(source_desc.float(), dim=1),
                    dino_w * F.normalize(dino_source_desc.float(), dim=1),
                ],
                dim=1,
            )
            target_desc = torch.cat(
                [
                    iwm_w * F.normalize(target_desc.float(), dim=1),
                    dino_w * F.normalize(dino_target_desc.float(), dim=1),
                ],
                dim=1,
            )
            del dino_source_desc, dino_target_desc

        if method == "iwm":
            topk_idx, topk_score = _cosine_topk(
                source_desc=source_desc,
                target_desc=target_desc,
                topk=args.topk,
                chunk_size=args.match_chunk_size,
                device=device,
            )
        else:
            topk_idx, topk_score = _dino_ot_topk(
                source_desc=source_desc,
                target_desc=target_desc,
                topk=args.topk,
                chunk_size=args.match_chunk_size,
                ot_target_pool=args.ot_target_pool,
                device=device,
                ot_alpha=args.ot_alpha,
                ot_reg=args.ot_reg,
                ot_temperature=args.ot_temperature,
                ot_iters=args.ot_iters,
            )
    else:
        raise ValueError(f"Unsupported method: {method}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_outputs(
        source_paths=source_paths,
        target_paths=target_paths,
        topk_idx=topk_idx,
        topk_score=topk_score,
        out_dir=output_dir,
    )
    _save_visualizations(
        source_paths=source_paths,
        target_paths=target_paths,
        topk_idx=topk_idx,
        out_dir=output_dir,
        viz_count=args.viz_count,
        viz_label_mode=args.viz_label_mode,
    )

    meta = {
        "task": args.task,
        "method": method,
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "n_source": len(source_paths),
        "n_target": len(target_paths),
        "topk": int(args.topk),
        "viz_label_mode": str(args.viz_label_mode),
        "output_dir": str(output_dir),
        "dino_descriptor_include_hist": (not str(args.task).startswith("patch_")) if method in ("dino_ot", "hybrid_ot") else None,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else str(output_dir / ".resume_cache"),
        "cache_dtype": str(args.cache_dtype),
        "disable_resume": bool(args.disable_resume),
        "target_scan_chunk_size": int(args.target_scan_chunk_size),
        "hybrid_iwm_weight": float(args.hybrid_iwm_weight) if method == "hybrid_ot" else None,
        "hybrid_dino_weight": float(args.hybrid_dino_weight) if method == "hybrid_ot" else None,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Outputs in: {output_dir}")
    print(f"- {output_dir / 'matches_topk_long.csv'}")
    print(f"- {output_dir / 'matches_top1.csv'}")
    print(f"- {output_dir / 'matches_topk_wide.csv'}")
    print(f"- {output_dir / 'viz'}")


if __name__ == "__main__":
    main()
