from __future__ import annotations

import argparse
import random
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ROOT = _REPO_ROOT / "model"
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from losses import (
    VGG19StyleEncoder,
    channel_moment_loss,
    discriminator_hinge_loss,
    generator_hinge_loss,
    gram_style_loss,
    luma_chroma_hist_loss,
    safe_clamp01,
    total_variation_loss,
)
from models import UNetDiscriminatorSN
from train_utils import (
    build_input_pred_gt_grid,
    load_checkpoint_state,
    set_seed,
    to_device,
    torch_load_compat,
)


_VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _VALID_IMAGE_EXTS]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in {root}")
    return files


def _load_rgb_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _random_crop_or_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if out_h <= 0 or out_w <= 0:
        return np.ascontiguousarray(img)

    h, w = img.shape[:2]
    if h < out_h or w < out_w:
        scale = max(out_h / max(h, 1), out_w / max(w, 1))
        new_h = max(out_h, int(round(h * scale)))
        new_w = max(out_w, int(round(w * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    top = random.randint(0, h - out_h) if h > out_h else 0
    left = random.randint(0, w - out_w) if w > out_w else 0
    crop = img[top : top + out_h, left : left + out_w]
    return np.ascontiguousarray(crop)


def _parse_numeric_prefix(stem: str) -> Optional[str]:
    m = re.match(r"^(\d+)", stem)
    if m is None:
        return None
    return m.group(1).zfill(3)


def _remap_repo_path(path: Path) -> Path:
    """
    Remap paths copied from another machine's checkout of the same repo
    to the current local repo root, without host-specific hardcoding.
    """
    raw = str(path)

    repo_name = _REPO_ROOT.name
    marker_mid = f"/{repo_name}/"
    if marker_mid in raw:
        rel = raw.split(marker_mid, 1)[1]
        return _REPO_ROOT / rel
    if raw.endswith(f"/{repo_name}"):
        return _REPO_ROOT

    return path


def _normalize_runtime_paths(args: argparse.Namespace) -> None:
    for key in ("train_raw_dir", "train_rgb_dir", "pairing_npz", "checkpoint_dir", "tb_log_dir", "init_checkpoint"):
        value = getattr(args, key, None)
        if value is None:
            continue
        original = Path(value)
        remapped = _remap_repo_path(original)
        if remapped != original:
            print(f"[INFO] Remapped --{key}: {original} -> {remapped}")
        setattr(args, key, remapped)


def inverse_smoothstep_torch(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)


def gamma_expansion_torch(image: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return torch.clamp(image, min=1e-8).pow(gamma)


class LUT3DHead(nn.Module):
    """Differentiable 3D LUT color head with trilinear interpolation."""

    def __init__(self, lut_size: int = 33):
        super().__init__()
        if int(lut_size) <= 1:
            raise ValueError("lut_size must be > 1")
        self.lut_size = int(lut_size)
        coords = torch.linspace(0.0, 1.0, self.lut_size, dtype=torch.float32)
        rr, gg, bb = torch.meshgrid(coords, coords, coords, indexing="ij")
        identity = torch.stack([rr, gg, bb], dim=0).unsqueeze(0)  # [1, 3, L, L, L]
        self.register_buffer("identity_lut", identity)
        self.lut = nn.Parameter(identity.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"LUT3DHead expects [B,3,H,W], got {tuple(x.shape)}")
        rgb = x.clamp(0.0, 1.0).permute(0, 2, 3, 1)  # [B,H,W,3] as [R,G,B]
        grid = torch.stack(
            [
                rgb[..., 2] * 2.0 - 1.0,  # x indexes LUT width  (B)
                rgb[..., 1] * 2.0 - 1.0,  # y indexes LUT height (G)
                rgb[..., 0] * 2.0 - 1.0,  # z indexes LUT depth  (R)
            ],
            dim=-1,
        ).unsqueeze(1)  # [B,1,H,W,3]
        lut = self.lut.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1, -1)
        out = F.grid_sample(
            lut,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B,3,1,H,W]
        return out.squeeze(2)

    def identity_regularization(self) -> torch.Tensor:
        return F.mse_loss(self.lut, self.identity_lut)


class PrecomputedOTImagePairDataset(Dataset):
    """
    Source image + paired target image dataset using precomputed sparse OT edges.

    Expected sparse file keys:
      - offsets, target_ids, final_weight/patch_weight/full_weight, source_paths, target_paths
    """

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        pairing_npz: Path,
        target_size: int = 512,
        pairing_topk: int = 0,
        pairing_weight_key: str = "final_weight",
        pairing_sampling: str = "weighted",
        pairing_fallback_unpaired: bool = True,
        pairing_prefix_fallback: bool = False,
    ):
        self.source_dir = Path(source_dir)
        if not self.source_dir.exists():
            raise FileNotFoundError(f"source_dir does not exist: {self.source_dir}")
        self.source_files = _list_images(self.source_dir)

        self.target_dir = Path(target_dir)
        if not self.target_dir.exists():
            raise FileNotFoundError(f"target_dir does not exist: {self.target_dir}")
        self.target_files = _list_images(self.target_dir)

        self.target_size = int(target_size)
        self.pairing_topk = int(pairing_topk)
        self.pairing_sampling = str(pairing_sampling).lower()
        if self.pairing_sampling not in ("weighted", "top1", "uniform"):
            raise ValueError("pairing_sampling must be one of: weighted, top1, uniform")
        self.pairing_fallback_unpaired = bool(pairing_fallback_unpaired)
        self.pairing_prefix_fallback = bool(pairing_prefix_fallback)

        self.offsets: np.ndarray
        self.target_ids: np.ndarray
        self.weights: np.ndarray
        self.source_paths: list[str]
        self.target_paths: list[str]
        self._load_pairing_arrays(Path(pairing_npz), pairing_weight_key=pairing_weight_key)

        target_by_name: dict[str, Path] = {}
        target_by_stem: dict[str, list[Path]] = {}
        for p in self.target_files:
            target_by_name.setdefault(p.name, p)
            target_by_stem.setdefault(p.stem, []).append(p)

        self.target_id_to_path: list[Optional[Path]] = []
        unresolved_targets = 0
        for t_path in self.target_paths:
            p = Path(t_path)
            resolved = p if p.exists() else target_by_name.get(p.name)
            if resolved is None:
                stem_matches = target_by_stem.get(p.stem, [])
                if len(stem_matches) == 1:
                    resolved = stem_matches[0]
            if resolved is None:
                unresolved_targets += 1
            self.target_id_to_path.append(resolved)

        source_name_to_ids: dict[str, list[int]] = {}
        source_stem_to_ids: dict[str, list[int]] = {}
        source_prefix_to_ids: dict[str, list[int]] = {}
        for sid, src_path in enumerate(self.source_paths):
            p = Path(src_path)
            source_name_to_ids.setdefault(p.name, []).append(sid)
            source_stem_to_ids.setdefault(p.stem, []).append(sid)
            pref = _parse_numeric_prefix(p.stem)
            if pref is not None:
                source_prefix_to_ids.setdefault(pref, []).append(sid)

        self.source_to_source_ids: list[np.ndarray] = []
        matched_name = 0
        matched_stem = 0
        matched_prefix = 0
        unmatched = 0

        for src in self.source_files:
            cands: list[int] = []
            name_ids = source_name_to_ids.get(src.name, [])
            if name_ids:
                cands = name_ids
                matched_name += 1
            else:
                stem_ids = source_stem_to_ids.get(src.stem, [])
                if stem_ids:
                    cands = stem_ids
                    matched_stem += 1
                elif self.pairing_prefix_fallback:
                    pref = _parse_numeric_prefix(src.stem)
                    if pref is not None:
                        pref_ids = source_prefix_to_ids.get(pref, [])
                        if pref_ids:
                            cands = pref_ids
                            matched_prefix += 1

            if cands:
                self.source_to_source_ids.append(np.asarray(cands, dtype=np.int64))
            else:
                self.source_to_source_ids.append(np.empty((0,), dtype=np.int64))
                unmatched += 1

        self.stats = {
            "num_source": len(self.source_files),
            "num_target": len(self.target_files),
            "num_source_meta": len(self.source_paths),
            "num_target_meta": len(self.target_paths),
            "num_edges": int(self.target_ids.shape[0]),
            "matched_name": matched_name,
            "matched_stem": matched_stem,
            "matched_prefix": matched_prefix,
            "unmatched_source": unmatched,
            "unresolved_target_ids": unresolved_targets,
        }

    def _load_pairing_arrays(self, pairing_npz: Path, pairing_weight_key: str) -> None:
        if not pairing_npz.exists():
            raise FileNotFoundError(f"pairing_npz not found: {pairing_npz}")
        required = {"offsets", "target_ids", pairing_weight_key, "source_paths", "target_paths"}

        with np.load(pairing_npz, allow_pickle=True) as sparse:
            missing = [k for k in required if k not in sparse.files]
            if missing:
                raise KeyError(f"Missing keys in pairing npz {pairing_npz}: {missing}")
            self.offsets = sparse["offsets"].astype(np.int64, copy=False)
            self.target_ids = sparse["target_ids"].astype(np.int64, copy=False)
            self.weights = sparse[pairing_weight_key].astype(np.float32, copy=False)
            source_raw = sparse["source_paths"].tolist()
            target_raw = sparse["target_paths"].tolist()
            self.source_paths = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in source_raw]
            self.target_paths = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in target_raw]

        if self.offsets.ndim != 1:
            raise ValueError("offsets must be 1D")
        if self.target_ids.ndim != 1 or self.weights.ndim != 1:
            raise ValueError("target_ids/weights must be 1D")
        if self.target_ids.shape[0] != self.weights.shape[0]:
            raise ValueError("target_ids and weights size mismatch")
        if len(self.offsets) != len(self.source_paths) + 1:
            raise ValueError("offsets length must be len(source_paths)+1")
        if int(self.offsets[-1]) != int(self.target_ids.shape[0]):
            raise ValueError("offsets[-1] must equal number of sparse edges")

    def __len__(self) -> int:
        return len(self.source_files)

    def _load_image_tensor(self, path: Path) -> torch.Tensor:
        rgb = _load_rgb_image(path)
        rgb = _random_crop_or_resize(rgb, self.target_size, self.target_size)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()

    def _sample_target_from_source_id(self, source_id: int) -> Optional[Path]:
        start = int(self.offsets[source_id])
        end = int(self.offsets[source_id + 1])
        if end <= start:
            return None

        edge_tids = self.target_ids[start:end]
        edge_w = self.weights[start:end]

        valid_idx = []
        for j, tid in enumerate(edge_tids):
            tid_int = int(tid)
            if 0 <= tid_int < len(self.target_id_to_path) and self.target_id_to_path[tid_int] is not None:
                valid_idx.append(j)
        if not valid_idx:
            return None

        idx = np.asarray(valid_idx, dtype=np.int64)
        tids = edge_tids[idx].astype(np.int64, copy=False)
        w = edge_w[idx].astype(np.float64, copy=False)

        if self.pairing_topk > 0 and tids.shape[0] > self.pairing_topk:
            keep = np.argpartition(w, -self.pairing_topk)[-self.pairing_topk:]
            tids = tids[keep]
            w = w[keep]

        if self.pairing_sampling == "top1":
            chosen_tid = int(tids[int(np.argmax(w))])
        elif self.pairing_sampling == "uniform":
            chosen_tid = int(tids[int(np.random.randint(0, tids.shape[0]))])
        else:
            prob = np.clip(w, a_min=0.0, a_max=None)
            denom = float(prob.sum())
            if not np.isfinite(denom) or denom <= 0:
                chosen_tid = int(tids[int(np.random.randint(0, tids.shape[0]))])
            else:
                prob = prob / denom
                chosen_idx = int(np.random.choice(np.arange(tids.shape[0]), p=prob))
                chosen_tid = int(tids[chosen_idx])

        return self.target_id_to_path[chosen_tid]

    def _sample_target_for_source(self, src_idx: int) -> tuple[torch.Tensor, str]:
        source_candidates = self.source_to_source_ids[src_idx]
        sampled_target: Optional[Path] = None

        if source_candidates.size > 0:
            chosen_source_id = int(source_candidates[int(np.random.randint(0, source_candidates.size))])
            sampled_target = self._sample_target_from_source_id(chosen_source_id)

        if sampled_target is None:
            if not self.pairing_fallback_unpaired:
                raise RuntimeError(
                    f"No valid target pairing for source '{self.source_files[src_idx]}' and fallback is disabled."
                )
            sampled_target = random.choice(self.target_files)

        return self._load_image_tensor(sampled_target), str(sampled_target)

    def _top1_target_for_source_id(self, source_id: int) -> tuple[Optional[Path], float]:
        start = int(self.offsets[source_id])
        end = int(self.offsets[source_id + 1])
        if end <= start:
            return None, -float("inf")

        edge_tids = self.target_ids[start:end]
        edge_w = self.weights[start:end]

        best_path: Optional[Path] = None
        best_w = -float("inf")
        for j, tid in enumerate(edge_tids):
            tid_int = int(tid)
            if not (0 <= tid_int < len(self.target_id_to_path)):
                continue
            cand_path = self.target_id_to_path[tid_int]
            if cand_path is None:
                continue
            w = float(edge_w[j])
            if w > best_w:
                best_w = w
                best_path = cand_path
        return best_path, best_w

    def _top1_target_for_source(self, src_idx: int) -> Optional[Path]:
        source_candidates = self.source_to_source_ids[src_idx]
        best_path: Optional[Path] = None
        best_w = -float("inf")
        for sid in source_candidates:
            cand_path, cand_w = self._top1_target_for_source_id(int(sid))
            if cand_path is not None and cand_w > best_w:
                best_path = cand_path
                best_w = cand_w
        return best_path

    def __getitem__(self, idx: int) -> dict:
        src_path = self.source_files[idx]
        src_tensor = self._load_image_tensor(src_path)
        tgt_tensor, tgt_path = self._sample_target_for_source(idx)
        top1_path = self._top1_target_for_source(idx)
        if top1_path is None:
            top1_path_str = tgt_path
        else:
            top1_path_str = str(top1_path)
        return {
            "source_rgb": src_tensor,
            "target_rgb": tgt_tensor,
            "source_path": str(src_path),
            "target_path": tgt_path,
            "target_top1_path": top1_path_str,
        }


class DemosaicCNN(nn.Module):
    """
    Lightweight residual CNN.
    Supports:
      - 3ch RGB input (direct mode)
      - 4ch RAW input (legacy demosaic mode)
    """

    def __init__(self, kernel_size: int = 3, hidden: int = 16, use_lut_head: bool = False, lut_size: int = 33):
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError("kernel_size must be 1 or 3")
        padding = 0 if kernel_size == 1 else 1
        conv_kwargs = {"padding_mode": "reflect"} if padding > 0 else {}
        self.conv1 = nn.Conv2d(3, hidden, kernel_size=kernel_size, padding=padding, **conv_kwargs)
        self.conv2 = nn.Conv2d(hidden, 3, kernel_size=kernel_size, padding=padding, **conv_kwargs)
        # CCM-like global color mixing head.
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
        return safe_clamp01(out)


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
    """
    Mid-size residual CNN variant for stronger color/style modeling.
    Supports:
      - 3ch RGB input (direct mode)
      - 4ch RAW input (legacy demosaic mode)
    """

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
        return safe_clamp01(out)


def _build_generator_from_args(args: argparse.Namespace) -> nn.Module:
    if args.model_type == "demosaiccnn":
        return DemosaicCNN(
            kernel_size=args.kernel_size,
            hidden=args.hidden,
            use_lut_head=args.use_lut_head,
            lut_size=args.lut_size,
        )
    if args.model_type == "bigger_demosaiccnn":
        return BiggerDemosaicCNN(
            kernel_size=args.kernel_size,
            hidden=args.hidden,
            num_res_blocks=args.num_res_blocks,
            use_lut_head=args.use_lut_head,
            lut_size=args.lut_size,
        )
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_raw_dir",
        type=Path,
        default=Path("data/train/raws_png_dmnet_ffdnet"),
        help="Source image directory (PNG/JPG). RAW npy is not used in this script.",
    )
    p.add_argument(
        "--train_rgb_dir",
        type=Path,
        default=Path("data/train/jpegs"),
        help="Target image directory (PNG/JPG).",
    )
    p.add_argument(
        "--pairing_npz",
        type=Path,
        default=Path("data_processing/patch_ot_all_weights_sparse.npz"),
        help="Precomputed sparse OT file from data_processing/build_patch_ot_all_from_full.py",
    )
    p.add_argument(
        "--pairing_weight_key",
        type=str,
        default="final_weight",
        choices=["final_weight", "patch_weight", "full_weight"],
        help="Which precomputed weight to use for target sampling.",
    )
    p.add_argument(
        "--pairing_topk",
        type=int,
        default=0,
        help="Top-k candidates per source from sparse OT. 0 means use all candidates.",
    )
    p.add_argument(
        "--pairing_sampling",
        type=str,
        default="weighted",
        choices=["weighted", "top1", "uniform"],
        help="Sampling strategy inside selected pairing candidates.",
    )
    p.add_argument("--pairing_prefix_fallback", action="store_true")
    p.add_argument("--pairing_fallback_unpaired", action="store_true", default=True)
    p.add_argument("--no_pairing_fallback_unpaired", action="store_false", dest="pairing_fallback_unpaired")

    p.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Random crop size. If source is smaller than target_size, resize first; <=0 keeps original size.",
    )
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--tone_mapping", type=str, default="smoothstep", choices=["smoothstep", "none"])
    p.add_argument(
        "--target_domain",
        type=str,
        default="linear",
        choices=["linear", "nonlinear"],
        help=(
            "Target supervision domain. "
            "'linear': apply inverse tone-mapping + gamma expansion to target. "
            "'nonlinear': use target RGB as-is."
        ),
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4, help="Generator LR.")
    p.add_argument("--lr_d", type=float, default=1e-4, help="Discriminator LR.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps_per_epoch", type=int, default=0)
    p.add_argument(
        "--model_type",
        type=str,
        default="demosaiccnn",
        choices=["demosaiccnn", "bigger_demosaiccnn"],
        help="Generator backbone choice.",
    )
    p.add_argument("--kernel_size", type=int, default=3, choices=[1, 3])
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument(
        "--num_res_blocks",
        type=int,
        default=4,
        choices=[3, 4, 5],
        help="Number of residual blocks for bigger_demosaiccnn.",
    )
    p.add_argument("--use_lut_head", action="store_true", help="Enable optional 3D LUT color head.")
    p.add_argument("--lut_size", type=int, default=33, help="3D LUT size per axis (e.g., 17/33).")
    p.add_argument(
        "--lut_identity_weight",
        type=float,
        default=0.0,
        help="Regularization weight for keeping LUT close to identity.",
    )

    p.add_argument("--moment_weight", type=float, default=1.0)
    p.add_argument("--gram_weight", type=float, default=1.0)
    p.add_argument("--hist_luma_weight", type=float, default=1.0)
    p.add_argument("--hist_chroma_weight", type=float, default=1.5)
    p.add_argument("--hist_bins_y", type=int, default=64)
    p.add_argument("--hist_bins_uv", type=int, default=32)
    p.add_argument("--tv_weight", type=float, default=0.05)
    p.add_argument("--gan_weight", type=float, default=0.003)
    p.add_argument("--disable_gan", action="store_true")
    p.add_argument("--d_num_feat", type=int, default=64)
    p.add_argument(
        "--cycle_weight",
        type=float,
        default=0.0,
        help="Cycle consistency weight (L1). >0 enables reverse generator training (target->source).",
    )
    p.add_argument(
        "--cycle_identity_weight",
        type=float,
        default=0.0,
        help="Identity regularization weight for cycle generators (L1).",
    )

    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--init_checkpoint", type=Path, default=None, help="Load generator weights before training.")
    p.add_argument(
        "--resume",
        type=str,
        default="none",
        help="Resume full training state. Use 'none', 'auto' (checkpoint_dir/last.pt), or a checkpoint path.",
    )
    p.add_argument("--strict", action="store_true", help="Strict checkpoint loading for --init_checkpoint.")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--tb_log_dir", type=Path, default=None)
    p.add_argument("--tb_image_log_every", type=int, default=1, help="Log images every N epochs.")
    p.add_argument("--tb_num_images", type=int, default=10, help="Number of images to log to TensorBoard.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _normalize_runtime_paths(args)
    if args.hist_bins_y <= 1 or args.hist_bins_uv <= 1:
        raise ValueError("hist_bins_y and hist_bins_uv must be > 1")
    if args.lut_size <= 1:
        raise ValueError("lut_size must be > 1")
    if args.lut_identity_weight < 0:
        raise ValueError("lut_identity_weight must be >= 0")
    if args.cycle_weight < 0:
        raise ValueError("cycle_weight must be >= 0")
    if args.cycle_identity_weight < 0:
        raise ValueError("cycle_identity_weight must be >= 0")
    set_seed(args.seed)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stop_requested = {"flag": False}

    def _signal_handler(_sig, _frame):
        stop_requested["flag"] = True
        print("Signal received. Will save checkpoint and stop after current iteration.")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            raise RuntimeError("TensorBoard is not available. Install tensorboard or disable --tensorboard.") from e
        tb_log_dir = args.tb_log_dir if args.tb_log_dir is not None else (args.checkpoint_dir / "tensorboard")
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard logging enabled at: {tb_log_dir}")

    train_set = PrecomputedOTImagePairDataset(
        source_dir=args.train_raw_dir,
        target_dir=args.train_rgb_dir,
        pairing_npz=args.pairing_npz,
        target_size=args.target_size,
        pairing_topk=args.pairing_topk,
        pairing_weight_key=args.pairing_weight_key,
        pairing_sampling=args.pairing_sampling,
        pairing_fallback_unpaired=args.pairing_fallback_unpaired,
        pairing_prefix_fallback=args.pairing_prefix_fallback,
    )
    print(
        "Pairing stats: "
        f"source={train_set.stats['num_source']}, "
        f"target={train_set.stats['num_target']}, "
        f"source_meta={train_set.stats['num_source_meta']}, "
        f"target_meta={train_set.stats['num_target_meta']}, "
        f"edges={train_set.stats['num_edges']}, "
        f"matched_name={train_set.stats['matched_name']}, "
        f"matched_stem={train_set.stats['matched_stem']}, "
        f"matched_prefix={train_set.stats['matched_prefix']}, "
        f"unmatched_source={train_set.stats['unmatched_source']}, "
        f"unresolved_target_ids={train_set.stats['unresolved_target_ids']}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )

    style_encoder = VGG19StyleEncoder(pretrained=True).to(args.device)
    model = _build_generator_from_args(args).to(args.device)
    use_cycle = (args.cycle_weight > 0) or (args.cycle_identity_weight > 0)
    model_cycle = _build_generator_from_args(args).to(args.device) if use_cycle else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Generator={args.model_type}, hidden={args.hidden}, params={n_params:,}, "
        f"lut_head={args.use_lut_head}, lut_size={args.lut_size}"
    )
    if use_cycle:
        n_params_cycle = sum(p.numel() for p in model_cycle.parameters() if p.requires_grad)
        print(
            "Cycle consistency enabled: "
            f"cycle_weight={args.cycle_weight}, cycle_identity_weight={args.cycle_identity_weight}, "
            f"reverse_params={n_params_cycle:,}"
        )

    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cycle = (
        torch.optim.AdamW(model_cycle.parameters(), lr=args.lr, weight_decay=args.weight_decay) if use_cycle else None
    )
    use_gan = (not args.disable_gan) and (args.gan_weight > 0)
    model_d = UNetDiscriminatorSN(num_in_ch=3, num_feat=args.d_num_feat).to(args.device) if use_gan else None
    optimizer_d = (
        torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, weight_decay=args.weight_decay) if use_gan else None
    )

    start_epoch = 0
    resume_path = None
    resume_arg = str(args.resume).strip()
    if resume_arg.lower() == "auto":
        candidate = args.checkpoint_dir / "last.pt"
        if candidate.exists():
            resume_path = candidate
    elif resume_arg.lower() not in ("none", ""):
        resume_input = Path(resume_arg)
        resume_path = _remap_repo_path(resume_input)
        if resume_path != resume_input:
            print(f"[INFO] Remapped --resume: {resume_input} -> {resume_path}")

    loaded_cycle_state = False
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        raw_resume = torch_load_compat(resume_path, map_location="cpu")
        state = load_checkpoint_state(resume_path)
        incompatible = model.load_state_dict(state, strict=args.strict)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(
            f"Resumed model checkpoint {resume_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)}, strict={args.strict})"
        )

        if isinstance(raw_resume, dict):
            if use_cycle and model_cycle is not None:
                cycle_state = raw_resume.get("model_cycle")
                if isinstance(cycle_state, dict):
                    incompatible_cycle = model_cycle.load_state_dict(cycle_state, strict=args.strict)
                    missing_cycle = getattr(incompatible_cycle, "missing_keys", [])
                    unexpected_cycle = getattr(incompatible_cycle, "unexpected_keys", [])
                    loaded_cycle_state = True
                    print(
                        "Restored reverse cycle generator "
                        f"(missing={len(missing_cycle)}, unexpected={len(unexpected_cycle)}, strict={args.strict})."
                    )
                else:
                    print("[WARN] model_cycle state missing in resume checkpoint.")

            opt_g = raw_resume.get("optimizer_g")
            if isinstance(opt_g, dict):
                optimizer_g.load_state_dict(opt_g)
                print("Restored optimizer_g state.")
            else:
                print("[WARN] optimizer_g state missing in resume checkpoint.")

            if use_cycle and optimizer_cycle is not None:
                opt_cycle = raw_resume.get("optimizer_cycle")
                if isinstance(opt_cycle, dict):
                    optimizer_cycle.load_state_dict(opt_cycle)
                    print("Restored optimizer_cycle state.")
                else:
                    print("[WARN] optimizer_cycle state missing in resume checkpoint.")

            if use_gan and optimizer_d is not None:
                opt_d = raw_resume.get("optimizer_d")
                if isinstance(opt_d, dict):
                    optimizer_d.load_state_dict(opt_d)
                    print("Restored optimizer_d state.")
                else:
                    print("[WARN] optimizer_d state missing in resume checkpoint.")

            epoch_val = raw_resume.get("epoch")
            if isinstance(epoch_val, int):
                start_epoch = epoch_val + 1
        print(f"Continuing from epoch index {start_epoch}.")
        if args.init_checkpoint is not None:
            print("[INFO] --resume is set; ignoring --init_checkpoint.")
    elif args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(f"init checkpoint not found: {args.init_checkpoint}")
        raw_init = torch_load_compat(args.init_checkpoint, map_location="cpu")
        state = load_checkpoint_state(args.init_checkpoint)
        incompatible = model.load_state_dict(state, strict=args.strict)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(
            f"Loaded init checkpoint {args.init_checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)}, strict={args.strict})"
        )
        if use_cycle and model_cycle is not None and isinstance(raw_init, dict):
            init_cycle_state = raw_init.get("model_cycle")
            if isinstance(init_cycle_state, dict):
                incompatible_cycle = model_cycle.load_state_dict(init_cycle_state, strict=args.strict)
                missing_cycle = getattr(incompatible_cycle, "missing_keys", [])
                unexpected_cycle = getattr(incompatible_cycle, "unexpected_keys", [])
                loaded_cycle_state = True
                print(
                    "Loaded reverse cycle generator from init checkpoint "
                    f"(missing={len(missing_cycle)}, unexpected={len(unexpected_cycle)}, strict={args.strict})."
                )

    if use_cycle and model_cycle is not None and not loaded_cycle_state:
        model_cycle.load_state_dict(model.state_dict(), strict=True)
        print("[INFO] Initialized reverse cycle generator from forward generator weights.")

    run_start = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        if use_cycle and model_cycle is not None:
            model_cycle.train()
        epoch_total = []
        epoch_gram = []
        epoch_g_adv = []
        epoch_d = []
        epoch_moment = []
        epoch_luma = []
        epoch_chroma = []
        epoch_lut_reg = []
        epoch_tv = []
        epoch_cycle = []
        epoch_cycle_id = []
        logged_images = False

        pbar = tqdm(train_loader, desc=f"train_cnn_new epoch {epoch}", leave=False)

        for step, batch in enumerate(pbar):
            batch = to_device(batch, args.device)
            source_rgb = batch["source_rgb"]
            target_rgb = batch["target_rgb"]

            if not torch.isfinite(source_rgb).all():
                print("nonfinite source", batch.get("source_path", "unknown"))
                break
            if not torch.isfinite(target_rgb).all():
                print("nonfinite target", batch.get("target_path", "unknown"))
                break

            pred = model(source_rgb)
            if not torch.isfinite(pred).all():
                print("nonfinite pred")
                print("source min/max", source_rgb.min().item(), source_rgb.max().item())
                break

            if args.target_domain == "linear":
                if args.tone_mapping == "smoothstep":
                    target_linear = inverse_smoothstep_torch(target_rgb)
                elif args.tone_mapping in ("none", "identity"):
                    target_linear = target_rgb
                else:
                    raise ValueError(f"Unsupported tone_mapping: {args.tone_mapping}")
                target_train = gamma_expansion_torch(target_linear, gamma=args.gamma)
            elif args.target_domain == "nonlinear":
                target_train = target_rgb
            else:
                raise ValueError(f"Unsupported target_domain: {args.target_domain}")

            if use_gan:
                optimizer_d.zero_grad(set_to_none=True)
                pred_detached = pred.detach()
                d_loss = discriminator_hinge_loss(model_d(target_train), model_d(pred_detached))
                if torch.isfinite(d_loss):
                    d_loss.backward()
                    optimizer_d.step()
            else:
                d_loss = pred.new_tensor(0.0)

            moment_val = channel_moment_loss(pred, target_train, mode="batch")
            if args.gram_weight > 0:
                gram_val = gram_style_loss(pred, target_train, style_encoder, mode="pair")
            else:
                gram_val = pred.new_tensor(0.0)

            luma_val, chroma_val = luma_chroma_hist_loss(
                pred,
                target_train,
                mode="pair",
                bins_y=args.hist_bins_y,
                bins_uv=args.hist_bins_uv,
                sigma_y=0.02,
                sigma_uv=0.03,
            )
            if args.lut_identity_weight > 0 and getattr(model, "lut_head", None) is not None:
                lut_reg_val = model.lut_head.identity_regularization()
            else:
                lut_reg_val = pred.new_tensor(0.0)
            tv_val = total_variation_loss(pred) if args.tv_weight > 0 else pred.new_tensor(0.0)
            if use_gan:
                for p in model_d.parameters():
                    p.requires_grad_(False)
                g_adv = generator_hinge_loss(model_d(pred))
                for p in model_d.parameters():
                    p.requires_grad_(True)
            else:
                g_adv = pred.new_tensor(0.0)

            if use_cycle and model_cycle is not None:
                source_cycle = model_cycle(pred)
                target_to_source = model_cycle(target_train)
                target_cycle = model(target_to_source)
                cycle_val = F.l1_loss(source_cycle, source_rgb) + F.l1_loss(target_cycle, target_train)
                if args.cycle_identity_weight > 0:
                    cycle_id_val = F.l1_loss(model(target_train), target_train) + F.l1_loss(
                        model_cycle(source_rgb), source_rgb
                    )
                else:
                    cycle_id_val = pred.new_tensor(0.0)
            else:
                cycle_val = pred.new_tensor(0.0)
                cycle_id_val = pred.new_tensor(0.0)

            for name, val in [
                ("gram", gram_val),
                ("luma", luma_val),
                ("chroma", chroma_val),
                ("lut_reg", lut_reg_val),
                ("moment", moment_val),
                ("g_adv", g_adv),
                ("cycle", cycle_val),
                ("cycle_id", cycle_id_val),
            ]:
                if not torch.isfinite(val):
                    print("nonfinite loss", name)
                    break

            loss = (
                args.moment_weight * moment_val
                + args.gram_weight * gram_val
                + args.hist_luma_weight * luma_val
                + args.hist_chroma_weight * chroma_val
                + args.lut_identity_weight * lut_reg_val
                + args.tv_weight * tv_val
                + args.gan_weight * g_adv
                + args.cycle_weight * cycle_val
                + args.cycle_identity_weight * cycle_id_val
            )

            if not torch.isfinite(loss):
                optimizer_g.zero_grad(set_to_none=True)
                if optimizer_cycle is not None:
                    optimizer_cycle.zero_grad(set_to_none=True)
                continue

            optimizer_g.zero_grad(set_to_none=True)
            if optimizer_cycle is not None:
                optimizer_cycle.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_g.step()
            if optimizer_cycle is not None:
                optimizer_cycle.step()

            epoch_total.append(loss.item())
            epoch_moment.append(moment_val.item())
            epoch_gram.append(gram_val.item())
            epoch_g_adv.append(g_adv.item())
            epoch_d.append(d_loss.item())
            epoch_luma.append(luma_val.item())
            epoch_chroma.append(chroma_val.item())
            epoch_lut_reg.append(lut_reg_val.item())
            epoch_tv.append(tv_val.item())
            epoch_cycle.append(cycle_val.item())
            epoch_cycle_id.append(cycle_id_val.item())

            postfix = {
                "total": f"{np.mean(epoch_total):.4f}",
                "g_adv": f"{np.mean(epoch_g_adv):.4f}",
                "d": f"{np.mean(epoch_d):.4f}",
            }
            if use_cycle:
                postfix["cyc"] = f"{np.mean(epoch_cycle):.4f}"
                postfix["cyc_id"] = f"{np.mean(epoch_cycle_id):.4f}"
            pbar.set_postfix(**postfix)

            if tb_writer is not None and (epoch + 1) % args.tb_image_log_every == 0 and not logged_images:
                n_img = min(args.tb_num_images, source_rgb.shape[0], pred.shape[0], target_rgb.shape[0])
                target_for_tb = target_rgb
                top1_paths = batch.get("target_top1_path")
                if isinstance(top1_paths, list) and len(top1_paths) >= n_img:
                    h = int(source_rgb.shape[-2])
                    w = int(source_rgb.shape[-1])
                    top1_tensors = []
                    for p in top1_paths[:n_img]:
                        img = _load_rgb_image(Path(p))
                        img = _random_crop_or_resize(img, h, w)
                        top1_tensors.append(torch.from_numpy(img).permute(2, 0, 1).float())
                    if top1_tensors:
                        target_for_tb = torch.stack(top1_tensors, dim=0)

                grid = build_input_pred_gt_grid(source_rgb, pred, target_for_tb, max_images=n_img)
                tb_writer.add_image("train/first_batch_input_pred_target", grid, epoch)
                logged_images = True

            if stop_requested["flag"]:
                break
            if args.steps_per_epoch > 0 and (step + 1) >= args.steps_per_epoch:
                break

        mean_total = float(np.mean(epoch_total)) if epoch_total else float("nan")
        mean_gram = float(np.mean(epoch_gram)) if epoch_gram else float("nan")
        mean_g_adv = float(np.mean(epoch_g_adv)) if epoch_g_adv else float("nan")
        mean_d = float(np.mean(epoch_d)) if epoch_d else float("nan")
        mean_moment = float(np.mean(epoch_moment)) if epoch_moment else float("nan")
        mean_luma = float(np.mean(epoch_luma)) if epoch_luma else float("nan")
        mean_chroma = float(np.mean(epoch_chroma)) if epoch_chroma else float("nan")
        mean_lut_reg = float(np.mean(epoch_lut_reg)) if epoch_lut_reg else float("nan")
        mean_tv = float(np.mean(epoch_tv)) if epoch_tv else float("nan")
        mean_cycle = float(np.mean(epoch_cycle)) if epoch_cycle else float("nan")
        mean_cycle_id = float(np.mean(epoch_cycle_id)) if epoch_cycle_id else float("nan")

        if tb_writer is not None:
            tb_writer.add_scalar("train/total", mean_total, epoch)
            tb_writer.add_scalar("train/gram", mean_gram, epoch)
            tb_writer.add_scalar("train/g_adv", mean_g_adv, epoch)
            tb_writer.add_scalar("train/d_loss", mean_d, epoch)
            tb_writer.add_scalar("train/moment", mean_moment, epoch)
            tb_writer.add_scalar("train/luma", mean_luma, epoch)
            tb_writer.add_scalar("train/chroma", mean_chroma, epoch)
            tb_writer.add_scalar("train/lut_reg", mean_lut_reg, epoch)
            tb_writer.add_scalar("train/tv", mean_tv, epoch)
            tb_writer.add_scalar("train/cycle", mean_cycle, epoch)
            tb_writer.add_scalar("train/cycle_identity", mean_cycle_id, epoch)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "model_cycle": model_cycle.state_dict() if model_cycle is not None else None,
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_cycle": optimizer_cycle.state_dict() if optimizer_cycle is not None else None,
                "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
                "args": vars(args),
            },
            args.checkpoint_dir / "last.pt",
        )
        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "model_cycle": model_cycle.state_dict() if model_cycle is not None else None,
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_cycle": optimizer_cycle.state_dict() if optimizer_cycle is not None else None,
                    "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
                    "args": vars(args),
                },
                args.checkpoint_dir / f"epoch_{epoch:04d}.pt",
            )

        print(
            "epoch={} total={:.7f} moment={:.7f} gram={:.7f} g_adv={:.7f} d={:.7f} luma={:.7f} chroma={:.7f} lut_reg={:.7f} tv={:.7f} cycle={:.7f} cycle_id={:.7f}".format(
                epoch,
                mean_total,
                mean_moment,
                mean_gram,
                mean_g_adv,
                mean_d,
                mean_luma,
                mean_chroma,
                mean_lut_reg,
                mean_tv,
                mean_cycle,
                mean_cycle_id,
            )
        )

        if stop_requested["flag"]:
            print("Stopped early after checkpointing.")
            break

        if args.steps_per_epoch > 0:
            elapsed_h = (time.time() - run_start) / 3600.0
            if elapsed_h > 0:
                pass

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
