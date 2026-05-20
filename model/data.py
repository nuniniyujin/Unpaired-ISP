from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from raw_pipeline import (
    PseudoISPConfig,
    load_raw_npy,
    make_pseudo_rgb_from_raw4,
    reorder_raw4_to_canonical,
)


_VALID_TARGET_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_target_images(target_dir: Path) -> list[Path]:
    images = [p for p in target_dir.rglob("*") if p.suffix.lower() in _VALID_TARGET_EXTS]
    images = sorted(images)
    if not images:
        raise FileNotFoundError(f"No target RGB files found in {target_dir}")
    return images


def _load_rgb_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _random_crop_or_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape[:2]

    # Ensure minimum size for cropping.
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


class ISPUnpairedDataset(Dataset):
    """
    RAW npy dataset for unpaired ISP training.

    - Raw input is expected as raw4_hwc npy with shape (H, W, 4).
    - Canonical channel order inside the pipeline is [R, Gr, Gb, B].
    - Pseudo RGB is generated from the same raw via simple ISP steps.
    - If target_dir is provided, target RGB is sampled independently (unpaired).
    """

    def __init__(
        self,
        raw_dir: Path,
        raw4_order: str,
        pseudo_cfg: PseudoISPConfig,
        target_dir: Optional[Path] = None,
        target_size: int = 512,
        input_use_wb: bool = False,
        wb_randomize: bool = False,
        wb_jitter_r: float = 0.0,
        wb_jitter_b: float = 0.0,
    ):
        self.raw_dir = Path(raw_dir)
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"raw_dir does not exist: {self.raw_dir}")
        self.raw_files = sorted(self.raw_dir.rglob("*.npy"))
        if not self.raw_files:
            raise FileNotFoundError(f"No npy files found in raw_dir: {self.raw_dir}")

        self.raw4_order = raw4_order
        self.pseudo_cfg = pseudo_cfg
        self.target_size = int(target_size)
        self.input_use_wb = bool(input_use_wb)
        self.wb_randomize = bool(wb_randomize)
        self.wb_jitter_r = float(wb_jitter_r)
        self.wb_jitter_b = float(wb_jitter_b)

        self.target_files: list[Path] = []
        if target_dir is not None:
            target_root = Path(target_dir)
            if not target_root.exists():
                raise FileNotFoundError(f"target_dir does not exist: {target_root}")
            self.target_files = _list_target_images(target_root)

    def __len__(self) -> int:
        return len(self.raw_files)

    def _load_target_unpaired(self) -> tuple[torch.Tensor, str]:
        target_path = random.choice(self.target_files)
        rgb = _load_rgb_image(target_path)
        rgb = _random_crop_or_resize(rgb, self.target_size, self.target_size)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return tensor, str(target_path)

    def __getitem__(self, idx: int) -> dict:
        raw_path = self.raw_files[idx]
        raw4 = load_raw_npy(str(raw_path))
        raw4_canonical = reorder_raw4_to_canonical(raw4, self.raw4_order)

        cfg_local = self.pseudo_cfg
        if self.wb_randomize and (self.wb_jitter_r > 0.0 or self.wb_jitter_b > 0.0):
            wb = cfg_local.white_balance.astype(np.float32).copy()
            if self.wb_jitter_r > 0.0:
                wb[0] *= random.uniform(max(0.01, 1.0 - self.wb_jitter_r), 1.0 + self.wb_jitter_r)
            if self.wb_jitter_b > 0.0:
                wb[3] *= random.uniform(max(0.01, 1.0 - self.wb_jitter_b), 1.0 + self.wb_jitter_b)
            cfg_local = replace(cfg_local, white_balance=wb)

        raw4_norm, pseudo_rgb = make_pseudo_rgb_from_raw4(raw4_canonical, cfg_local)

        # Optional: feed WB-applied input to model. Default is raw normalized input.
        if self.input_use_wb:
            raw_input = raw4_norm.copy()
            raw_input[..., 0] *= float(self.pseudo_cfg.white_balance[0])
            raw_input[..., 1] *= float(self.pseudo_cfg.white_balance[1])
            raw_input[..., 2] *= float(self.pseudo_cfg.white_balance[2])
            raw_input[..., 3] *= float(self.pseudo_cfg.white_balance[3])
            raw_input = np.clip(raw_input, 0.0, 1.0)
        else:
            raw_input = raw4_norm

        sample = {
            "raw": torch.from_numpy(raw_input).permute(2, 0, 1).float(),
            "pseudo_rgb": torch.from_numpy(pseudo_rgb).permute(2, 0, 1).float(),
            "raw_path": str(raw_path),
        }

        if self.target_files:
            target_rgb, target_path = self._load_target_unpaired()
            sample["target_rgb"] = target_rgb
            sample["target_path"] = target_path

        return sample
