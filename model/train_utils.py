from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader


def torch_load_compat(path: str | Path, map_location: str = "cpu"):
    # PyTorch >=2.6 default changed to weights_only=True.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_checkpoint_state(raw: Any) -> dict[str, Any]:
    state = raw
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state = raw["state_dict"]
        elif "model" in raw and isinstance(raw["model"], dict):
            state = raw["model"]
        elif "model_g" in raw and isinstance(raw["model_g"], dict):
            state = raw["model_g"]

    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format.")

    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state = {k[len("module.") :]: v for k, v in state.items()}
        keys = list(state.keys())
    if keys and all(k.startswith("model.") for k in keys):
        state = {k[len("model.") :]: v for k, v in state.items()}
    return state


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
    raw = torch_load_compat(path, map_location="cpu")
    return normalize_checkpoint_state(raw)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def tensor_psnr(pred: Tensor, target: Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def build_input_pred_gt_grid(raw4: Tensor, pred: Tensor, target: Tensor, max_images: int = 1) -> Tensor:
    """Build a TensorBoard-friendly grid: [input_preview | pred | target]."""
    n = min(max_images, raw4.shape[0], pred.shape[0], target.shape[0])
    rows = []
    for i in range(n):
        raw_preview = raw4[i, :3].detach().cpu().clamp(0, 1)
        if raw_preview.shape[-2:] != pred.shape[-2:]:
            raw_preview = F.interpolate(
                raw_preview.unsqueeze(0), size=pred.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(0)
        row = torch.cat(
            [
                raw_preview,
                pred[i].detach().cpu().clamp(0, 1),
                target[i].detach().cpu().clamp(0, 1),
            ],
            dim=2,
        )
        rows.append(row)
    return torch.cat(rows, dim=1)


def evaluate_stage_a(model: nn.Module, loader: DataLoader, device: str, max_batches: int = 0) -> dict[str, float]:
    model.eval()
    l1_vals = []
    psnr_vals = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = to_device(batch, device)
            pred = torch.clamp(model(batch["raw"]), 0.0, 1.0)
            target = batch["pseudo_rgb"]
            l1_vals.append(F.l1_loss(pred, target).item())
            psnr_vals.append(tensor_psnr(pred, target))
            if max_batches > 0 and (i + 1) >= max_batches:
                break

    return {
        "val_l1": float(np.mean(l1_vals)) if l1_vals else float("nan"),
        "val_psnr": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
    }


def ensure_csv_header(csv_path: Path, header: list[str]):
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv_row(csv_path: Path, row: list[Any]):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
