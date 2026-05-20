from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LUT3DHead(nn.Module):
    """Differentiable 3D LUT color head with trilinear interpolation."""

    def __init__(self, lut_size: int = 33):
        super().__init__()
        if int(lut_size) <= 1:
            raise ValueError("lut_size must be > 1")
        self.lut_size = int(lut_size)
        coords = torch.linspace(0.0, 1.0, self.lut_size, dtype=torch.float32)
        rr, gg, bb = torch.meshgrid(coords, coords, coords, indexing="ij")
        identity = torch.stack([rr, gg, bb], dim=0).unsqueeze(0)  # [1,3,L,L,L]
        self.lut = nn.Parameter(identity.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"LUT3DHead expects [B,3,H,W], got {tuple(x.shape)}")
        rgb = x.clamp(0.0, 1.0).permute(0, 2, 3, 1)
        grid = torch.stack(
            [
                rgb[..., 2] * 2.0 - 1.0,  # x -> B axis
                rgb[..., 1] * 2.0 - 1.0,  # y -> G axis
                rgb[..., 0] * 2.0 - 1.0,  # z -> R axis
            ],
            dim=-1,
        ).unsqueeze(1)
        lut = self.lut.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1, -1)
        out = F.grid_sample(
            lut,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return out.squeeze(2)


class DemosaicCNN(nn.Module):
    """Submission model: demosaiccnn only."""

    def __init__(self, kernel_size: int = 3, hidden: int = 16, use_lut_head: bool = False, lut_size: int = 33):
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError("kernel_size must be 1 or 3")
        padding = 0 if kernel_size == 1 else 1
        conv_kwargs = {"padding_mode": "reflect"} if padding > 0 else {}
        self.conv1 = nn.Conv2d(3, hidden, kernel_size=kernel_size, padding=padding, **conv_kwargs)
        self.conv2 = nn.Conv2d(hidden, 3, kernel_size=kernel_size, padding=padding, **conv_kwargs)
        self.ccm_head = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=True)
        with torch.no_grad():
            self.ccm_head.weight.zero_()
            self.ccm_head.bias.zero_()
            self.ccm_head.weight.copy_(torch.eye(3, dtype=self.ccm_head.weight.dtype).view(3, 3, 1, 1))
        self.lut_head = LUT3DHead(lut_size=lut_size) if use_lut_head else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input with shape [B,C,H,W], got shape={tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels for submission model input, got {x.shape[1]}")
        h = F.relu(self.conv1(x), inplace=True)
        delta = self.conv2(h)
        out = x + delta
        out = self.ccm_head(out)
        if self.lut_head is not None:
            out = self.lut_head(out)
        return torch.clamp(out, 0.0, 1.0)

