from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

try:
    from pytorch_msssim import ssim as _ssim_fn
except Exception:
    _ssim_fn = None

try:
    import kornia
except Exception:
    kornia = None

try:
    from torchvision.models import VGG19_Weights, vgg19
except Exception:
    VGG19_Weights = None
    vgg19 = None


def safe_clamp01(x: Tensor) -> Tensor:
    return torch.clamp(x, 0.0, 1.0)


def reconstruction_loss_l1(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target)


def reconstruction_loss_ssim(pred: Tensor, target: Tensor) -> Tensor:
    if _ssim_fn is None:
        # Graceful fallback if pytorch_msssim is not installed.
        return pred.new_tensor(0.0)
    ssim_val = _ssim_fn(pred, target, data_range=1.0, size_average=True)
    return 1.0 - ssim_val


def sobel_edge_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    c = pred.shape[1]
    kx = pred.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = pred.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    kx = kx.expand(c, 1, 3, 3)
    ky = ky.expand(c, 1, 3, 3)

    pred_gx = F.conv2d(pred, kx, padding=1, groups=c)
    pred_gy = F.conv2d(pred, ky, padding=1, groups=c)
    target_gx = F.conv2d(target, kx, padding=1, groups=c)
    target_gy = F.conv2d(target, ky, padding=1, groups=c)

    pred_mag = torch.sqrt(pred_gx.square() + pred_gy.square() + 1e-12)
    target_mag = torch.sqrt(target_gx.square() + target_gy.square() + 1e-12)
    return F.l1_loss(pred_mag, target_mag)


def total_variation_loss(x: Tensor) -> Tensor:
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dx + dy


def discriminator_hinge_loss(pred_real: Tensor, pred_fake: Tensor) -> Tensor:
    return 0.5 * (F.relu(1.0 - pred_real).mean() + F.relu(1.0 + pred_fake).mean())


def generator_hinge_loss(pred_fake: Tensor) -> Tensor:
    return -pred_fake.mean()


def channel_moment_loss(pred: Tensor, target: Tensor, mode: str = "batch") -> Tensor:
    """
    Match color moments between generated and unpaired target batches.

    mode="batch": compare batch-level RGB moments (recommended for unpaired training).
    mode="pair": compare sample-wise moments at the same batch index.
    """
    if mode == "batch":
        pred_mean = pred.mean(dim=(0, 2, 3))
        target_mean = target.mean(dim=(0, 2, 3))
        pred_std = pred.std(dim=(0, 2, 3), unbiased=False)
        target_std = target.std(dim=(0, 2, 3), unbiased=False)
    elif mode == "pair":
        pred_mean = pred.mean(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        pred_std = pred.std(dim=(2, 3), unbiased=False)
        target_std = target.std(dim=(2, 3), unbiased=False)
    else:
        raise ValueError(f"Unsupported channel moment mode: {mode}")
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def lab_color_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Approximate color fidelity loss in Lab space (surrogate for DeltaE)."""
    if kornia is None:
        return pred.new_tensor(0.0)
    pred_lab = kornia.color.rgb_to_lab(pred)
    target_lab = kornia.color.rgb_to_lab(target)
    return F.l1_loss(pred_lab, target_lab)


class VGG19StyleEncoder(nn.Module):
    """
    VGG19 feature extractor for style loss.
    Captures relu{1..5}_1 as in Johnson et al. / Neural Photo Finishing.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if vgg19 is None:
            raise RuntimeError("torchvision is required for VGG19 style loss.")
        if pretrained and VGG19_Weights is not None:
            weights = VGG19_Weights.IMAGENET1K_V1
            backbone = vgg19(weights=weights).features
        else:
            backbone = vgg19(weights=None).features
        self.features = backbone.eval()
        for p in self.features.parameters():
            p.requires_grad = False
        # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 indices in torchvision VGG19 features.
        self.capture_ids = {1, 6, 11, 20, 29}
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> list[Tensor]:
        x = safe_clamp01(x)
        x = (x - self.imagenet_mean) / self.imagenet_std
        feats = []
        h = x
        for idx, layer in enumerate(self.features):
            h = layer(h)
            if idx in self.capture_ids:
                feats.append(h)
        return feats


def gram_matrix(feat: Tensor) -> Tensor:
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / float(c * h * w)


def gram_style_loss(pred: Tensor, style: Tensor, encoder: nn.Module, mode: str = "batch") -> Tensor:
    """
    batch: averages Grams first, then computes one MSE.
    pair: computes MSE on per-image Grams and then averages over images.
    """
    pred_feats = encoder(pred)
    style_feats = encoder(style)
    if mode == "batch":
        loss = pred.new_tensor(0.0)
        for pf, sf in zip(pred_feats, style_feats):#loop on layers
            pred_gram_avg = gram_matrix(pf).mean(dim=0)
            style_gram_avg = gram_matrix(sf).mean(dim=0)
            loss = loss + F.mse_loss(pred_gram_avg, style_gram_avg)
    elif mode == "pair":
        loss = pred.new_tensor(0.0)
        for pf, sf in zip(pred_feats, style_feats):#loop on layers
            loss = loss + F.mse_loss(gram_matrix(pf), gram_matrix(sf))
    else:
        raise ValueError(f"Unsupported channel gram mode: {mode}")
    
    return loss


def rgb_to_yuv_bt601(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Full-range BT.601-like transform with U/V approximately in [-0.5, 0.5].
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b
    v = 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, u, v


def soft_histogram_1d(x: Tensor, bins: int = 64, sigma: float = 0.02, eps: float = 1e-8) -> Tensor:
    # x: [B, 1, H, W] in [0, 1]
    b = x.shape[0]
    values = x.reshape(b, -1)
    centers = torch.linspace(0.0, 1.0, bins, device=x.device, dtype=x.dtype).view(1, 1, bins)
    d = (values.unsqueeze(-1) - centers) / max(float(sigma), 1e-6)
    w = torch.exp(-0.5 * d.square())
    hist = w.sum(dim=1)
    hist = hist / (hist.sum(dim=-1, keepdim=True) + eps)
    return hist


def soft_histogram_2d(
    u: Tensor,
    v: Tensor,
    bins_u: int = 32,
    bins_v: int = 32,
    sigma_u: float = 0.03,
    sigma_v: float = 0.03,
    eps: float = 1e-8,
) -> Tensor:
    # u/v: [B, 1, H, W] in [0, 1]
    b = u.shape[0]
    u_vals = u.reshape(b, -1)
    v_vals = v.reshape(b, -1)
    u_centers = torch.linspace(0.0, 1.0, bins_u, device=u.device, dtype=u.dtype).view(1, 1, bins_u)
    v_centers = torch.linspace(0.0, 1.0, bins_v, device=v.device, dtype=v.dtype).view(1, 1, bins_v)

    du = (u_vals.unsqueeze(-1) - u_centers) / max(float(sigma_u), 1e-6)
    dv = (v_vals.unsqueeze(-1) - v_centers) / max(float(sigma_v), 1e-6)
    wu = torch.exp(-0.5 * du.square())
    wv = torch.exp(-0.5 * dv.square())

    hist = torch.einsum("bnk,bnl->bkl", wu, wv)
    hist = hist / (hist.sum(dim=(1, 2), keepdim=True) + eps)
    return hist


def luma_chroma_hist_loss(
    pred: Tensor,
    style: Tensor,
    bins_y: int = 64,
    bins_uv: int = 32,
    sigma_y: float = 0.02,
    sigma_uv: float = 0.03,
    mode: str = "batch"
) -> tuple[Tensor, Tensor]:
    pred = safe_clamp01(pred)
    style = safe_clamp01(style)

    y_p, u_p, v_p = rgb_to_yuv_bt601(pred)
    y_s, u_s, v_s = rgb_to_yuv_bt601(style)

    # Y is already approximately [0, 1].
    y_p = y_p.clamp(0.0, 1.0)
    y_s = y_s.clamp(0.0, 1.0)

    # Shift U/V from [-0.5, 0.5] -> [0, 1].
    u_p = (u_p + 0.5).clamp(0.0, 1.0)
    u_s = (u_s + 0.5).clamp(0.0, 1.0)
    v_p = (v_p + 0.5).clamp(0.0, 1.0)
    v_s = (v_s + 0.5).clamp(0.0, 1.0)
    #print("y_p y_s size", y_p.size(), y_s.size())
    hist_y_p = soft_histogram_1d(y_p, bins=bins_y, sigma=sigma_y)
    hist_y_s = soft_histogram_1d(y_s, bins=bins_y, sigma=sigma_y)
    hist_uv_p = soft_histogram_2d(
        u_p, v_p, bins_u=bins_uv, bins_v=bins_uv, sigma_u=sigma_uv, sigma_v=sigma_uv
    )
    hist_uv_s = soft_histogram_2d(
        u_s, v_s, bins_u=bins_uv, bins_v=bins_uv, sigma_u=sigma_uv, sigma_v=sigma_uv
    )
    if mode == "batch":
        loss_luma = F.mse_loss(hist_y_p.mean(dim=0), hist_y_s.mean(dim=0))
        loss_chroma = F.mse_loss(hist_uv_p.mean(dim=0), hist_uv_s.mean(dim=0))
    elif mode == "pair":
        loss_luma = F.mse_loss(hist_y_p, hist_y_s)
        loss_chroma = F.mse_loss(hist_uv_p, hist_uv_s)
    else:
        raise ValueError(f"Unsupported channel luma, chroma mode: {mode}")
    
    return loss_luma, loss_chroma


def wasserstein_1d_cdf_l1(hist_pred: Tensor, hist_target: Tensor) -> Tensor:
    """
    Differentiable 1D Wasserstein-1 distance from soft histograms:
    W1 ~ L1 distance between CDFs (uniform bin width).
    """
    if hist_pred.shape != hist_target.shape:
        raise ValueError(f"hist shape mismatch: {tuple(hist_pred.shape)} vs {tuple(hist_target.shape)}")
    cdf_pred = torch.cumsum(hist_pred, dim=-1)
    cdf_target = torch.cumsum(hist_target, dim=-1)
    return torch.mean(torch.abs(cdf_pred - cdf_target))


def luma_chroma_emd_loss(
    pred: Tensor,
    style: Tensor,
    bins_y: int = 64,
    bins_uv: int = 32,
    sigma_y: float = 0.02,
    sigma_uv: float = 0.03,
    mode: str = "batch",
) -> tuple[Tensor, Tensor]:
    """
    Differentiable luma/chroma EMD (Wasserstein-1) using soft histograms.
    - Luma: W1 over Y histogram.
    - Chroma: average W1 over U and V histograms.
    """
    pred = safe_clamp01(pred)
    style = safe_clamp01(style)

    y_p, u_p, v_p = rgb_to_yuv_bt601(pred)
    y_s, u_s, v_s = rgb_to_yuv_bt601(style)

    y_p = y_p.clamp(0.0, 1.0)
    y_s = y_s.clamp(0.0, 1.0)
    u_p = (u_p + 0.5).clamp(0.0, 1.0)
    u_s = (u_s + 0.5).clamp(0.0, 1.0)
    v_p = (v_p + 0.5).clamp(0.0, 1.0)
    v_s = (v_s + 0.5).clamp(0.0, 1.0)

    hist_y_p = soft_histogram_1d(y_p, bins=bins_y, sigma=sigma_y)
    hist_y_s = soft_histogram_1d(y_s, bins=bins_y, sigma=sigma_y)
    hist_u_p = soft_histogram_1d(u_p, bins=bins_uv, sigma=sigma_uv)
    hist_u_s = soft_histogram_1d(u_s, bins=bins_uv, sigma=sigma_uv)
    hist_v_p = soft_histogram_1d(v_p, bins=bins_uv, sigma=sigma_uv)
    hist_v_s = soft_histogram_1d(v_s, bins=bins_uv, sigma=sigma_uv)

    if mode == "batch":
        hist_y_p = hist_y_p.mean(dim=0, keepdim=True)
        hist_y_s = hist_y_s.mean(dim=0, keepdim=True)
        hist_u_p = hist_u_p.mean(dim=0, keepdim=True)
        hist_u_s = hist_u_s.mean(dim=0, keepdim=True)
        hist_v_p = hist_v_p.mean(dim=0, keepdim=True)
        hist_v_s = hist_v_s.mean(dim=0, keepdim=True)
    elif mode != "pair":
        raise ValueError(f"Unsupported luma/chroma EMD mode: {mode}")

    loss_luma = wasserstein_1d_cdf_l1(hist_y_p, hist_y_s)
    loss_chroma_u = wasserstein_1d_cdf_l1(hist_u_p, hist_u_s)
    loss_chroma_v = wasserstein_1d_cdf_l1(hist_v_p, hist_v_s)
    loss_chroma = 0.5 * (loss_chroma_u + loss_chroma_v)
    return loss_luma, loss_chroma


#@lru_cache(maxsize=4)
def load_dinov2_model(model_name: str = "dinov2_vits14") -> nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def semantic_descriptors_dinov2(
    x: Tensor,
    model: nn.Module,
    resize: int = 224,
) -> Tensor:
    """
    Compute per-image semantic descriptors with DINOv2.
    """
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W] input, got {tuple(x.shape)}")

    x = safe_clamp01(x)
    x = F.interpolate(x, size=(resize, resize), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std

    model = model.to(device=x.device)
    with torch.no_grad():
        out = model(x)

    if isinstance(out, dict):
        cls = out.get("x_norm_clstoken", None)
        patches = out.get("x_norm_patchtokens", None)
        parts: list[Tensor] = []
        if isinstance(cls, torch.Tensor):
            parts.append(cls)
        if isinstance(patches, torch.Tensor):
            parts.append(patches.mean(dim=1))
        if parts:
            return torch.cat(parts, dim=1)
        raise ValueError("DINOv2 dict output missing x_norm_clstoken/x_norm_patchtokens.")

    if isinstance(out, torch.Tensor):
        if out.ndim == 2:
            return out
        if out.ndim == 3:
            return out.mean(dim=1)
        if out.ndim == 4:
            return out.flatten(start_dim=2).mean(dim=2)
        raise ValueError(f"Unsupported DINOv2 tensor shape: {tuple(out.shape)}")

    raise ValueError("Unsupported DINOv2 output type.")


def semantic_gram_luv_descriptors(
    x: Tensor,
    style_encoder: nn.Module | None = None,
    semantic_encoder: nn.Module | None = None,
    bins_l: int = 64,
    bins_uv: int = 32,
    sigma_l: float = 0.02,
    sigma_uv: float = 0.03,
) -> Tensor:
    """
    Build per-image descriptors by concatenating:
    1) flattened Gram matrices (from style_encoder features or RGB tensor directly)
    2) DINOv2 semantic descriptors
    3) Y channel soft histogram
    4) UV 2D soft histogram
    """
    x = safe_clamp01(x)
    
    semantic_part = [semantic_descriptors_dinov2(x, semantic_encoder)]

    feats = style_encoder(x) if style_encoder is not None else [x]
    gram_parts = []
    for f in feats:
        gram = gram_matrix(f)
        i, j = torch.triu_indices(gram.size(1), gram.size(2), device=gram.device)
        gram_parts.append(gram[:, i, j])
    
    l, u, v = rgb_to_yuv_bt601(x)
    l = l.clamp(0.0, 1.0)
    u = (u + 0.5).clamp(0.0, 1.0)
    v = (v + 0.5).clamp(0.0, 1.0)

    hist_l = soft_histogram_1d(l, bins=bins_l, sigma=sigma_l)
    hist_uv = soft_histogram_2d(u, v, bins_u=bins_uv, bins_v=bins_uv, sigma_u=sigma_uv, sigma_v=sigma_uv)
    hist_part = torch.cat([hist_l, hist_uv.flatten(start_dim=1)], dim=1)

    return torch.cat([*semantic_part, *gram_parts, hist_part], dim=1)
