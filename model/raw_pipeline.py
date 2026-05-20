from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for demosaicnet path
    torch = None

try:
    import demosaicnet
except Exception:  # pragma: no cover - optional dependency for demosaicnet path
    demosaicnet = None

_FFDNET_MODELS: Dict[tuple[str, str], "torch.nn.Module"] = {}


def _parse_csv_floats(value: str, n: int, name: str) -> np.ndarray:
    arr = np.array([float(x.strip()) for x in value.split(",")], dtype=np.float32)
    if arr.size != n:
        raise ValueError(f"{name} must have {n} comma-separated values, got {arr.size}: {value}")
    return arr


def parse_levels(csv_4: str, name: str) -> np.ndarray:
    return _parse_csv_floats(csv_4, 4, name)


def parse_white_balance(csv_4: str) -> np.ndarray:
    return _parse_csv_floats(csv_4, 4, "white_balance")


def parse_raw4_order(raw4_order: str) -> Tuple[int, int, int, int]:
    """
    Return indices mapping to canonical order [R, Gr, Gb, B].

    Example:
      raw4_order='R,Gr,Gb,B' -> (0,1,2,3)
      raw4_order='B,Gr,R,Gb' -> (2,1,3,0)
    """
    labels = [x.strip() for x in raw4_order.split(",")]
    if len(labels) != 4:
        raise ValueError(f"raw4_order must have 4 labels, got {len(labels)}: {raw4_order}")

    normalized = []
    for label in labels:
        key = label.lower()
        if key in ("r",):
            normalized.append("R")
        elif key in ("gr", "g1"):
            normalized.append("Gr")
        elif key in ("gb", "g2"):
            normalized.append("Gb")
        elif key in ("b",):
            normalized.append("B")
        else:
            raise ValueError(f"Unsupported raw4 channel label: {label}")

    required = ["R", "Gr", "Gb", "B"]
    if sorted(normalized) != sorted(required):
        raise ValueError(
            f"raw4_order must contain exactly {required}; got {normalized}"
        )

    # canonical -> source idx
    return (
        normalized.index("R"),
        normalized.index("Gr"),
        normalized.index("Gb"),
        normalized.index("B"),
    )


def load_raw_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(
            f"Expected raw4 npy with shape (H, W, 4), got shape={arr.shape} for {path}"
        )
    return arr


def reorder_raw4_to_canonical(raw4_hwc: np.ndarray, raw4_order: str) -> np.ndarray:
    idx_r, idx_gr, idx_gb, idx_b = parse_raw4_order(raw4_order)
    out = np.stack(
        [
            raw4_hwc[..., idx_r],
            raw4_hwc[..., idx_gr],
            raw4_hwc[..., idx_gb],
            raw4_hwc[..., idx_b],
        ],
        axis=-1,
    )
    return out.astype(np.float32)


def normalize_raw4(raw4_canonical: np.ndarray, black_levels: np.ndarray, white_levels: np.ndarray) -> np.ndarray:
    out = np.empty_like(raw4_canonical, dtype=np.float32)
    for c in range(4):
        denom = max(float(white_levels[c] - black_levels[c]), 1e-6)
        out[..., c] = (raw4_canonical[..., c].astype(np.float32) - float(black_levels[c])) / denom
    return np.clip(out, 0.0, 1.0)


def apply_wb_raw4(raw4_norm: np.ndarray, white_balance: np.ndarray) -> np.ndarray:
    wb = raw4_norm.astype(np.float32).copy()
    for c in range(4):
        wb[..., c] *= float(white_balance[c])
    return np.clip(wb, 0.0, 1.0)


def raw4_to_mosaic_rggb(raw4_canonical: np.ndarray) -> np.ndarray:
    """
    raw4 canonical channel order is [R, Gr, Gb, B].
    Output mosaic follows RGGB:
      R G
      G B
    """
    h, w, _ = raw4_canonical.shape
    mosaic = np.empty((h * 2, w * 2), dtype=np.float32)
    mosaic[0::2, 0::2] = raw4_canonical[..., 0]  # R
    mosaic[0::2, 1::2] = raw4_canonical[..., 1]  # Gr
    mosaic[1::2, 0::2] = raw4_canonical[..., 2]  # Gb
    mosaic[1::2, 1::2] = raw4_canonical[..., 3]  # B
    return mosaic


def demosaic_rggb_bilinear(mosaic: np.ndarray) -> np.ndarray:
    x = np.clip(mosaic, 0.0, 1.0)
    x16 = (x * 65535.0).astype(np.uint16)
    # OpenCV COLOR_BAYER_RG2RGB performs bilinear demosaicing.
    rgb16 = cv2.cvtColor(x16, cv2.COLOR_BAYER_RG2RGB)
    rgb = rgb16.astype(np.float32) / 65535.0
    return np.clip(rgb, 0.0, 1.0)


_EA_FALLBACK_WARNED = False


def demosaic_rggb_opencv_ea(mosaic: np.ndarray) -> np.ndarray:
    """
    OpenCV edge-aware demosaicing for RGGB.
    Falls back to bilinear if EA flag is unavailable in this OpenCV build.
    """
    global _EA_FALLBACK_WARNED
    x = np.clip(mosaic, 0.0, 1.0)
    x16 = (x * 65535.0).astype(np.uint16)

    # Prefer the explicit RGGB EA code available in newer OpenCV versions.
    ea_code = getattr(cv2, "COLOR_BayerRGGB2RGB_EA", None)
    if ea_code is None:
        if not _EA_FALLBACK_WARNED:
            print("[WARN] OpenCV EA demosaic flag is unavailable; falling back to bilinear.")
            _EA_FALLBACK_WARNED = True
        return demosaic_rggb_bilinear(mosaic)

    rgb16 = cv2.cvtColor(x16, ea_code)
    rgb = rgb16.astype(np.float32) / 65535.0
    return np.clip(rgb, 0.0, 1.0)


def demosaic_rggb_opencv(mosaic: np.ndarray) -> np.ndarray:
    """
    Backward-compatible alias.
    Historically "opencv" in this repo meant OpenCV bilinear demosaicing.
    """
    return demosaic_rggb_bilinear(mosaic)


_DEMOSAICNET_MODEL = None
_DEMOSAICNET_FALLBACK_WARNED = False


def _mosaicked2atrous_torch(mosaic_t: "torch.Tensor") -> "torch.Tensor":
    # Input: (B,1,H,W) RGGB mosaic, output: (B,3,H,W) sparse atrous layout.
    b, _, h, w = mosaic_t.shape
    atrous = torch.zeros((b, 3, h, w), device=mosaic_t.device, dtype=mosaic_t.dtype)
    atrous[:, 0:1, 0::2, 0::2] = mosaic_t[:, :, 0::2, 0::2]  # R
    atrous[:, 1:2, 0::2, 1::2] = mosaic_t[:, :, 0::2, 1::2]  # G (Gr)
    atrous[:, 1:2, 1::2, 0::2] = mosaic_t[:, :, 1::2, 0::2]  # G (Gb)
    atrous[:, 2:3, 1::2, 1::2] = mosaic_t[:, :, 1::2, 1::2]  # B
    return atrous


def _get_demosaicnet_model():
    global _DEMOSAICNET_MODEL
    if _DEMOSAICNET_MODEL is not None:
        return _DEMOSAICNET_MODEL

    if torch is None:
        raise RuntimeError("torch is not available for demosaicnet path.")
    if demosaicnet is None:
        raise RuntimeError("demosaicnet is not installed.")

    model = demosaicnet.BayerDemosaick(pad=True).to("cpu").eval()
    _DEMOSAICNET_MODEL = model
    return _DEMOSAICNET_MODEL


def demosaic_rggb_demosaicnet(mosaic: np.ndarray) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is not available for demosaicnet path.")
    model = _get_demosaicnet_model()
    x = np.clip(mosaic, 0.0, 1.0).astype(np.float32)
    with torch.no_grad():
        m = torch.from_numpy(x).view(1, 1, x.shape[0], x.shape[1]).to("cpu")
        atrous = _mosaicked2atrous_torch(m).flip(-1)
        rgb = model(atrous).flip(-1)
        rgb_np = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(rgb_np.astype(np.float32), 0.0, 1.0)


def demosaic_rggb(mosaic: np.ndarray, method: str) -> np.ndarray:
    global _DEMOSAICNET_FALLBACK_WARNED
    method_l = method.lower()
    if method_l in ("opencv", "bilinear"):
        return demosaic_rggb_bilinear(mosaic)
    if method_l in ("opencv_ea", "ea", "edge_aware"):
        return demosaic_rggb_opencv_ea(mosaic)
    if method_l == "demosaicnet":
        try:
            return demosaic_rggb_demosaicnet(mosaic)
        except Exception as e:
            if not _DEMOSAICNET_FALLBACK_WARNED:
                print(f"[WARN] demosaicnet failed ({e}). Falling back to OpenCV demosaic.")
                _DEMOSAICNET_FALLBACK_WARNED = True
            return demosaic_rggb_bilinear(mosaic)
    raise ValueError(f"Unsupported demosaic method: {method}")


def _resolve_existing_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    candidates = [
        p,
        Path.cwd() / p,
        Path(__file__).resolve().parents[2] / p,  # repo root + p
        Path(__file__).resolve().parents[1] / p,  # model/ + p
        Path(__file__).resolve().parents[1] / "third_party" / "ffdnet_pytorch" / "models" / "net_rgb.pth",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return p


def _remove_dataparallel_wrapper(state_dict: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _load_ffdnet_models_module(third_party_dir: Path):
    if str(third_party_dir) not in sys.path:
        sys.path.insert(0, str(third_party_dir))
    spec = importlib.util.spec_from_file_location(
        "ffdnet_models_local",
        third_party_dir / "models.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load FFDNet models.py from {third_party_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_ffdnet_model(weights_path: str | Path, device: str = "cpu"):
    if torch is None:
        raise RuntimeError("torch is required for FFDNet denoising.")

    d = str(device).lower()
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available():
        d = "cpu"
    device_obj = torch.device(d)

    wpath = _resolve_existing_path(weights_path)
    if not wpath.exists():
        raise FileNotFoundError(f"FFDNet weights not found: {weights_path}")

    cache_key = (str(wpath), str(device_obj))
    if cache_key in _FFDNET_MODELS:
        return _FFDNET_MODELS[cache_key]

    third_party_dir = wpath.parent.parent  # .../ffdnet_pytorch
    models_py = third_party_dir / "models.py"
    if not models_py.exists():
        # fallback to repo copy
        third_party_dir = Path(__file__).resolve().parents[1] / "third_party" / "ffdnet_pytorch"
        models_py = third_party_dir / "models.py"
    if not models_py.exists():
        raise FileNotFoundError(
            "FFDNet models.py not found. Expected under model/third_party/ffdnet_pytorch."
        )

    ffd_module = _load_ffdnet_models_module(third_party_dir)
    model = ffd_module.FFDNet(num_input_channels=3)

    state_dict = torch.load(str(wpath), map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = _remove_dataparallel_wrapper(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device_obj).eval()

    _FFDNET_MODELS[cache_key] = model
    return model


def ffdnet_denoise_rgb_np(
    rgb: np.ndarray,
    noise_sigma: float = 0.03,
    weights_path: str | Path = "model/third_party/ffdnet_pytorch/models/net_rgb.pth",
    device: str = "cpu",
) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"FFDNet expects HxWx3 RGB input, got {x.shape}")

    x = np.clip(x, 0.0, 1.0)
    h, w, _ = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    d = str(device).lower()
    if d == "auto":
        d = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    if d.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        d = "cpu"
    device_obj = torch.device(d)

    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device_obj, dtype=torch.float32)
    nsigma = torch.tensor([float(noise_sigma)], device=device_obj, dtype=torch.float32)
    model = _get_ffdnet_model(weights_path=weights_path, device=str(device_obj))

    with torch.no_grad():
        pred_noise = model(t, nsigma)
        out = torch.clamp(t - pred_noise, 0.0, 1.0)
    out_np = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    if pad_h:
        out_np = out_np[:-pad_h, :, :]
    if pad_w:
        out_np = out_np[:, :-pad_w, :]
    return np.clip(out_np, 0.0, 1.0).astype(np.float32)


def smoothstep_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def gamma_compress_np(x: np.ndarray, gamma: float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.power(x, 1.0 / gamma)


def apply_ccm_np(rgb: np.ndarray, cam2rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    out = flat @ cam2rgb.T
    out = out.reshape(h, w, 3)
    return np.clip(out, 0.0, 1.0)


def random_cam2rgb(seed: int = 0) -> np.ndarray:
    xyz2cams = np.array(
        [
            [[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469], [-0.0703, 0.2188, 0.6406]],
            [[0.4913, -0.0541, -0.0202], [-0.6130, 1.3513, 0.2906], [-0.1564, 0.2151, 0.7183]],
            [[0.8380, -0.2630, -0.0639], [-0.2887, 1.0725, 0.2496], [-0.0627, 0.1427, 0.5438]],
            [[0.6596, -0.2079, -0.0562], [-0.4782, 1.3016, 0.1933], [-0.0970, 0.1581, 0.5181]],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.0, 1.0, size=(xyz2cams.shape[0], 1, 1)).astype(np.float32)
    xyz2cam = (xyz2cams * weights).sum(axis=0) / np.maximum(weights.sum(), 1e-6)

    rgb2xyz = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    rgb2cam = xyz2cam @ rgb2xyz
    cam2rgb = np.linalg.inv(rgb2cam)

    cam2rgb = cam2rgb / np.maximum(cam2rgb.sum(axis=1, keepdims=True), 1e-8)
    return cam2rgb.astype(np.float32)


@dataclass
class PseudoISPConfig:
    white_balance: np.ndarray
    black_levels: np.ndarray
    white_levels: np.ndarray
    demosaic_method: str = "opencv"  # bilinear | opencv(alias) | opencv_ea | demosaicnet
    denoise_method: str = "none"  # none | ffdnet
    ffdnet_noise_sigma: float = 0.01
    ffdnet_weights: str = "model/third_party/ffdnet_pytorch/models/net_rgb.pth"
    ffdnet_device: str = "cpu"  # cpu | cuda | auto
    gamma: float = 2.2
    tone_mapping: str = "smoothstep"
    ccm_mode: str = "none"  # none | camera_pipeline_random
    ccm_seed: int = 0


def make_pseudo_rgb_from_raw4(
    raw4_canonical: np.ndarray,
    cfg: PseudoISPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      (raw4_norm_for_model_input, pseudo_rgb_target)
    """
    raw4_norm = normalize_raw4(raw4_canonical, cfg.black_levels, cfg.white_levels)
    raw4_wb = apply_wb_raw4(raw4_norm, cfg.white_balance)

    mosaic = raw4_to_mosaic_rggb(raw4_wb)
    rgb = demosaic_rggb(mosaic, method=cfg.demosaic_method)
    if cfg.denoise_method in ("none", "identity"):
        pass
    elif cfg.denoise_method == "ffdnet":
        rgb = ffdnet_denoise_rgb_np(
            rgb,
            noise_sigma=float(cfg.ffdnet_noise_sigma),
            weights_path=cfg.ffdnet_weights,
            device=cfg.ffdnet_device,
        )
    else:
        raise ValueError(f"Unsupported denoise_method: {cfg.denoise_method}")

    if cfg.ccm_mode == "camera_pipeline_random":
        cam2rgb = random_cam2rgb(seed=cfg.ccm_seed)
        rgb = apply_ccm_np(rgb, cam2rgb)
    elif cfg.ccm_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported ccm_mode: {cfg.ccm_mode}")

    rgb = gamma_compress_np(rgb, gamma=cfg.gamma)
    rgb = np.clip(rgb, 0.0, 1.0)

    if cfg.tone_mapping == "smoothstep":
        rgb = smoothstep_np(rgb)
    elif cfg.tone_mapping in ("none", "identity"):
        pass
    else:
        raise ValueError(f"Unsupported tone_mapping: {cfg.tone_mapping}")

    rgb = np.clip(rgb, 0.0, 1.0)
    return raw4_norm, rgb
