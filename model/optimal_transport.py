import torch

def sinkhorn_transport_plan(
    cost: torch.Tensor,
    reg: float = 0.05,
    iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Entropic OT transport plan with uniform marginals."""
    bp, bt = cost.shape
    a = torch.full((bp,), 1.0 / max(bp, 1), device=cost.device, dtype=cost.dtype)
    b = torch.full((bt,), 1.0 / max(bt, 1), device=cost.device, dtype=cost.dtype)
    k = torch.exp(-cost / max(reg, eps)).clamp_min(eps)

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(max(iters, 1)):
        u = a / (k @ v).clamp_min(eps)
        v = b / (k.transpose(0, 1) @ u).clamp_min(eps)
    return (u[:, None] * k) * v[None, :]


def soft_pseudo_target_ot_FGW(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    reg: float = 0.05,
    temperature: float = 1.0,
    sinkhorn_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Row-stochastic soft pseudo-target weights from feature vectors via fused GW-like OT."""
    with torch.no_grad():
        pred = pred.detach().float().flatten(1).contiguous()      # [Bp, D]
        target = target.detach().float().flatten(1).contiguous()  # [Bt, D]

        d_feat = torch.cdist(pred, target, p=2).pow(2)  # [Bp, Bt]

        cp = torch.cdist(pred, pred, p=2).pow(2)   # [Bp, Bp]
        ct = torch.cdist(target, target, p=2).pow(2)  # [Bt, Bt]
        cp = cp / cp.mean(dim=1, keepdim=True).clamp_min(eps)
        ct = ct / ct.mean(dim=1, keepdim=True).clamp_min(eps)
        d_struct = torch.cdist(cp, ct, p=2).pow(2) / max(cp.shape[1], 1)

        cost = alpha * d_struct + (1.0 - alpha) * d_feat
        cost = cost / cost.mean().clamp_min(eps)
        cost = cost / max(temperature, eps)

        plan = sinkhorn_transport_plan(cost, reg=reg, iters=sinkhorn_iters, eps=eps)
        weights = plan / plan.sum(dim=1, keepdim=True).clamp_min(eps)
    return weights