"""OCP-style MXFP4 helpers: E2M1 elements + E8M0 scale per group of 32.

Scale modes:
  - rtn (default): round(log2(amax/6)) — may clamp amax slightly above 6*scale
  - floor: floor(log2(amax/6)) — OCP-style, only enlarges error, no clamp of amax
"""

from __future__ import annotations

from typing import Literal

import torch

GROUP_SIZE = 32
ScaleMode = Literal["rtn", "floor"]

# Positive E2M1 magnitudes
_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Process-wide default; scripts may set via set_scale_mode / cfg
_SCALE_MODE: ScaleMode = "rtn"


def get_scale_mode() -> ScaleMode:
    return _SCALE_MODE


def set_scale_mode(mode: ScaleMode) -> None:
    global _SCALE_MODE
    if mode not in ("rtn", "floor"):
        raise ValueError(f"scale_mode must be 'rtn' or 'floor', got {mode!r}")
    _SCALE_MODE = mode


def e8m0_scale_from_amax(
    amax: torch.Tensor,
    mode: ScaleMode | None = None,
) -> torch.Tensor:
    """Power-of-two scale shared by a block from amax / E2M1 max (6)."""
    mode = mode or _SCALE_MODE
    amax = amax.to(torch.float32).clamp_min(1e-12)
    log_ratio = torch.log2(amax / 6.0)
    if mode == "rtn":
        exp = torch.floor(log_ratio + 0.5).clamp(-127, 127)
    elif mode == "floor":
        exp = torch.floor(log_ratio).clamp(-127, 127)
    else:
        raise ValueError(mode)
    return torch.pow(torch.tensor(2.0, device=amax.device, dtype=torch.float32), exp)


def _quantize_to_e2m1(y: torch.Tensor) -> torch.Tensor:
    """Fast nearest E2M1 quantization via clamp + half-step rounding on abs."""
    # levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    # midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    a = y.abs()
    q = torch.zeros_like(a)
    q = torch.where(a >= 0.25, torch.tensor(0.5, device=a.device, dtype=a.dtype), q)
    q = torch.where(a >= 0.75, torch.tensor(1.0, device=a.device, dtype=a.dtype), q)
    q = torch.where(a >= 1.25, torch.tensor(1.5, device=a.device, dtype=a.dtype), q)
    q = torch.where(a >= 1.75, torch.tensor(2.0, device=a.device, dtype=a.dtype), q)
    q = torch.where(a >= 2.5, torch.tensor(3.0, device=a.device, dtype=a.dtype), q)
    q = torch.where(a >= 3.5, torch.tensor(4.0, device=a.device, dtype=a.dtype), q)
    q = torch.where(a >= 5.0, torch.tensor(6.0, device=a.device, dtype=a.dtype), q)
    return q * y.sign().where(y != 0, torch.ones_like(y))


def quantize_mxfp4(
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
    ste: bool = True,
    scale_mode: ScaleMode | None = None,
) -> torch.Tensor:
    """Quantize to MXFP4 then dequantize (fake-quant) along the last dimension."""
    if x.numel() == 0:
        return x

    orig_shape = x.shape
    orig_dtype = x.dtype
    x_f = x.float().reshape(-1, orig_shape[-1])
    n, d = x_f.shape

    pad = (group_size - (d % group_size)) % group_size
    if pad:
        x_f = torch.nn.functional.pad(x_f, (0, pad))

    d_pad = x_f.shape[-1]
    xg = x_f.view(n, d_pad // group_size, group_size)
    amax = xg.abs().amax(dim=-1)
    scale = e8m0_scale_from_amax(amax, mode=scale_mode).unsqueeze(-1)
    y = xg / scale
    q = _quantize_to_e2m1(y)
    y_dq = q * scale
    y_dq = y_dq.view(n, d_pad)[:, :d].reshape(orig_shape).to(orig_dtype)

    if ste and x.requires_grad:
        return x + (y_dq - x).detach()
    return y_dq


def quantize_weight_mxfp4_static(
    weight: torch.Tensor,
    group_size: int = GROUP_SIZE,
    scale_mode: ScaleMode | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PTQ: return dequantized weight and per-group E8M0 scales [out, G]."""
    w = weight.detach().float()
    out_f, in_f = w.shape
    pad = (group_size - (in_f % group_size)) % group_size
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    in_pad = w.shape[-1]
    wg = w.view(out_f, in_pad // group_size, group_size)
    amax = wg.abs().amax(dim=-1)
    scale = e8m0_scale_from_amax(amax, mode=scale_mode)
    y = wg / scale.unsqueeze(-1)
    q = _quantize_to_e2m1(y)
    w_dq = (q * scale.unsqueeze(-1)).view(out_f, in_pad)[:, :in_f]
    return w_dq.to(weight.dtype), scale


def is_on_mxfp4_grid(
    weight: torch.Tensor,
    group_size: int = GROUP_SIZE,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    scale_mode: ScaleMode | None = None,
) -> bool:
    """
    True if each group is E2M1_level * 2^e for a shared e (within tol).

    Vectorized per-group over a small exponent window (no Python nested loops
    over every element).
    """
    del scale_mode
    levels = torch.tensor(list(_E2M1_POS), dtype=torch.float32, device=weight.device)
    w = weight.detach().float()
    if w.ndim == 1:
        w = w.unsqueeze(0)
    out_f, in_f = w.shape[0], w.shape[-1]
    pad = (group_size - (in_f % group_size)) % group_size
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    in_pad = w.shape[-1]
    n_groups = in_pad // group_size
    wg = w.view(out_f, n_groups, group_size)
    amax = wg.abs().amax(dim=-1)  # [out, G]
    # Skip near-zero groups
    active = amax > atol
    if not active.any():
        return True
    e0 = torch.floor(torch.log2((amax.clamp_min(1e-12)) / 6.0) + 0.5).to(torch.int64)
    # Try e0 + {-3..+3}
    ok_any = torch.zeros_like(amax, dtype=torch.bool)
    for de in range(-3, 4):
        e = (e0 + de).clamp(-127, 127)
        scale = torch.pow(torch.tensor(2.0, device=w.device), e.float()).unsqueeze(-1)
        y = wg / scale
        a = y.abs().unsqueeze(-1)  # [O,G,32,1]
        dist = (a - levels).abs().amin(dim=-1)  # [O,G,32]
        thresh = atol + rtol * a.squeeze(-1).clamp_min(1e-12)
        ok = (dist <= thresh).all(dim=-1)  # [O,G]
        ok_any = ok_any | ok
    return bool((ok_any | ~active).all().item())


__all__ = [
    "GROUP_SIZE",
    "ScaleMode",
    "quantize_mxfp4",
    "quantize_weight_mxfp4_static",
    "e8m0_scale_from_amax",
    "get_scale_mode",
    "set_scale_mode",
    "is_on_mxfp4_grid",
]
