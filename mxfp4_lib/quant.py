"""OCP-style MXFP4 helpers: E2M1 elements + E8M0 scale per group of 32."""

from __future__ import annotations

import torch

GROUP_SIZE = 32

# Positive E2M1 magnitudes
_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e8m0_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Power-of-two scale shared by a block: round(log2(amax / 6))."""
    amax = amax.to(torch.float32).clamp_min(1e-12)
    exp = torch.floor(torch.log2(amax / 6.0) + 0.5).clamp(-127, 127)
    return torch.pow(torch.tensor(2.0, device=amax.device, dtype=torch.float32), exp)


def _quantize_to_e2m1(y: torch.Tensor) -> torch.Tensor:
    """Fast nearest E2M1 quantization via clamp + half-step rounding on abs."""
    # Map |y| onto the discrete set by piecewise thresholds midpoints:
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
    scale = e8m0_scale_from_amax(amax).unsqueeze(-1)
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
    scale = e8m0_scale_from_amax(amax)
    y = wg / scale.unsqueeze(-1)
    q = _quantize_to_e2m1(y)
    w_dq = (q * scale.unsqueeze(-1)).view(out_f, in_pad)[:, :in_f]
    return w_dq.to(weight.dtype), scale


__all__ = [
    "GROUP_SIZE",
    "quantize_mxfp4",
    "quantize_weight_mxfp4_static",
    "e8m0_scale_from_amax",
]
