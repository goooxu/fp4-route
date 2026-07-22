"""Runtime assertions for eval / PTQ correctness."""

from __future__ import annotations

import torch
import torch.nn as nn

from .linear import Mxfp4Linear
from .quant import GROUP_SIZE, _E2M1_POS, is_on_mxfp4_grid
from .replace import _get_embed_and_lm_head


def assert_default_tie_scope(model: nn.Module, *, expect_tied: bool = True) -> None:
    """
    Default quant scope: lm_head and embed share storage when tied; neither is Mxfp4Linear.
    """
    embed, lm_head = _get_embed_and_lm_head(model)
    if embed is None or lm_head is None:
        raise AssertionError("model missing embed_tokens or lm_head")
    if isinstance(lm_head, Mxfp4Linear):
        raise AssertionError("lm_head must remain nn.Linear under default quant scope")
    if isinstance(embed, Mxfp4Linear):
        raise AssertionError("embed_tokens must not be Mxfp4Linear")
    if expect_tied:
        if lm_head.weight.data_ptr() != embed.weight.data_ptr():
            raise AssertionError(
                "expected tied lm_head.weight is embed_tokens.weight "
                f"(ptrs {lm_head.weight.data_ptr()} vs {embed.weight.data_ptr()})"
            )


def assert_mxfp4_weights_on_grid(
    model: nn.Module,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    *,
    max_modules: int = 8,
) -> int:
    """
    For PTQ Mxfp4Linear (train_fq=False), weights must lie on the MXFP4 grid.

    Checks up to `max_modules` layers (full check is too slow for large models).
    """
    n = 0
    checked = 0
    for name, m in model.named_modules():
        if not isinstance(m, Mxfp4Linear):
            continue
        if m.train_fq:
            continue
        n += 1
        if checked >= max_modules:
            continue
        if not is_on_mxfp4_grid(m.weight.data, m.group_size, atol=atol, rtol=rtol):
            raise AssertionError(f"Mxfp4Linear {name!r} weights not on MXFP4 grid")
        checked += 1
    return n


def fast_grid_ok(weight: torch.Tensor, group_size: int = GROUP_SIZE, atol: float = 1e-4) -> bool:
    """Vectorized-ish grid check via re-quant with floor mode (more idempotent)."""
    from .quant import quantize_weight_mxfp4_static, set_scale_mode, get_scale_mode

    prev = get_scale_mode()
    try:
        set_scale_mode("floor")
        w_dq, _ = quantize_weight_mxfp4_static(weight, group_size, scale_mode="floor")
        # For floor-packed weights, one more pass should be close; for rtn-packed,
        # fall back to searching membership on a random subset of rows.
        if torch.allclose(weight.float(), w_dq.float(), atol=atol, rtol=1e-3):
            return True
    finally:
        set_scale_mode(prev)
    # Sample up to 32 rows
    w = weight.detach().float()
    rows = min(32, w.shape[0])
    idx = torch.linspace(0, w.shape[0] - 1, rows).long()
    return is_on_mxfp4_grid(w[idx], group_size, atol=atol)


__all__ = ["assert_default_tie_scope", "assert_mxfp4_weights_on_grid", "fast_grid_ok"]
