"""Unit tests for MXFP4 quant + tie-scope replace."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mxfp4_lib.linear import Mxfp4Linear
from mxfp4_lib.quant import (
    GROUP_SIZE,
    e8m0_scale_from_amax,
    is_on_mxfp4_grid,
    quantize_mxfp4,
    quantize_weight_mxfp4_static,
    set_scale_mode,
)
from mxfp4_lib.replace import replace_linears_with_mxfp4


def test_roundtrip_error_bound():
    set_scale_mode("rtn")
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 3
    y = quantize_mxfp4(x, ste=False)
    # Dequant should be finite and not explode
    assert torch.isfinite(y).all()
    # Relative error: MXFP4 is coarse; allow large but bounded MSE vs scale
    mse = (x - y).pow(2).mean().item()
    assert mse < 50.0


def test_scale_amax_zero_and_pow2():
    set_scale_mode("rtn")
    z = torch.zeros(4)
    s = e8m0_scale_from_amax(z)
    assert torch.isfinite(s).all()
    # amax = 6 * 2^k → exact
    for k in (-2, 0, 3):
        amax = torch.tensor([6.0 * (2**k)])
        s = e8m0_scale_from_amax(amax, mode="rtn")
        assert abs(s.item() - (2.0**k)) < 1e-6


def test_floor_vs_rtn_difference():
    amax = torch.tensor([7.0])  # slightly above 6 → rtn may round up or down
    s_rtn = e8m0_scale_from_amax(amax, mode="rtn")
    s_floor = e8m0_scale_from_amax(amax, mode="floor")
    # floor(log2(7/6)) = floor(small positive) = 0 → scale 1
    assert s_floor.item() == 1.0
    assert s_rtn.item() in (1.0, 2.0)


def test_ste_gradient_identity():
    set_scale_mode("rtn")
    x = torch.randn(4, 32, requires_grad=True)
    y = quantize_mxfp4(x, ste=True)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    # STE: grad ≈ ones (identity through fake-quant)
    assert torch.allclose(x.grad, torch.ones_like(x.grad), atol=1e-5)


def test_weight_on_grid_after_static():
    set_scale_mode("rtn")
    w = torch.randn(16, 64)
    w_dq, _ = quantize_weight_mxfp4_static(w)
    assert is_on_mxfp4_grid(w_dq)


def test_tie_skip_lm_head():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(32, 8)
            self.model.block = nn.Linear(8, 8)
            self.lm_head = nn.Linear(8, 32, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight
            self.config = type("C", (), {"tie_word_embeddings": True})()

        def tie_weights(self):
            self.lm_head.weight = self.model.embed_tokens.weight

    m = Tiny()
    n = replace_linears_with_mxfp4(m, train_fq=True, include_lm_head=False)
    assert n == 1
    assert isinstance(m.model.block, Mxfp4Linear)
    assert isinstance(m.lm_head, nn.Linear)
    assert not isinstance(m.lm_head, Mxfp4Linear)
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()


def test_include_lm_head_unties():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(32, 8)
            self.model.block = nn.Linear(8, 8)
            self.lm_head = nn.Linear(8, 32, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight
            self.config = type("C", (), {"tie_word_embeddings": True})()

        def tie_weights(self):
            self.lm_head.weight = self.model.embed_tokens.weight

    m = Tiny()
    n = replace_linears_with_mxfp4(m, train_fq=False, include_lm_head=True)
    assert n == 2  # block + lm_head
    assert isinstance(m.lm_head, Mxfp4Linear)
    assert m.config.tie_word_embeddings is False
    assert m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr()


if __name__ == "__main__":
    test_roundtrip_error_bound()
    test_scale_amax_zero_and_pow2()
    test_floor_vs_rtn_difference()
    test_ste_gradient_identity()
    test_weight_on_grid_after_static()
    test_tie_skip_lm_head()
    test_include_lm_head_unties()
    print("all tests passed")
