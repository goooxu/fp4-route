"""Replace nn.Linear modules with Mxfp4Linear across a model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .linear import Mxfp4Linear


def replace_linears_with_mxfp4(model: nn.Module, train_fq: bool = True) -> int:
    """Recursively replace every nn.Linear with Mxfp4Linear. Returns count."""
    count = 0
    for name, module in list(model.named_children()):
        if isinstance(module, Mxfp4Linear):
            continue
        if isinstance(module, nn.Linear):
            setattr(model, name, Mxfp4Linear.from_linear(module, train_fq=train_fq))
            count += 1
        else:
            count += replace_linears_with_mxfp4(module, train_fq=train_fq)
    return count


def revert_mxfp4_to_linear(model: nn.Module) -> int:
    """Convert Mxfp4Linear back to nn.Linear (copy master weights) for HF save."""
    count = 0
    for name, module in list(model.named_children()):
        if isinstance(module, Mxfp4Linear):
            lin = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            with torch.no_grad():
                lin.weight.copy_(module.weight.data)
                if module.bias is not None:
                    lin.bias.copy_(module.bias.data)
            setattr(model, name, lin)
            count += 1
        else:
            count += revert_mxfp4_to_linear(module)
    return count


def pack_all_mxfp4_weights(model: nn.Module) -> int:
    """PTQ-pack all Mxfp4Linear weights in-place."""
    n = 0
    for m in model.modules():
        if isinstance(m, Mxfp4Linear):
            m.pack_weights()
            n += 1
    return n


__all__ = [
    "replace_linears_with_mxfp4",
    "revert_mxfp4_to_linear",
    "pack_all_mxfp4_weights",
]
