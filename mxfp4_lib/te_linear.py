"""Transformer Engine Linear replace for hardware low-precision (NVFP4/MXFP8/FP8).

Primary recipe on Blackwell: NVFP4BlockScaling (Tensor Core GEMM).
Software fake-quant (OCP MXFP4 STE) has been removed from this project.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


def te_available() -> bool:
    try:
        import transformer_engine.pytorch  # noqa: F401

        return True
    except Exception:
        return False


def get_preferred_recipe():
    """Return (recipe, name) preferring NVFP4 → MXFP8 → DelayedScaling FP8."""
    import transformer_engine.pytorch as te  # noqa: F401
    from transformer_engine.common.recipe import (
        DelayedScaling,
        Format,
        MXFP8BlockScaling,
        NVFP4BlockScaling,
    )

    candidates = [
        ("NVFP4BlockScaling", NVFP4BlockScaling()),
        ("MXFP8BlockScaling", MXFP8BlockScaling(fp8_format=Format.E4M3)),
        ("DelayedScaling", DelayedScaling(fp8_format=Format.HYBRID)),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TE recipes require CUDA")

    # Use dims divisible by 32 (MXFP8 block) and realistic GEMM shapes for NVFP4
    # (tiny 16x64 probes falsely fail NVFP4 cuBLAS algorithm selection).
    for name, recipe in candidates:
        try:
            lin = te.Linear(512, 512, bias=True).to(device).bfloat16()
            x = torch.randn(64, 512, device=device, dtype=torch.bfloat16)
            with te.autocast(enabled=True, recipe=recipe):
                y = lin(x)
            y.mean().backward()
            del lin, x, y
            torch.cuda.empty_cache()
            print(f"[te] recipe {name} OK", flush=True)
            return recipe, name
        except Exception as e:
            print(f"[te] recipe {name} failed: {type(e).__name__}: {e}", flush=True)
            continue
    raise RuntimeError("No TE low-precision recipe works on this GPU/build")


def make_te_autocast_ctx(recipe) -> Callable:
    import transformer_engine.pytorch as te

    def ctx():
        return te.autocast(enabled=True, recipe=recipe)

    return ctx


def _should_skip(name: str, module: nn.Module, include_lm_head: bool) -> bool:
    lname = name.lower()
    if "embed" in lname:
        return True
    if not include_lm_head and ("lm_head" in lname or lname.endswith("lm_head")):
        return True
    return False


def replace_linears_with_te(
    model: nn.Module,
    *,
    include_lm_head: bool = False,
) -> int:
    """Replace nn.Linear in transformer blocks with te.Linear (BF16 params)."""
    import transformer_engine.pytorch as te

    count = 0

    def _recurse(module: nn.Module, prefix: str = "") -> None:
        nonlocal count
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not _should_skip(full, child, include_lm_head):
                # TE Linear wants dims divisible by 16 for FP8/FP4 paths
                if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                    print(
                        f"[te] skip {full}: dims {child.in_features}x{child.out_features} "
                        f"not divisible by 16",
                        flush=True,
                    )
                    continue
                te_lin = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                )
                with torch.no_grad():
                    te_lin.weight.copy_(child.weight.data)
                    if child.bias is not None and te_lin.bias is not None:
                        te_lin.bias.copy_(child.bias.data)
                te_lin = te_lin.to(device=child.weight.device, dtype=child.weight.dtype)
                setattr(module, name, te_lin)
                count += 1
            else:
                _recurse(child, full)

    _recurse(model)
    return count


def count_te_linears(model: nn.Module) -> int:
    try:
        import transformer_engine.pytorch as te
    except Exception:
        return 0
    n = 0
    for m in model.modules():
        if isinstance(m, te.Linear):
            n += 1
    return n


def revert_te_to_linear(model: nn.Module) -> int:
    """Copy te.Linear weights into nn.Linear for HF save_pretrained."""
    import transformer_engine.pytorch as te

    count = 0

    def _recurse(module: nn.Module) -> None:
        nonlocal count
        for name, child in list(module.named_children()):
            if isinstance(child, te.Linear):
                lin = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                with torch.no_grad():
                    lin.weight.copy_(child.weight.detach())
                    if child.bias is not None and lin.bias is not None:
                        lin.bias.copy_(child.bias.detach())
                setattr(module, name, lin)
                count += 1
            else:
                _recurse(child)

    _recurse(model)
    return count


__all__ = [
    "count_te_linears",
    "get_preferred_recipe",
    "make_te_autocast_ctx",
    "replace_linears_with_te",
    "revert_te_to_linear",
    "te_available",
]
