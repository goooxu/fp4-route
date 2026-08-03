"""Transformer Engine Linear replace for hardware low-precision (NVFP4 / FP8 recipes).

Primary recipe on Blackwell: NVFP4BlockScaling (Tensor Core GEMM).
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


def _make_recipe(kind: str):
    """Build TE recipe by kind: nvfp4 | mxfp8 | delayed."""
    from transformer_engine.common.recipe import (
        DelayedScaling,
        Format,
        MXFP8BlockScaling,
        NVFP4BlockScaling,
    )

    k = kind.lower().replace("-", "").replace("_", "")
    if k in ("nvfp4", "nvfp4blockscaling", "fp4"):
        return NVFP4BlockScaling(), "NVFP4BlockScaling"
    if k in ("mxfp8", "mxfp8blockscaling", "fp8mx"):
        return MXFP8BlockScaling(fp8_format=Format.E4M3), "MXFP8BlockScaling"
    if k in ("delayed", "delayedscaling", "fp8"):
        return DelayedScaling(fp8_format=Format.HYBRID), "DelayedScaling"
    raise ValueError(f"unknown TE recipe kind: {kind!r} (use nvfp4|mxfp8|delayed)")


def _probe_recipe(recipe, name: str) -> bool:
    import transformer_engine.pytorch as te

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TE recipes require CUDA")
    try:
        # Dims friendly to both NVFP4 and MXFP8 block constraints
        lin = te.Linear(512, 512, bias=True).to(device).bfloat16()
        x = torch.randn(64, 512, device=device, dtype=torch.bfloat16)
        with te.autocast(enabled=True, recipe=recipe):
            y = lin(x)
        y.mean().backward()
        del lin, x, y
        torch.cuda.empty_cache()
        print(f"[te] recipe {name} OK", flush=True)
        return True
    except Exception as e:
        print(f"[te] recipe {name} failed: {type(e).__name__}: {e}", flush=True)
        return False


def get_recipe(kind: str, *, probe: bool = True):
    """Return (recipe, name) for an explicit kind: nvfp4 | mxfp8 | delayed."""
    recipe, name = _make_recipe(kind)
    if probe and not _probe_recipe(recipe, name):
        raise RuntimeError(f"TE recipe {name} not usable on this GPU/build")
    return recipe, name


def get_preferred_recipe():
    """Return (recipe, name) preferring NVFP4 → MXFP8 → DelayedScaling FP8."""
    for kind in ("nvfp4", "mxfp8", "delayed"):
        recipe, name = _make_recipe(kind)
        if _probe_recipe(recipe, name):
            return recipe, name
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


def _effective_bias(module: nn.Module) -> Optional[torch.Tensor]:
    """Return a real bias tensor, or None.

    Transformer Engine often registers ``bias`` even when the layer was built with
    ``bias=False``; in that case the Parameter is an empty tensor (numel==0).
    Treating that as ``bias is not None`` makes nn.Linear(bias=True) and then
    ``copy_`` fails with a shape mismatch (e.g. 960 vs 0).
    """
    b = getattr(module, "bias", None)
    if b is None or not torch.is_tensor(b):
        return None
    try:
        if b.numel() == 0:
            return None
    except Exception:
        return None
    return b


def revert_te_to_linear(model: nn.Module) -> int:
    """Copy te.Linear weights into nn.Linear for HF save_pretrained."""
    import transformer_engine.pytorch as te

    count = 0

    def _recurse(module: nn.Module) -> None:
        nonlocal count
        for name, child in list(module.named_children()):
            if isinstance(child, te.Linear):
                w = child.weight.detach()
                b = _effective_bias(child)
                # Prefer weight shape over attributes (more reliable across TE versions)
                out_f, in_f = int(w.shape[0]), int(w.shape[1])
                lin = nn.Linear(
                    in_f,
                    out_f,
                    bias=b is not None,
                    device=w.device,
                    dtype=w.dtype,
                )
                with torch.no_grad():
                    if lin.weight.shape != w.shape:
                        raise RuntimeError(
                            f"te.Linear {name}: weight shape {tuple(w.shape)} "
                            f"!= nn.Linear {tuple(lin.weight.shape)}"
                        )
                    lin.weight.copy_(w)
                    if b is not None and lin.bias is not None:
                        if lin.bias.shape != b.shape:
                            print(
                                f"[te] skip bias copy for {name}: "
                                f"nn {tuple(lin.bias.shape)} vs te {tuple(b.shape)}",
                                flush=True,
                            )
                        else:
                            lin.bias.copy_(b.detach())
                setattr(module, name, lin)
                count += 1
            else:
                _recurse(child)

    _recurse(model)
    return count


__all__ = [
    "count_te_linears",
    "get_preferred_recipe",
    "get_recipe",
    "make_te_autocast_ctx",
    "replace_linears_with_te",
    "revert_te_to_linear",
    "te_available",
]
