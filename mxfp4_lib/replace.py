"""Replace nn.Linear modules with Mxfp4Linear across a model.

Default industrial scope: keep embedding + lm_head in BF16/FP16; only quantize
transformer-block Linears. Optional include_lm_head unties weights first.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .linear import Mxfp4Linear

# Module *names* (last path segment) skipped under the default quant scope.
DEFAULT_SKIP_NAMES = frozenset({"lm_head", "embed_tokens"})


def _get_embed_and_lm_head(model: nn.Module):
    """Return (embed_tokens, lm_head) or (None, None) if missing."""
    embed = None
    lm_head = getattr(model, "lm_head", None)
    inner = getattr(model, "model", None)
    if inner is not None:
        embed = getattr(inner, "embed_tokens", None)
    if embed is None:
        embed = getattr(model, "embed_tokens", None)
    return embed, lm_head


def untie_and_copy_lm_head(model: nn.Module) -> None:
    """
    Force untied lm_head: set tie_word_embeddings=False and copy embed → lm_head.
    Required before quantizing lm_head on tied models (e.g. SmolLM2).
    """
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
    embed, lm_head = _get_embed_and_lm_head(model)
    if embed is None or lm_head is None:
        raise RuntimeError("Cannot untie: missing embed_tokens or lm_head")
    # Detach shared storage by cloning into a fresh Parameter if needed
    with torch.no_grad():
        if lm_head.weight.data_ptr() == embed.weight.data_ptr():
            lm_head.weight = nn.Parameter(embed.weight.detach().clone())
        else:
            lm_head.weight.data.copy_(embed.weight.data)
    if hasattr(model, "tie_weights"):
        # Prevent HF from re-tying on save/load paths that call tie_weights()
        def _no_tie():
            return None

        model.tie_weights = _no_tie  # type: ignore[method-assign]


def replace_linears_with_mxfp4(
    model: nn.Module,
    train_fq: bool = True,
    *,
    include_lm_head: bool = False,
    skip_names: frozenset[str] | None = None,
) -> int:
    """
    Recursively replace nn.Linear with Mxfp4Linear.

    Default: skip embed_tokens and lm_head (keep BF16).
    If include_lm_head=True: untie + copy embed into lm_head, then quantize lm_head
    (still skip embed_tokens).
    """
    skip = set(DEFAULT_SKIP_NAMES if skip_names is None else skip_names)
    if include_lm_head:
        untie_and_copy_lm_head(model)
        skip.discard("lm_head")

    def _walk(module: nn.Module, prefix: str = "") -> int:
        count = 0
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            leaf = name
            if isinstance(child, Mxfp4Linear):
                continue
            if isinstance(child, nn.Linear):
                if leaf in skip or full in skip:
                    continue
                setattr(module, name, Mxfp4Linear.from_linear(child, train_fq=train_fq))
                count += 1
            else:
                count += _walk(child, full)
        return count

    return _walk(model)


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


def count_mxfp4_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, Mxfp4Linear))


__all__ = [
    "DEFAULT_SKIP_NAMES",
    "untie_and_copy_lm_head",
    "replace_linears_with_mxfp4",
    "revert_mxfp4_to_linear",
    "pack_all_mxfp4_weights",
    "count_mxfp4_linears",
]
