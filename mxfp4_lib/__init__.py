"""MXFP4 / NVFP4 route experiment library.

Hardware low-precision uses Transformer Engine (NVFP4). Software fake-quant is removed.
"""

from .te_linear import (
    count_te_linears,
    get_preferred_recipe,
    make_te_autocast_ctx,
    replace_linears_with_te,
    revert_te_to_linear,
    te_available,
)

__all__ = [
    "count_te_linears",
    "get_preferred_recipe",
    "make_te_autocast_ctx",
    "replace_linears_with_te",
    "revert_te_to_linear",
    "te_available",
]
