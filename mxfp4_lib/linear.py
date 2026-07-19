"""Linear layers with MXFP4 fake-quant (train) or static PTQ weights (eval)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import GROUP_SIZE, quantize_mxfp4, quantize_weight_mxfp4_static


class Mxfp4Linear(nn.Module):
    """
    Drop-in for nn.Linear.
    - train_fq=True: fake-quant weights + activations each forward (STE).
    - train_fq=False: weights assumed already PTQ'd (or call pack_weights());
      activations still dynamically MXFP4-quantized (W4A4 semantics).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = GROUP_SIZE,
        train_fq: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.train_fq = train_fq
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear: nn.Linear, train_fq: bool = True) -> "Mxfp4Linear":
        m = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            train_fq=train_fq,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            m.weight.copy_(linear.weight)
            if linear.bias is not None:
                m.bias.copy_(linear.bias)
        return m

    @torch.no_grad()
    def pack_weights(self) -> None:
        """In-place PTQ of stored weights to MXFP4-dequant floats."""
        w_dq, _ = quantize_weight_mxfp4_static(self.weight.data, self.group_size)
        self.weight.data.copy_(w_dq)
        self.train_fq = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # train_fq: quantize master weights each forward (STE only while training)
        # not train_fq: weights already PTQ-packed; still dynamic-quant activations
        if self.train_fq:
            w = quantize_mxfp4(self.weight, self.group_size, ste=self.training)
        else:
            w = self.weight
        x_q = quantize_mxfp4(x, self.group_size, ste=self.training)
        return F.linear(x_q, w, self.bias)


__all__ = ["Mxfp4Linear"]
