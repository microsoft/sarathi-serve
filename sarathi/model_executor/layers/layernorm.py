"""Custom normalization layers."""

from typing import Optional

import torch
import torch.nn as nn

from sarathi import layernorm_ops
from sarathi.metrics.cuda_timer import CudaTimer


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_name: Optional[str] = None,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._norm_timer = CudaTimer(norm_name, layer_id=layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self._norm_timer:
            out = torch.empty_like(x)
            layernorm_ops.rms_norm(
                out,
                x,
                self.weight.data,
                self.variance_epsilon,
            )
            return out
