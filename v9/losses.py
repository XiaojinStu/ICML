"""Loss wrappers for v9 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from ane_core import AngularEntropy


@dataclass
class LossPack:
    name: str


class AngularEntropyLoss(LossPack):
    """Angular Numerical Entropy (ANE) loss."""

    def __init__(
        self,
        num_idx: List[int],
        embed_layer: torch.nn.Module,
        use_float32: bool = True,
        eps: float = 1e-4,
        cache_embeddings: bool = True,
        distance_mode: str = "angle",
    ) -> None:
        super().__init__(name="ane")
        self.core = AngularEntropy(
            num_idx,
            embed_layer,
            use_float32=use_float32,
            eps=eps,
            cache_embeddings=cache_embeddings,
            distance_mode=distance_mode,
        )

    def to(self, device: torch.device) -> "AngularEntropyLoss":
        self.core.to(device)
        return self

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return self.core(logits)
