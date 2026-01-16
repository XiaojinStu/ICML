"""Losses and decoding baselines for v5.1 ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from ane_core import AngularEntropy


@dataclass
class LossPack:
    name: str


class AngularEntropyLoss(LossPack):
    def __init__(
        self,
        num_idx: List[int],
        embed_layer: torch.nn.Module,
        use_float32: bool = True,
        eps: float = 1e-4,
        cache_embeddings: bool = True,
    ) -> None:
        super().__init__(name="ane")
        self.core = AngularEntropy(
            num_idx,
            embed_layer,
            use_float32=use_float32,
            eps=eps,
            cache_embeddings=cache_embeddings,
        )

    def to(self, device: torch.device) -> "AngularEntropyLoss":
        self.core.to(device)
        return self

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return self.core(logits)


class RealDistanceEntropyLoss(LossPack):
    """Wasserstein-1 on the real line using token numeric values."""

    def __init__(
        self,
        num_idx: List[int],
        num_values: List[int],
        eps: float = 1e-8,
        normalize: bool = True,
    ) -> None:
        super().__init__(name="real_distance")
        if len(num_idx) != len(num_values):
            raise ValueError("num_idx and num_values must align")
        self.num_idx_tensor = torch.tensor(num_idx, dtype=torch.long)
        self.values = torch.tensor(num_values, dtype=torch.float32)
        self.eps = eps
        self.normalize = normalize
        vmin = float(min(num_values)) if num_values else 0.0
        vmax = float(max(num_values)) if num_values else 1.0
        self.scale = max(self.eps, vmax - vmin)

    def to(self, device: torch.device) -> "RealDistanceEntropyLoss":
        self.num_idx_tensor = self.num_idx_tensor.to(device)
        self.values = self.values.to(device)
        return self

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        num_idx = self.num_idx_tensor
        if num_idx.device != logits.device:
            num_idx = num_idx.to(logits.device)
        values = self.values
        if values.device != logits.device:
            values = values.to(logits.device)

        num_logits = logits[num_idx].float()
        probs = F.softmax(num_logits, dim=-1)

        anchor = torch.dot(probs, values)
        dist = torch.abs(values - anchor)
        loss = torch.dot(probs, dist)
        if self.normalize:
            loss = loss / self.scale

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(self.eps, device=logits.device, dtype=torch.float32, requires_grad=True)
        return loss


def mean_embedding_decode(
    logits: torch.Tensor,
    num_idx_tensor: torch.Tensor,
    embed_layer: torch.nn.Module,
) -> int:
    """One-shot decode: anchor = E_p[phi(t)], predict nearest embedding to anchor."""
    if num_idx_tensor.device != logits.device:
        num_idx_tensor = num_idx_tensor.to(logits.device)

    num_logits = logits[num_idx_tensor].float()
    probs = F.softmax(num_logits, dim=-1)

    num_embeds = embed_layer.weight[num_idx_tensor].float()
    anchor = torch.einsum("n,nd->d", probs, num_embeds)

    num_embeds_n = F.normalize(num_embeds, p=2, dim=-1)
    anchor_n = F.normalize(anchor, p=2, dim=-1)

    cos_sim = torch.matmul(num_embeds_n, anchor_n)
    best_pos = int(torch.argmax(cos_sim).item())
    return int(num_idx_tensor[best_pos].item())
