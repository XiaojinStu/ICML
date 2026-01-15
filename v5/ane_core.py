"""ANE core: stable angular numerical entropy with optional embedding cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def stable_arccos(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Stable arccos with bounded gradients near |x|=1."""
    return torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps))


@dataclass
class ANEStats:
    max_prob: float
    anchor_norm: float
    cos_min: float
    cos_max: float
    angle_min: float
    angle_max: float


class AngularEntropy:
    """Angular Numerical Entropy (ANE) with optional cached embeddings."""

    def __init__(
        self,
        num_idx: List[int],
        embed_layer: torch.nn.Module,
        use_float32: bool = True,
        eps: float = 1e-4,
        cache_embeddings: bool = True,
    ) -> None:
        self.num_idx = num_idx
        self.embed_layer = embed_layer
        self.use_float32 = use_float32
        self.eps = eps
        self.cache_embeddings = cache_embeddings
        self.num_idx_tensor = torch.tensor(num_idx, dtype=torch.long)
        self._cached = False
        if cache_embeddings:
            self.refresh()

    def to(self, device: torch.device) -> "AngularEntropy":
        self.num_idx_tensor = self.num_idx_tensor.to(device)
        if self._cached:
            self.num_embeds = self.num_embeds.to(device)
            self.num_norms = self.num_norms.to(device)
        return self

    def refresh(self) -> None:
        device = self.embed_layer.weight.device
        self.num_idx_tensor = self.num_idx_tensor.to(device)
        with torch.no_grad():
            num_embeds = self.embed_layer.weight[self.num_idx_tensor]
            if self.use_float32:
                num_embeds = num_embeds.float()
            num_norms = torch.norm(num_embeds, p=2, dim=-1)
            num_norms = torch.clamp(num_norms, min=self.eps)
        self.num_embeds = num_embeds
        self.num_norms = num_norms
        self._cached = True

    def __call__(self, logits: torch.Tensor, return_stats: bool = False):
        num_idx = self.num_idx_tensor
        if num_idx.device != logits.device:
            num_idx = num_idx.to(logits.device)

        num_logits = logits[num_idx]
        if self.use_float32:
            num_logits = num_logits.float()
        probs = F.softmax(num_logits, dim=-1)

        if self.cache_embeddings and self._cached:
            num_embeds = self.num_embeds
            num_norms = self.num_norms
        else:
            num_embeds = self.embed_layer.weight[num_idx]
            if self.use_float32:
                num_embeds = num_embeds.float()
            num_norms = torch.norm(num_embeds, p=2, dim=-1)
            num_norms = torch.clamp(num_norms, min=self.eps)

        anchor = torch.einsum("n,nd->d", probs, num_embeds)
        anchor_norm = torch.clamp(torch.norm(anchor, p=2), min=self.eps)

        dot = torch.matmul(num_embeds, anchor)
        cos_sim = dot / (num_norms * anchor_norm)
        angles = stable_arccos(cos_sim, eps=self.eps)

        loss = torch.dot(probs, angles)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(
                self.eps,
                device=logits.device,
                dtype=torch.float32 if self.use_float32 else logits.dtype,
                requires_grad=True,
            )

        if not return_stats:
            return loss

        stats = ANEStats(
            max_prob=float(probs.max().item()),
            anchor_norm=float(anchor_norm.item()),
            cos_min=float(cos_sim.min().item()),
            cos_max=float(cos_sim.max().item()),
            angle_min=float(angles.min().item()),
            angle_max=float(angles.max().item()),
        )
        return loss, stats


def compute_gradient_stats(params: List[torch.nn.Parameter]) -> Dict[str, float]:
    grads = [p.grad.view(-1) for p in params if p.grad is not None]
    if not grads:
        return {"grad_norm": 0.0, "grad_max": 0.0, "has_nan": False, "has_inf": False}
    flat = torch.cat(grads)
    return {
        "grad_norm": float(torch.norm(flat).item()),
        "grad_max": float(torch.max(torch.abs(flat)).item()),
        "has_nan": bool(torch.isnan(flat).any().item()),
        "has_inf": bool(torch.isinf(flat).any().item()),
    }


def check_numerical_health(
    loss: torch.Tensor, params: List[torch.nn.Parameter], max_grad_norm: float = 1e6
) -> Tuple[bool, str]:
    if torch.isnan(loss):
        return False, "loss_nan"
    if torch.isinf(loss):
        return False, "loss_inf"

    stats = compute_gradient_stats(params)
    if stats["has_nan"]:
        return False, "grad_nan"
    if stats["has_inf"]:
        return False, "grad_inf"
    if stats["grad_norm"] > max_grad_norm:
        return False, "grad_exploding"
    return True, "ok"
