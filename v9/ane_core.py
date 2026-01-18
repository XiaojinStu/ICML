"""ANE core: stable angular numerical entropy with optional embedding cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


def stable_arccos(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Stable arccos with bounded gradients near |x|=1."""
    return torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps))


@dataclass
class ANEDiagnostics:
    max_prob: float
    anchor_norm: float
    cos_min: float
    cos_max: float
    angle_min: float
    angle_max: float
    anchor_argmax_pos: int


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

    def __call__(self, logits: torch.Tensor, *, return_diag: bool = False, return_anchor: bool = False):
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

        # Anchor direction on the unit sphere.
        anchor_dir = anchor / anchor_norm

        if not return_diag and not return_anchor:
            return loss

        out = [loss]
        if return_diag:
            diag = ANEDiagnostics(
                max_prob=float(probs.max().item()),
                anchor_norm=float(anchor_norm.item()),
                cos_min=float(cos_sim.min().item()),
                cos_max=float(cos_sim.max().item()),
                angle_min=float(angles.min().item()),
                angle_max=float(angles.max().item()),
                anchor_argmax_pos=int(torch.argmax(cos_sim).item()),
            )
            out.append(diag)
        if return_anchor:
            out.append(anchor_dir)

        return tuple(out) if len(out) > 1 else out[0]


def compute_gradient_stats(params: List[torch.nn.Parameter]) -> Tuple[float, bool, bool]:
    """Compute (global) grad-norm and NaN/Inf flags without concatenating tensors."""
    total_sq = None
    has_nan = False
    has_inf = False

    for p in params:
        g = p.grad
        if g is None:
            continue
        if not has_nan and torch.isnan(g).any().item():
            has_nan = True
        if not has_inf and torch.isinf(g).any().item():
            has_inf = True

        gn = torch.norm(g)
        gn2 = gn.float() * gn.float()
        total_sq = gn2 if total_sq is None else total_sq + gn2

    if total_sq is None:
        return 0.0, False, False
    return float(torch.sqrt(total_sq).item()), has_nan, has_inf


def check_numerical_health(
    loss: torch.Tensor,
    params: List[torch.nn.Parameter],
    max_grad_norm: float = 1e6,
) -> Tuple[bool, str]:
    if torch.isnan(loss):
        return False, "loss_nan"
    if torch.isinf(loss):
        return False, "loss_inf"

    grad_norm, has_nan, has_inf = compute_gradient_stats(params)
    if has_nan:
        return False, "grad_nan"
    if has_inf:
        return False, "grad_inf"
    if grad_norm > max_grad_norm:
        return False, "grad_exploding"
    return True, "ok"
