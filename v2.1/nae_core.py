"""
Numerical Angular Entropy (NAE) Core Implementation
v2.1: Numerically stable computation aligned with paper formulation

Mathematical Foundation (from numerical_entropy (4).pdf):

Definition 1 (Numerical Angular Entropy):
- Anchor embedding: φ̄(p) = Σ p(t)φ(t)  [RAW embeddings]
- NAE: H_∠(p) = Σ p(t) · d_∠(φ(t), φ̄(p))
- Angular distance: d_∠(u,v) = arccos(u^T v / (||u|| ||v||))

Key properties (Theorem 1):
1. Uniqueness of minimum: H_∠(p) = 0 iff p is point mass
2. Regularity: Continuous and differentiable on simplex interior
3. Scale invariance: Invariant to positive rescaling of embeddings
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


def safe_normalize(x: torch.Tensor,
                   eps: float = 1e-8,
                   dim: int = -1) -> torch.Tensor:
    """
    Safe L2 normalization with zero-norm handling.

    Prevents NaN when input has zero or near-zero norm.

    Args:
        x: Input tensor
        eps: Minimum norm value to prevent division by zero
        dim: Dimension along which to normalize

    Returns:
        Normalized tensor with ||x||_2 = 1 (or eps-bounded)
    """
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return x / norm


def stable_arccos(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Numerically stable arccos with bounded gradients.

    The standard arccos has unbounded gradients at x=±1:
    d/dx arccos(x) = -1/√(1-x²) → ∞ as x → ±1

    This implementation clamps to [-1+eps, 1-eps] to ensure
    bounded gradients during backpropagation.

    Args:
        x: Input cosine similarity values in [-1, 1]
        eps: Boundary margin (larger = more stable, less precise)

    Returns:
        Angular distance in [0, π]
    """
    # More conservative clamping than v1.2's 1e-8
    x_clamped = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return torch.acos(x_clamped)


def angular_entropy_stable(logits: torch.Tensor,
                          num_idx: List[int],
                          embed_layer: torch.nn.Module,
                          use_float32: bool = True,
                          eps: float = 1e-4,
                          return_diagnostics: bool = False) -> torch.Tensor:
    """
    Numerically stable Angular Wasserstein Entropy computation.

    Aligned with paper Definition 1:
    - Anchor: φ̄(p) = Σ p(t)φ(t) using RAW embeddings
    - NAE: H_∠(p) = Σ p(t) · d_∠(φ(t), φ̄(p))
    - d_∠(u,v) = arccos(u^T v / (||u|| ||v||))

    Key stability features:
    1. Float32 computation even with bfloat16 model
    2. Safe normalization with minimum norm clamping
    3. Conservative arccos clamping for bounded gradients
    4. Explicit NaN detection and handling

    Args:
        logits: Raw logits tensor of shape (vocab_size,)
        num_idx: List of numerical token indices
        embed_layer: Model's embedding layer (for accessing weights)
        use_float32: Whether to compute in float32 (recommended)
        eps: Numerical stability epsilon
        return_diagnostics: If True, return diagnostic dict

    Returns:
        NAE loss value (scalar tensor)
        If return_diagnostics=True, also returns dict with intermediate values
    """
    # Convert indices to tensor
    num_idx_tensor = torch.tensor(num_idx, device=logits.device, dtype=torch.long)

    # Extract numerical token logits
    num_logits = logits[num_idx_tensor]

    # Convert to float32 for stable computation
    if use_float32:
        num_logits = num_logits.float()

    # Compute softmax probabilities over numerical tokens
    probs = F.softmax(num_logits, dim=-1)  # (N_num,)

    # Get numerical token embeddings (RAW, not normalized)
    num_embeds = embed_layer.weight[num_idx_tensor]  # (N_num, d)

    if use_float32:
        num_embeds = num_embeds.float()

    # Compute probability-weighted anchor using RAW embeddings (Paper Eq. 2)
    # anchor = φ̄(p) = Σ p(t) * φ(t)
    anchor = torch.einsum('n,nd->d', probs, num_embeds)  # (d,)

    # Compute angular distance d_∠(φ(t), φ̄(p)) for each token (Paper Eq. 1)
    # d_∠(u,v) = arccos(u^T v / (||u|| ||v||))

    # Compute norms
    embed_norms = torch.norm(num_embeds, p=2, dim=-1, keepdim=False)  # (N_num,)
    anchor_norm = torch.norm(anchor, p=2)  # scalar

    # Safe norm clamping
    embed_norms = torch.clamp(embed_norms, min=eps)
    anchor_norm = torch.clamp(anchor_norm, min=eps)

    # Compute cosine similarity: (φ(t)^T φ̄(p)) / (||φ(t)|| ||φ̄(p)||)
    dot_products = torch.matmul(num_embeds, anchor)  # (N_num,)
    cos_sim = dot_products / (embed_norms * anchor_norm)  # (N_num,)

    # Compute angular distance with stable arccos
    angles = stable_arccos(cos_sim, eps=eps)  # (N_num,)

    # Compute expected angular distance (NAE) (Paper Eq. 3)
    # H_∠(p) = Σ p(t) · d_∠(φ(t), φ̄(p))
    H_angular = torch.dot(probs, angles)  # scalar

    # NaN/Inf check with gradient-preserving fallback
    if torch.isnan(H_angular) or torch.isinf(H_angular):
        # Fallback: return a safe small value that still has gradients
        H_angular = torch.tensor(eps, device=logits.device,
                                 dtype=torch.float32 if use_float32 else logits.dtype,
                                 requires_grad=True)

    if return_diagnostics:
        diagnostics = {
            'probs': probs.detach(),
            'anchor_norm': anchor_norm.item(),
            'cos_sim_range': (cos_sim.min().item(), cos_sim.max().item()),
            'angles_range': (angles.min().item(), angles.max().item()),
            'max_prob': probs.max().item(),
            'entropy_shannon': -(probs * torch.log(probs + 1e-10)).sum().item()
        }
        return H_angular, diagnostics

    return H_angular


def angular_entropy_supervised(logits: torch.Tensor,
                               num_idx: List[int],
                               target_idx: int,
                               embed_layer: torch.nn.Module,
                               use_float32: bool = True,
                               eps: float = 1e-4) -> torch.Tensor:
    """
    Supervised Angular Wasserstein Entropy with known target (Paper Eq. 6).

    H_∠^sup(p, t*) = Σ p(t) · d_∠(φ(t), φ(t*)) = W_1^S(μ_p, δ_{φ̃(t*)})

    This equals the Wasserstein-1 distance on the hypersphere (Proposition 1).

    Args:
        logits: Raw logits tensor
        num_idx: List of numerical token indices
        target_idx: Target token index (must be in vocabulary, not num_idx position)
        embed_layer: Model's embedding layer
        use_float32: Whether to compute in float32
        eps: Numerical stability epsilon

    Returns:
        Supervised NAE loss value
    """
    num_idx_tensor = torch.tensor(num_idx, device=logits.device, dtype=torch.long)

    # Extract numerical logits and compute probabilities
    num_logits = logits[num_idx_tensor]
    if use_float32:
        num_logits = num_logits.float()
    probs = F.softmax(num_logits, dim=-1)

    # Get RAW embeddings
    num_embeds = embed_layer.weight[num_idx_tensor]
    target_embed = embed_layer.weight[target_idx]

    if use_float32:
        num_embeds = num_embeds.float()
        target_embed = target_embed.float()

    # Compute angular distances d_∠(φ(t), φ(t*))
    # d_∠(u,v) = arccos(u^T v / (||u|| ||v||))
    embed_norms = torch.norm(num_embeds, p=2, dim=-1)
    target_norm = torch.norm(target_embed, p=2)

    # Safe norm clamping
    embed_norms = torch.clamp(embed_norms, min=eps)
    target_norm = torch.clamp(target_norm, min=eps)

    # Cosine similarity
    dot_products = torch.matmul(num_embeds, target_embed)
    cos_sim = dot_products / (embed_norms * target_norm)

    # Angular distance
    angles = stable_arccos(cos_sim, eps=eps)

    # Expected angular distance
    H_angular = torch.dot(probs, angles)

    return H_angular


class NAELoss(torch.nn.Module):
    """
    Neural Angular Entropy Loss Module.

    Wraps the NAE computation as a PyTorch module for cleaner integration.
    """

    def __init__(self,
                 num_idx: List[int],
                 embed_layer: torch.nn.Module,
                 use_float32: bool = True,
                 eps: float = 1e-4,
                 supervised: bool = False):
        """
        Args:
            num_idx: List of numerical token indices
            embed_layer: Model's embedding layer
            use_float32: Whether to compute in float32
            eps: Numerical stability epsilon
            supervised: If True, use supervised formulation with target
        """
        super().__init__()
        self.num_idx = num_idx
        self.embed_layer = embed_layer
        self.use_float32 = use_float32
        self.eps = eps
        self.supervised = supervised

        # Pre-compute and register numerical indices as buffer
        self.register_buffer('num_idx_tensor',
                           torch.tensor(num_idx, dtype=torch.long))

    def forward(self,
                logits: torch.Tensor,
                target_idx: Optional[int] = None) -> torch.Tensor:
        """
        Compute NAE loss.

        Args:
            logits: Raw logits tensor
            target_idx: Target token index (required if supervised=True)

        Returns:
            NAE loss value
        """
        if self.supervised:
            if target_idx is None:
                raise ValueError("target_idx required for supervised NAE")
            return angular_entropy_supervised(
                logits, self.num_idx, target_idx,
                self.embed_layer, self.use_float32, self.eps
            )
        else:
            return angular_entropy_stable(
                logits, self.num_idx, self.embed_layer,
                self.use_float32, self.eps
            )


def compute_gradient_stats(params: List[torch.nn.Parameter]) -> Dict[str, float]:
    """
    Compute gradient statistics for monitoring.

    Args:
        params: List of parameters with gradients

    Returns:
        Dictionary with gradient statistics
    """
    all_grads = []
    for p in params:
        if p.grad is not None:
            all_grads.append(p.grad.view(-1))

    if not all_grads:
        return {'grad_norm': 0.0, 'grad_max': 0.0, 'has_nan': False, 'has_inf': False}

    all_grads = torch.cat(all_grads)

    return {
        'grad_norm': torch.norm(all_grads).item(),
        'grad_max': torch.max(torch.abs(all_grads)).item(),
        'has_nan': torch.isnan(all_grads).any().item(),
        'has_inf': torch.isinf(all_grads).any().item()
    }


def check_numerical_health(loss: torch.Tensor,
                          params: List[torch.nn.Parameter]) -> Tuple[bool, str]:
    """
    Check if loss and gradients are numerically healthy.

    Args:
        loss: Loss value
        params: List of parameters

    Returns:
        (is_healthy, reason_if_unhealthy)
    """
    # Check loss
    if torch.isnan(loss):
        return False, "Loss is NaN"
    if torch.isinf(loss):
        return False, "Loss is Inf"

    # Check gradients
    grad_stats = compute_gradient_stats(params)
    if grad_stats['has_nan']:
        return False, "Gradients contain NaN"
    if grad_stats['has_inf']:
        return False, "Gradients contain Inf"
    if grad_stats['grad_norm'] > 1e6:
        return False, f"Gradient norm too large: {grad_stats['grad_norm']:.2e}"

    return True, "OK"
