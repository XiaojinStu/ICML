"""
Numerical Token Utilities for NAE
Handles token extraction and numerical subspace constraint
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def get_numerical_tokens(tokenizer) -> Tuple[List[int], List[str]]:
    """
    Extract pure numerical tokens (digits only) from tokenizer vocabulary.

    Returns:
        num_idx: List of token indices that are pure digits
        num_tok: List of corresponding token strings
    """
    num_idx, num_tok = [], []
    for i in range(tokenizer.vocab_size):
        try:
            token = tokenizer.decode([i])
            # Only pure digit tokens (0-9, or multi-digit like "12", "456")
            if token and all(c in '0123456789' for c in token) and len(token) > 0:
                num_idx.append(i)
                num_tok.append(token)
        except:
            continue
    return num_idx, num_tok


def get_single_digit_tokens(tokenizer) -> Tuple[List[int], List[str]]:
    """
    Extract only single digit tokens (0-9) from tokenizer vocabulary.

    Returns:
        digit_idx: List of token indices for single digits
        digit_tok: List of corresponding digit strings
    """
    digit_idx, digit_tok = [], []
    for i in range(tokenizer.vocab_size):
        try:
            token = tokenizer.decode([i])
            if token in '0123456789':
                digit_idx.append(i)
                digit_tok.append(token)
        except:
            continue
    return digit_idx, digit_tok


def apply_numerical_mask(logits: torch.Tensor,
                         num_idx: List[int],
                         return_masked: bool = True) -> torch.Tensor:
    """
    Apply hard mask to constrain logits to numerical subspace.
    Sets all non-numerical token logits to -inf.

    This ensures:
    1. Output can only be numerical tokens
    2. Gradients flow only through numerical token logits
    3. Softmax probability is distributed only among numerical tokens

    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        num_idx: List of numerical token indices
        return_masked: If True, return masked logits; if False, return mask

    Returns:
        Masked logits with non-numerical positions set to -inf
    """
    # Create mask with -inf for non-numerical tokens
    mask = torch.full_like(logits, float('-inf'))

    # Convert to tensor for advanced indexing
    num_idx_tensor = torch.tensor(num_idx, device=logits.device, dtype=torch.long)

    # Keep numerical token logits unchanged
    if logits.dim() == 1:
        mask[num_idx_tensor] = logits[num_idx_tensor]
    else:
        # Handle batched logits
        mask[..., num_idx_tensor] = logits[..., num_idx_tensor]

    return mask


def get_numerical_logits(logits: torch.Tensor,
                         num_idx: List[int]) -> torch.Tensor:
    """
    Extract only numerical token logits (no masking, just extraction).

    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        num_idx: List of numerical token indices

    Returns:
        Numerical logits of shape (..., len(num_idx))
    """
    num_idx_tensor = torch.tensor(num_idx, device=logits.device, dtype=torch.long)

    if logits.dim() == 1:
        return logits[num_idx_tensor]
    else:
        return logits[..., num_idx_tensor]


def numerical_softmax(logits: torch.Tensor,
                      num_idx: List[int],
                      temperature: float = 1.0) -> torch.Tensor:
    """
    Compute softmax over numerical tokens only.

    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        num_idx: List of numerical token indices
        temperature: Softmax temperature (default 1.0)

    Returns:
        Probability distribution over numerical tokens, shape (..., len(num_idx))
    """
    num_logits = get_numerical_logits(logits, num_idx)
    return F.softmax(num_logits / temperature, dim=-1)


def is_numerical_prediction(logits: torch.Tensor,
                           num_idx: List[int],
                           threshold: float = 0.5) -> bool:
    """
    Check if the model is likely predicting a numerical token.

    Args:
        logits: Raw logits tensor
        num_idx: List of numerical token indices
        threshold: Probability mass threshold for numerical tokens

    Returns:
        True if numerical tokens have probability mass > threshold
    """
    probs = F.softmax(logits, dim=-1)
    num_idx_tensor = torch.tensor(num_idx, device=logits.device, dtype=torch.long)

    if probs.dim() == 1:
        num_mass = probs[num_idx_tensor].sum()
    else:
        num_mass = probs[..., num_idx_tensor].sum(dim=-1)

    return (num_mass > threshold).item() if num_mass.dim() == 0 else num_mass > threshold


def create_numerical_index_map(num_idx: List[int],
                               vocab_size: int,
                               device: torch.device) -> torch.Tensor:
    """
    Create a mapping tensor for fast numerical token lookup.

    Args:
        num_idx: List of numerical token indices
        vocab_size: Size of full vocabulary
        device: Target device

    Returns:
        Tensor of shape (vocab_size,) where numerical positions contain
        their index in num_idx, and non-numerical positions contain -1
    """
    index_map = torch.full((vocab_size,), -1, device=device, dtype=torch.long)
    for i, idx in enumerate(num_idx):
        index_map[idx] = i
    return index_map
