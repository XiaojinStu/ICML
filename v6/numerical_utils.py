"""Numerical token utilities (v5.1)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def _is_digit_token(token: str, allow_prefix_space: bool) -> bool:
    if not token:
        return False
    if allow_prefix_space:
        token = token.strip()
    return token.isdigit()


def get_numerical_tokens(
    tokenizer,
    allow_prefix_space: bool = False,
    vocab_size: int | None = None,
) -> Tuple[List[int], List[str], List[int]]:
    """Return numerical token ids, strings, and parsed int values."""

    if vocab_size is None:
        vocab_size = tokenizer.vocab_size

    num_idx: List[int] = []
    num_tok: List[str] = []
    num_val: List[int] = []

    for i in range(vocab_size):
        try:
            tok = tokenizer.decode([i])
        except Exception:
            continue

        if not _is_digit_token(tok, allow_prefix_space=allow_prefix_space):
            continue

        s = tok.strip() if allow_prefix_space else tok
        try:
            v = int(s)
        except Exception:
            continue

        num_idx.append(i)
        num_tok.append(s)
        num_val.append(v)

    return num_idx, num_tok, num_val


def build_num_mask(vocab_size: int, num_idx: List[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    idx = torch.tensor(num_idx, dtype=torch.long, device=device)
    mask[idx] = True
    return mask


def mask_logits_to_num(logits: torch.Tensor, num_mask: torch.Tensor) -> torch.Tensor:
    if num_mask.device != logits.device:
        num_mask = num_mask.to(logits.device)
    return logits.masked_fill(~num_mask, float("-inf"))


def numerical_softmax(logits: torch.Tensor, num_idx_tensor: torch.Tensor) -> torch.Tensor:
    if num_idx_tensor.device != logits.device:
        num_idx_tensor = num_idx_tensor.to(logits.device)
    return F.softmax(logits[num_idx_tensor].float(), dim=-1)


def build_id_to_pos(num_idx: List[int], vocab_size: int, device: torch.device) -> torch.Tensor:
    """Return tensor map: token_id -> position in num_idx, else -1."""
    mapping = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    for pos, idx in enumerate(num_idx):
        mapping[idx] = pos
    return mapping


def compute_num_prob_rank(
    logits: torch.Tensor,
    target_idx: int,
    num_idx_tensor: torch.Tensor,
    id_to_pos: torch.Tensor,
) -> Tuple[float, int]:
    """Probability/rank within numerical subset (softmax over numerical tokens only)."""
    if id_to_pos.device != logits.device:
        id_to_pos = id_to_pos.to(logits.device)
    if num_idx_tensor.device != logits.device:
        num_idx_tensor = num_idx_tensor.to(logits.device)

    pos = int(id_to_pos[target_idx].item()) if 0 <= target_idx < id_to_pos.shape[0] else -1
    num_logits = logits[num_idx_tensor].float()

    if pos < 0:
        return 0.0, int(num_logits.shape[0] + 1)

    target_logit = num_logits[pos]
    lse = torch.logsumexp(num_logits, dim=-1)
    prob = torch.exp(target_logit - lse).item()
    rank = int((num_logits > target_logit).sum().item() + 1)
    return float(prob), int(rank)
