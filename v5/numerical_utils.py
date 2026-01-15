"""Numerical token utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def is_digit_token(token: str, allow_prefix_space: bool = False) -> bool:
    if not token:
        return False
    if allow_prefix_space:
        token = token.strip()
    return token.isdigit()


def get_numerical_tokens(tokenizer, allow_prefix_space: bool = False) -> Tuple[List[int], List[str]]:
    num_idx, num_tok = [], []
    for i in range(tokenizer.vocab_size):
        try:
            tok = tokenizer.decode([i])
        except Exception:
            continue
        if is_digit_token(tok, allow_prefix_space=allow_prefix_space):
            num_idx.append(i)
            num_tok.append(tok.strip() if allow_prefix_space else tok)
    return num_idx, num_tok


def build_num_mask(num_idx: List[int], vocab_size: int, device: torch.device) -> torch.Tensor:
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


def build_num_lookup(num_idx: List[int], num_tokens: List[str]) -> Dict[int, str]:
    return {idx: tok for idx, tok in zip(num_idx, num_tokens)}
