"""Metric helpers for v7."""

from __future__ import annotations

import re
from typing import List, Tuple


_INT_RE = re.compile(r"^[0-9]+$")


def normalize_int_answer(text: str) -> str:
    """Normalize an integer answer string (supports optional leading '-')."""
    s = str(text).strip()
    s = s.replace(",", "")
    s = s.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    if s.startswith("+"):
        s = s[1:]
    sign = ""
    if s.startswith("-"):
        sign = "-"
        s = s[1:]
    if s.endswith("."):
        s = s[:-1]
    if not s:
        return sign
    if not _INT_RE.match(s):
        return sign + s
    return sign + s


def split_sign_digits(s: str) -> Tuple[str, str]:
    s = normalize_int_answer(s)
    if s.startswith("-"):
        return "-", s[1:]
    return "", s


def encode_int_answer(tokenizer, gold_norm: str) -> List[int]:
    """Encode gold answer into token ids, forcing '-' to be a separate token if present."""
    gold_norm = normalize_int_answer(gold_norm)
    if gold_norm.startswith("-") and len(gold_norm) > 1:
        sign_ids = tokenizer.encode("-", add_special_tokens=False)
        rest_ids = tokenizer.encode(gold_norm[1:], add_special_tokens=False)
        return list(sign_ids) + list(rest_ids)
    return list(tokenizer.encode(gold_norm, add_special_tokens=False))


def digit_accuracy(pred: str, gold: str) -> float:
    """Right-aligned digit accuracy for integer strings (handles different lengths)."""
    ps, pd = split_sign_digits(pred)
    gs, gd = split_sign_digits(gold)
    if ps != gs:
        return 0.0
    if not gd:
        return 0.0
    if not pd:
        return 0.0
    a = list(reversed(pd))
    b = list(reversed(gd))
    n = max(len(a), len(b))
    match = 0
    for i in range(n):
        ca = a[i] if i < len(a) else None
        cb = b[i] if i < len(b) else None
        match += int(ca == cb)
    return match / max(1, n)

