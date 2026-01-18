"""Metric helpers for v9 (numeric-string answers).

v9 supports answers that are numeric strings containing digits and separators,
e.g. integers, decimals, fractions, scientific notation.

Important: Token-level metrics are computed ONLY on *digit-only tokens*
(numerical sub-vocab). Separators like '.', '/', 'e' are treated as fixed
(teacher-forced) positions.
"""

from __future__ import annotations

from typing import List, Tuple


def normalize_answer(text: str) -> str:
    """Normalize answer string for comparison/encoding (keep separators)."""
    s = str(text).strip()
    s = s.replace(",", "")
    s = s.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    if s.startswith("+"):
        s = s[1:]
    # Some tokenizers may decode a trailing '.' after numbers; keep it for now
    # but strip a lone trailing period like '13.' -> '13' to reduce noise.
    if s.endswith(".") and len(s) > 1 and s[:-1].replace("-", "").isdigit():
        s = s[:-1]
    return s


def split_sign_body(s: str) -> Tuple[str, str]:
    s = normalize_answer(s)
    if s.startswith("-"):
        return "-", s[1:]
    return "", s


def digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def encode_answer(tokenizer, gold: str) -> List[int]:
    """Encode gold answer into token ids.

    Key: force leading '-' to be a separate token (so it can be teacher-forced),
    while digit segments remain eligible for numeric-token TTA.
    """

    gold = normalize_answer(gold)
    if gold.startswith("-") and len(gold) > 1:
        sign_ids = tokenizer.encode("-", add_special_tokens=False)
        rest_ids = tokenizer.encode(gold[1:], add_special_tokens=False)
        return list(sign_ids) + list(rest_ids)
    return list(tokenizer.encode(gold, add_special_tokens=False))


def digit_accuracy(pred: str, gold: str) -> float:
    """Right-aligned digit accuracy over digit characters (ignores separators).

    - Sign must match, otherwise 0.
    - Compare digit sequences after removing non-digits.
    """

    ps, pb = split_sign_body(pred)
    gs, gb = split_sign_body(gold)
    if ps != gs:
        return 0.0

    pd = digits_only(pb)
    gd = digits_only(gb)
    if not gd or not pd:
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

