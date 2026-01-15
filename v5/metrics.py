"""Metrics utilities for ANE-TTA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TokenRecord:
    correct: bool
    rank: int
    prob: float


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def summarize_tokens(tokens: List[TokenRecord], topk_list: List[int]) -> Dict:
    if not tokens:
        return {
            "token_count": 0,
            "token_acc": 0.0,
            "token_acc_at_k": {str(k): 0.0 for k in topk_list},
            "token_prob_mean": 0.0,
            "rank_mean": 0.0,
            "rank_median": 0.0,
            "mrr": 0.0,
        }

    ranks = [t.rank for t in tokens]
    probs = [t.prob for t in tokens]
    acc = _safe_mean([1.0 if t.correct else 0.0 for t in tokens])
    acc_at_k = {str(k): _safe_mean([1.0 if t.rank <= k else 0.0 for t in tokens]) for k in topk_list}
    mrr = _safe_mean([1.0 / max(1, t.rank) for t in tokens])

    return {
        "token_count": len(tokens),
        "token_acc": acc,
        "token_acc_at_k": acc_at_k,
        "token_prob_mean": _safe_mean(probs),
        "rank_mean": _safe_mean(ranks),
        "rank_median": _safe_median(ranks),
        "mrr": mrr,
    }


def summarize_sequences(samples: List[List[TokenRecord]], topk_list: List[int]) -> Dict:
    if not samples:
        return {
            "sample_count": 0,
            "seq_acc": 0.0,
            "seq_acc_at_k": {str(k): 0.0 for k in topk_list},
            "avg_len": 0.0,
            "median_len": 0.0,
        }

    lengths = [len(s) for s in samples]

    seq_acc = _safe_mean([1.0 if all(t.correct for t in s) else 0.0 for s in samples])
    seq_acc_at_k = {
        str(k): _safe_mean([1.0 if all(t.rank <= k for t in s) else 0.0 for s in samples])
        for k in topk_list
    }

    return {
        "sample_count": len(samples),
        "seq_acc": seq_acc,
        "seq_acc_at_k": seq_acc_at_k,
        "avg_len": _safe_mean(lengths),
        "median_len": _safe_median(lengths),
    }


def build_metrics(samples: List[List[TokenRecord]], topk_list: List[int]) -> Dict:
    tokens = [t for s in samples for t in s]
    token_stats = summarize_tokens(tokens, topk_list)
    seq_stats = summarize_sequences(samples, topk_list)
    return {**token_stats, **seq_stats}
