"""Dataset loading utilities for v8.3.

For the main experiments, we standardize to a simple local JSON format:
  [{"id": "...", "question": "...", "answer": "13"}, ...]

Current supported datasets:
- addition_50 (local)
- gsm8k (local JSON produced by v6.x scripts)
- bigbench_crt (BIG-bench chinese_remainder_theorem; either original dict or converted list)

Other datasets can be plugged in by converting to the same JSON schema.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from metrics import normalize_int_answer


def load_json_any(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_json_list(path: str | Path) -> List[Dict[str, Any]]:
    data = load_json_any(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def _slice_shuffle(items: List[Dict[str, Any]], num_samples: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    if shuffle:
        rng = random.Random(seed)
        idx = list(range(len(items)))
        rng.shuffle(idx)
        items = [items[i] for i in idx]
    # num_samples <= 0 means "use all"
    if num_samples is not None and num_samples > 0:
        items = items[:num_samples]
    return items


def load_addition_50(*, data_path: str | None, num_samples: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    path = data_path or "data/addition_problems_dataset(1-50)(1).json"
    data = load_json_list(path)
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(data):
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue
        out.append({"id": ex.get("id", f"addition-50-{i}"), "question": str(q), "answer": normalize_int_answer(a)})
    return _slice_shuffle(out, num_samples=num_samples, seed=seed, shuffle=shuffle)


def load_gsm8k_local(*, data_path: str | None, num_samples: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    path = data_path or "data/gsm8k/gsm8k_test_50.json"
    data = load_json_list(path)
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(data):
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue
        out.append({"id": ex.get("id", f"gsm8k-local-{i}"), "question": str(q), "answer": normalize_int_answer(a)})
    return _slice_shuffle(out, num_samples=num_samples, seed=seed, shuffle=shuffle)


def load_bigbench_crt(*, data_path: str | None, num_samples: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    """BIG-bench chinese_remainder_theorem.

    Supports:
    - Original BIG-bench format (a dict with `examples: [{input, target}, ...]`)
    - A converted JSON list in our standard format.
    """

    path = data_path or "data/bigbench_chinese_remainder_theorem_500.json"
    raw = load_json_any(path)

    examples: List[Dict[str, Any]]
    if isinstance(raw, dict) and isinstance(raw.get("examples"), list):
        examples = raw["examples"]
        out: List[Dict[str, Any]] = []
        for i, ex in enumerate(examples):
            q = ex.get("input")
            a = ex.get("target")
            if q is None or a is None:
                continue
            out.append({"id": f"bigbench-crt-{i}", "question": str(q), "answer": normalize_int_answer(a)})
        return _slice_shuffle(out, num_samples=num_samples, seed=seed, shuffle=shuffle)

    if isinstance(raw, list):
        out2: List[Dict[str, Any]] = []
        for i, ex in enumerate(raw):
            if not isinstance(ex, dict):
                continue
            q = ex.get("question", ex.get("input"))
            a = ex.get("answer", ex.get("target"))
            if q is None or a is None:
                continue
            out2.append({"id": ex.get("id", f"bigbench-crt-{i}"), "question": str(q), "answer": normalize_int_answer(a)})
        return _slice_shuffle(out2, num_samples=num_samples, seed=seed, shuffle=shuffle)

    raise ValueError(f"Unsupported JSON schema in {path}: expected dict(examples=...) or list")


def load_dataset(
    *,
    name: str,
    data_path: str | None,
    num_samples: int,
    seed: int,
    shuffle: bool,
) -> List[Dict[str, Any]]:
    name = name.strip().lower()
    if name in {"addition_50", "addition50", "add50"}:
        return load_addition_50(data_path=data_path, num_samples=num_samples, seed=seed, shuffle=shuffle)
    if name in {"gsm8k"}:
        return load_gsm8k_local(data_path=data_path, num_samples=num_samples, seed=seed, shuffle=shuffle)
    if name in {"bigbench_crt", "bb_crt", "bigbench-chinese-remainder-theorem", "chinese_remainder_theorem", "crt"}:
        return load_bigbench_crt(data_path=data_path, num_samples=num_samples, seed=seed, shuffle=shuffle)
    raise ValueError(f"Unknown dataset: {name}")
