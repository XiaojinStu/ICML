"""Dataset loading utilities for v9.

v9 expects a unified local JSON format:
  [{"id": "...", "question": "...", "answer": "<numeric-string>"}, ...]

Default dataset paths come from `datasets_final/main/*.json` (generated from
`data/final_raw_data`).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from metrics import normalize_answer


def load_json_any(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_json_list(path: str | Path) -> List[Dict[str, Any]]:
    obj = load_json_any(path)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return obj


def _slice_shuffle(items: List[Dict[str, Any]], *, num_samples: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    if shuffle:
        rng = random.Random(seed)
        idx = list(range(len(items)))
        rng.shuffle(idx)
        items = [items[i] for i in idx]
    if num_samples is not None and num_samples > 0:
        items = items[:num_samples]
    return items


def load_standard_json(
    *,
    data_path: str,
    num_samples: int,
    seed: int,
    shuffle: bool,
) -> List[Dict[str, Any]]:
    data = load_json_list(data_path)
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            continue
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue
        out.append({"id": ex.get("id", f"sample-{i}"), "question": str(q), "answer": normalize_answer(a)})
    return _slice_shuffle(out, num_samples=num_samples, seed=seed, shuffle=shuffle)


def _final_dataset_path(name: str) -> Optional[str]:
    mapping = {
        # Main datasets (numeric-string answers)
        "addition50_v1": "datasets_final/main/addition50_v1.json",
        "gsm8k_test500_v1": "datasets_final/main/gsm8k_test500_v1.json",
        "bigbench_arithmetic200_v1": "datasets_final/main/bigbench_arithmetic200_v1.json",
        "bigbench_mixed_number_string300_seed42_v1": "datasets_final/main/bigbench_mixed_number_string300_seed42_v1.json",
        "math401_all401_v1": "datasets_final/main/math401_all401_v1.json",
        "nupa_test440_v1": "datasets_final/main/nupa_test440_v1.json",
        "numericbench_all31200_v1": "datasets_final/main/numericbench_all31200_v1.json",
    }
    return mapping.get(name)


def load_dataset(
    *,
    name: str,
    data_path: str | None,
    num_samples: int,
    seed: int,
    shuffle: bool,
) -> List[Dict[str, Any]]:
    name = name.strip().lower()

    aliases = {
        "addition_50": "addition50_v1",
        "addition50": "addition50_v1",
        "add50": "addition50_v1",
        "gsm8k": "gsm8k_test500_v1",
        "bigbench_arithmetic": "bigbench_arithmetic200_v1",
        "bb_arithmetic": "bigbench_arithmetic200_v1",
        "bigbench_mixed_number_string": "bigbench_mixed_number_string300_seed42_v1",
        "bb_mixed_number_string": "bigbench_mixed_number_string300_seed42_v1",
        "math401": "math401_all401_v1",
        "nupa": "nupa_test440_v1",
        "numericbench": "numericbench_all31200_v1",
    }
    name = aliases.get(name, name)

    if name in {"generic", "json", "standard"}:
        if not data_path:
            raise ValueError("generic/standard requires --data_path")
        return load_standard_json(data_path=data_path, num_samples=num_samples, seed=seed, shuffle=shuffle)

    final_path = _final_dataset_path(name)
    if final_path:
        return load_standard_json(data_path=data_path or final_path, num_samples=num_samples, seed=seed, shuffle=shuffle)

    raise ValueError(f"Unknown dataset: {name}")

