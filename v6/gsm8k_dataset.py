"""GSM8K adapter -> local JSON [{id, question, answer}].

This repo's experiment code expects a list of dicts with at least:
- question: str
- answer: str  (non-negative integer, no commas/spaces)
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


_INT_RE = re.compile(r"^[0-9]+$")


def normalize_int_answer(text: str) -> Optional[str]:
    """Return a normalized non-negative integer string, else None."""
    if text is None:
        return None
    s = str(text).strip()
    s = s.replace(",", "")
    s = s.replace(" ", "")
    if s.startswith("+"):
        s = s[1:]
    if s.startswith("-"):
        return None
    if s.endswith("."):
        s = s[:-1]
    if not _INT_RE.match(s):
        return None
    return s


def extract_gsm8k_final_answer(answer_text: str) -> Optional[str]:
    """Extract GSM8K final answer after '####' and normalize to int."""
    if not answer_text:
        return None
    m = re.search(r"####\s*([^\n\r]+)", answer_text)
    if not m:
        return None
    raw = m.group(1).strip().lstrip("$")
    return normalize_int_answer(raw)


def load_local_json(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def save_json(path: str | Path, data: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _sample(items: List[Dict[str, Any]], num_samples: int, seed: int) -> List[Dict[str, Any]]:
    if num_samples <= 0 or num_samples >= len(items):
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:num_samples]]


def load_gsm8k(
    *,
    split: str = "test",
    num_samples: int = 50,
    seed: int = 42,
    cache_dir: str | None = None,
) -> List[Dict[str, Any]]:
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    items: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        ans = extract_gsm8k_final_answer(ex.get("answer", ""))
        if ans is None:
            continue
        items.append({"id": f"gsm8k-{split}-{i}", "question": ex["question"], "answer": ans})
    return _sample(items, num_samples, seed)

