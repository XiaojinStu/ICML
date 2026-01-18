"""SVAMP adapter -> local JSON [{id, question, answer}].

This mirrors the style of `gsm8k_dataset.py` to produce a list of dicts
with `id`, `question`, and integer `answer` as a string.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from gsm8k_dataset import normalize_int_answer


def _sample(items: List[Dict[str, Any]], num_samples: int, seed: int) -> List[Dict[str, Any]]:
    if num_samples <= 0 or num_samples >= len(items):
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:num_samples]]


def load_svamp(
    *,
    split: str = "test",
    num_samples: int = 500,
    seed: int = 42,
    data_dir: str | None = None,
    use_de: bool = False,
) -> List[Dict[str, Any]]:
    """Load SVAMP from local parquet files under a `SVAMP` dataset folder.

    The loader expects files at `<data_dir>/SVAMP/data/{split}-00000-of-00001.parquet`
    when `data_dir` is provided, or from a sibling `SVAMP` directory next to this file.
    """
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "SVAMP"
    parquet_path = base / "data" / f"{split}-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Could not find SVAMP parquet at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    items: List[Dict[str, Any]] = []
    for i, row in enumerate(df.itertuples(index=False)):
        # row field names vary; use dict-like access via pandas
        r = row._asdict() if hasattr(row, "_asdict") else dict(row._asdict())
        # prefer the concatenated question if present
        q = None
        if use_de and "question_concat_DE" in r and r.get("question_concat_DE"):
            q = r["question_concat_DE"]
        elif "question_concat" in r and r.get("question_concat"):
            q = r["question_concat"]
        else:
            # fallback to Body + Question
            body = r.get("Body") or r.get("Body_DE") or ""
            question = r.get("Question") or r.get("Question_DE") or ""
            q = (body + "\n" + question).strip()

        ans_raw = r.get("Answer")
        ans = normalize_int_answer(ans_raw)
        if ans is None:
            continue
        items.append({"id": f"svamp-{split}-{i}", "question": q, "answer": ans})

    return _sample(items, num_samples, seed)
