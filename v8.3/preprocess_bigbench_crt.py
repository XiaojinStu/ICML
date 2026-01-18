"""Preprocess BIG-bench chinese_remainder_theorem into our unified JSON list format.

Input (BIG-bench original):
  {
    ...,
    "examples": [{"input": "...", "target": "27043"}, ...]
  }

Output (our format):
  [{"id": "...", "question": "...", "answer": "27043"}, ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from metrics import normalize_int_answer


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def convert(obj: Any, *, limit: int = 0) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict) or not isinstance(obj.get("examples"), list):
        raise ValueError("Expected BIG-bench dict with `examples` list")

    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(obj["examples"]):
        if limit and len(out) >= int(limit):
            break
        if not isinstance(ex, dict):
            continue
        q = ex.get("input")
        a = ex.get("target")
        if q is None or a is None:
            continue
        out.append({"id": f"bigbench-crt-{i}", "question": str(q), "answer": normalize_int_answer(a)})
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess BIG-bench CRT dataset into JSON list format.")
    p.add_argument("--input", default="data/Big-bench_chinese_remainder_theorem.json")
    p.add_argument("--output", default="data/bigbench_chinese_remainder_theorem_500.json")
    p.add_argument("--limit", type=int, default=0, help="0 means all")
    return p


def main() -> None:
    args = build_parser().parse_args()
    inp = Path(args.input)
    out = Path(args.output)

    obj = load_json(inp)
    converted = convert(obj, limit=int(args.limit))
    write_json(out, converted)
    print(f"Wrote {len(converted)} examples to {out}")


if __name__ == "__main__":
    main()

