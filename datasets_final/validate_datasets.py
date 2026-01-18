"""Validate datasets under `datasets_final/int` for v8.3 integer pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


INT_RE = re.compile(r"^-?[0-9]+$")


def load(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def validate_one(path: Path) -> Dict[str, Any]:
    obj = load(path)
    if not isinstance(obj, list):
        raise ValueError(f"{path}: expected JSON list")
    n = len(obj)
    bad = 0
    missing = 0
    ids = set()
    dup = 0
    for i, ex in enumerate(obj):
        if not isinstance(ex, dict):
            bad += 1
            continue
        if "id" not in ex or "question" not in ex or "answer" not in ex:
            missing += 1
            continue
        ex_id = str(ex["id"])
        if ex_id in ids:
            dup += 1
        ids.add(ex_id)
        ans = str(ex["answer"]).strip().replace(",", "")
        if not INT_RE.fullmatch(ans):
            bad += 1
    return {"file": path.name, "n": n, "missing_keys": missing, "bad_answer": bad, "dup_id": dup}


def main() -> None:
    root = Path("datasets_final/int")
    if not root.exists():
        raise SystemExit("datasets_final/int not found")
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.glob("*.json")):
        rows.append(validate_one(p))

    ok = True
    for r in rows:
        print(r)
        if r["missing_keys"] or r["bad_answer"]:
            ok = False
    if not ok:
        sys.exit(2)
    print("OK")


if __name__ == "__main__":
    main()

