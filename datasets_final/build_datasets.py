"""Build finalized datasets for main experiments from `data/final_raw_data`.

Goal: produce unified JSON list format:
  [{"id": "...", "question": "...", "answer": "13"}, ...]

Notes:
- Current v8.3 pipeline assumes integer answers (optionally negative).
- Some raw datasets contain floats/fractions; we keep them in `datasets_final/raw`
  and additionally provide integer-only subsets when well-defined.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


INTISH_FLOAT_RE = re.compile(r"^([+-]?)(\d+)\.(\d+)$")


def read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_intish(answer: Any) -> Tuple[bool, Optional[str]]:
    """Return (ok, normalized_integer_string_or_none).

    Accepts:
    - int / str integers: "-13", "13"
    - floats with all-zero fraction: "3350.0000" -> "3350"
    """

    s = str(answer).strip()
    s = s.replace(",", "")
    s = s.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    if s.startswith("+"):
        s = s[1:]
    if not s:
        return False, None
    if s.startswith("-"):
        sign = "-"
        rest = s[1:]
    else:
        sign = ""
        rest = s

    if rest.isdigit():
        return True, sign + rest

    m = INTISH_FLOAT_RE.fullmatch(s)
    if m:
        sign2, intpart, frac = m.group(1), m.group(2), m.group(3)
        if set(frac) == {"0"}:
            return True, ("-" if sign2 == "-" else "") + intpart

    return False, None


def to_std_list(items: Iterable[Dict[str, Any]], *, id_prefix: str, id_key: str = "id") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(items):
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue
        ok, norm = normalize_intish(a)
        if not ok or norm is None:
            continue
        out.append(
            {
                "id": ex.get(id_key) or f"{id_prefix}-{i}",
                "question": str(q),
                "answer": norm,
            }
        )
    return out


def sample_items(items: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    idx = idx[:n]
    return [items[i] for i in idx]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    src: Path
    kind: str  # list | dict_data


def build_all(*, raw_root: Path, out_root: Path, seed: int, mixed_n: int) -> Dict[str, Path]:
    out_paths: Dict[str, Path] = {}

    out_int = out_root / "int"
    out_raw = out_root / "raw"
    out_int.mkdir(parents=True, exist_ok=True)
    out_raw.mkdir(parents=True, exist_ok=True)

    # 1) Already-integer datasets (or integer answers)
    for key, fname in [
        ("addition50", "addition_problems_dataset(1-50)(1).json"),
        ("gsm8k_test500", "gsm8k_test_500.json"),
        ("bigbench_arithmetic200", "bigbench_arithmetic_200.json"),
    ]:
        src = raw_root / fname
        data = read_json(src)
        if not isinstance(data, list):
            raise ValueError(f"{src} expected list")
        std = []
        for i, ex in enumerate(data):
            q = ex.get("question")
            a = ex.get("answer")
            if q is None or a is None:
                continue
            ok, norm = normalize_intish(a)
            if not ok or norm is None:
                continue
            std.append({"id": ex.get("id", f"{key}-{i}"), "question": str(q), "answer": norm})
        out = out_int / f"{key}_v1.json"
        write_json(out, std)
        out_paths[key] = out

    # 1b) BIG-bench CRT already used in v8.3 (if present in repo root data/)
    crt_src_candidates = [
        Path("data/bigbench_chinese_remainder_theorem_500.json"),
        Path("data/Big-bench_chinese_remainder_theorem.json"),
    ]
    for crt_src in crt_src_candidates:
        if not crt_src.exists():
            continue
        crt_obj = read_json(crt_src)
        if isinstance(crt_obj, dict) and isinstance(crt_obj.get("examples"), list):
            # BIG-bench original format
            crt_list = []
            for i, ex in enumerate(crt_obj["examples"]):
                if not isinstance(ex, dict):
                    continue
                q = ex.get("input")
                a = ex.get("target")
                if q is None or a is None:
                    continue
                ok, norm = normalize_intish(a)
                if not ok or norm is None:
                    continue
                crt_list.append({"id": f"bigbench-crt-{i}", "question": str(q), "answer": norm})
        elif isinstance(crt_obj, list):
            crt_list = []
            for i, ex in enumerate(crt_obj):
                if not isinstance(ex, dict):
                    continue
                q = ex.get("question", ex.get("input"))
                a = ex.get("answer", ex.get("target"))
                if q is None or a is None:
                    continue
                ok, norm = normalize_intish(a)
                if not ok or norm is None:
                    continue
                crt_list.append({"id": ex.get("id", f"bigbench-crt-{i}"), "question": str(q), "answer": norm})
        else:
            continue

        out_crt = out_int / "bigbench_crt_test500_v1.json"
        write_json(out_crt, crt_list)
        out_paths["bigbench_crt_test500"] = out_crt
        break

    # 2) BIG-bench mixed_number_string: dict with `data` list; sample 300
    mixed_src = raw_root / "bigbench_mixed_number_string_500_per_sample.json"
    mixed_obj = read_json(mixed_src)
    if not isinstance(mixed_obj, dict) or not isinstance(mixed_obj.get("data"), list):
        raise ValueError(f"{mixed_src} expected dict with key `data` list")
    mixed_list = mixed_obj["data"]
    # Standardize first, then sample.
    mixed_std_all = []
    for i, ex in enumerate(mixed_list):
        if not isinstance(ex, dict):
            continue
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue
        ok, norm = normalize_intish(a)
        if not ok or norm is None:
            continue
        ex_id = ex.get("id") or ex.get("idx") or f"mixed-{i}"
        mixed_std_all.append({"id": f"bb-mns-{ex_id}", "question": str(q), "answer": norm})

    mixed_std = sample_items(mixed_std_all, mixed_n, seed)
    out = out_int / f"bigbench_mixed_number_string{mixed_n}_v1_seed{seed}.json"
    write_json(out, mixed_std)
    out_paths[f"bigbench_mixed_number_string{mixed_n}"] = out

    # 3) math401: keep raw; also produce intish subset
    math_src = raw_root / "math401_all.json"
    math = read_json(math_src)
    if not isinstance(math, list):
        raise ValueError(f"{math_src} expected list")
    write_json(out_raw / "math401_all_raw_v1.json", math)
    math_int = []
    for i, ex in enumerate(math):
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue
        ok, norm = normalize_intish(a)
        if not ok or norm is None:
            continue
        math_int.append(
            {
                "id": ex.get("id", f"math401-{i}"),
                "question": str(q),
                "answer": norm,
                "answer_raw": ex.get("answer"),
            }
        )
    out_math_int = out_int / f"math401_int{len(math_int)}_v1.json"
    write_json(out_math_int, math_int)
    out_paths["math401_int"] = out_math_int

    # 4) nupa: keep raw; also produce intish subset
    nupa_src = raw_root / "nupa_test_440.json"
    nupa = read_json(nupa_src)
    if not isinstance(nupa, list):
        raise ValueError(f"{nupa_src} expected list")
    write_json(out_raw / "nupa_test440_raw_v1.json", nupa)
    nupa_int = []
    for i, ex in enumerate(nupa):
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue
        ok, norm = normalize_intish(a)
        if not ok or norm is None:
            continue
        nupa_int.append(
            {
                "id": ex.get("id", f"nupa-{i}"),
                "question": str(q),
                "answer": norm,
                "answer_raw": ex.get("answer"),
                "category": ex.get("category"),
                "subcategory": ex.get("subcategory"),
            }
        )
    out_nupa_int = out_int / f"nupa_test440_int{len(nupa_int)}_v1.json"
    write_json(out_nupa_int, nupa_int)
    out_paths["nupa_int"] = out_nupa_int

    # 5) NumericBench: keep raw only (float task); do NOT coerce to integer for main pipeline
    nb_src = raw_root / "NumericBench_all.json"
    nb = read_json(nb_src)
    if not isinstance(nb, list):
        raise ValueError(f"{nb_src} expected list")
    write_json(out_raw / "numericbench_all_raw_v1.json", nb)
    out_paths["numericbench_raw"] = out_raw / "numericbench_all_raw_v1.json"

    return out_paths


def compute_basic_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    q_lens = [len(str(ex.get("question", ""))) for ex in items]
    a_strs = [str(ex.get("answer", "")) for ex in items]
    digit_lens = []
    neg = 0
    for s in a_strs:
        s = s.strip()
        if s.startswith("-"):
            neg += 1
            s = s[1:]
        digit_lens.append(len(s))
    return {
        "n": n,
        "q_len_mean": float(sum(q_lens) / max(1, n)),
        "q_len_p50": float(np_percentile(q_lens, 50)),
        "answer_neg_rate": float(neg / max(1, n)),
        "answer_digit_len_counts": dict(Counter(digit_lens)),
    }


def np_percentile(xs: List[int], p: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * (p / 100.0)
    f = int(k)
    c = min(len(xs2) - 1, f + 1)
    if f == c:
        return float(xs2[f])
    return float(xs2[f] + (xs2[c] - xs2[f]) * (k - f))


def render_readme(out_root: Path) -> None:
    lines = []
    lines.append("# Final Datasets (Main Experiments)\n")
    lines.append("本文件夹由 `datasets_final/build_datasets.py` 从 `data/final_raw_data` 生成。\n")
    lines.append("统一格式：JSON 列表，每条样本至少包含 `id/question/answer`，其中 `answer` 为整数（字符串）。\n")
    lines.append("\n## 目录结构\n")
    lines.append("- `int/`：可直接接入当前 v8.3（整数输出）主实验流水线\n")
    lines.append("- `raw/`：原始数据（包含小数/分数等），需要后续扩展 float/fraction 支持\n")

    lines.append("\n## 数据集统计（int/）\n")
    lines.append("| dataset_key | filename | n | q_len_mean | q_len_p50 | neg_rate |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")

    for p in sorted((out_root / "int").glob("*.json")):
        items = read_json(p)
        if not isinstance(items, list):
            continue
        st = compute_basic_stats(items)
        lines.append(
            f"| {p.stem} | `{p.name}` | {st['n']} | {st['q_len_mean']:.1f} | {st['q_len_p50']:.0f} | {st['answer_neg_rate']:.3f} |\n"
        )

    lines.append("\n## 备注\n")
    lines.append("- `math401_all`/`nupa_test_440` 在 raw 里包含大量非整数答案；在 int 里我们保留了可严格视作整数的子集（例如 `3350.0000 -> 3350`）。\n")
    lines.append("- `NumericBench_all` 主要是两位小数输出任务，目前只保留 raw 版本，暂不纳入整数主流水线。\n")
    (out_root / "README.md").write_text("".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build finalized datasets folder for main experiments.")
    p.add_argument("--raw_root", default="data/final_raw_data")
    p.add_argument("--out_root", default="datasets_final")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_n", type=int, default=300)
    return p


def main() -> None:
    args = build_parser().parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    build_all(raw_root=raw_root, out_root=out_root, seed=int(args.seed), mixed_n=int(args.mixed_n))
    render_readme(out_root)
    print(f"Wrote datasets to {out_root}")


if __name__ == "__main__":
    main()
