"""Build finalized datasets for main experiments from `data/final_raw_data`.

Output schema (unified):
  [{"id": "...", "question": "...", "answer": "<numeric-string>"}, ...]

Where `answer` is a *numeric string* that may include:
- integers: `13`, `-13`
- decimals: `8.21`, `-1034.0000`
- fractions: `94801/36312`
- scientific notation (seen in NUPA): `9.300618e25`

We intentionally keep non-digit separators as-is, because v9's decoding/TTA
only targets *digit-only* tokens and treats separators (e.g. '.', '/', 'e')
as fixed (teacher-forced) positions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_numeric_string(answer: Any) -> str:
    """Normalize a numeric-string answer for consistent tokenization/metrics."""
    s = str(answer).strip()
    s = s.replace(",", "")
    s = s.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    if s.startswith("+"):
        s = s[1:]
    return s


def to_std_list(items: Iterable[Dict[str, Any]], *, id_prefix: str, id_key: str = "id") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(items):
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue
        out.append(
            {
                "id": ex.get(id_key) or f"{id_prefix}-{i}",
                "question": str(q),
                "answer": normalize_numeric_string(a),
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


def build_all(*, raw_root: Path, out_root: Path, seed: int, mixed_n: int) -> Dict[str, Path]:
    out_paths: Dict[str, Path] = {}

    out_main = out_root / "main"
    out_raw = out_root / "raw"
    out_main.mkdir(parents=True, exist_ok=True)
    out_raw.mkdir(parents=True, exist_ok=True)

    # 1) Datasets already in list format
    for key, fname in [
        ("addition50", "addition_problems_dataset(1-50)(1).json"),
        ("gsm8k_test500", "gsm8k_test_500.json"),
        ("bigbench_arithmetic200", "bigbench_arithmetic_200.json"),
        ("math401_all401", "math401_all.json"),
        ("nupa_test440", "nupa_test_440.json"),
        ("numericbench_all31200", "NumericBench_all.json"),
    ]:
        src = raw_root / fname
        data = read_json(src)
        if not isinstance(data, list):
            raise ValueError(f"{src} expected list")
        std = []
        for i, ex in enumerate(data):
            if not isinstance(ex, dict):
                continue
            q = ex.get("question")
            a = ex.get("answer")
            if q is None or a is None:
                continue
            std.append(
                {
                    "id": ex.get("id", f"{key}-{i}"),
                    "question": str(q),
                    "answer": normalize_numeric_string(a),
                    **({k: ex[k] for k in ["category", "subcategory"] if k in ex} if key.startswith("nupa") else {}),
                }
            )

        out = out_main / f"{key}_v1.json"
        write_json(out, std)
        out_paths[key] = out

    # 2) BIG-bench mixed_number_string: dict with `data` list; sample 300
    mixed_src = raw_root / "bigbench_mixed_number_string_500_per_sample.json"
    mixed_obj = read_json(mixed_src)
    if not isinstance(mixed_obj, dict) or not isinstance(mixed_obj.get("data"), list):
        raise ValueError(f"{mixed_src} expected dict with key `data` list")
    mixed_list = mixed_obj["data"]
    mixed_std_all = []
    for i, ex in enumerate(mixed_list):
        if not isinstance(ex, dict):
            continue
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue
        ex_id = ex.get("id") or ex.get("idx") or f"mixed-{i}"
        mixed_std_all.append(
            {
                "id": f"bb-mns-{ex_id}",
                "question": str(q),
                "answer": normalize_numeric_string(a),
                **({k: ex[k] for k in ["ability", "question_index"] if k in ex}),
            }
        )

    mixed_std = sample_items(mixed_std_all, mixed_n, seed)
    out = out_main / f"bigbench_mixed_number_string{mixed_n}_seed{seed}_v1.json"
    write_json(out, mixed_std)
    out_paths[f"bigbench_mixed_number_string{mixed_n}"] = out

    # 3) Also keep raw snapshots for reference
    for key, fname, out_name in [
        ("math401_all", "math401_all.json", "math401_all_raw_v1.json"),
        ("nupa_test_440", "nupa_test_440.json", "nupa_test440_raw_v1.json"),
        ("numericbench_all", "NumericBench_all.json", "numericbench_all_raw_v1.json"),
        ("bigbench_mixed_number_string", "bigbench_mixed_number_string_500_per_sample.json", "bigbench_mixed_number_string_raw_v1.json"),
    ]:
        src = raw_root / fname
        write_json(out_raw / out_name, read_json(src))

    return out_paths


def compute_basic_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    q_lens = [len(str(ex.get("question", ""))) for ex in items]
    a_strs = [str(ex.get("answer", "")) for ex in items]
    ans_char_lens = [len(s) for s in a_strs]
    digits_lens = [sum(ch.isdigit() for ch in s) for s in a_strs]
    neg = sum(str(s).strip().startswith("-") for s in a_strs)
    has_dot = sum("." in s for s in a_strs)
    has_slash = sum("/" in s for s in a_strs)
    has_sci = sum(("e" in s.lower()) for s in a_strs)
    return {
        "n": n,
        "q_len_mean": float(sum(q_lens) / max(1, n)),
        "q_len_p50": float(np_percentile(q_lens, 50)),
        "ans_char_len_mean": float(sum(ans_char_lens) / max(1, n)),
        "ans_char_len_p50": float(np_percentile(ans_char_lens, 50)),
        "ans_digits_len_mean": float(sum(digits_lens) / max(1, n)),
        "ans_digits_len_p50": float(np_percentile(digits_lens, 50)),
        "answer_neg_rate": float(neg / max(1, n)),
        "dot_rate": float(has_dot / max(1, n)),
        "slash_rate": float(has_slash / max(1, n)),
        "sci_rate": float(has_sci / max(1, n)),
        "ans_digits_len_counts": dict(Counter(digits_lens)),
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
    lines.append("统一格式：JSON 列表，每条样本至少包含 `id/question/answer`。\n")
    lines.append("其中 `answer` 是“数值字符串”，可能包含小数点/分数/科学计数法。\n")
    lines.append("\n## 目录结构\n")
    lines.append("- `main/`：主实验直接使用（统一 schema）\n")
    lines.append("- `raw/`：原始数据快照（对齐来源，便于溯源）\n")

    lines.append("\n## 数据集统计（main/）\n")
    lines.append("| dataset_key | filename | n | q_len_p50 | ans_chars_p50 | ans_digits_p50 | neg | dot | slash | sci |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for p in sorted((out_root / "main").glob("*.json")):
        items = read_json(p)
        if not isinstance(items, list):
            continue
        st = compute_basic_stats(items)
        lines.append(
            f"| {p.stem} | `{p.name}` | {st['n']} | {st['q_len_p50']:.0f} | {st['ans_char_len_p50']:.0f} | {st['ans_digits_len_p50']:.0f} | "
            f"{st['answer_neg_rate']:.3f} | {st['dot_rate']:.3f} | {st['slash_rate']:.3f} | {st['sci_rate']:.3f} |\n"
        )

    lines.append("\n## 备注\n")
    lines.append("- v9 实验会只在“digit-only token”的位置做 TTA；非数字 token（如 `.`, `/`, `e`）视为分隔符并在评测时 teacher-forcing。\n")
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
