"""Aggregate v9.5 main experiment runs into a table + simple token-acc curve plots.

v9.5 spec:
- Run v9 TF experiments with total steps=10.
- Record token accuracy checkpoints at steps {2,5} (and 0,10).
- Use curated datasets with per-model-group learning rates.
- Use ANE metric = cosine.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
            "figure.dpi": 200,
            "savefig.dpi": 520,
            "savefig.bbox": "tight",
            "axes.linewidth": 1.0,
        }
    )


def _full_box(ax) -> None:
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)


def find_json(root: Path) -> List[Path]:
    paths = [p for p in root.rglob("*.json") if p.is_file()]
    paths = [p for p in paths if ".ipynb_checkpoints" not in str(p)]
    return sorted(paths)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _pass_at(summary: Dict[str, Any], k: int, which: str) -> Optional[float]:
    d = summary.get(f"pass@k_acc_{which}", {})
    if not isinstance(d, dict):
        return None
    v = d.get(str(int(k)))
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def token_acc_at(run: Dict[str, Any], step: int) -> float:
    totals = 0
    correct = 0
    for sample in run.get("results", []):
        for tok in sample.get("tokens", []):
            m = tok.get("metrics", {}) or {}
            ranks = m.get("target_rank", [])
            if not isinstance(ranks, list) or step < 0 or step >= len(ranks):
                continue
            totals += 1
            if int(ranks[step]) == 1:
                correct += 1
    return float(correct / totals) if totals else 0.0


def write_csv(rows: List[Dict[str, Any]], out_path: Path, fields: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def plot_curves(rows: List[Dict[str, Any]], out_dir: Path, checkpoints: List[int]) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted({r["dataset"] for r in rows})
    llama_models = sorted({r["model"] for r in rows if "llama" in str(r["model"]).lower()})
    qwen_models = sorted({r["model"] for r in rows if "qwen" in str(r["model"]).lower()})

    def _plot_group(group_models: List[str], out_name: str) -> None:
        if not group_models:
            return
        fig, axes = plt.subplots(len(datasets), 1, figsize=(9.5, 3.6 * len(datasets)), squeeze=False)
        xs = np.array(checkpoints, dtype=int)

        for i, ds in enumerate(datasets):
            ax = axes[i][0]
            _full_box(ax)
            for model in group_models:
                # Each (dataset, model) should have exactly one run in v9.5 main.
                cand = [r for r in rows if r["dataset"] == ds and r["model"] == model]
                if not cand:
                    continue
                r = cand[0]
                ys = np.array([float(r.get(f"token_acc@{s}", 0.0)) for s in checkpoints], dtype=float)
                label = f"{model} (lr={r.get('lr')})"
                ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=4.5, label=label)

            ax.set_title(ds, fontweight="bold")
            ax.set_xlabel("TTA steps", fontweight="bold")
            ax.set_ylabel("Token Acc (TF)", fontweight="bold")
            ax.set_xticks(xs)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="lower right", frameon=True)

        fig.tight_layout()
        fig.savefig(out_dir / out_name)
        plt.close(fig)

    _plot_group(llama_models, "token_acc_curves_llama.png")
    _plot_group(qwen_models, "token_acc_curves_qwen.png")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate v9.5 runs")
    p.add_argument("--root", required=True, help="Root dir containing v9 JSON outputs.")
    p.add_argument("--out_dir", required=True, help="Output directory for table + plots.")
    p.add_argument("--checkpoints", default="0,2,5,10", help="Comma-separated step checkpoints.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    checkpoints = [int(x.strip()) for x in str(args.checkpoints).split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for path in find_json(root):
        run = load_json(path)
        s = run.get("summary", {}) or {}
        if str(s.get("eval_mode", "")) != "tf":
            continue

        # v9.5 uses only steps_total=10 runs
        steps_total = int(s.get("steps", 0))
        if steps_total <= 0:
            continue

        row: Dict[str, Any] = {
            "json_path": str(path),
            "dataset": str(s.get("dataset")),
            "model": str(s.get("model")),
            "steps_total": steps_total,
            "lr": float(s.get("lr", 0.0)),
            "lr_schedule": str(s.get("lr_schedule")),
            "lr_norm": str(s.get("lr_norm", "none")),
            "ane_metric": str(s.get("ane_metric", "")),
            "token_total": int(s.get("token_total", 0)),
            "seq_total": int(s.get("seq_total", 0)),
            "token_acc_before": float(s.get("token_acc_before", 0.0)),
            "token_acc_after": float(s.get("token_acc_after", 0.0)),
            "digit_acc_before": float(s.get("digit_acc_before", 0.0)),
            "digit_acc_after": float(s.get("digit_acc_after", 0.0)),
            "em_before": float(s.get("em_before", 0.0)),
            "em_after": float(s.get("em_after", 0.0)),
            "pass2_before": _pass_at(s, 2, "before"),
            "pass2_after": _pass_at(s, 2, "after"),
            "pass5_before": _pass_at(s, 5, "before"),
            "pass5_after": _pass_at(s, 5, "after"),
            "runtime_seconds": float(s.get("runtime_seconds", 0.0)),
            "runtime_per_token": float(s.get("runtime_seconds", 0.0)) / max(1, int(s.get("token_total", 0))),
        }

        # token acc checkpoints
        for ck in checkpoints:
            row[f"token_acc@{ck}"] = token_acc_at(run, ck)

        rows.append(row)

    rows.sort(key=lambda r: (r["dataset"], r["model"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset",
        "model",
        "lr",
        "steps_total",
        "ane_metric",
        "token_total",
        "seq_total",
        "token_acc_before",
        *[f"token_acc@{ck}" for ck in checkpoints],
        "token_acc_after",
        "pass2_before",
        "pass2_after",
        "pass5_before",
        "pass5_after",
        "em_before",
        "em_after",
        "digit_acc_before",
        "digit_acc_after",
        "runtime_per_token",
        "json_path",
    ]
    write_csv(rows, out_dir / "table_main_v9_5.csv", fields)
    plot_curves(rows, out_dir / "plots", checkpoints)
    print(f"[v9.5] Wrote: {out_dir/'table_main_v9_5.csv'}")
    print(f"[v9.5] Plots: {out_dir/'plots'}")


if __name__ == "__main__":
    main()

