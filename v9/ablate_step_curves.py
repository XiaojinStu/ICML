"""Aggregate token-accuracy-at-step curves from v9 JSON outputs.

This script is for ablations where we run a *single* long-TTA run (e.g. steps=30)
and want to report token accuracy at intermediate checkpoints (e.g. steps 5/10/15/20/25).

Definition (TF mode):
- For each numerical target token position, v9 records `target_rank` over steps (0..S).
- Token is correct at step t iff `target_rank[t] == 1` (among numerical sub-vocab).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
            "legend.fontsize": 12,
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
    # Ignore notebook checkpoints and other transient artifacts.
    paths = [p for p in paths if ".ipynb_checkpoints" not in str(p)]
    return sorted(paths)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def token_acc_at_steps(run: Dict[str, Any], steps: List[int]) -> Dict[int, float]:
    want = sorted(set(int(s) for s in steps))
    totals = {s: 0 for s in want}
    correct = {s: 0 for s in want}

    results = run.get("results", [])
    for sample in results:
        for tok in sample.get("tokens", []):
            m = tok.get("metrics", {}) or {}
            ranks = m.get("target_rank", [])
            if not isinstance(ranks, list) or not ranks:
                continue
            for s in want:
                if s < 0 or s >= len(ranks):
                    continue
                totals[s] += 1
                if int(ranks[s]) == 1:
                    correct[s] += 1

    out: Dict[int, float] = {}
    for s in want:
        out[s] = float(correct[s] / totals[s]) if totals[s] else 0.0
    return out


def write_csv(rows: List[Dict[str, Any]], out_path: Path, fields: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def plot_curves(rows: List[Dict[str, Any]], out_path: Path, *, datasets: List[str], models: List[str], curve_steps: List[int]) -> None:
    setup_style()
    ds_list = datasets
    m_list = models

    fig, axes = plt.subplots(len(ds_list), len(m_list), figsize=(5.8 * len(m_list), 3.9 * len(ds_list)), squeeze=False)
    x = np.array(curve_steps, dtype=int)

    # stable ordering for legend
    def _cfg_key(r: Dict[str, Any]) -> Tuple:
        return (r.get("variant", ""), float(r.get("lr", 0.0)))

    for i, ds in enumerate(ds_list):
        for j, model in enumerate(m_list):
            ax = axes[i][j]
            _full_box(ax)
            subset = [r for r in rows if r.get("dataset") == ds and r.get("model") == model]
            subset.sort(key=_cfg_key)

            for r in subset:
                y = np.array([float(r.get(f"acc@{s}", 0.0)) for s in curve_steps], dtype=float)
                label = f"{r.get('variant')} lr={r.get('lr')}"
                ax.plot(x, y, marker="o", linewidth=2.2, markersize=4.5, label=label)

            ax.set_title(f"{ds} | {model}", fontweight="bold")
            ax.set_xlabel("TTA steps", fontweight="bold")
            ax.set_ylabel("Token Acc (TF)", fontweight="bold")
            ax.set_xticks(x)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            if subset:
                ax.legend(loc="lower right", frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate v9 step-curve token accuracy")
    p.add_argument("--root", required=True, help="Root folder containing v9 JSON outputs.")
    p.add_argument("--out_dir", required=True, help="Output folder for CSV + plot.")
    p.add_argument("--steps", default="0,5,10,15,20,25,30", help="Comma-separated checkpoint steps.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)

    curve_steps = [int(x.strip()) for x in str(args.steps).split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for path in find_json(root):
        run = load_json(path)
        s = run.get("summary", {}) or {}
        if str(s.get("eval_mode", "")) != "tf":
            continue
        if int(s.get("steps", 0)) <= 0:
            continue

        accs = token_acc_at_steps(run, curve_steps)
        variant = "constant"
        if str(s.get("lr_schedule")) == "cosine" or str(s.get("lr_norm")) != "none":
            variant = f"{s.get('lr_norm', 'none')}+{s.get('lr_schedule', '')}".strip("+")
        rows.append(
            {
                "json_path": str(path),
                "dataset": str(s.get("dataset")),
                "model": str(s.get("model")),
                "lr": float(s.get("lr", 0.0)),
                "steps_total": int(s.get("steps", 0)),
                "lr_schedule": str(s.get("lr_schedule")),
                "lr_norm": str(s.get("lr_norm")),
                "variant": variant,
                **{f"acc@{k}": float(v) for k, v in accs.items()},
            }
        )

    # normalize dataset/model lists for plotting
    datasets = sorted({r["dataset"] for r in rows})
    models = sorted({r["model"] for r in rows})

    # write CSV
    fields = ["dataset", "model", "variant", "lr", "steps_total", "lr_schedule", "lr_norm"] + [f"acc@{s}" for s in curve_steps] + ["json_path"]
    write_csv(rows, out_dir / "step_curves.csv", fields)

    # plot
    plot_curves(rows, out_dir / "step_curves.png", datasets=datasets, models=models, curve_steps=curve_steps)

    print(f"[ablate] Wrote: {out_dir/'step_curves.csv'}")
    print(f"[ablate] Wrote: {out_dir/'step_curves.png'}")


if __name__ == "__main__":
    main()
