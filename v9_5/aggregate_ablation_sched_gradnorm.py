"""Aggregate Llama-3B ablation runs (schedule + lr_norm) into a table + plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

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


def plot_by_dataset(rows: List[Dict[str, Any]], out_dir: Path, checkpoints: List[int]) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted({r["dataset"] for r in rows})
    variants = sorted({r["variant"] for r in rows})
    xs = np.array(checkpoints, dtype=int)

    fig, axes = plt.subplots(len(datasets), 1, figsize=(9.8, 3.6 * len(datasets)), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i][0]
        _full_box(ax)
        sub = [r for r in rows if r["dataset"] == ds]
        sub.sort(key=lambda r: r["variant"])
        for r in sub:
            ys = np.array([float(r.get(f"token_acc@{s}", 0.0)) for s in checkpoints], dtype=float)
            label = r["variant"]
            ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=4.5, label=label)
        ax.set_title(ds, fontweight="bold")
        ax.set_xlabel("TTA steps", fontweight="bold")
        ax.set_ylabel("Token Acc (TF)", fontweight="bold")
        ax.set_xticks(xs)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "token_acc_curves_by_dataset.png")
    plt.close(fig)

    # Summary bar: delta at final step
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.0))
    _full_box(ax)
    width = 0.18
    x0 = np.arange(len(datasets), dtype=float)
    for j, var in enumerate(variants):
        vals = []
        for ds in datasets:
            r = next((x for x in rows if x["dataset"] == ds and x["variant"] == var), None)
            vals.append(float(r.get("delta_final", 0.0)) if r else 0.0)
        ax.bar(x0 + (j - (len(variants) - 1) / 2) * width, vals, width=width, label=var)
    ax.set_xticks(x0)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("Δ Token Acc (after - before)", fontweight="bold")
    ax.set_title("Llama-3B: schedule/lr_norm ablation (Δ at step=10)", fontweight="bold")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_token_acc_bar.png")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate Llama3B schedule+grad_norm ablation")
    p.add_argument("--root", required=True, help="Root folder containing JSONs under variant/*/*/dataset/model/*.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoints", default="0,2,5,10")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    checkpoints = [int(x.strip()) for x in str(args.checkpoints).split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for path in find_json(root):
        # infer variant from folder name: root/<variant>/...
        try:
            variant = path.relative_to(root).parts[0]
        except Exception:
            variant = "unknown"

        run = load_json(path)
        s = run.get("summary", {}) or {}
        if str(s.get("eval_mode", "")) != "tf":
            continue

        row: Dict[str, Any] = {
            "variant": variant,
            "dataset": str(s.get("dataset")),
            "model": str(s.get("model")),
            "lr": float(s.get("lr", 0.0)),
            "lr_schedule": str(s.get("lr_schedule")),
            "lr_norm": str(s.get("lr_norm", "none")),
            "ane_metric": str(s.get("ane_metric", "")),
            "token_total": int(s.get("token_total", 0)),
            "seq_total": int(s.get("seq_total", 0)),
            "token_acc_before": float(s.get("token_acc_before", 0.0)),
            "token_acc_after": float(s.get("token_acc_after", 0.0)),
            "em_before": float(s.get("em_before", 0.0)),
            "em_after": float(s.get("em_after", 0.0)),
            "runtime_seconds": float(s.get("runtime_seconds", 0.0)),
            "runtime_per_token": float(s.get("runtime_seconds", 0.0)) / max(1, int(s.get("token_total", 0))),
            "json_path": str(path),
        }
        for ck in checkpoints:
            row[f"token_acc@{ck}"] = token_acc_at(run, ck)
        row["delta_final"] = row["token_acc_after"] - row["token_acc_before"]
        rows.append(row)

    rows.sort(key=lambda r: (r["dataset"], r["variant"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset",
        "variant",
        "lr",
        "lr_schedule",
        "lr_norm",
        "ane_metric",
        "token_total",
        "seq_total",
        "token_acc_before",
        *[f"token_acc@{ck}" for ck in checkpoints],
        "token_acc_after",
        "delta_final",
        "em_before",
        "em_after",
        "runtime_per_token",
        "json_path",
    ]
    write_csv(rows, out_dir / "table_llama3b_sched_gradnorm.csv", fields)
    plot_by_dataset(rows, out_dir / "plots", checkpoints)
    print(f"[ablate] Wrote: {out_dir/'table_llama3b_sched_gradnorm.csv'}")
    print(f"[ablate] Plots: {out_dir/'plots'}")


if __name__ == "__main__":
    main()

