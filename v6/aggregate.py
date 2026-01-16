"""Aggregate v6 results and generate comparison charts/tables."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 320, "savefig.bbox": "tight"})


def load_jsons(paths: List[str]) -> Dict[str, Dict]:
    out = {}
    for p in paths:
        name = os.path.basename(p).replace(".json", "")
        with open(p, "r") as f:
            out[name] = json.load(f)
    return out


def find_jsons(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))


def collect_rows(results: Dict[str, Dict]) -> List[Dict]:
    rows = []
    for name, data in results.items():
        s = data.get("summary", {})
        row = {
            "exp": name,
            "task": s.get("task", ""),
            "method": s.get("method", ""),
            "model": s.get("model", ""),
            "steps": s.get("steps", ""),
            "lr": s.get("lr", ""),
            "update_target": s.get("update_target", ""),
            "num_layers": s.get("num_layers", ""),
            "eval_mode": s.get("eval_mode", ""),
            "tta_reset": s.get("tta_reset", ""),
            "token_total": s.get("token_total", 0),
            "token_correct_before": s.get("token_correct_before", 0),
            "token_correct_after": s.get("token_correct_after", 0),
            "token_acc_before": s.get("token_acc_before", 0.0),
            "token_acc_after": s.get("token_acc_after", 0.0),
            "token_acc@5_before": s.get("token_topk_acc_before", {}).get("5", 0.0),
            "token_acc@5_after": s.get("token_topk_acc_after", {}).get("5", 0.0),
            "seq_total": s.get("seq_total", 0),
            "seq_correct_before": s.get("seq_correct_before", 0),
            "seq_correct_after": s.get("seq_correct_after", 0),
            "seq_acc_before": s.get("seq_acc_before", 0.0),
            "seq_acc_after": s.get("seq_acc_after", 0.0),
            "str_total": s.get("str_total", 0),
            "str_correct_before": s.get("str_correct_before", 0),
            "str_correct_after": s.get("str_correct_after", 0),
            "str_acc_before": s.get("str_acc_before", 0.0),
            "str_acc_after": s.get("str_acc_after", 0.0),
            "flipped": s.get("flipped_count", 0),
            "minutes": s.get("elapsed_minutes", 0.0),
            "trainable_params": s.get("trainable", {}).get("trainable_params", 0),
            "trainable_pct": s.get("trainable", {}).get("trainable_pct", 0.0),
            "trainable_layer_count": s.get("trainable", {}).get("layer_count", 0),
            "total_layers": s.get("trainable", {}).get("total_layers", 0),
        }
        rows.append(row)
    return rows


def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_bars(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return

    # keep stable order
    rows = sorted(rows, key=lambda r: r.get("str_acc_after", 0.0), reverse=True)

    labels = [r["method"] for r in rows]
    str_after = [r.get("str_acc_after", 0.0) * 100 for r in rows]
    token_after = [r["token_acc_after"] * 100 for r in rows]
    token5_after = [r["token_acc@5_after"] * 100 for r in rows]
    seq_after = [r["seq_acc_after"] * 100 for r in rows]
    minutes = [r["minutes"] for r in rows]

    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(x, str_after, color="#2171b5")
    ax.set_title("String Accuracy (After)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axes[0, 1]
    ax.bar(x, token_after, color="#238b45")
    ax.set_title("Token Accuracy (After)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axes[1, 0]
    ax.bar(x, seq_after, color="#756bb1")
    ax.set_title("Sequence Accuracy (After)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axes[1, 1]
    ax.bar(x, minutes, color="#fd8d3c")
    ax.set_title("Runtime")
    ax.set_ylabel("Minutes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()



def plot_scatter(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return

    x = [r.get("trainable_pct", 0.0) for r in rows]
    y = [r.get("str_acc_after", 0.0) * 100 for r in rows]
    c = [r.get("minutes", 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=70, alpha=0.9, edgecolors="white", linewidths=0.6)
    for r in rows:
        ax.annotate(r.get("method", ""), (r.get("trainable_pct", 0.0), r.get("str_acc_after", 0.0) * 100), fontsize=8, alpha=0.85)

    ax.set_xlabel("Trainable Params (%)")
    ax.set_ylabel("String Accuracy After (%)")
    ax.set_title("String Accuracy vs. Trainable Subset (color=runtime minutes)")
    ax.grid(True, linestyle="--", alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Minutes")

    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()


def _group_rows(rows: List[Dict], keys: List[str]) -> Dict[Tuple, List[Dict]]:
    grouped: Dict[Tuple, List[Dict]] = {}
    for r in rows:
        k = tuple(r.get(x) for x in keys)
        grouped.setdefault(k, []).append(r)
    return grouped


def plot_lr_sweeps(rows: List[Dict], out_dir: str) -> None:
    """Per (model, update_target): str/token/seq acc after vs LR (log-x)."""
    lr_rows = [r for r in rows if str(r.get("method", "")).startswith("lrsearch_")]
    if not lr_rows:
        return

    grouped = _group_rows(lr_rows, ["model", "update_target", "steps", "tta_reset"])
    for (model, update_target, steps, reset), rs in grouped.items():
        rs = sorted(rs, key=lambda x: float(x.get("lr", 0.0)))
        lrs = [float(r.get("lr", 0.0)) for r in rs]
        str_after = [float(r.get("str_acc_after", 0.0)) * 100 for r in rs]
        tok_after = [float(r.get("token_acc_after", 0.0)) * 100 for r in rs]
        seq_after = [float(r.get("seq_acc_after", 0.0)) * 100 for r in rs]

        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        ax.plot(lrs, str_after, marker="o", lw=2.2, color="#2171b5", label="str(after)")
        ax.plot(lrs, tok_after, marker="o", lw=2.0, color="#238b45", label="token(after)")
        ax.plot(lrs, seq_after, marker="o", lw=2.0, color="#756bb1", label="seq(after)")
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate (log scale)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"LR sweep | {model} | {update_target} | steps={steps} | reset={reset}")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(frameon=False)

        out_path = os.path.join(out_dir, f"lr_sweep_{model}_{update_target}_steps{steps}_{reset}.png".replace("/", "_"))
        plt.tight_layout()
        plt.savefig(out_path, facecolor="white")
        plt.close()


def plot_steps_curves(rows: List[Dict], out_dir: str) -> None:
    """Per (model, update_target): str acc after vs steps."""
    grid_rows = [r for r in rows if str(r.get("method", "")).startswith("gsm8k500_")]
    if not grid_rows:
        return

    grouped = _group_rows(grid_rows, ["model", "update_target", "lr", "tta_reset"])
    for (model, update_target, lr, reset), rs in grouped.items():
        rs = sorted(rs, key=lambda x: int(x.get("steps", 0)))
        steps = [int(r.get("steps", 0)) for r in rs]
        str_before = [float(r.get("str_acc_before", 0.0)) * 100 for r in rs]
        str_after = [float(r.get("str_acc_after", 0.0)) * 100 for r in rs]

        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        ax.plot(steps, str_before, marker="o", lw=2.0, color="#9ecae1", label="before")
        ax.plot(steps, str_after, marker="o", lw=2.3, color="#2171b5", label="after")
        ax.set_xlabel("Steps")
        ax.set_ylabel("String accuracy (%)")
        ax.set_title(f"Steps sweep | {model} | {update_target} | lr={lr} | reset={reset}")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(frameon=False)

        out_path = os.path.join(out_dir, f"steps_sweep_{model}_{update_target}_lr{lr}_{reset}.png".replace("/", "_"))
        plt.tight_layout()
        plt.savefig(out_path, facecolor="white")
        plt.close()


def plot_method_bars(rows: List[Dict], out_dir: str) -> None:
    """Compare methods per model: best-after bars (str_acc_after)."""
    grid_rows = [r for r in rows if str(r.get("method", "")).startswith("gsm8k500_")]
    if not grid_rows:
        return

    by_model = _group_rows(grid_rows, ["model"])
    for (model,), rs in by_model.items():
        # pick stable order: (mlp, ln) and steps ascending
        rs = sorted(rs, key=lambda r: (r.get("update_target", ""), int(r.get("steps", 0))))
        labels = [f"{r.get('update_target')}-s{r.get('steps')}" for r in rs]
        vals = [float(r.get("str_acc_after", 0.0)) * 100 for r in rs]
        base = [float(r.get("str_acc_before", 0.0)) * 100 for r in rs]

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 4.8))
        ax.bar(x, base, color="#c6dbef", label="before")
        ax.bar(x, vals, color="#2171b5", alpha=0.9, label="after")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("String accuracy (%)")
        ax.set_title(f"GSM8K-500 | {model}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.legend(frameon=False)

        out_path = os.path.join(out_dir, f"compare_{model}.png".replace("/", "_"))
        plt.tight_layout()
        plt.savefig(out_path, facecolor="white")
        plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate v6 results")
    p.add_argument("--root", required=True, help="root directory to scan for json")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paths = find_jsons(args.root)
    results = load_jsons(paths)

    rows = collect_rows(results)
    save_csv(rows, os.path.join(args.out_dir, "summary.csv"))
    plot_bars(rows, os.path.join(args.out_dir, "summary_bars.png"))
    plot_scatter(rows, os.path.join(args.out_dir, "summary_scatter.png"))
    plot_lr_sweeps(rows, args.out_dir)
    plot_steps_curves(rows, args.out_dir)
    plot_method_bars(rows, args.out_dir)

    print(f"Found {len(rows)} experiments. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
