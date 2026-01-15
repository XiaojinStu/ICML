"""Aggregate v5.1 ablation results and generate comparison charts/tables."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


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
            "method": s.get("method", ""),
            "model": s.get("model", ""),
            "steps": s.get("steps", ""),
            "lr": s.get("lr", ""),
            "update_target": s.get("update_target", ""),
            "num_layers": s.get("num_layers", ""),
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
            "flipped": s.get("flipped_count", 0),
            "minutes": s.get("elapsed_minutes", 0.0),
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
    rows = sorted(rows, key=lambda r: r["method"])

    labels = [r["method"] for r in rows]
    token_after = [r["token_acc_after"] * 100 for r in rows]
    token5_after = [r["token_acc@5_after"] * 100 for r in rows]
    seq_after = [r["seq_acc_after"] * 100 for r in rows]
    minutes = [r["minutes"] for r in rows]

    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(x, token_after, color="#2171b5")
    ax.set_title("Token Accuracy (After)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax = axes[0, 1]
    ax.bar(x, token5_after, color="#238b45")
    ax.set_title("Token Accuracy@5 (After)")
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


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate v5.1 results")
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

    print(f"Found {len(rows)} experiments. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
