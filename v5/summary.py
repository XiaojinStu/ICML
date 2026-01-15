"""Aggregate experiment results into tables and charts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(input_dir: str) -> Dict[str, Dict]:
    results = {}
    for path in glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True):
        name = os.path.basename(path).replace(".json", "")
        with open(path, "r") as f:
            results[name] = json.load(f)
    return results


def collect_rows(results: Dict[str, Dict]) -> List[Dict]:
    rows = []
    for name, data in results.items():
        summary = data.get("summary", {})
        metrics = summary.get("metrics", {})
        config = summary.get("config", {})
        for mode, stats in metrics.items():
            row = {
                "exp_name": name,
                "model": summary.get("model", ""),
                "mode": mode,
                "token_acc": stats.get("token_acc", 0.0),
                "token_acc_at_5": stats.get("token_acc_at_k", {}).get("5", 0.0),
                "seq_acc": stats.get("seq_acc", 0.0),
                "seq_acc_at_5": stats.get("seq_acc_at_k", {}).get("5", 0.0),
                "rank_mean": stats.get("rank_mean", 0.0),
                "mrr": stats.get("mrr", 0.0),
                "token_prob_mean": stats.get("token_prob_mean", 0.0),
                "token_count": stats.get("token_count", 0),
                "sample_count": stats.get("sample_count", 0),
                "runtime_min": summary.get("runtime_minutes", 0.0),
                "flipped_count": summary.get("flipped_count", 0),
                "steps": config.get("steps", ""),
                "lr": config.get("lr", ""),
                "update_target": config.get("update_target", ""),
                "num_layers": config.get("num_layers", ""),
                "layer_stride": config.get("layer_stride", ""),
            }
            rows.append(row)
    return rows


def save_csv(rows: List[Dict], output_path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_latex(rows: List[Dict], output_path: str) -> None:
    if not rows:
        return

    # Build a compact table with TF/AR accuracies per experiment
    grouped = {}
    for row in rows:
        grouped.setdefault(row["exp_name"], {})[row["mode"]] = row

    lines = []
    header = (
        "exp & model & tf acc & tf+tta acc & tf seq & tf+tta seq & "
        "ar acc & ar+tta acc & ar seq & ar+tta seq \\\\"
    )
    lines.append("\\begin{tabular}{l l r r r r r r r r}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")

    for exp, modes in grouped.items():
        tf = modes.get("tf", {})
        tf_tta = modes.get("tf_tta", {})
        ar = modes.get("ar", {})
        ar_tta = modes.get("ar_tta", {})
        model = tf.get("model", "") or tf_tta.get("model", "")
        line = (
            f"{exp} & {model} & "
            f"{tf.get('token_acc',0):.3f} & {tf_tta.get('token_acc',0):.3f} & "
            f"{tf.get('seq_acc',0):.3f} & {tf_tta.get('seq_acc',0):.3f} & "
            f"{ar.get('token_acc',0):.3f} & {ar_tta.get('token_acc',0):.3f} & "
            f"{ar.get('seq_acc',0):.3f} & {ar_tta.get('seq_acc',0):.3f} \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def plot_summary(results: Dict[str, Dict], output_path: str) -> None:
    if not results:
        return

    names = list(results.keys())
    tf_acc = []
    tf_tta_acc = []
    ar_acc = []
    ar_tta_acc = []
    tf_seq = []
    tf_tta_seq = []
    ar_seq = []
    ar_tta_seq = []

    for name in names:
        metrics = results[name].get("summary", {}).get("metrics", {})
        tf = metrics.get("tf", {})
        tf_tta = metrics.get("tf_tta", {})
        ar = metrics.get("ar", {})
        ar_tta = metrics.get("ar_tta", {})

        tf_acc.append(tf.get("token_acc", 0) * 100)
        tf_tta_acc.append(tf_tta.get("token_acc", 0) * 100)
        ar_acc.append(ar.get("token_acc", 0) * 100)
        ar_tta_acc.append(ar_tta.get("token_acc", 0) * 100)

        tf_seq.append(tf.get("seq_acc", 0) * 100)
        tf_tta_seq.append(tf_tta.get("seq_acc", 0) * 100)
        ar_seq.append(ar.get("seq_acc", 0) * 100)
        ar_tta_seq.append(ar_tta.get("seq_acc", 0) * 100)

    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(x - width / 2, tf_acc, width, label="TF", color="#bdbdbd")
    ax.bar(x + width / 2, tf_tta_acc, width, label="TF+TTA", color="#2171b5")
    ax.set_title("Token Accuracy (TF)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(x - width / 2, ar_acc, width, label="AR", color="#bdbdbd")
    ax.bar(x + width / 2, ar_tta_acc, width, label="AR+TTA", color="#2171b5")
    ax.set_title("Token Accuracy (AR)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(x - width / 2, tf_seq, width, label="TF", color="#bdbdbd")
    ax.bar(x + width / 2, tf_tta_seq, width, label="TF+TTA", color="#2171b5")
    ax.set_title("Sequence Accuracy (TF)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(x - width / 2, ar_seq, width, label="AR", color="#bdbdbd")
    ax.bar(x + width / 2, ar_tta_seq, width, label="AR+TTA", color="#2171b5")
    ax.set_title("Sequence Accuracy (AR)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, facecolor="white")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="ANE-TTA summary")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="summary")
    return parser.parse_args()


def main():
    args = parse_args()
    results = load_results(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = collect_rows(results)
    save_csv(rows, os.path.join(args.output_dir, "summary.csv"))
    save_latex(rows, os.path.join(args.output_dir, "summary_table.tex"))
    plot_summary(results, os.path.join(args.output_dir, "summary_plot.png"))

    print(f"Saved summary to {args.output_dir}")


if __name__ == "__main__":
    main()
