"""Visualization utilities for ANE-TTA v5 (publication-focused)."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


setup_style()


def create_prob_cmap() -> LinearSegmentedColormap:
    colors = ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c"]
    return LinearSegmentedColormap.from_list("prob_blue", colors, N=256)


def create_rank_cmap() -> LinearSegmentedColormap:
    colors = ["#00441b", "#238b45", "#74c476", "#c7e9c0", "#f7fcf5"]
    return LinearSegmentedColormap.from_list("rank_green", colors, N=256)


def collect_tf_tokens(results: Dict) -> Tuple[List[Dict], List[int]]:
    entries: List[Dict] = []
    boundaries = [0]
    for sample in results.get("results", []):
        tf_tokens = sample.get("tf", {}).get("tokens", [])
        for tok in tf_tokens:
            metrics = tok.get("metrics", {})
            prob = metrics.get("target_prob", [])
            rank = metrics.get("target_rank", [])
            if not prob or not rank:
                continue
            entries.append(
                {
                    "token": tok.get("target_token", "?"),
                    "prob": prob,
                    "rank": rank,
                    "answer_len": tok.get("answer_len", sample.get("answer_len", None)),
                    "correct_before": tok.get("baseline", {}).get("correct", False),
                    "correct_after": tok.get("tta", {}).get("correct", False),
                    "metrics": metrics,
                    "pred_before": tok.get("baseline", {}).get("pred_token", ""),
                    "pred_after": tok.get("tta", {}).get("pred_token", ""),
                }
            )
        boundaries.append(len(entries))
    return entries, boundaries


def pad_matrix(rows: List[List[float]], target_len: int) -> np.ndarray:
    mat = np.zeros((len(rows), target_len))
    for i, row in enumerate(rows):
        if not row:
            continue
        row = list(row)
        if len(row) >= target_len:
            mat[i, :] = row[:target_len]
        else:
            mat[i, : len(row)] = row
            mat[i, len(row) :] = row[-1]
    return mat


def text_color_from_cmap(cmap, norm, value, vmin: float, vmax: float) -> str:
    if norm is None:
        norm_val = (value - vmin) / (vmax - vmin + 1e-9)
    else:
        norm_val = norm(value)
    r, g, b, _ = cmap(norm_val)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def render_heatmap(
    mat: np.ndarray,
    title: str,
    output_path: str,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    x_labels: List[str],
    y_labels: List[str],
    annotate: bool,
    fmt,
    cbar_label: str,
    boundaries: List[int] | None = None,
    norm=None,
) -> None:
    n_rows, n_cols = mat.shape
    fig_w = max(9, n_cols * 0.30)
    fig_h = max(6, n_rows * 0.16)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if norm is None:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    if annotate:
        font_size = max(5, min(8, 240 // max(n_rows, n_cols)))
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                color = text_color_from_cmap(cmap, norm, val, vmin, vmax)
                text = fmt(val)
                if text:
                    ax.text(j, i, text, ha="center", va="center", fontsize=font_size, color=color)

    ax.set_title(title, pad=8)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Token Index")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0, fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)

    if boundaries:
        for b in boundaries[1:-1]:
            if 0 <= b < n_rows:
                ax.axhline(b - 0.5, color="#222222", lw=0.8, alpha=0.6)

    cbar = plt.colorbar(im, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.subplots_adjust(left=0.14)
    plt.savefig(output_path, facecolor="white")
    plt.close()


def visualize_prob_heatmap(results: Dict, output_dir: str, exp_name: str, tokens_per_page: int = 200) -> None:
    entries, boundaries = collect_tf_tokens(results)
    if not entries:
        return

    probs = [e["prob"] for e in entries]
    n_steps = max(len(p) for p in probs)
    if tokens_per_page is None or tokens_per_page <= 0:
        tokens_per_page = len(entries)
    n_pages = math.ceil(len(entries) / tokens_per_page)
    cmap = create_prob_cmap()

    for page in range(n_pages):
        start = page * tokens_per_page
        end = min(len(entries), start + tokens_per_page)
        page_entries = entries[start:end]
        page_probs = [e["prob"] for e in page_entries]
        mat = pad_matrix(page_probs, n_steps)

        x_labels = [str(i) for i in range(n_steps)]
        y_labels = [f"{start+i:03d}:{e['token']}" for i, e in enumerate(page_entries)]
        page_boundaries = [b - start for b in boundaries if start <= b <= end]

        suffix = f"_p{page+1}" if n_pages > 1 else ""
        out_path = os.path.join(output_dir, f"{exp_name}_prob_heatmap{suffix}.png")

        render_heatmap(
            mat,
            title=f"Target Probability Evolution ({exp_name}{suffix})",
            output_path=out_path,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            x_labels=x_labels,
            y_labels=y_labels,
            annotate=True,
            fmt=lambda v: f"{v:.2f}" if v >= 0.01 else "",
            cbar_label="Target Probability",
            boundaries=page_boundaries,
        )


def visualize_rank_heatmap(results: Dict, output_dir: str, exp_name: str, tokens_per_page: int = 200) -> None:
    entries, boundaries = collect_tf_tokens(results)
    if not entries:
        return

    ranks = [e["rank"] for e in entries]
    n_steps = max(len(r) for r in ranks)
    if tokens_per_page is None or tokens_per_page <= 0:
        tokens_per_page = len(entries)
    n_pages = math.ceil(len(entries) / tokens_per_page)
    cmap = create_rank_cmap()
    max_rank = max(max(r) for r in ranks if r)

    for page in range(n_pages):
        start = page * tokens_per_page
        end = min(len(entries), start + tokens_per_page)
        page_entries = entries[start:end]
        page_ranks = [e["rank"] for e in page_entries]
        mat = pad_matrix(page_ranks, n_steps)

        x_labels = [str(i) for i in range(n_steps)]
        y_labels = [f"{start+i:03d}:{e['token']}" for i, e in enumerate(page_entries)]
        page_boundaries = [b - start for b in boundaries if start <= b <= end]

        suffix = f"_p{page+1}" if n_pages > 1 else ""
        out_path = os.path.join(output_dir, f"{exp_name}_rank_heatmap{suffix}.png")

        render_heatmap(
            mat,
            title=f"Target Rank Evolution ({exp_name}{suffix})",
            output_path=out_path,
            cmap=cmap,
            vmin=1.0,
            vmax=float(max_rank),
            x_labels=x_labels,
            y_labels=y_labels,
            annotate=True,
            fmt=lambda v: f"{int(v)}",
            cbar_label=f"Target Rank (1-{max_rank})",
            boundaries=page_boundaries,
            norm=LogNorm(vmin=1.0, vmax=float(max_rank)),
        )


def collect_cases(entries: List[Dict]) -> List[Dict]:
    cases = []
    for e in entries:
        prob = e["prob"]
        rank = e["rank"]
        if not prob or not rank:
            continue
        before_prob = prob[0]
        after_prob = prob[-1]
        before_rank = rank[0]
        after_rank = rank[-1]
        cases.append(
            {
                "token": e["token"],
                "pred_before": e["pred_before"],
                "pred_after": e["pred_after"],
                "correct_before": e["correct_before"],
                "correct_after": e["correct_after"],
                "metrics": e["metrics"],
                "delta_prob": after_prob - before_prob,
                "delta_rank": before_rank - after_rank,
            }
        )
    return cases


def select_cases(cases: List[Dict], n_cases: int, prefer_flipped: bool = True) -> List[Dict]:
    if not cases:
        return []
    flipped = [c for c in cases if not c["correct_before"] and c["correct_after"]]
    pool = flipped if (prefer_flipped and flipped) else cases
    pool.sort(key=lambda x: (x["delta_rank"], x["delta_prob"]), reverse=True)
    return pool[:n_cases]


def visualize_subspace_evolution(results: Dict, output_dir: str, exp_name: str, n_cases: int = 8) -> None:
    entries, _ = collect_tf_tokens(results)
    cases = select_cases(collect_cases(entries), n_cases, prefer_flipped=True)
    if not cases:
        return

    steps = None
    for c in cases:
        snaps = c["metrics"].get("num_topk_snapshots", [])
        if snaps:
            steps = [s["step"] for s in snaps]
            break
    if not steps:
        return

    n_rows = len(cases)
    n_cols = len(steps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, case in enumerate(cases):
        snap_map = {s["step"]: s["tokens"] for s in case["metrics"].get("num_topk_snapshots", [])}
        target = case["token"]
        for c_idx, step in enumerate(steps):
            ax = axes[r, c_idx]
            tokens = snap_map.get(step, [])
            if not tokens:
                ax.axis("off")
                continue
            labels = [t["token"] for t in tokens]
            probs = [t["prob"] for t in tokens]
            colors = ["#1f77b4" if lbl != target else "#d62728" for lbl in labels]

            ax.barh(range(len(labels)), probs, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlim(0, 1.0)
            ax.invert_yaxis()
            if r == 0:
                ax.set_title(f"Step {step}")
            if c_idx == 0:
                ax.set_ylabel(f"Case {r+1}")

            for idx, val in enumerate(probs):
                if val >= 0.05:
                    ax.text(val + 0.02, idx, f"{val:.2f}", va="center", fontsize=6)

    plt.suptitle(f"Numerical Subspace Evolution (top-k) - {exp_name}", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{exp_name}_subspace_evolution.png")
    plt.savefig(out_path, facecolor="white")
    plt.close()


def visualize_topk_trajectories(results: Dict, output_dir: str, exp_name: str, n_cases: int = 6) -> None:
    entries, _ = collect_tf_tokens(results)
    cases = select_cases(collect_cases(entries), n_cases, prefer_flipped=True)
    if not cases:
        return

    n_cols = 2
    n_rows = math.ceil(len(cases) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.4 * n_cols, 3.8 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, case in enumerate(cases):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        metrics = case["metrics"]
        tracked = metrics.get("tracked_topk", {})
        probs = np.array(tracked.get("probs", []))
        tokens = tracked.get("tokens", [])
        if probs.size == 0:
            ax.axis("off")
            continue

        steps = np.arange(probs.shape[0])
        colors = plt.cm.tab10(np.linspace(0, 0.9, probs.shape[1]))
        for j in range(probs.shape[1]):
            label = tokens[j] if j < len(tokens) else f"T{j}"
            ax.plot(steps, probs[:, j], color=colors[j], linewidth=1.2, alpha=0.6, label=label)

        target_prob = metrics.get("target_prob", [])
        if target_prob:
            ax.plot(steps, target_prob, color="#d62728", linewidth=2.4, label="target")
            marker_steps = [0, len(target_prob) // 2, len(target_prob) - 1]
            for ms in marker_steps:
                if 0 <= ms < len(target_prob):
                    ax.scatter(ms, target_prob[ms], color="#d62728", s=80, marker="*", edgecolors="white", linewidths=0.6)

        title = f"{case['pred_before']} -> {case['pred_after']} (target {case['token']})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    for idx in range(len(cases), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{exp_name}_topk_trajectories.png")
    plt.savefig(out_path, facecolor="white")
    plt.close()


def visualize_flipped_curves(results: Dict, output_dir: str, exp_name: str, n_cases: int = 9) -> None:
    entries, _ = collect_tf_tokens(results)
    cases = select_cases(collect_cases(entries), n_cases, prefer_flipped=True)
    if not cases:
        return

    n_cols = 3
    n_rows = math.ceil(len(cases) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.6 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, case in enumerate(cases):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        metrics = case["metrics"]
        probs = metrics.get("target_prob", [])
        ranks = metrics.get("target_rank", [])
        steps = np.arange(len(probs))

        ax.plot(steps, probs, color="#2171b5", linewidth=2.2, marker="o", markersize=4)
        ax.fill_between(steps, probs, color="#6baed6", alpha=0.2)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Step")
        ax.set_ylabel("Target Prob")

        if ranks:
            ax2 = ax.twinx()
            ax2.plot(steps, ranks, color="#238b45", linewidth=1.8, linestyle="--", marker="s", markersize=3)
            ax2.set_ylabel("Rank")
            ax2.invert_yaxis()

        title = f"{case['pred_before']} -> {case['pred_after']} (target {case['token']})"
        ax.set_title(title, fontsize=10)

    for idx in range(len(cases), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{exp_name}_flipped_curves.png")
    plt.savefig(out_path, facecolor="white")
    plt.close()


def visualize_stats(results: Dict, output_dir: str, exp_name: str) -> None:
    entries, _ = collect_tf_tokens(results)
    if not entries:
        return

    cases = collect_cases(entries)
    before_prob = np.array([c["metrics"]["target_prob"][0] for c in cases])
    after_prob = np.array([c["metrics"]["target_prob"][-1] for c in cases])
    before_rank = np.array([c["metrics"]["target_rank"][0] for c in cases])
    after_rank = np.array([c["metrics"]["target_rank"][-1] for c in cases])
    delta_prob = after_prob - before_prob
    delta_rank = before_rank - after_rank

    lengths = np.array([c.get("answer_len", 0) or 0 for c in entries])
    bins = [0, 5, 10, 20, 50, 100]
    labels = ["1-5", "6-10", "11-20", "21-50", "50+"]
    bin_ids = np.digitize(lengths, bins, right=True)

    acc_before = []
    acc_after = []
    for i in range(1, len(bins)):
        mask = bin_ids == i
        if not np.any(mask):
            acc_before.append(0.0)
            acc_after.append(0.0)
            continue
        acc_before.append(np.mean([e["correct_before"] for e, m in zip(entries, mask) if m]))
        acc_after.append(np.mean([e["correct_after"] for e, m in zip(entries, mask) if m]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    ax.hist(delta_prob, bins=20, color="#3182bd", alpha=0.8)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_title("Delta Probability")
    ax.set_xlabel("After - Before")
    ax.set_ylabel("Count")

    ax = axes[0, 1]
    ax.hist(delta_rank, bins=20, color="#31a354", alpha=0.8)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_title("Delta Rank (positive is better)")
    ax.set_xlabel("Before - After")
    ax.set_ylabel("Count")

    ax = axes[1, 0]
    ax.scatter(before_rank, after_rank, s=10, alpha=0.5, color="#756bb1")
    max_rank = max(before_rank.max(), after_rank.max()) if len(before_rank) else 1
    ax.plot([1, max_rank], [1, max_rank], color="#333333", linestyle="--", lw=1)
    ax.set_title("Rank Before vs After")
    ax.set_xlabel("Before")
    ax.set_ylabel("After")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[1, 1]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, np.array(acc_before) * 100, width, label="Before", color="#bdbdbd")
    ax.bar(x + width / 2, np.array(acc_after) * 100, width, label="After", color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Answer Length")
    ax.legend(fontsize=8)

    plt.suptitle(f"ANE-TTA Summary Stats ({exp_name})", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{exp_name}_stats.png")
    plt.savefig(out_path, facecolor="white")
    plt.close()


def visualize_all(results: Dict, output_dir: str, exp_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    visualize_prob_heatmap(results, output_dir, exp_name)
    visualize_rank_heatmap(results, output_dir, exp_name)
    visualize_subspace_evolution(results, output_dir, exp_name)
    visualize_topk_trajectories(results, output_dir, exp_name)
    visualize_flipped_curves(results, output_dir, exp_name)
    visualize_stats(results, output_dir, exp_name)
