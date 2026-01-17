"""Publication-style visualization for v7 experiments."""

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
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


setup_style()


def create_prob_cmap() -> LinearSegmentedColormap:
    colors = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"]
    return LinearSegmentedColormap.from_list("prob_blue", colors, N=256)


def create_rank_cmap() -> LinearSegmentedColormap:
    colors = ["#00441b", "#238b45", "#74c476", "#c7e9c0", "#f7fcf5"]
    return LinearSegmentedColormap.from_list("rank_green", colors, N=256)


def _text_color_from_bg(cmap, norm, value: float, vmin: float, vmax: float) -> str:
    if norm is None:
        nv = (value - vmin) / (vmax - vmin + 1e-9)
    else:
        nv = float(norm(value))
    r, g, b, _ = cmap(nv)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def _pad_matrix(rows: List[List[float]], target_len: int) -> np.ndarray:
    mat = np.zeros((len(rows), target_len), dtype=float)
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


def _collect_token_series(output: Dict) -> Tuple[List[Dict], List[int]]:
    entries = []
    boundaries = [0]
    for sample in output.get("results", []):
        for tok in sample.get("tokens", []):
            metrics = tok.get("metrics", {})
            p = metrics.get("target_prob", [])
            r = metrics.get("target_rank", [])
            if p and r:
                entries.append(
                    {
                        "target_token": tok.get("target_token", "?"),
                        "prob": p,
                        "rank": r,
                        "pred_before": tok.get("pred_before", ""),
                        "pred_after": tok.get("pred_after", ""),
                        "correct_before": tok.get("correct_before", False),
                        "correct_after": tok.get("correct_after", False),
                        "metrics": metrics,
                    }
                )
        boundaries.append(len(entries))
    return entries, boundaries


def _collect_cases(entries: List[Dict]) -> List[Dict]:
    cases = []
    for e in entries:
        prob = e.get("prob", [])
        rank = e.get("rank", [])
        if not prob or not rank:
            continue
        cases.append(
            {
                "token": e.get("target_token", "?"),
                "pred_before": e.get("pred_before", ""),
                "pred_after": e.get("pred_after", ""),
                "correct_before": bool(e.get("correct_before", False)),
                "correct_after": bool(e.get("correct_after", False)),
                "metrics": e.get("metrics", {}),
                "delta_prob": float(prob[-1]) - float(prob[0]),
                "delta_rank": float(rank[0]) - float(rank[-1]),
            }
        )
    return cases


def _select_cases(cases: List[Dict], n_cases: int, prefer_flipped: bool = True) -> List[Dict]:
    if not cases:
        return []
    flipped = [c for c in cases if (not c["correct_before"]) and c["correct_after"]]
    pool = flipped if (prefer_flipped and flipped) else cases
    pool.sort(key=lambda x: (x["delta_rank"], x["delta_prob"]), reverse=True)
    return pool[:n_cases]


def visualize_prob_heatmap(output: Dict, out_dir: str, exp_name: str, tokens_per_page: int = 120) -> None:
    entries, boundaries = _collect_token_series(output)
    if not entries:
        return

    probs = [e["prob"] for e in entries]
    n_steps = max(len(p) for p in probs)
    cmap = create_prob_cmap()

    n_pages = math.ceil(len(entries) / tokens_per_page)
    for page in range(n_pages):
        start = page * tokens_per_page
        end = min(len(entries), start + tokens_per_page)
        page_entries = entries[start:end]
        mat = _pad_matrix([e["prob"] for e in page_entries], n_steps)

        fig_h = min(22, max(7, mat.shape[0] * 0.18))
        fig_w = min(18, max(11, mat.shape[1] * 0.55))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

        annotate = mat.shape[0] <= 80 and mat.shape[1] <= 25
        if annotate:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = float(mat[i, j])
                    if val < 0.01:
                        continue
                    color = _text_color_from_bg(cmap, None, val, 0.0, 1.0)
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

        page_boundaries = [b - start for b in boundaries if start <= b <= end]
        for b in page_boundaries[1:-1]:
            ax.axhline(b - 0.5, color="#222222", lw=1.0, alpha=0.7)

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Target Probability (numerical sub-vocab)")

        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Token Index")
        ax.set_xticks(list(range(n_steps)))
        ax.set_xticklabels([str(i) for i in range(n_steps)])
        ax.set_yticks(list(range(mat.shape[0])))
        ax.set_yticklabels([f"{start+i:03d}:{e['target_token']}" for i, e in enumerate(page_entries)], fontsize=10)

        suffix = f"_p{page+1}" if n_pages > 1 else ""
        ax.set_title(f"Target Probability Evolution ({exp_name}{suffix})")

        plt.tight_layout()
        plt.subplots_adjust(left=0.14)
        plt.savefig(os.path.join(out_dir, f"{exp_name}_prob_heatmap{suffix}.png"), facecolor="white")
        plt.close()


def visualize_rank_heatmap(output: Dict, out_dir: str, exp_name: str, tokens_per_page: int = 120) -> None:
    entries, boundaries = _collect_token_series(output)
    if not entries:
        return

    ranks = [e["rank"] for e in entries]
    n_steps = max(len(r) for r in ranks)
    max_rank = int(max(max(r) for r in ranks if r))

    cmap = create_rank_cmap()
    norm = LogNorm(vmin=1.0, vmax=float(max_rank))

    n_pages = math.ceil(len(entries) / tokens_per_page)
    for page in range(n_pages):
        start = page * tokens_per_page
        end = min(len(entries), start + tokens_per_page)
        page_entries = entries[start:end]
        mat = _pad_matrix([e["rank"] for e in page_entries], n_steps)

        fig_h = min(22, max(7, mat.shape[0] * 0.18))
        fig_w = min(18, max(11, mat.shape[1] * 0.55))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

        annotate = mat.shape[0] <= 80 and mat.shape[1] <= 25
        if annotate:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = int(mat[i, j])
                    color = _text_color_from_bg(cmap, norm, float(val), 1.0, float(max_rank))
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

        page_boundaries = [b - start for b in boundaries if start <= b <= end]
        for b in page_boundaries[1:-1]:
            ax.axhline(b - 0.5, color="#222222", lw=1.0, alpha=0.7)

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(f"Target Rank in numerical sub-vocab (1-{max_rank})")

        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Token Index")
        ax.set_xticks(list(range(n_steps)))
        ax.set_xticklabels([str(i) for i in range(n_steps)])
        ax.set_yticks(list(range(mat.shape[0])))
        ax.set_yticklabels([f"{start+i:03d}:{e['target_token']}" for i, e in enumerate(page_entries)], fontsize=10)

        suffix = f"_p{page+1}" if n_pages > 1 else ""
        ax.set_title(f"Target Rank Evolution ({exp_name}{suffix})")

        plt.tight_layout()
        plt.subplots_adjust(left=0.14)
        plt.savefig(os.path.join(out_dir, f"{exp_name}_rank_heatmap{suffix}.png"), facecolor="white")
        plt.close()


def visualize_top10_case_evolution(output: Dict, out_dir: str, exp_name: str, n_cases: int = 10) -> None:
    """Top cases: tracked-topk probability trajectories + target probability."""
    entries, _ = _collect_token_series(output)
    cases = _select_cases(_collect_cases(entries), n_cases, prefer_flipped=True)
    if not cases:
        return

    n_cols = 2
    n_rows = math.ceil(len(cases) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.4 * n_cols, 4.8 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, case in enumerate(cases):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        metrics = case.get("metrics", {})
        tracked = metrics.get("tracked_topk", {})
        probs = np.array(tracked.get("probs", []), dtype=float)
        tokens = tracked.get("tokens", [])
        if probs.size == 0:
            ax.axis("off")
            continue

        steps = np.arange(probs.shape[0])
        colors = plt.cm.tab10(np.linspace(0, 0.9, probs.shape[1]))
        for j in range(probs.shape[1]):
            label = tokens[j] if j < len(tokens) else f"T{j}"
            ax.plot(steps, probs[:, j], color=colors[j], linewidth=1.6, alpha=0.55, label=label)

        target_prob = np.array(metrics.get("target_prob", []), dtype=float)
        if target_prob.size > 0:
            ax.plot(steps, target_prob, color="#d62728", linewidth=3.0, linestyle="--", label="target")
            for ms in [0, len(steps) // 2, len(steps) - 1]:
                ax.scatter(ms, target_prob[ms], color="#d62728", s=140, marker="*", edgecolors="white", linewidths=0.9, zorder=5)

        title = f"{case['pred_before']} → {case['pred_after']} (target {case['token']})"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Prob (numerical sub-vocab)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(fontsize=9, ncol=2, frameon=False)

    for idx in range(len(cases), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_top10_case_evolution.png"), facecolor="white")
    plt.close()


def visualize_flipped_curves(output: Dict, out_dir: str, exp_name: str, n_cases: int = 9) -> None:
    """Flipped cases: target prob + target rank curve per token."""
    entries, _ = _collect_token_series(output)
    cases = _select_cases(_collect_cases(entries), n_cases, prefer_flipped=True)
    if not cases:
        return

    n_cols = 3
    n_rows = math.ceil(len(cases) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.0 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, case in enumerate(cases):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        metrics = case["metrics"]
        probs = np.array(metrics.get("target_prob", []), dtype=float)
        ranks = np.array(metrics.get("target_rank", []), dtype=float)
        if probs.size == 0:
            ax.axis("off")
            continue

        steps = np.arange(probs.shape[0])
        ax.plot(steps, probs, color="#2171b5", linewidth=2.6, marker="o", markersize=4)
        ax.fill_between(steps, probs, color="#6baed6", alpha=0.22)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Step")
        ax.set_ylabel("Target Prob")
        ax.grid(True, linestyle="--", alpha=0.25)

        if ranks.size > 0:
            ax2 = ax.twinx()
            ax2.plot(steps, ranks, color="#238b45", linewidth=2.0, linestyle="--", marker="s", markersize=3)
            ax2.set_ylabel("Rank")
            ax2.invert_yaxis()

        title = f"{case['pred_before']} → {case['pred_after']} (target {case['token']})"
        ax.set_title(title, fontsize=11)

    for idx in range(len(cases), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_flipped_curves.png"), facecolor="white")
    plt.close()


def visualize_subspace_evolution(output: Dict, out_dir: str, exp_name: str, n_cases: int = 6) -> None:
    """Per-step bar plots of numerical top-k distribution (each step as one column)."""
    entries, _ = _collect_token_series(output)
    cases = _select_cases(_collect_cases(entries), n_cases, prefer_flipped=True)
    if not cases:
        return

    # Determine the step list from the first case with snapshots.
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, case in enumerate(cases):
        snaps = case["metrics"].get("num_topk_snapshots", [])
        snap_map = {int(s["step"]): s["tokens"] for s in snaps if "tokens" in s}
        target = case["token"]

        for c_idx, step in enumerate(steps):
            ax = axes[r, c_idx]
            tokens = snap_map.get(int(step), [])
            if not tokens:
                ax.axis("off")
                continue

            labels = [t["token"] for t in tokens]
            probs = [float(t["prob"]) for t in tokens]
            colors = ["#1f77b4" if lbl != target else "#d62728" for lbl in labels]

            ax.barh(range(len(labels)), probs, color=colors, edgecolor="white", linewidth=0.6)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlim(0, 1.0)
            ax.invert_yaxis()
            if r == 0:
                ax.set_title(f"Step {step}", fontsize=12)
            if c_idx == 0:
                ax.set_ylabel(f"Case {r+1}", fontsize=12)

            for i, v in enumerate(probs):
                if v >= 0.05:
                    ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)

    plt.suptitle(f"Numerical Subspace Evolution (top-k, each step) — {exp_name}", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_subspace_evolution.png"), facecolor="white")
    plt.close()


def visualize_avg_curves(output: Dict, out_dir: str, exp_name: str) -> None:
    """Aggregate token-level curves: mean/median prob + rank over steps."""
    entries, _ = _collect_token_series(output)
    if not entries:
        return

    probs = [e["prob"] for e in entries if e.get("prob")]
    ranks = [e["rank"] for e in entries if e.get("rank")]
    if not probs or not ranks:
        return

    n_steps = max(max(len(p) for p in probs), max(len(r) for r in ranks))
    prob_by_step: List[List[float]] = [[] for _ in range(n_steps)]
    rank_by_step: List[List[float]] = [[] for _ in range(n_steps)]

    for p in probs:
        for i, v in enumerate(p):
            prob_by_step[i].append(float(v))
    for r in ranks:
        for i, v in enumerate(r):
            rank_by_step[i].append(float(v))

    steps = np.arange(n_steps)
    prob_mean = np.array([np.mean(v) if v else float("nan") for v in prob_by_step], dtype=float)
    prob_med = np.array([np.median(v) if v else float("nan") for v in prob_by_step], dtype=float)
    rank_mean = np.array([np.mean(v) if v else float("nan") for v in rank_by_step], dtype=float)
    rank_med = np.array([np.median(v) if v else float("nan") for v in rank_by_step], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = axes[0]
    ax.plot(steps, prob_mean, color="#2171b5", lw=2.4, label="mean")
    ax.plot(steps, prob_med, color="#084594", lw=2.2, linestyle="--", label="median")
    ax.set_title("Target Probability (avg over tokens)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(steps, rank_mean, color="#238b45", lw=2.4, label="mean")
    ax.plot(steps, rank_med, color="#00441b", lw=2.2, linestyle="--", label="median")
    ax.set_title("Target Rank (avg over tokens)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rank (log scale)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_avg_curves.png"), facecolor="white")
    plt.close()


def visualize_entropy_vs_correctness(output: Dict, out_dir: str, exp_name: str) -> None:
    """ANE@step0 distribution for correct vs wrong tokens (sanity/insight)."""
    entries, _ = _collect_token_series(output)
    if not entries:
        return

    ane0 = []
    correct0 = []
    for e in entries:
        m = e.get("metrics", {})
        a = m.get("ane", [])
        if not a:
            continue
        ane0.append(float(a[0]))
        correct0.append(bool(e.get("correct_before", False)))

    if not ane0:
        return

    ane0 = np.array(ane0, dtype=float)
    correct0 = np.array(correct0, dtype=bool)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.hist(ane0[correct0], bins=35, alpha=0.75, color="#238b45", label="correct", edgecolor="white", linewidth=0.5)
    ax.hist(ane0[~correct0], bins=35, alpha=0.60, color="#d62728", label="wrong", edgecolor="white", linewidth=0.5)
    ax.set_title("Baseline ANE vs correctness (token-level)")
    ax.set_xlabel("ANE at step=0")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_ane_vs_correct.png"), facecolor="white")
    plt.close()


def visualize_all(output: Dict, out_dir: str, exp_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    visualize_avg_curves(output, out_dir, exp_name)
    visualize_entropy_vs_correctness(output, out_dir, exp_name)
    visualize_prob_heatmap(output, out_dir, exp_name)
    visualize_rank_heatmap(output, out_dir, exp_name)
    visualize_top10_case_evolution(output, out_dir, exp_name)
    visualize_flipped_curves(output, out_dir, exp_name)
    visualize_subspace_evolution(output, out_dir, exp_name)

