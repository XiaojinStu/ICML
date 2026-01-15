"""Publication-style visualization for v5.1 teacher-forcing experiments."""

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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
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
    # low rank (good) -> dark green; high rank -> near white
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


def visualize_prob_heatmap(output: Dict, out_dir: str, exp_name: str, tokens_per_page: int = 60) -> None:
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

        fig_h = max(8, mat.shape[0] * 0.20)
        fig_w = max(12, mat.shape[1] * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = float(mat[i, j])
                if val < 0.01:
                    continue
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

        # sample boundaries
        page_boundaries = [b - start for b in boundaries if start <= b <= end]
        for b in page_boundaries[1:-1]:
            ax.axhline(b - 0.5, color="#222222", lw=1.0, alpha=0.7)

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Target Probability (numerical sub-vocab)")

        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Token Index")
        ax.set_xticks(list(range(n_steps)))
        ax.set_xticklabels([str(i) for i in range(n_steps)], fontsize=9)
        ax.set_yticks(list(range(mat.shape[0])))
        ax.set_yticklabels([f"{start+i:03d}:{e['target_token']}" for i, e in enumerate(page_entries)], fontsize=9)

        suffix = f"_p{page+1}" if n_pages > 1 else ""
        ax.set_title(f"Target Probability Evolution ({exp_name}{suffix})")

        plt.tight_layout()
        plt.subplots_adjust(left=0.14)
        plt.savefig(os.path.join(out_dir, f"{exp_name}_prob_heatmap{suffix}.png"), facecolor="white")
        plt.close()


def visualize_rank_heatmap(output: Dict, out_dir: str, exp_name: str, tokens_per_page: int = 60) -> None:
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

        fig_h = max(8, mat.shape[0] * 0.20)
        fig_w = max(12, mat.shape[1] * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = int(mat[i, j])
                color = _text_color_from_bg(cmap, norm, float(val), 1.0, float(max_rank))
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

        page_boundaries = [b - start for b in boundaries if start <= b <= end]
        for b in page_boundaries[1:-1]:
            ax.axhline(b - 0.5, color="#222222", lw=1.0, alpha=0.7)

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(f"Target Rank in numerical sub-vocab (1-{max_rank})")

        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Token Index")
        ax.set_xticks(list(range(n_steps)))
        ax.set_xticklabels([str(i) for i in range(n_steps)], fontsize=9)
        ax.set_yticks(list(range(mat.shape[0])))
        ax.set_yticklabels([f"{start+i:03d}:{e['target_token']}" for i, e in enumerate(page_entries)], fontsize=9)

        suffix = f"_p{page+1}" if n_pages > 1 else ""
        ax.set_title(f"Target Rank Evolution ({exp_name}{suffix})")

        plt.tight_layout()
        plt.subplots_adjust(left=0.14)
        plt.savefig(os.path.join(out_dir, f"{exp_name}_rank_heatmap{suffix}.png"), facecolor="white")
        plt.close()


def visualize_top10_cases(output: Dict, out_dir: str, exp_name: str, n_cases: int = 6) -> None:
    entries, _ = _collect_token_series(output)
    if not entries:
        return

    cases = []
    for e in entries:
        if not e["correct_before"] and e["correct_after"]:
            cases.append(e)
    if not cases:
        # fallback: pick biggest rank improvement
        for e in entries:
            r = e["rank"]
            if r:
                cases.append(e)
        cases.sort(key=lambda x: (x["rank"][0] - x["rank"][-1]), reverse=True)

    cases = cases[:n_cases]
    if not cases:
        return

    n_cols = 2
    n_rows = math.ceil(len(cases) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2 * n_cols, 4.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, case in enumerate(cases):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        metrics = case.get("metrics", {})

        topk_probs = np.array(metrics.get("num_topk_probs", []), dtype=float)
        target_prob = np.array(metrics.get("target_prob", []), dtype=float)
        if topk_probs.size == 0 or target_prob.size == 0:
            ax.axis("off")
            continue

        steps = np.arange(target_prob.shape[0])
        n_lines = min(10, topk_probs.shape[1] if topk_probs.ndim == 2 else 0)

        colors = plt.cm.tab10(np.linspace(0, 0.9, n_lines))
        for j in range(n_lines):
            ax.plot(steps, topk_probs[:, j], color=colors[j], lw=1.4, alpha=0.55)

        # target highlighted
        ax.plot(steps, target_prob, color="#d62728", lw=2.6, linestyle="--", label="target")
        for ms in [0, len(steps) // 2, len(steps) - 1]:
            ax.scatter(ms, target_prob[ms], color="#d62728", s=120, marker="*", edgecolors="white", linewidths=0.8, zorder=5)

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Step")
        ax.set_ylabel("Probability")
        ax.grid(True, linestyle="--", alpha=0.25)

        title = f"{case['pred_before']} -> {case['pred_after']} (target {case['target_token']})"
        ax.set_title(title, fontsize=11)

    for idx in range(len(cases), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_top10_cases.png"), facecolor="white")
    plt.close()


def visualize_all(output: Dict, out_dir: str, exp_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    visualize_prob_heatmap(output, out_dir, exp_name)
    visualize_rank_heatmap(output, out_dir, exp_name)
    visualize_top10_cases(output, out_dir, exp_name)
