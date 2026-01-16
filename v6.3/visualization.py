"""Publication-style visualization for v6 experiments."""

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


def visualize_prob_heatmap(
    output: Dict,
    out_dir: str,
    exp_name: str,
    tokens_per_page: int = 120,
    max_pages: int | None = None,
) -> None:
    entries, boundaries = _collect_token_series(output)
    if not entries:
        return

    probs = [e["prob"] for e in entries]
    n_steps = max(len(p) for p in probs)
    cmap = create_prob_cmap()

    n_pages = math.ceil(len(entries) / tokens_per_page)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)
    for page in range(n_pages):
        start = page * tokens_per_page
        end = min(len(entries), start + tokens_per_page)
        page_entries = entries[start:end]
        mat = _pad_matrix([e["prob"] for e in page_entries], n_steps)

        fig_h = min(22, max(7, mat.shape[0] * 0.18))
        fig_w = min(18, max(11, mat.shape[1] * 0.55))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

        annotate = mat.shape[0] <= 90 and mat.shape[1] <= 25
        if annotate:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = float(mat[i, j])
                    if val < 0.01:
                        continue
                    color = _text_color_from_bg(cmap, None, val, 0.0, 1.0)
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


def visualize_rank_heatmap(
    output: Dict,
    out_dir: str,
    exp_name: str,
    tokens_per_page: int = 120,
    max_pages: int | None = None,
) -> None:
    entries, boundaries = _collect_token_series(output)
    if not entries:
        return

    ranks = [e["rank"] for e in entries]
    n_steps = max(len(r) for r in ranks)
    max_rank = int(max(max(r) for r in ranks if r))

    cmap = create_rank_cmap()
    norm = LogNorm(vmin=1.0, vmax=float(max_rank))

    n_pages = math.ceil(len(entries) / tokens_per_page)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)
    for page in range(n_pages):
        start = page * tokens_per_page
        end = min(len(entries), start + tokens_per_page)
        page_entries = entries[start:end]
        mat = _pad_matrix([e["rank"] for e in page_entries], n_steps)

        fig_h = min(22, max(7, mat.shape[0] * 0.18))
        fig_w = min(18, max(11, mat.shape[1] * 0.55))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

        annotate = mat.shape[0] <= 90 and mat.shape[1] <= 25
        if annotate:
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


def _percentile_band(values_by_step: List[List[float]], lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    lo_v = []
    hi_v = []
    for step_vals in values_by_step:
        if not step_vals:
            lo_v.append(float("nan"))
            hi_v.append(float("nan"))
            continue
        a = np.array(step_vals, dtype=float)
        lo_v.append(float(np.nanpercentile(a, lo)))
        hi_v.append(float(np.nanpercentile(a, hi)))
    return np.array(lo_v, dtype=float), np.array(hi_v, dtype=float)


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
    # values_by_step[k] = list of values across tokens at step k
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

    prob_lo, prob_hi = _percentile_band(prob_by_step, 25, 75)
    rank_lo, rank_hi = _percentile_band(rank_by_step, 25, 75)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))

    ax = axes[0]
    ax.plot(steps, prob_mean, color="#2171b5", lw=2.2, label="mean")
    ax.plot(steps, prob_med, color="#084594", lw=2.0, linestyle="--", label="median")
    ax.fill_between(steps, prob_lo, prob_hi, color="#c6dbef", alpha=0.6, label="IQR")
    ax.set_title("Target Probability (avg over tokens)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(steps, rank_mean, color="#238b45", lw=2.2, label="mean")
    ax.plot(steps, rank_med, color="#00441b", lw=2.0, linestyle="--", label="median")
    ax.fill_between(steps, rank_lo, rank_hi, color="#c7e9c0", alpha=0.6, label="IQR")
    ax.set_title("Target Rank (avg over tokens)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rank (log scale)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_avg_curves.png"), facecolor="white")
    plt.close()


def visualize_delta_hist(output: Dict, out_dir: str, exp_name: str) -> None:
    """Distributions of delta prob and delta rank (after - before)."""
    entries, _ = _collect_token_series(output)
    if not entries:
        return

    dprob = []
    drank = []
    for e in entries:
        p = e.get("prob", [])
        r = e.get("rank", [])
        if p and r:
            dprob.append(float(p[-1]) - float(p[0]))
            drank.append(float(r[-1]) - float(r[0]))

    if not dprob:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))

    ax = axes[0]
    ax.hist(dprob, bins=35, color="#2171b5", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axvline(0.0, color="#222222", lw=1.0)
    ax.set_title("Δ Target Probability (after - before)")
    ax.set_xlabel("Δ prob")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.2)

    ax = axes[1]
    ax.hist(drank, bins=35, color="#238b45", alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axvline(0.0, color="#222222", lw=1.0)
    ax.set_title("Δ Target Rank (after - before)")
    ax.set_xlabel("Δ rank (negative is better)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_delta_hist.png"), facecolor="white")
    plt.close()


def visualize_sample_transitions(output: Dict, out_dir: str, exp_name: str) -> None:
    """Sample-level before/after correctness transitions (token-seq + string)."""
    results = output.get("results", [])
    if not results:
        return

    seq_counts = {"c->c": 0, "c->w": 0, "w->c": 0, "w->w": 0}
    str_counts = {"c->c": 0, "c->w": 0, "w->c": 0, "w->w": 0}

    for sample in results:
        toks = sample.get("tokens", [])
        if toks:
            all_before = all(bool(t.get("correct_before", False)) for t in toks)
            all_after = all(bool(t.get("correct_after", False)) for t in toks)
            key = ("c" if all_before else "w") + "->" + ("c" if all_after else "w")
            seq_counts[key] += 1

        cb = sample.get("correct_before_str")
        ca = sample.get("correct_after_str")
        if cb is None or ca is None:
            continue
        key = ("c" if cb else "w") + "->" + ("c" if ca else "w")
        str_counts[key] += 1

    def _bar(ax, counts, title):
        labels = ["c->c", "c->w", "w->c", "w->w"]
        vals = [counts[k] for k in labels]
        colors = ["#2171b5", "#9ecae1", "#238b45", "#c7e9c0"]
        ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.6, str(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", linestyle="--", alpha=0.2)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
    _bar(axes[0], seq_counts, "Token-seq correctness transitions")
    _bar(axes[1], str_counts, "String correctness transitions (AR)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_sample_transitions.png"), facecolor="white")
    plt.close()

def visualize_stats(output: Dict, out_dir: str, exp_name: str) -> None:
    s = output.get("summary", {})
    if not s:
        return

    metrics = [
        ("Token", float(s.get("token_acc_before", 0.0)), float(s.get("token_acc_after", 0.0))),
        ("Seq", float(s.get("seq_acc_before", 0.0)), float(s.get("seq_acc_after", 0.0))),
        ("Str", float(s.get("str_acc_before", 0.0)), float(s.get("str_acc_after", 0.0))),
    ]

    labels = [m[0] for m in metrics]
    before = [m[1] * 100 for m in metrics]
    after = [m[2] * 100 for m in metrics]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, before, width, label="before", color="#9ecae1")
    ax.bar(x + width / 2, after, width, label="after", color="#2171b5")

    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - width / 2, b + 0.6, f"{b:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, a + 0.6, f"{a:.1f}", ha="center", va="bottom", fontsize=9)

    title = f"{s.get('model', '')} | {s.get('update_target', '')} | steps={s.get('steps', '')} | reset={s.get('tta_reset', '')}"
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(after + before + [0.0]) + 8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{exp_name}_stats.png"), facecolor="white")
    plt.close()


def visualize_all(output: Dict, out_dir: str, exp_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    visualize_stats(output, out_dir, exp_name)
    visualize_sample_transitions(output, out_dir, exp_name)
    visualize_avg_curves(output, out_dir, exp_name)
    visualize_delta_hist(output, out_dir, exp_name)
    visualize_prob_heatmap(output, out_dir, exp_name)
    visualize_rank_heatmap(output, out_dir, exp_name)
    visualize_top10_cases(output, out_dir, exp_name)
