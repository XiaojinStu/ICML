"""Publication-style visualization for v9 experiments."""

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
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.dpi": 200,
            "savefig.dpi": 520,
            "savefig.bbox": "tight",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.0,
        }
    )


setup_style()


def create_prob_cmap() -> LinearSegmentedColormap:
    colors = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"]
    return LinearSegmentedColormap.from_list("prob_blue", colors, N=256)


def create_rank_cmap() -> LinearSegmentedColormap:
    colors = ["#00441b", "#238b45", "#74c476", "#c7e9c0", "#f7fcf5"]
    return LinearSegmentedColormap.from_list("rank_green", colors, N=256)


def _ensure_2d_axes(axes, n_rows: int, n_cols: int):
    """Normalize matplotlib axes output into a 2D ndarray (n_rows x n_cols)."""
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return np.array([axes])
    if n_cols == 1:
        return np.array([[a] for a in axes])
    return axes


def _full_box(ax) -> None:
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)


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


def visualize_prob_heatmap(output: Dict, out_dir: str, exp_name: str, tokens_per_page: int = 80) -> None:
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
        _full_box(ax)

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
        cbar.set_label("Target Probability (numerical sub-vocab)", fontweight="bold")

        ax.set_xlabel("Optimization Step", fontweight="bold")
        ax.set_ylabel("Token Index", fontweight="bold")
        ax.set_xticks(list(range(n_steps)))
        ax.set_xticklabels([str(i) for i in range(n_steps)])
        ax.set_yticks(list(range(mat.shape[0])))
        ax.set_yticklabels([f"{start+i:03d}:{e['target_token']}" for i, e in enumerate(page_entries)], fontsize=10)

        suffix = f"_p{page+1}" if n_pages > 1 else ""
        ax.set_title(_paper_label(output))

        plt.tight_layout(pad=0.4)
        plt.subplots_adjust(left=0.14)
        plt.savefig(os.path.join(out_dir, f"{exp_name}_prob_heatmap{suffix}.png"), facecolor="white")
        plt.close()


def visualize_rank_heatmap(output: Dict, out_dir: str, exp_name: str, tokens_per_page: int = 80) -> None:
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
        _full_box(ax)

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
        cbar.set_label(f"Target Rank in numerical sub-vocab (1-{max_rank})", fontweight="bold")

        ax.set_xlabel("Optimization Step", fontweight="bold")
        ax.set_ylabel("Token Index", fontweight="bold")
        ax.set_xticks(list(range(n_steps)))
        ax.set_xticklabels([str(i) for i in range(n_steps)])
        ax.set_yticks(list(range(mat.shape[0])))
        ax.set_yticklabels([f"{start+i:03d}:{e['target_token']}" for i, e in enumerate(page_entries)], fontsize=10)

        suffix = f"_p{page+1}" if n_pages > 1 else ""
        ax.set_title(_paper_label(output))

        plt.tight_layout(pad=0.4)
        plt.subplots_adjust(left=0.14)
        plt.savefig(os.path.join(out_dir, f"{exp_name}_rank_heatmap{suffix}.png"), facecolor="white")
        plt.close()


def visualize_top10_case_evolution(output: Dict, out_dir: str, exp_name: str, n_cases: int = 10) -> None:
    """Top cases: tracked-topk probability trajectories + target probability."""
    entries, _ = _collect_token_series(output)
    cases = _select_cases(_collect_cases(entries), n_cases * 3, prefer_flipped=True)
    # Require tracked_topk to exist (compact save mode keeps it only for selected tokens).
    cases = [c for c in cases if c.get("metrics", {}).get("tracked_topk", {}).get("probs")]
    cases = cases[:n_cases]
    if not cases:
        return

    n_cols = 2
    n_rows = math.ceil(len(cases) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.4 * n_cols, 4.8 * n_rows))
    axes = _ensure_2d_axes(axes, n_rows, n_cols)

    for idx, case in enumerate(cases):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        _full_box(ax)
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
    axes = _ensure_2d_axes(axes, n_rows, n_cols)

    for idx, case in enumerate(cases):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        _full_box(ax)
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
            _full_box(ax2)
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
    cases = _select_cases(_collect_cases(entries), n_cases * 3, prefer_flipped=True)
    cases = [c for c in cases if c.get("metrics", {}).get("num_topk_snapshots")]
    cases = cases[:n_cases]
    if not cases:
        return

    # Determine the available steps from the first case with snapshots.
    available_steps = None
    for c in cases:
        snaps = c["metrics"].get("num_topk_snapshots", [])
        if snaps:
            available_steps = [int(s["step"]) for s in snaps if "step" in s]
            break
    if not available_steps:
        return

    wanted = output.get("summary", {}).get("checkpoint_steps")
    if isinstance(wanted, list) and wanted:
        steps = [int(t) for t in wanted if int(t) in set(available_steps)]
    else:
        steps = _default_step_checkpoints(int(output.get("summary", {}).get("steps", max(available_steps))))
        steps = [t for t in steps if t in set(available_steps)]
    if not steps:
        steps = available_steps

    n_rows = len(cases)
    n_cols = len(steps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows))
    axes = _ensure_2d_axes(axes, n_rows, n_cols)

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
                ax.set_title(f"t={step}", fontsize=12, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"Case {r+1}", fontsize=12)

            for i, v in enumerate(probs):
                if v >= 0.05:
                    ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)

    plt.suptitle(f"{_paper_label(output)} — Numerical Subspace Evolution (top-k)", y=1.02, fontsize=14, fontweight="bold")
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
    _full_box(ax)
    ax.plot(steps, prob_mean, color="#2171b5", lw=2.4, label="mean")
    ax.plot(steps, prob_med, color="#084594", lw=2.2, linestyle="--", label="median")
    ax.set_title("Target Probability (avg over tokens)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    _full_box(ax)
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
    _full_box(ax)
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


def _pca_2d(x: np.ndarray) -> np.ndarray:
    """Simple PCA (SVD) to 2D without sklearn. x: (n, d)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    w = vt[:2].T  # (d,2)
    return x @ w


def _paper_dataset_name(dataset: str) -> str:
    d = str(dataset or "").strip()
    dl = d.lower()
    if "gsm8k" in dl:
        return "GSM8K"
    if "svamp" in dl:
        return "SVAMP"
    if "nupa" in dl:
        return "NUPA"
    if "math401" in dl:
        return "MATH401"
    if "numericbench" in dl:
        return "NumericBench"
    if "bigbench_arithmetic" in dl or "bigbench-arithmetic" in dl or "big-bench-arithmetic" in dl or dl.startswith("bb_arith"):
        return "BIG-Bench Arithmetic"
    if "bigbench_mixed" in dl or "bigbench-mixed" in dl or "big-bench-mixed" in dl or dl.startswith("bb_mixed"):
        return "BIG-Bench Mixed"
    if "addition" in dl:
        return "Addition-50"
    return d or "Dataset"


def _paper_model_name(model: str) -> str:
    m = str(model or "").strip()
    base = os.path.basename(m) if m else ""
    base = base or m
    for suf in ["-Instruct-2507", "-Instruct"]:
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base or "Model"


def _paper_label(output: Dict) -> str:
    s = output.get("summary", {}) or {}
    return f"{_paper_model_name(s.get('model'))} | {_paper_dataset_name(s.get('dataset'))}"


def _default_step_checkpoints(steps: int) -> List[int]:
    base = [0, 1, 2, 5, 10, 15, 20, 25, 30]
    steps = int(steps)
    keep = sorted({t for t in base if 0 <= t <= steps} | {0, steps})
    return keep


def visualize_anchor_traces(output: Dict, out_dir: str, exp_name: str, n_cases: int = 4, topk_show: int = 10) -> None:
    """High-density anchor embedding visualizations for (typically) flipped tokens."""

    candidates = []
    for sample in output.get("results", []):
        for tok in sample.get("tokens", []):
            m = tok.get("metrics", {})
            tr = m.get("anchor_trace")
            if not tr:
                continue
            ranks = m.get("target_rank", [])
            delta_rank = float(ranks[0]) - float(ranks[-1]) if ranks else 0.0
            flipped = bool(tr.get("flipped", False))
            candidates.append((1 if flipped else 0, delta_rank, sample, tok))

    if not candidates:
        return

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    cases = candidates[: max(1, int(n_cases))]

    run_label = _paper_label(output)
    n_rows = len(cases)
    fig, axes = plt.subplots(n_rows, 3, figsize=(19.0, 5.2 * n_rows))
    axes = _ensure_2d_axes(axes, n_rows, 3)

    for r, (_, __, sample, tok) in enumerate(cases):
        m = tok.get("metrics", {})
        tr = m.get("anchor_trace", {}) or {}
        embeds = tr.get("embeds", {}) or {}

        anchor_dir_all = np.asarray(tr.get("anchor_dir", []), dtype=float)  # (T,D)
        cos_all = np.asarray(tr.get("cos_to_target", []), dtype=float)
        ang_all = np.asarray(tr.get("angle_to_target", []), dtype=float)
        nearest = tr.get("anchor_nearest_token", [])
        n_steps = max(0, int(len(cos_all) - 1))
        keep_steps = _default_step_checkpoints(n_steps)
        keep_steps = [t for t in keep_steps if t < len(cos_all)]

        anchor_dir = anchor_dir_all[keep_steps] if anchor_dir_all.size and anchor_dir_all.shape[0] > max(keep_steps) else np.zeros((0, 0), dtype=float)
        cos_to_target = cos_all[keep_steps] if cos_all.size else np.zeros((0,), dtype=float)
        angle_deg = (ang_all[keep_steps] * (180.0 / math.pi)) if ang_all.size else np.zeros((0,), dtype=float)
        x_steps = np.asarray(keep_steps, dtype=int)

        case_title = f"{tok.get('pred_before')}→{tok.get('pred_after')} (target {tok.get('target_token')})"
        case_meta = f"id={sample.get('id')}  pos={tok.get('position')}"

        # (1) Angle-to-target curve (+ best checkpoint)
        ax = axes[r, 0]
        _full_box(ax)
        if angle_deg.size:
            ax.plot(x_steps, angle_deg, color="#d62728", lw=2.6, marker="o", markersize=5)
            best_i = int(np.nanargmin(angle_deg)) if np.isfinite(angle_deg).any() else 0
            ax.scatter([x_steps[best_i]], [angle_deg[best_i]], marker="*", s=220, color="#238b45", edgecolors="white", linewidths=0.8, zorder=5)
            ax.set_ylabel("angle(anchor, target) [deg]", fontweight="bold")
        ax.set_title(case_title, fontsize=15, fontweight="bold")
        ax.text(0.01, 0.98, run_label, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
        ax.text(0.01, 0.91, case_meta, transform=ax.transAxes, ha="left", va="top", fontsize=11, color="#333333")
        ax.set_xlabel("Step", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_xticks(list(x_steps))
        if nearest and len(nearest) > max(keep_steps):
            nearest_keep = [nearest[t] for t in keep_steps]
            changes = [keep_steps[i] for i in range(1, len(nearest_keep)) if nearest_keep[i] != nearest_keep[i - 1]]
            for t in changes:
                ax.axvline(t, color="#111111", lw=1.4, alpha=0.18)

        # (2) PCA projection: anchor trajectory + tokens
        ax = axes[r, 1]
        _full_box(ax)
        tgt_dir = embeds.get("target_dir")
        pb_dir = embeds.get("pred_before_dir")
        pa_dir = embeds.get("pred_after_dir")
        topk_dirs = embeds.get("topk_dirs", []) or []

        vecs: List[np.ndarray] = []
        if anchor_dir.size:
            vecs.extend([np.asarray(v, dtype=float) for v in anchor_dir])
        extras = []
        for name, v in [("target", tgt_dir), ("pred_before", pb_dir), ("pred_after", pa_dir)]:
            if v is not None:
                extras.append(name)
                vecs.append(np.asarray(v, dtype=float))
        for row in topk_dirs:
            v = row.get("dir")
            if v is None:
                continue
            vecs.append(np.asarray(v, dtype=float))

        xy = _pca_2d(np.asarray(vecs, dtype=float)) if vecs else np.zeros((0, 2), dtype=float)

        t_anchor = anchor_dir.shape[0] if anchor_dir.size else 0
        if t_anchor > 0:
            pts = xy[:t_anchor]
            # Trajectory with checkpoint labels and color by angle-to-target.
            for i in range(1, t_anchor):
                ax.plot(pts[i - 1 : i + 1, 0], pts[i - 1 : i + 1, 1], color="#1f77b4", lw=2.4, alpha=0.75)
            if angle_deg.size == t_anchor:
                sc = ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=70,
                    c=angle_deg,
                    cmap="viridis_r",
                    edgecolors="white",
                    linewidths=0.7,
                    label="anchor (checkpoints)",
                )
                cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label("angle to target [deg]", fontweight="bold")
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=70, color="#1f77b4", edgecolors="white", linewidths=0.7, label="anchor (checkpoints)")
            for i, step in enumerate(keep_steps[:t_anchor]):
                ax.text(pts[i, 0], pts[i, 1], str(step), fontsize=10, fontweight="bold", ha="center", va="center", color="white")

        idx_base = t_anchor
        name_to_style = {
            "target": dict(marker="*", s=220, color="#d62728", edgecolors="white", linewidths=0.8),
            "pred_before": dict(marker="X", s=140, color="#7f7f7f", edgecolors="white", linewidths=0.8),
            "pred_after": dict(marker="P", s=140, color="#238b45", edgecolors="white", linewidths=0.8),
        }
        for k, name in enumerate(extras):
            p = xy[idx_base + k]
            ax.scatter([p[0]], [p[1]], label=name, **name_to_style[name])

        # Top-k tokens: color by Δprob (probT - prob0), size by max(prob)
        tok_start = idx_base + len(extras)
        if topk_dirs and xy.shape[0] > tok_start:
            deltas = []
            sizes = []
            xs = []
            ys = []
            for i, row in enumerate(topk_dirs):
                p = xy[tok_start + i]
                xs.append(p[0])
                ys.append(p[1])
                d = float(row.get("probT", 0.0) - row.get("prob0", 0.0))
                deltas.append(d)
                sizes.append(80 + 600 * float(max(row.get("prob0", 0.0), row.get("probT", 0.0))))
            sc = ax.scatter(xs, ys, c=deltas, cmap="coolwarm", s=sizes, alpha=0.75, edgecolors="white", linewidths=0.5)
            cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("Δ prob (final - step0)", fontweight="bold")

        ax.set_title("Anchor trajectory (PCA-2D)", fontsize=15, fontweight="bold")
        ax.set_xlabel("PC1", fontweight="bold")
        ax.set_ylabel("PC2", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.22)
        ax.legend(frameon=False, loc="best")

        # (3) Distribution change (top-k) at step0 vs best checkpoint
        ax = axes[r, 2]
        _full_box(ax)
        snaps = m.get("num_topk_snapshots", []) or []
        snap_map = {int(s.get("step")): list(s.get("tokens", [])) for s in snaps if s.get("tokens")}
        step0 = 0
        step_best = int(x_steps[int(np.nanargmin(angle_deg))]) if angle_deg.size and np.isfinite(angle_deg).any() else int(x_steps[-1]) if x_steps.size else 0
        top0 = snap_map.get(step0, [])
        topB = snap_map.get(step_best, [])

        if top0 and topB:
            p0 = {str(t.get("token")): float(t.get("prob", 0.0)) for t in top0}
            pB = {str(t.get("token")): float(t.get("prob", 0.0)) for t in topB}
            labels = sorted(set(p0.keys()) | set(pB.keys()), key=lambda k: max(p0.get(k, 0.0), pB.get(k, 0.0)), reverse=True)[: int(topk_show)]
            y = np.arange(len(labels))
            v0 = np.array([p0.get(k, 0.0) for k in labels], dtype=float)
            vB = np.array([pB.get(k, 0.0) for k in labels], dtype=float)
            target_tok = str(tok.get("target_token"))
            colors0 = ["#d62728" if k == target_tok else "#6baed6" for k in labels]
            colorsB = ["#d62728" if k == target_tok else "#fd8d3c" for k in labels]

            ax.barh(y - 0.18, v0, height=0.34, color=colors0, label="step0")
            ax.barh(y + 0.18, vB, height=0.34, color=colorsB, label=f"step{step_best}")
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontweight="bold")
            ax.invert_yaxis()
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("prob (numerical sub-vocab)", fontweight="bold")
            ax.set_title("Top-k prob shift", fontsize=15, fontweight="bold")
            ax.grid(True, axis="x", linestyle="--", alpha=0.25)
            ax.legend(frameon=False)
        else:
            ax.axis("off")

    plt.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    plt.savefig(os.path.join(out_dir, f"{exp_name}_anchor_traces.png"), facecolor="white")
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
    visualize_anchor_traces(output, out_dir, exp_name)
