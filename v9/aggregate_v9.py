"""Aggregate v9 experiment outputs into paper-ready tables + high-res dashboards.

Reads JSON files produced by:
- `v9/experiment_v9.py`
- `v9/run_grid_v9.py`

Writes:
- `summary_runs.csv`: one row per run (eval_mode-specific)
- `table_main.csv`: merged TF + AR metrics per (dataset, model, steps, lr, ...)
- `plots/*__dashboard.png`: a 2x3 grid of heatmaps (steps x lr) for key metrics

Optional:
- `selected_viz/**`: render detailed per-run visualizations for the best config per (dataset, model, mode).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


setup_style()


RUN_FIELDS = [
    "json_path",
    "dataset",
    "model",
    "eval_mode",
    "steps",
    "lr",
    "lr_schedule",
    "update_target",
    "num_layers",
    "tta_reset",
    "seq_total",
    "token_total",
    "token_acc_before",
    "token_acc_after",
    "pass2_before",
    "pass2_after",
    "pass5_before",
    "pass5_after",
    "em_before",
    "em_after",
    "digit_acc_before",
    "digit_acc_after",
    "runtime_seconds",
    "runtime_per_token",
    "tokens_per_second",
    "flipped_count",
]


MAIN_FIELDS = [
    "dataset",
    "model",
    "steps",
    "lr",
    "lr_schedule",
    "update_target",
    "num_layers",
    "tta_reset",
    "json_tf",
    "json_ar",
    "acc_tf_before",
    "acc_tf_after",
    "pass2_before",
    "pass2_after",
    "pass5_before",
    "pass5_after",
    "runtime_per_token_tf",
    "acc_ar_before",
    "acc_ar_after",
    "em_before",
    "em_after",
    "digit_acc_before",
    "digit_acc_after",
    "runtime_per_token_ar",
]


@dataclass(frozen=True)
class Run:
    path: Path
    summary: Dict[str, Any]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def find_json(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def write_csv(rows: List[Dict[str, Any]], out_path: Path, fields: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _pass_at(summary: Dict[str, Any], k: int, *, which: str) -> Optional[float]:
    d = summary.get(f"pass@k_acc_{which}", {})
    if not isinstance(d, dict):
        return None
    return _safe_float(d.get(str(int(k))))


def _runtime_per_token(summary: Dict[str, Any]) -> Optional[float]:
    t = _safe_float(summary.get("runtime_seconds"))
    n = _safe_float(summary.get("token_total"))
    if t is None or n is None or n <= 0:
        return None
    return float(t / n)


def _config_key(summary: Dict[str, Any]) -> Tuple:
    return (
        summary.get("dataset"),
        summary.get("model"),
        int(summary.get("steps", 0)),
        float(summary.get("lr", 0.0)),
        summary.get("lr_schedule"),
        summary.get("update_target"),
        summary.get("num_layers"),
        summary.get("tta_reset"),
    )


def _format_cell(after: Optional[float], before: Optional[float], *, percent: bool) -> str:
    if after is None:
        return ""
    if percent:
        a = 100 * float(after)
        if before is None:
            return f"{a:.1f}"
        b = 100 * float(before)
        return f"{a:.1f}\n({a-b:+.1f})"
    if before is None:
        return f"{after:.3g}"
    return f"{after:.3g}\n({after-before:+.3g})"


def _full_box(ax) -> None:
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)


def plot_heatmap(
    ax,
    mat: np.ndarray,
    before: Optional[np.ndarray],
    *,
    steps_list: List[int],
    lr_list: List[float],
    title: str,
    percent: bool,
    cmap: str,
) -> None:
    vals = mat[~np.isnan(mat)]
    if vals.size == 0:
        ax.axis("off")
        return
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if abs(vmax - vmin) < 1e-8:
        vmin -= 1e-4
        vmax += 1e-4

    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    _full_box(ax)
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xticks(range(len(lr_list)))
    ax.set_xticklabels([f"{lr:g}" for lr in lr_list], fontweight="bold")
    ax.set_yticks(range(len(steps_list)))
    ax.set_yticklabels([str(s) for s in steps_list], fontweight="bold")
    ax.set_xlabel("lr", fontweight="bold")
    ax.set_ylabel("steps", fontweight="bold")

    # Annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            a = mat[i, j]
            if np.isnan(a):
                continue
            b = None
            if before is not None:
                bb = before[i, j]
                b = float(bb) if not np.isnan(bb) else None
            ax.text(j, i, _format_cell(float(a), b, percent=percent), ha="center", va="center", fontsize=12, fontweight="bold", color="white" if a < (vmin + vmax) / 2 else "black")

    return im


def build_mats(runs: List[Run], *, dataset: str, model: str, eval_mode: str, key_after: str, key_before: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray], List[int], List[float]]:
    subset = [r for r in runs if r.summary.get("dataset") == dataset and r.summary.get("model") == model and r.summary.get("eval_mode") == eval_mode]
    steps_list = sorted({int(r.summary.get("steps", 0)) for r in subset})
    lr_list = sorted({float(r.summary.get("lr", 0.0)) for r in subset})
    mat = np.full((len(steps_list), len(lr_list)), np.nan, dtype=float)
    before = np.full_like(mat, np.nan) if key_before else None
    for r in subset:
        s = r.summary
        i = steps_list.index(int(s.get("steps", 0)))
        j = lr_list.index(float(s.get("lr", 0.0)))
        a = _safe_float(s.get(key_after))
        if a is not None:
            mat[i, j] = a
        if key_before and before is not None:
            b = _safe_float(s.get(key_before))
            if b is not None:
                before[i, j] = b
    return mat, before, steps_list, lr_list


def plot_dashboard(runs: List[Run], *, out_path: Path, dataset: str, model: str) -> None:
    mats: List[Tuple[str, str, bool, str, str, Optional[str]]] = [
        ("TF token acc", "tf", True, "viridis", "token_acc_after", "token_acc_before"),
        ("TF pass@2", "tf", True, "viridis", "pass2_after", "pass2_before"),
        ("TF pass@5", "tf", True, "viridis", "pass5_after", "pass5_before"),
        ("AR token acc", "ar", True, "viridis", "token_acc_after", "token_acc_before"),
        ("AR EM", "ar", True, "viridis", "em_after", "em_before"),
        ("AR digit acc", "ar", True, "viridis", "digit_acc_after", "digit_acc_before"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.8))
    im = None
    for ax, (title, mode, percent, cmap, k_after, k_before) in zip(axes.flat, mats):
        # We allow computed fields pass2/pass5 injected at runtime; handle separately.
        if k_after.startswith("pass"):
            tmp = []
            k = 2 if "pass2" in k_after else 5
            for r in runs:
                if r.summary.get("dataset") == dataset and r.summary.get("model") == model and r.summary.get("eval_mode") == mode:
                    s2 = dict(r.summary)
                    s2["pass2_after"] = _pass_at(r.summary, 2, which="after")
                    s2["pass2_before"] = _pass_at(r.summary, 2, which="before")
                    s2["pass5_after"] = _pass_at(r.summary, 5, which="after")
                    s2["pass5_before"] = _pass_at(r.summary, 5, which="before")
                    tmp.append(Run(path=r.path, summary=s2))
            mat, before, steps_list, lr_list = build_mats(tmp, dataset=dataset, model=model, eval_mode=mode, key_after=k_after, key_before=k_before)
        else:
            mat, before, steps_list, lr_list = build_mats(runs, dataset=dataset, model=model, eval_mode=mode, key_after=k_after, key_before=k_before)

        im = plot_heatmap(ax, mat, before, steps_list=steps_list, lr_list=lr_list, title=title, percent=percent, cmap=cmap)

    # Single shared colorbar (for percentage metrics).
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("score (after), with Δ in parentheses", fontweight="bold")

    fig.suptitle(f"{dataset} | {model} | v9 dashboard (steps × lr)", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate v9 results into tables + dashboards.")
    p.add_argument("--root", default="v9/results_main")
    p.add_argument("--out_dir", default="v9/summary_main")
    p.add_argument("--filter_models", default=None, help="Comma-separated model basenames to include.")
    p.add_argument("--filter_datasets", default=None, help="Comma-separated datasets to include.")
    p.add_argument("--render_viz_top_n", type=int, default=1, help="Per (dataset,model,mode), render detailed viz for top-N runs. 0 disables.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_filter = None
    if args.filter_models:
        model_filter = {m.strip() for m in args.filter_models.split(",") if m.strip()}
    dataset_filter = None
    if args.filter_datasets:
        dataset_filter = {d.strip() for d in args.filter_datasets.split(",") if d.strip()}

    runs: List[Run] = []
    for p in find_json(root):
        obj = load_json(p)
        s = obj.get("summary", {})
        if not s or s.get("version") != "v9":
            continue
        if model_filter is not None and s.get("model") not in model_filter:
            continue
        if dataset_filter is not None and s.get("dataset") not in dataset_filter:
            continue
        runs.append(Run(path=p, summary=s))

    if not runs:
        raise SystemExit(f"No v9 results found under {root}")

    # One row per run.
    run_rows: List[Dict[str, Any]] = []
    for r in runs:
        s = r.summary
        run_rows.append(
            {
                "json_path": str(r.path),
                "dataset": s.get("dataset"),
                "model": s.get("model"),
                "eval_mode": s.get("eval_mode"),
                "steps": s.get("steps"),
                "lr": s.get("lr"),
                "lr_schedule": s.get("lr_schedule"),
                "update_target": s.get("update_target"),
                "num_layers": s.get("num_layers"),
                "tta_reset": s.get("tta_reset"),
                "seq_total": s.get("seq_total"),
                "token_total": s.get("token_total"),
                "token_acc_before": s.get("token_acc_before"),
                "token_acc_after": s.get("token_acc_after"),
                "pass2_before": _pass_at(s, 2, which="before"),
                "pass2_after": _pass_at(s, 2, which="after"),
                "pass5_before": _pass_at(s, 5, which="before"),
                "pass5_after": _pass_at(s, 5, which="after"),
                "em_before": s.get("em_before"),
                "em_after": s.get("em_after"),
                "digit_acc_before": s.get("digit_acc_before"),
                "digit_acc_after": s.get("digit_acc_after"),
                "runtime_seconds": s.get("runtime_seconds"),
                "runtime_per_token": _runtime_per_token(s),
                "tokens_per_second": s.get("tokens_per_second"),
                "flipped_count": s.get("flipped_count"),
            }
        )

    write_csv(run_rows, out_dir / "summary_runs.csv", RUN_FIELDS)

    # Merge TF + AR into main table.
    by_key: Dict[Tuple, Dict[str, Run]] = {}
    for r in runs:
        by_key.setdefault(_config_key(r.summary), {})
        by_key[_config_key(r.summary)][str(r.summary.get("eval_mode"))] = r

    main_rows: List[Dict[str, Any]] = []
    for key, modes in by_key.items():
        tf = modes.get("tf")
        ar = modes.get("ar")
        base = (tf.summary if tf else (ar.summary if ar else {})) or {}
        main_rows.append(
            {
                "dataset": base.get("dataset"),
                "model": base.get("model"),
                "steps": base.get("steps"),
                "lr": base.get("lr"),
                "lr_schedule": base.get("lr_schedule"),
                "update_target": base.get("update_target"),
                "num_layers": base.get("num_layers"),
                "tta_reset": base.get("tta_reset"),
                "json_tf": str(tf.path) if tf else "",
                "json_ar": str(ar.path) if ar else "",
                "acc_tf_before": tf.summary.get("token_acc_before") if tf else None,
                "acc_tf_after": tf.summary.get("token_acc_after") if tf else None,
                "pass2_before": _pass_at(tf.summary, 2, which="before") if tf else None,
                "pass2_after": _pass_at(tf.summary, 2, which="after") if tf else None,
                "pass5_before": _pass_at(tf.summary, 5, which="before") if tf else None,
                "pass5_after": _pass_at(tf.summary, 5, which="after") if tf else None,
                "runtime_per_token_tf": _runtime_per_token(tf.summary) if tf else None,
                "acc_ar_before": ar.summary.get("token_acc_before") if ar else None,
                "acc_ar_after": ar.summary.get("token_acc_after") if ar else None,
                "em_before": ar.summary.get("em_before") if ar else None,
                "em_after": ar.summary.get("em_after") if ar else None,
                "digit_acc_before": ar.summary.get("digit_acc_before") if ar else None,
                "digit_acc_after": ar.summary.get("digit_acc_after") if ar else None,
                "runtime_per_token_ar": _runtime_per_token(ar.summary) if ar else None,
            }
        )

    main_rows.sort(key=lambda r: (r["dataset"], r["model"], float(r["lr"]), int(r["steps"])))
    write_csv(main_rows, out_dir / "table_main.csv", MAIN_FIELDS)

    # Dashboards.
    plots_dir = out_dir / "plots"
    datasets = sorted({r.summary.get("dataset") for r in runs})
    models = sorted({r.summary.get("model") for r in runs})
    for d in datasets:
        for m in models:
            plot_dashboard(runs, out_path=plots_dir / f"{d}__{m}__dashboard.png", dataset=d, model=m)

    # Optional: detailed per-run visualizations.
    top_n = int(args.render_viz_top_n)
    if top_n > 0:
        try:
            from visualization import visualize_all
        except Exception:
            visualize_all = None  # type: ignore
        if visualize_all is not None:
            selected_dir = out_dir / "selected_viz"
            for d in datasets:
                for m in models:
                    for mode, metric in [("tf", "token_acc_after"), ("ar", "em_after")]:
                        subset = [r for r in runs if r.summary.get("dataset") == d and r.summary.get("model") == m and r.summary.get("eval_mode") == mode]
                        subset.sort(key=lambda r: float(r.summary.get(metric, 0.0) or 0.0), reverse=True)
                        for r in subset[:top_n]:
                            obj = load_json(r.path)
                            exp_name = r.path.stem
                            visualize_all(obj, str(selected_dir / d / m), exp_name)

    print(f"Wrote: {out_dir/'summary_runs.csv'}")
    print(f"Wrote: {out_dir/'table_main.csv'}")
    print(f"Wrote plots to: {plots_dir}")


if __name__ == "__main__":
    main()

