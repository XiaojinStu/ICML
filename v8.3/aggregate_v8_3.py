"""Aggregate v8.3 experiment outputs into paper-ready tables + grid heatmaps.

Reads JSON files produced by:
- `v8.3/experiment_v8_3.py`
- `v8.3/run_grid_v8_3.py`

Writes:
- `summary_runs.csv`: one row per run (eval_mode-specific)
- `table_main.csv`: merged TF + AR metrics per config (paper table)
- `plots/*.png`: grid heatmaps over (steps, lr) for key metrics

Optional:
- Render rich per-token visualizations for top-N runs per (dataset, model, eval_mode).
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


RUN_FIELDS = [
    "json_path",
    "dataset",
    "model",
    "eval_mode",
    "steps",
    "lr",
    "lr_schedule",
    "lr_min",
    "optimizer",
    "momentum",
    "update_target",
    "num_layers",
    "layer_stride",
    "tta_reset",
    "seq_total",
    "token_total",
    "token_acc_before",
    "token_acc_after",
    "pass2_before",
    "pass2_after",
    "pass5_before",
    "pass5_after",
    "target_rank_avg_before",
    "target_rank_avg_after",
    "target_prob_avg_before",
    "target_prob_avg_after",
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
    "lr_min",
    "optimizer",
    "momentum",
    "update_target",
    "num_layers",
    "layer_stride",
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
    v = d.get(str(int(k)))
    return _safe_float(v)


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
        float(summary.get("lr_min", 0.0)),
        summary.get("optimizer"),
        float(summary.get("momentum", 0.0)),
        summary.get("update_target"),
        summary.get("num_layers"),
        int(summary.get("layer_stride", 1)),
        summary.get("tta_reset"),
    )


def write_csv(rows: List[Dict[str, Any]], out_path: Path, fields: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


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


def plot_grid_heatmap(
    runs: List[Run],
    *,
    out_path: Path,
    dataset: str,
    model: str,
    eval_mode: str,
    metric_after: str,
    metric_before: Optional[str],
    title: str,
    percent: bool = True,
) -> None:
    subset = [r for r in runs if r.summary.get("dataset") == dataset and r.summary.get("model") == model and r.summary.get("eval_mode") == eval_mode]
    if not subset:
        return

    steps_list = sorted({int(r.summary.get("steps", 0)) for r in subset})
    lr_list = sorted({float(r.summary.get("lr", 0.0)) for r in subset})

    mat = np.full((len(steps_list), len(lr_list)), np.nan, dtype=float)
    before_mat = np.full_like(mat, np.nan)
    for r in subset:
        s = r.summary
        i = steps_list.index(int(s.get("steps", 0)))
        j = lr_list.index(float(s.get("lr", 0.0)))
        a = _safe_float(s.get(metric_after))
        b = _safe_float(s.get(metric_before)) if metric_before else None
        if a is not None:
            mat[i, j] = a
        if b is not None:
            before_mat[i, j] = b

    vals = mat[~np.isnan(mat)]
    if vals.size == 0:
        return
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if abs(vmax - vmin) < 1e-8:
        vmin = max(0.0, vmin - 1e-3)
        vmax = min(1.0, vmax + 1e-3) if percent else vmax + 1e-3

    fig, ax = plt.subplots(figsize=(2.8 + 1.3 * len(lr_list), 2.4 + 1.0 * len(steps_list)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(lr_list)))
    ax.set_xticklabels([f"{lr:g}" for lr in lr_list])
    ax.set_yticks(range(len(steps_list)))
    ax.set_yticklabels([str(s) for s in steps_list])
    ax.set_xlabel("lr")
    ax.set_ylabel("steps")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            a = mat[i, j]
            if np.isnan(a):
                continue
            b = before_mat[i, j] if metric_before else np.nan
            text = _format_cell(float(a), float(b) if (metric_before and not np.isnan(b)) else None, percent=percent)
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color="white" if a < (vmin + vmax) / 2 else "black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate v8.3 results into tables + heatmaps.")
    p.add_argument("--root", default="v8.3/results_crt_main")
    p.add_argument("--out_dir", default="v8.3/summary_crt_main")
    p.add_argument("--filter_models", default=None, help="Comma-separated model basenames to include.")
    p.add_argument("--filter_datasets", default=None, help="Comma-separated datasets to include.")
    p.add_argument("--render_viz_top_n", type=int, default=1, help="Per (dataset,model,mode), render rich viz for top-N runs. 0 disables.")
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
        if not s or s.get("version") != "v8.3":
            continue
        if model_filter is not None and s.get("model") not in model_filter:
            continue
        if dataset_filter is not None and s.get("dataset") not in dataset_filter:
            continue
        runs.append(Run(path=p, summary=s))

    if not runs:
        raise SystemExit(f"No v8.3 results found under {root}")

    # One row per run.
    run_rows: List[Dict[str, Any]] = []
    for r in runs:
        s = r.summary
        row: Dict[str, Any] = {
            "json_path": str(r.path),
            "dataset": s.get("dataset"),
            "model": s.get("model"),
            "eval_mode": s.get("eval_mode"),
            "steps": s.get("steps"),
            "lr": s.get("lr"),
            "lr_schedule": s.get("lr_schedule"),
            "lr_min": s.get("lr_min"),
            "optimizer": s.get("optimizer"),
            "momentum": s.get("momentum"),
            "update_target": s.get("update_target"),
            "num_layers": s.get("num_layers"),
            "layer_stride": s.get("layer_stride"),
            "tta_reset": s.get("tta_reset"),
            "seq_total": s.get("seq_total"),
            "token_total": s.get("token_total"),
            "token_acc_before": s.get("token_acc_before"),
            "token_acc_after": s.get("token_acc_after"),
            "pass2_before": _pass_at(s, 2, which="before"),
            "pass2_after": _pass_at(s, 2, which="after"),
            "pass5_before": _pass_at(s, 5, which="before"),
            "pass5_after": _pass_at(s, 5, which="after"),
            "target_rank_avg_before": s.get("target_rank_avg_before"),
            "target_rank_avg_after": s.get("target_rank_avg_after"),
            "target_prob_avg_before": s.get("target_prob_avg_before"),
            "target_prob_avg_after": s.get("target_prob_avg_after"),
            "em_before": s.get("em_before"),
            "em_after": s.get("em_after"),
            "digit_acc_before": s.get("digit_acc_before"),
            "digit_acc_after": s.get("digit_acc_after"),
            "runtime_seconds": s.get("runtime_seconds"),
            "runtime_per_token": _runtime_per_token(s),
            "tokens_per_second": s.get("tokens_per_second"),
            "flipped_count": s.get("flipped_count"),
        }
        run_rows.append(row)

    write_csv(run_rows, out_dir / "summary_runs.csv", RUN_FIELDS)

    # Merge TF + AR into paper table.
    by_key: Dict[Tuple, Dict[str, Run]] = {}
    for r in runs:
        key = _config_key(r.summary)
        by_key.setdefault(key, {})
        by_key[key][str(r.summary.get("eval_mode"))] = r

    main_rows: List[Dict[str, Any]] = []
    for key, modes in by_key.items():
        tf = modes.get("tf")
        ar = modes.get("ar")
        base = (tf.summary if tf else (ar.summary if ar else {})) or {}

        row = {
            "dataset": base.get("dataset"),
            "model": base.get("model"),
            "steps": base.get("steps"),
            "lr": base.get("lr"),
            "lr_schedule": base.get("lr_schedule"),
            "lr_min": base.get("lr_min"),
            "optimizer": base.get("optimizer"),
            "momentum": base.get("momentum"),
            "update_target": base.get("update_target"),
            "num_layers": base.get("num_layers"),
            "layer_stride": base.get("layer_stride"),
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
        main_rows.append(row)

    main_rows.sort(key=lambda r: (r["dataset"], r["model"], float(r["lr"]), int(r["steps"])))
    write_csv(main_rows, out_dir / "table_main.csv", MAIN_FIELDS)

    # Grid heatmaps (steps x lr) per model.
    datasets = sorted({r.summary.get("dataset") for r in runs})
    models = sorted({r.summary.get("model") for r in runs})

    plots_dir = out_dir / "plots"
    for d in datasets:
        for m in models:
            # TF (token-level)
            plot_grid_heatmap(
                runs,
                out_path=plots_dir / f"{d}__{m}__tf__acc.png",
                dataset=d,
                model=m,
                eval_mode="tf",
                metric_after="token_acc_after",
                metric_before="token_acc_before",
                title=f"{d} | {m} | TF token acc",
                percent=True,
            )
            # AR (sequence-level)
            plot_grid_heatmap(
                runs,
                out_path=plots_dir / f"{d}__{m}__ar__acc.png",
                dataset=d,
                model=m,
                eval_mode="ar",
                metric_after="token_acc_after",
                metric_before="token_acc_before",
                title=f"{d} | {m} | AR token acc",
                percent=True,
            )
            plot_grid_heatmap(
                runs,
                out_path=plots_dir / f"{d}__{m}__ar__em.png",
                dataset=d,
                model=m,
                eval_mode="ar",
                metric_after="em_after",
                metric_before="em_before",
                title=f"{d} | {m} | AR EM",
                percent=True,
            )
            plot_grid_heatmap(
                runs,
                out_path=plots_dir / f"{d}__{m}__ar__digit.png",
                dataset=d,
                model=m,
                eval_mode="ar",
                metric_after="digit_acc_after",
                metric_before="digit_acc_before",
                title=f"{d} | {m} | AR digit acc",
                percent=True,
            )

    # Pass@k are stored as dicts; plot them separately to keep code explicit.
    for d in datasets:
        for m in models:
            for k in [2, 5]:
                subset = [r for r in runs if r.summary.get("dataset") == d and r.summary.get("model") == m and r.summary.get("eval_mode") == "tf"]
                if not subset:
                    continue
                # Inject temporary fields for plotting function.
                tmp_runs = []
                for r in subset:
                    s2 = dict(r.summary)
                    s2["pass_k_after"] = _pass_at(r.summary, k, which="after")
                    s2["pass_k_before"] = _pass_at(r.summary, k, which="before")
                    tmp_runs.append(Run(path=r.path, summary=s2))
                plot_grid_heatmap(
                    tmp_runs,
                    out_path=plots_dir / f"{d}__{m}__tf__pass{k}.png",
                    dataset=d,
                    model=m,
                    eval_mode="tf",
                    metric_after="pass_k_after",
                    metric_before="pass_k_before",
                    title=f"{d} | {m} | TF pass@{k}",
                    percent=True,
                )

    # Optional: render rich per-token viz for best runs per model/mode.
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
