"""Aggregate v10 runs into per-run tables and mean/std across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 200,
            "savefig.dpi": 520,
            "savefig.bbox": "tight",
            "axes.linewidth": 1.0,
        }
    )


def _full_box(ax) -> None:
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)


def find_json(root: Path) -> List[Path]:
    paths = [p for p in root.rglob("*.json") if p.is_file()]
    paths = [p for p in paths if ".ipynb_checkpoints" not in str(p)]
    return sorted(paths)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _pass_at(summary: Dict[str, Any], k: int, which: str) -> Optional[float]:
    d = summary.get(f"pass@k_acc_{which}", {})
    if not isinstance(d, dict):
        return None
    v = d.get(str(int(k)))
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def token_acc_at(run: Dict[str, Any], step: int) -> float:
    totals = 0
    correct = 0
    for sample in run.get("results", []):
        for tok in sample.get("tokens", []):
            m = tok.get("metrics", {}) or {}
            ranks = m.get("target_rank", [])
            if not isinstance(ranks, list) or step < 0 or step >= len(ranks):
                continue
            totals += 1
            correct += int(int(ranks[step]) == 1)
    return float(correct / totals) if totals else 0.0


def _parse_tag_seed(path: Path, root: Path) -> Tuple[str, Optional[int]]:
    try:
        rel = path.relative_to(root)
        parts = list(rel.parts)
    except Exception:
        parts = list(path.parts)

    tag = parts[0] if parts else ""
    seed: Optional[int] = None
    for p in parts:
        m = re.fullmatch(r"seed(\d+)", str(p))
        if m:
            seed = int(m.group(1))
            break
    return tag, seed


def _mean_std(xs: List[float]) -> Tuple[float, Optional[float]]:
    if not xs:
        return 0.0, None
    if len(xs) == 1:
        return float(xs[0]), 0.0
    arr = np.array(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def write_csv(rows: List[Dict[str, Any]], out_path: Path, fields: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _plot_curves_mean_std(rows: List[Dict[str, Any]], out_dir: Path, checkpoints: List[int]) -> None:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = sorted({r["tag"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    models = sorted({r["model"] for r in rows})

    xs = np.array(checkpoints, dtype=int)

    def _group_models(kind: str) -> List[str]:
        if kind == "llama":
            return [m for m in models if "llama" in m.lower()]
        if kind == "qwen":
            return [m for m in models if "qwen" in m.lower()]
        return models

    for tag in tags:
        for kind in ["llama", "qwen"]:
            mlist = _group_models(kind)
            if not mlist:
                continue
            fig, axes = plt.subplots(len(datasets), 1, figsize=(9.8, 3.6 * len(datasets)), squeeze=False)
            for i, ds in enumerate(datasets):
                ax = axes[i][0]
                _full_box(ax)
                for model in mlist:
                    subset = [r for r in rows if r["tag"] == tag and r["dataset"] == ds and r["model"] == model and r["eval_mode"] == "tf"]
                    if not subset:
                        continue
                    # mean/std over seeds
                    ys_mean = []
                    ys_std = []
                    for s in checkpoints:
                        vals = [float(rr.get(f"token_acc@{s}", 0.0)) for rr in subset]
                        mu, sd = _mean_std(vals)
                        ys_mean.append(mu)
                        ys_std.append(sd or 0.0)
                    ys_mean = np.array(ys_mean, dtype=float)
                    ys_std = np.array(ys_std, dtype=float)
                    ax.plot(xs, ys_mean, marker="o", linewidth=2.0, markersize=4.0, label=model)
                    ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, alpha=0.18)

                ax.set_title(f"{tag} | {ds}", fontweight="bold")
                ax.set_xlabel("TTA steps", fontweight="bold")
                ax.set_ylabel("Token Acc (TF)", fontweight="bold")
                ax.set_xticks(xs)
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.25)
                if mlist:
                    ax.legend(loc="lower right", frameon=True)

            fig.tight_layout()
            fig.savefig(out_dir / f"token_acc_curves_{tag}_{kind}.png", facecolor="white")
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate v10 runs (per-run + mean/std across seeds)")
    p.add_argument("--root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoints", default="0,1,2,5,10,15,20,25,30")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    checkpoints = [int(x.strip()) for x in str(args.checkpoints).split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for path in find_json(root):
        run = load_json(path)
        s = run.get("summary", {}) or {}
        tag, seed = _parse_tag_seed(path, root)

        row: Dict[str, Any] = {
            "tag": tag,
            "seed": "" if seed is None else int(seed),
            "json_path": str(path),
            "dataset": str(s.get("dataset", "")),
            "model": str(s.get("model", "")),
            "eval_mode": str(s.get("eval_mode", "")),
            "steps_total": int(s.get("steps", 0) or 0),
            # NOTE: best_step can be 0, so avoid `or -1` which would corrupt it.
            "best_step": int(-1 if s.get("best_step", -1) is None else s.get("best_step", -1)),
            "lr": float(s.get("lr", 0.0) or 0.0),
            "lr_schedule": str(s.get("lr_schedule", "")),
            "lr_norm": str(s.get("lr_norm", "none")),
            "update_target": str(s.get("update_target", "")),
            "num_layers": str(s.get("num_layers", "")),
            "tta_reset": str(s.get("tta_reset", "")),
            "ane_metric": str(s.get("ane_metric", "")),
            "token_total": int(s.get("token_total", 0) or 0),
            "seq_total": int(s.get("seq_total", 0) or 0),
            "token_acc_before": float(s.get("token_acc_before", 0.0) or 0.0),
            "token_acc_after": float(s.get("token_acc_after", 0.0) or 0.0),
            "pass2_before": _pass_at(s, 2, "before"),
            "pass2_after": _pass_at(s, 2, "after"),
            "pass5_before": _pass_at(s, 5, "before"),
            "pass5_after": _pass_at(s, 5, "after"),
            "em_before": float(s.get("em_before", 0.0) or 0.0),
            "em_after": float(s.get("em_after", 0.0) or 0.0),
            "digit_acc_before": float(s.get("digit_acc_before", 0.0) or 0.0),
            "digit_acc_after": float(s.get("digit_acc_after", 0.0) or 0.0),
            "runtime_seconds": float(s.get("runtime_seconds", 0.0) or 0.0),
        }

        row["runtime_per_token"] = float(row["runtime_seconds"]) / max(1, int(row["token_total"]))
        row["delta_acc"] = float(row["token_acc_after"]) - float(row["token_acc_before"])
        row["delta_em"] = float(row["em_after"]) - float(row["em_before"])
        row["delta_digit_acc"] = float(row["digit_acc_after"]) - float(row["digit_acc_before"])
        row["delta_pass2"] = (
            (float(row["pass2_after"]) - float(row["pass2_before"])) if (row["pass2_before"] is not None and row["pass2_after"] is not None) else None
        )
        row["delta_pass5"] = (
            (float(row["pass5_after"]) - float(row["pass5_before"])) if (row["pass5_before"] is not None and row["pass5_after"] is not None) else None
        )

        for ck in checkpoints:
            row[f"token_acc@{ck}"] = token_acc_at(run, ck)

        rows.append(row)

    rows.sort(key=lambda r: (r["tag"], r["dataset"], r["model"], r["eval_mode"], r["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    RUN_FIELDS = [
        "tag",
        "seed",
        "dataset",
        "model",
        "eval_mode",
        "ane_metric",
        "steps_total",
        "best_step",
        "lr",
        "lr_schedule",
        "lr_norm",
        "update_target",
        "num_layers",
        "tta_reset",
        "token_total",
        "seq_total",
        "token_acc_before",
        *[f"token_acc@{ck}" for ck in checkpoints],
        "token_acc_after",
        "delta_acc",
        "pass2_before",
        "pass2_after",
        "delta_pass2",
        "pass5_before",
        "pass5_after",
        "delta_pass5",
        "em_before",
        "em_after",
        "delta_em",
        "digit_acc_before",
        "digit_acc_after",
        "delta_digit_acc",
        "runtime_per_token",
        "json_path",
    ]
    write_csv(rows, out_dir / "table_runs_v10.csv", RUN_FIELDS)

    # mean/std across seeds (group by config, exclude seed/json_path/runtime_seconds)
    def _group_key(r: Dict[str, Any]) -> Tuple:
        return (
            r["tag"],
            r["dataset"],
            r["model"],
            r["eval_mode"],
            r["ane_metric"],
            r["steps_total"],
            r["lr"],
            r["lr_schedule"],
            r["lr_norm"],
            r["update_target"],
            r["num_layers"],
            r["tta_reset"],
        )

    grouped: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(_group_key(r), []).append(r)

    mean_rows: List[Dict[str, Any]] = []
    metrics = [
        "token_acc_before",
        "token_acc_after",
        "delta_acc",
        "pass2_before",
        "pass2_after",
        "delta_pass2",
        "pass5_before",
        "pass5_after",
        "delta_pass5",
        "em_before",
        "em_after",
        "delta_em",
        "digit_acc_before",
        "digit_acc_after",
        "delta_digit_acc",
        "runtime_per_token",
        *[f"token_acc@{ck}" for ck in checkpoints],
    ]

    for key, items in grouped.items():
        base = items[0]
        out: Dict[str, Any] = {k: base[k] for k in ["tag", "dataset", "model", "eval_mode", "ane_metric", "steps_total", "lr", "lr_schedule", "lr_norm", "update_target", "num_layers", "tta_reset"]}
        out["n"] = len(items)
        out["seeds"] = ",".join(str(x.get("seed", "")) for x in items if x.get("seed", "") != "")

        for m in metrics:
            vals = []
            for it in items:
                v = it.get(m, None)
                if v is None or v == "":
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            mu, sd = _mean_std(vals)
            out[f"{m}_mean"] = mu
            out[f"{m}_std"] = sd if sd is not None else ""

        mean_rows.append(out)

    mean_rows.sort(key=lambda r: (r["tag"], r["dataset"], r["model"], r["eval_mode"], float(r.get("lr", 0.0))))

    MEAN_FIELDS = [
        "tag",
        "dataset",
        "model",
        "eval_mode",
        "ane_metric",
        "steps_total",
        "lr",
        "lr_schedule",
        "lr_norm",
        "update_target",
        "num_layers",
        "tta_reset",
        "n",
        "seeds",
    ] + [f"{m}_{suf}" for m in metrics for suf in ["mean", "std"]]
    write_csv(mean_rows, out_dir / "table_mean_std_v10.csv", MEAN_FIELDS)

    _plot_curves_mean_std(rows, out_dir / "plots", checkpoints)

    print(f"[v10] Wrote: {out_dir/'table_runs_v10.csv'}")
    print(f"[v10] Wrote: {out_dir/'table_mean_std_v10.csv'}")
    print(f"[v10] Plots: {out_dir/'plots'}")


if __name__ == "__main__":
    main()
