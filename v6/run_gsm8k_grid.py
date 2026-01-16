"""Run a GSM8K grid for v6 (LR sweep + steps sweep) and save results/viz.

This script is intentionally simple: it shells out to `experiment_v6.py`, then
reads the written JSON to pick the best learning rate per (model, update_target).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "gsm8k" / "gsm8k_test_500.json"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "gsm8k500"


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    path: str
    extra_args: List[str]


def _default_models() -> List[ModelSpec]:
    return [
        ModelSpec(tag="llama3.2-1b", path="/home/jinsk/Models/Llama-3.2-1B-Instruct", extra_args=[]),
        ModelSpec(tag="llama3.2-3b", path="/home/jinsk/Models/Llama-3.2-3B-Instruct", extra_args=[]),
        # 8B is heavier: enable gradient checkpointing by default.
        ModelSpec(tag="llama3.1-8b", path="/home/jinsk/Models/Llama-3.1-8B-Instruct", extra_args=["--gradient_checkpointing"]),
    ]


def _ensure_gsm8k_500(path: Path, seed: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = path.parent / "gsm8k_test_500.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "v6" / "prepare_gsm8k.py"),
        "--out_dir",
        str(path.parent),
        "--split",
        "test",
        "--num_samples",
        "500",
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, check=True)
    if path.exists():
        return
    if expected.exists() and expected != path:
        expected.replace(path)
        return
    raise RuntimeError(f"Expected dataset file at {path}, but it was not created.")


def _fmt_lr(lr: float) -> str:
    # 0.001 -> 1e-3, 0.0003 -> 3e-4
    s = f"{lr:.0e}"
    return s.replace("+", "")


def _run_one(
    *,
    model: ModelSpec,
    data_path: Path,
    num_samples: int,
    update_target: str,
    steps: int,
    lr: float,
    output_root: Path,
    config_prefix: str,
    dtype: str,
    device_map: str,
    optimizer: str,
    momentum: float,
    tta_reset: str,
    topk_list: str,
    backup_on_cpu: bool,
    no_viz: bool,
    force: bool,
) -> Tuple[Path, Dict]:
    lr_tag = _fmt_lr(lr)
    config_name = f"{config_prefix}_{model.tag}_{update_target}_all_steps{steps}_lr{lr_tag}"
    out_dir = output_root / model.tag / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{config_name}.json"
    out_log = out_dir / "run.log"

    if out_json.exists() and not force:
        with open(out_json, "r") as f:
            return out_json, json.load(f)

    print(f"[run] {config_name} (model={model.tag}, target={update_target}, steps={steps}, lr={lr})")
    sys.stdout.flush()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "v6" / "experiment_v6.py"),
        "--data_path",
        str(data_path),
        "--num_samples",
        str(num_samples),
        "--model",
        model.path,
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--optimizer",
        optimizer,
        "--momentum",
        str(momentum),
        "--update_target",
        update_target,
        "--num_layers",
        "all",
        "--eval_mode",
        "ar",
        "--tta_reset",
        tta_reset,
        "--topk_list",
        topk_list,
        "--dtype",
        dtype,
        "--device_map",
        device_map,
        "--output_dir",
        str(out_dir),
        "--config_name",
        config_name,
    ]

    if backup_on_cpu:
        cmd.append("--backup_on_cpu")
    if no_viz:
        cmd.append("--no_viz")

    cmd.extend(model.extra_args)

    with open(out_log, "w") as f:
        f.write("CMD:\n" + " ".join(cmd) + "\n\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)

    with open(out_json, "r") as f:
        return out_json, json.load(f)


def _pick_best_lr(results: List[Tuple[float, Dict]]) -> float:
    # Primary: string accuracy after; tie-break: token accuracy after; then smaller LR.
    def key(item):
        lr, j = item
        s = j.get("summary", {})
        return (
            float(s.get("str_acc_after", 0.0)),
            float(s.get("token_acc_after", 0.0)),
            -lr,
        )

    best_lr, _ = max(results, key=key)
    return float(best_lr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run v6 GSM8K (500) grid: LR sweep + steps sweep.")
    p.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))

    p.add_argument("--steps_list", default="5,10,30")
    p.add_argument("--targets", default="mlp,ln")

    p.add_argument("--lr_candidates", default="0.0001,0.0003,0.001,0.003,0.01")
    p.add_argument("--lr_search_steps", type=int, default=10)

    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--tta_reset", default="sample", choices=["sample", "token"])
    p.add_argument("--topk_list", default="5")
    p.add_argument(
        "--backup_on_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store base parameter backup on CPU (recommended).",
    )

    p.add_argument("--skip_lr_search", action="store_true")
    p.add_argument("--only_lr_search", action="store_true", help="Run LR search only, then exit.")
    p.add_argument("--only_grid", action="store_true", help="Run steps grid only (requires chosen_lrs.json).")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no_viz", action="store_true", help="Disable per-run visualization (useful for LR search).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.only_lr_search and args.only_grid:
        raise ValueError("--only_lr_search and --only_grid are mutually exclusive.")
    if args.only_grid:
        args.skip_lr_search = True

    data_path = Path(args.data_path)
    _ensure_gsm8k_500(data_path, seed=args.seed)

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    steps_list = [int(x.strip()) for x in args.steps_list.split(",") if x.strip()]
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    lr_candidates = [float(x.strip()) for x in args.lr_candidates.split(",") if x.strip()]

    models = _default_models()

    chosen_lrs: Dict[str, Dict[str, float]] = {}

    # Stage 1: LR search on the full 500-sample set (user requirement).
    if not args.skip_lr_search:
        for m in models:
            chosen_lrs[m.tag] = {}
            for tgt in targets:
                runs: List[Tuple[float, Dict]] = []
                for lr in lr_candidates:
                    _, j = _run_one(
                        model=m,
                        data_path=data_path,
                        num_samples=500,
                        update_target=tgt,
                        steps=args.lr_search_steps,
                        lr=lr,
                        output_root=results_root / "lr_search",
                        config_prefix="lrsearch_gsm8k500",
                        dtype=args.dtype,
                        device_map=args.device_map,
                        optimizer=args.optimizer,
                        momentum=args.momentum,
                        tta_reset=args.tta_reset,
                        topk_list=args.topk_list,
                        backup_on_cpu=args.backup_on_cpu,
                        no_viz=True,  # keep LR search light
                        force=args.force,
                    )
                    runs.append((lr, j))
                best = _pick_best_lr(runs)
                chosen_lrs[m.tag][tgt] = best

        with open(results_root / "chosen_lrs.json", "w") as f:
            json.dump(chosen_lrs, f, indent=2)
        print("Chosen LRs:", json.dumps(chosen_lrs, indent=2))
    else:
        path = results_root / "chosen_lrs.json"
        if not path.exists():
            raise RuntimeError("--skip_lr_search set, but chosen_lrs.json not found.")
        with open(path, "r") as f:
            chosen_lrs = json.load(f)

    if args.only_lr_search:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "v6" / "aggregate.py"),
                "--root",
                str(results_root),
                "--out_dir",
                str(results_root / "summary"),
            ],
            check=True,
        )
        return

    # Stage 2: full grid on 500 samples.
    for m in models:
        for tgt in targets:
            lr = float(chosen_lrs[m.tag][tgt])
            for steps in steps_list:
                _run_one(
                    model=m,
                    data_path=data_path,
                    num_samples=500,
                    update_target=tgt,
                    steps=steps,
                    lr=lr,
                    output_root=results_root / "grid",
                    config_prefix="gsm8k500",
                    dtype=args.dtype,
                    device_map=args.device_map,
                    optimizer=args.optimizer,
                    momentum=args.momentum,
                    tta_reset=args.tta_reset,
                    topk_list=args.topk_list,
                    backup_on_cpu=args.backup_on_cpu,
                    no_viz=args.no_viz,
                    force=args.force,
                )

    # Aggregate plots/tables.
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "v6" / "aggregate.py"),
            "--root",
            str(results_root),
            "--out_dir",
            str(results_root / "summary"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
