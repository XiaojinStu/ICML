"""Batch runner for ANE-TTA experiments."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch

from experiment import run_experiment, set_seed


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def merge_config(common: dict, override: dict) -> dict:
    merged = dict(common)
    merged.update(override)
    return merged


def to_namespace(cfg: dict) -> SimpleNamespace:
    return SimpleNamespace(**cfg)


def parse_snapshot_steps(value, steps: int) -> list[int]:
    if isinstance(value, list):
        return value
    if value == "auto":
        mid = max(1, steps // 2)
        return [0, mid, steps]
    items = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if 0 not in items:
        items.insert(0, 0)
    if steps not in items:
        items.append(steps)
    return sorted(set(items))


def normalize_config(cfg: dict) -> dict:
    cfg = dict(cfg)
    if isinstance(cfg.get("dtype"), str):
        if cfg["dtype"] == "bf16":
            cfg["dtype"] = torch.bfloat16
        elif cfg["dtype"] == "fp16":
            cfg["dtype"] = torch.float16
        else:
            cfg["dtype"] = torch.float32
    cfg["snapshot_steps"] = parse_snapshot_steps(cfg.get("snapshot_steps", "auto"), cfg.get("steps", 0))
    cfg.setdefault("backup_on_cpu", None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="ANE-TTA batch runner")
    parser.add_argument("--config", required=True, help="Path to experiments json")
    parser.add_argument("--output_root", default="results_batch", help="Root output directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_viz", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    common = config.get("common", {})
    experiments = config.get("experiments", [])

    if not experiments:
        raise ValueError("No experiments found in config")

    os.makedirs(args.output_root, exist_ok=True)

    for exp in experiments:
        exp_cfg = merge_config(common, exp)
        exp_cfg = normalize_config(exp_cfg)
        exp_name = exp_cfg["exp_name"]
        exp_dir = os.path.join(args.output_root, exp_cfg.get("output_dir", "results"))
        exp_cfg["output_dir"] = exp_dir
        exp_cfg["no_viz"] = exp_cfg.get("no_viz", False) or args.no_viz

        output_path = os.path.join(exp_dir, f"{exp_name}.json")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"[skip] {exp_name} (exists)")
            continue

        print(f"[run] {exp_name}")
        seed = exp_cfg.get("seed", None)
        if seed is not None:
            set_seed(seed)
        run_experiment(to_namespace(exp_cfg))


if __name__ == "__main__":
    main()
