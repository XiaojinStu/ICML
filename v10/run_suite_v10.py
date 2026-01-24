#!/usr/bin/env python3
"""Run a per-dataset-config suite without reloading the model per dataset.

This is a v10 wrapper around v9 core code:
- Loads a model once
- Runs multiple datasets with per-dataset hyperparams (lr/schedule/norm)
- Writes v9-format JSON outputs + per-run visualizations
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "v9"))

from dataset_loader import load_standard_json  # noqa: E402
from experiment_v9 import evaluate, parse_topk_list, set_seed  # noqa: E402
from numerical_utils import build_id_to_pos, build_num_mask, get_numerical_tokens  # noqa: E402


def _dtype_from_arg(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _build_lastk_device_map(model_path: str, *, trust_remote_code: bool, k_last: int) -> dict:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    if n_layers <= 0:
        raise ValueError("Could not infer num_hidden_layers from config")
    k = max(0, min(int(k_last), n_layers))
    start = n_layers - k

    d = {"model.embed_tokens": 0, "model.norm": 0, "lm_head": 0}
    for i in range(n_layers):
        d[f"model.layers.{i}"] = 0 if i >= start else "cpu"
    return d


def _build_num_tables(model, tokenizer, allow_prefix_space: bool) -> Tuple[List[int], List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab_size = model.get_input_embeddings().num_embeddings
    num_idx, num_tokens, _ = get_numerical_tokens(tokenizer, allow_prefix_space=allow_prefix_space, vocab_size=vocab_size)
    if not num_idx:
        raise RuntimeError("No numerical tokens found")
    num_mask = build_num_mask(vocab_size, num_idx, model.device)
    num_idx_tensor = torch.tensor(num_idx, dtype=torch.long, device=model.device)
    id_to_pos = build_id_to_pos(num_idx, vocab_size, model.device)
    return num_idx, num_tokens, num_mask, num_idx_tensor, id_to_pos


def _config_name(
    dataset: str,
    model_base: str,
    eval_mode: str,
    *,
    steps: int,
    lr: float,
    lr_schedule: str,
    lr_min: float,
    lr_norm: str,
    ane_metric: str,
    optimizer: str,
    tta_reset: str,
    layer_stride: int,
    update_target: str,
    num_layers: str,
) -> str:
    sched_suffix = ""
    if str(lr_schedule) != "constant":
        sched_suffix = f"_{lr_schedule}"
        if str(lr_schedule) == "cosine":
            sched_suffix += f"_lrmin{lr_min:g}"
    lrnorm_suffix = "" if str(lr_norm) == "none" else f"_{lr_norm}"
    anem_suffix = "" if str(ane_metric) == "angle" else f"_ane{ane_metric}"
    stride = f"_stride{int(layer_stride)}" if int(layer_stride) != 1 else ""
    return (
        f"{dataset}_{model_base}_{eval_mode}_steps{int(steps)}_lr{lr:g}{sched_suffix}{lrnorm_suffix}{anem_suffix}"
        f"_{optimizer}_{tta_reset}{stride}_{update_target}_{num_layers}"
    )


def _load_suite(path: str) -> Dict[str, Any]:
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict) or "datasets" not in obj:
        raise ValueError("Suite must be a JSON object with a `datasets` field")
    if not isinstance(obj["datasets"], list) or not obj["datasets"]:
        raise ValueError("Suite `datasets` must be a non-empty list")
    return obj


def _validate_dataset_entry(d: Dict[str, Any]) -> None:
    for k in ["name", "path", "lr"]:
        if k not in d:
            raise ValueError(f"Dataset entry missing `{k}`: {d}")


def _safe_tag(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s.strip("_") or "run"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v10 suite runner (per-dataset configs, reuse loaded model)")
    p.add_argument("--models", required=True, help="Comma-separated model paths.")
    p.add_argument("--suite", required=True, help="Path to suite JSON (datasets + per-dataset hyperparams).")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=0, help="<=0 means all (applies to all datasets)")
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--eval_modes", default="tf", help="Comma-separated: tf,ar")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--ane_metric", choices=["angle", "cosine"], default="angle")

    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1e6)

    p.add_argument("--update_target", default="mlp+ln")
    p.add_argument("--num_layers", default="all")
    p.add_argument("--layer_stride", type=int, default=1)
    p.add_argument("--tta_reset", default="token", choices=["token", "sample", "none"])

    p.add_argument("--topk_list", default="2,5")
    p.add_argument("--num_topk", type=int, default=10)
    p.add_argument("--tracked_topk", type=int, default=10)
    p.add_argument("--snapshot_stride", type=int, default=1)
    p.add_argument("--snapshot_steps", default="", help="Comma-separated explicit snapshot steps (overrides stride).")
    p.add_argument("--anchor_log", choices=["none", "flipped", "all"], default="flipped")
    p.add_argument("--anchor_trace_max", type=int, default=20)

    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--gpu_last_layers_k", type=int, default=0)
    p.add_argument("--offload_folder", default="v9/offload")

    p.add_argument("--allow_prefix_space", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--backup_on_cpu", action="store_true")

    p.add_argument("--output_root", required=True)
    p.add_argument("--no_viz", action="store_true")
    p.add_argument("--save_prompts", action="store_true")
    p.add_argument("--save_rendered_prompt", action="store_true")
    return p


def run_one_model(*, args, model_path: str, suite: Dict[str, Any]) -> None:
    model_path = str(model_path)
    model_base = os.path.basename(model_path.rstrip("/"))
    print(f"[v10-suite] Loading model: {model_base}")

    dtype = _dtype_from_arg(args.dtype)
    device_map = args.device_map
    if int(args.gpu_last_layers_k) > 0:
        device_map = _build_lastk_device_map(model_path, trust_remote_code=bool(args.trust_remote_code), k_last=int(args.gpu_last_layers_k))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=bool(args.trust_remote_code), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=device_map,
        offload_folder=str(args.offload_folder),
        low_cpu_mem_usage=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    num_idx, num_tokens, num_mask, num_idx_tensor, id_to_pos = _build_num_tables(model, tokenizer, allow_prefix_space=args.allow_prefix_space)
    topk_list = parse_topk_list(args.topk_list)
    snapshot_steps = None
    if str(args.snapshot_steps).strip():
        snapshot_steps = sorted({int(s) for s in str(args.snapshot_steps).split(",") if s.strip()})

    eval_modes = [m.strip() for m in str(args.eval_modes).split(",") if m.strip()]
    if not eval_modes:
        eval_modes = ["tf"]

    for entry in suite["datasets"]:
        if not isinstance(entry, dict):
            continue
        _validate_dataset_entry(entry)

        dataset_name = str(entry["name"])
        data_path = str(entry["path"])
        lr = float(entry["lr"])
        lr_schedule = str(entry.get("lr_schedule", "constant"))
        lr_min = float(entry.get("lr_min", 1e-4))
        lr_norm = str(entry.get("lr_norm", "none"))
        lr_norm_eps = float(entry.get("lr_norm_eps", 1e-6))
        lr_norm_min = float(entry.get("lr_norm_min", 1e-5))
        lr_norm_max = float(entry.get("lr_norm_max", 0.05))
        steps = int(entry.get("steps", int(args.steps)))

        data = load_standard_json(
            data_path=data_path,
            num_samples=int(args.num_samples),
            seed=int(args.seed),
            shuffle=bool(args.shuffle),
        )

        out_dir = Path(args.output_root) / dataset_name / model_base
        out_dir.mkdir(parents=True, exist_ok=True)

        for eval_mode in eval_modes:
            if eval_mode not in {"tf", "ar"}:
                raise ValueError(f"Unknown eval_mode: {eval_mode}")

            cfg = _config_name(
                dataset_name,
                model_base,
                eval_mode,
                steps=steps,
                lr=lr,
                lr_schedule=lr_schedule,
                lr_min=lr_min,
                lr_norm=lr_norm,
                ane_metric=str(args.ane_metric),
                optimizer=str(args.optimizer),
                tta_reset=str(args.tta_reset),
                layer_stride=int(args.layer_stride),
                update_target=str(args.update_target),
                num_layers=str(args.num_layers),
            )
            out_path = out_dir / f"{cfg}.json"
            if out_path.exists():
                print(f"[v10-suite] Skip existing: {out_path}")
                continue

            output = evaluate(
                dataset=dataset_name,
                eval_mode=eval_mode,
                model_name=model_base,
                model=model,
                tokenizer=tokenizer,
                data=data,
                num_mask=num_mask,
                num_idx=num_idx,
                num_idx_tensor=num_idx_tensor,
                num_tokens=num_tokens,
                id_to_pos=id_to_pos,
                steps=steps,
                lr=lr,
                lr_schedule=lr_schedule,
                lr_min=lr_min,
                lr_norm=lr_norm,
                lr_norm_eps=lr_norm_eps,
                lr_norm_min=lr_norm_min,
                lr_norm_max=lr_norm_max,
                optimizer=str(args.optimizer),
                momentum=float(args.momentum),
                grad_clip=float(args.grad_clip),
                max_grad_norm=float(args.max_grad_norm),
                update_target=str(args.update_target),
                num_layers=str(args.num_layers),
                layer_stride=int(args.layer_stride),
                tta_reset=str(args.tta_reset),
                topk_list=topk_list,
                snapshot_stride=int(args.snapshot_stride),
                snapshot_steps=snapshot_steps,
                num_topk=int(args.num_topk),
                tracked_topk=int(args.tracked_topk),
                backup_on_cpu=bool(args.backup_on_cpu),
                save_prompts=bool(args.save_prompts),
                save_rendered_prompt=bool(args.save_rendered_prompt),
                anchor_log=str(args.anchor_log),
                anchor_trace_max=int(args.anchor_trace_max),
                ane_metric=str(args.ane_metric),
                save_mode="compact",
                keep_full_metrics_tokens=50,
            )

            out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
            print(f"[v10-suite] Wrote: {out_path}")

            if not args.no_viz:
                from visualization import visualize_all  # type: ignore

                visualize_all(output, str(out_dir), cfg)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))

    suite = _load_suite(args.suite)
    args.output_root = str(Path(args.output_root))
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    for m in [x.strip() for x in str(args.models).split(",") if x.strip()]:
        run_one_model(args=args, model_path=m, suite=suite)


if __name__ == "__main__":
    main()
