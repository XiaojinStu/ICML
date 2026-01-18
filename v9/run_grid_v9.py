"""Run a grid of v9 experiments (single-process, single GPU).

Why this script:
- Avoid re-loading the model for each config (much faster).
- Save each config's JSON + per-config visualizations.
- Support resuming: skip configs whose JSON already exists.

This is intended for the "main experiments" grid (multiple models, multiple steps/lr).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dataset_loader import load_dataset
from experiment_v9 import evaluate, parse_topk_list, set_seed
from numerical_utils import build_id_to_pos, build_num_mask, get_numerical_tokens


@dataclass(frozen=True)
class GridSpec:
    steps_list: List[int]
    lr_list: List[float]
    lr_schedule: str
    lr_min: float
    update_target: str
    num_layers: str
    layer_stride: int
    tta_reset: str


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v9 grid runner (reuse loaded model)")

    p.add_argument(
        "--models",
        default=",".join(
            [
                "/home/jinsk/Models/Llama-3.2-1B-Instruct",
                "/home/jinsk/Models/Llama-3.2-3B-Instruct",
                "/home/jinsk/Models/Llama-3.1-8B-Instruct",
                "/home/jinsk/Models/Qwen3-4B-Instruct-2507",
            ]
        ),
        help="Comma-separated model paths.",
    )
    p.add_argument(
        "--datasets",
        default="addition50_v1,gsm8k_test500_v1,bigbench_arithmetic200_v1,bigbench_mixed_number_string300_seed42_v1,math401_all401_v1,nupa_test440_v1",
        help="Comma-separated dataset names (see datasets_final/README.md).",
    )
    p.add_argument("--dataset_paths", default="", help="Optional overrides: dataset=path,dataset=path,...")
    p.add_argument("--num_samples", type=int, default=0, help="<=0 means all (applies to all datasets)")
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--eval_modes", default="tf,ar", help="Comma-separated: tf,ar")
    p.add_argument("--eval_mode", choices=["tf", "ar"], default=None, help="Deprecated: use --eval_modes")
    p.add_argument("--steps_list", default="5,10,15")
    p.add_argument("--lr_list", default="0.001,0.0005")
    p.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--update_target", default="mlp+ln")
    p.add_argument("--num_layers", default="all")
    p.add_argument("--layer_stride", type=int, default=1)
    p.add_argument("--tta_reset", default="token", choices=["token", "sample", "none"])

    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1e6)

    p.add_argument("--topk_list", default="2,5")
    p.add_argument("--num_topk", type=int, default=10)
    p.add_argument("--tracked_topk", type=int, default=10)
    p.add_argument("--snapshot_stride", type=int, default=1)
    p.add_argument("--anchor_log", choices=["none", "flipped", "all"], default="flipped")
    p.add_argument("--anchor_trace_max", type=int, default=20)

    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--gpu_last_layers_k", type=int, default=0, help="If >0, place last-K transformer layers + embed/norm/head on GPU, others on CPU.")
    p.add_argument("--offload_folder", default="v9/offload", help="Used when offloading layers to CPU.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_prefix_space", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--backup_on_cpu", action="store_true")

    p.add_argument("--output_root", default="v9/results_main")
    p.add_argument("--no_viz", action="store_true")
    p.add_argument("--save_prompts", action="store_true")
    p.add_argument("--save_rendered_prompt", action="store_true")
    return p


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

    d = {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "lm_head": 0,
    }
    for i in range(n_layers):
        d[f"model.layers.{i}"] = 0 if i >= start else "cpu"
    return d


def _parse_dataset_paths(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in [x.strip() for x in str(s).split(",") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"--dataset_paths expects dataset=path pairs, got: {part}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _load_all_datasets(args) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    overrides = _parse_dataset_paths(getattr(args, "dataset_paths", ""))
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        out[name] = load_dataset(
            name=name,
            data_path=overrides.get(name),
            num_samples=int(args.num_samples),
            seed=args.seed,
            shuffle=bool(args.shuffle),
        )
    return out


def _build_num_tables(model, tokenizer, allow_prefix_space: bool) -> Tuple[List[int], List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab_size = model.get_input_embeddings().num_embeddings
    num_idx, num_tokens, _ = get_numerical_tokens(tokenizer, allow_prefix_space=allow_prefix_space, vocab_size=vocab_size)
    if not num_idx:
        raise RuntimeError("No numerical tokens found")
    num_mask = build_num_mask(vocab_size, num_idx, model.device)
    num_idx_tensor = torch.tensor(num_idx, dtype=torch.long, device=model.device)
    id_to_pos = build_id_to_pos(num_idx, vocab_size, model.device)
    return num_idx, num_tokens, num_mask, num_idx_tensor, id_to_pos


def _config_name(dataset: str, model_base: str, eval_mode: str, steps: int, lr: float, args) -> str:
    sched = str(args.lr_schedule)
    sched_suffix = ""
    if sched != "constant":
        sched_suffix = f"_{sched}"
        if sched == "cosine":
            sched_suffix += f"_lrmin{args.lr_min:g}"
    stride = f"_stride{int(args.layer_stride)}" if int(args.layer_stride) != 1 else ""
    return (
        f"{dataset}_{model_base}_{eval_mode}_steps{steps}_lr{lr:g}{sched_suffix}"
        f"_{args.optimizer}_{args.tta_reset}{stride}_{args.update_target}_{args.num_layers}"
    )


def run_one_model(args, model_path: str, datasets: Dict[str, List[Dict[str, Any]]], spec: GridSpec) -> None:
    model_base = os.path.basename(model_path.rstrip("/"))
    print(f"\n[v9-grid] Loading model: {model_base}")

    trust = bool(args.trust_remote_code) or ("qwen" in model_base.lower())

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust)

    device_map = args.device_map
    if int(args.gpu_last_layers_k) > 0:
        device_map = _build_lastk_device_map(model_path, trust_remote_code=trust, k_last=int(args.gpu_last_layers_k))

    offload_folder = str(args.offload_folder)
    Path(offload_folder).mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust,
        torch_dtype=_dtype_from_arg(args.dtype),
        device_map=device_map,
        offload_folder=offload_folder,
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

    for dataset_name, data in datasets.items():
        out_dir = Path(args.output_root) / dataset_name / model_base
        out_dir.mkdir(parents=True, exist_ok=True)

        eval_modes = [m.strip() for m in str(getattr(args, "eval_modes", "")).split(",") if m.strip()]
        if not eval_modes:
            eval_modes = [args.eval_mode] if args.eval_mode else ["tf"]

        for eval_mode in eval_modes:
            if eval_mode not in {"tf", "ar"}:
                raise ValueError(f"Unknown eval_mode: {eval_mode}")

            for steps in spec.steps_list:
                for lr in spec.lr_list:
                    cfg = _config_name(dataset_name, model_base, eval_mode, steps, lr, args)
                    out_path = out_dir / f"{cfg}.json"
                    if out_path.exists():
                        print(f"[v9-grid] Skip existing: {out_path}")
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
                        lr_schedule=spec.lr_schedule,
                        lr_min=spec.lr_min,
                        optimizer=args.optimizer,
                        momentum=args.momentum,
                        grad_clip=args.grad_clip,
                        max_grad_norm=args.max_grad_norm,
                        update_target=spec.update_target,
                        num_layers=spec.num_layers,
                        layer_stride=spec.layer_stride,
                        tta_reset=spec.tta_reset,
                        topk_list=topk_list,
                        snapshot_stride=args.snapshot_stride,
                        num_topk=args.num_topk,
                        tracked_topk=args.tracked_topk,
                        backup_on_cpu=bool(args.backup_on_cpu),
                        save_prompts=bool(args.save_prompts),
                        save_rendered_prompt=bool(getattr(args, "save_rendered_prompt", False)),
                        anchor_log=str(args.anchor_log),
                        anchor_trace_max=int(args.anchor_trace_max),
                        save_mode="compact",
                        keep_full_metrics_tokens=50,
                    )

                    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
                    print(f"[v9-grid] Wrote: {out_path}")

                    if not args.no_viz:
                        from visualization import visualize_all

                        visualize_all(output, str(out_dir), cfg)

    # cleanup
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    datasets = _load_all_datasets(args)

    spec = GridSpec(
        steps_list=_parse_int_list(args.steps_list),
        lr_list=_parse_float_list(args.lr_list),
        lr_schedule=str(args.lr_schedule),
        lr_min=float(args.lr_min),
        update_target=args.update_target,
        num_layers=args.num_layers,
        layer_stride=int(args.layer_stride),
        tta_reset=args.tta_reset,
    )

    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        run_one_model(args, m, datasets, spec)


if __name__ == "__main__":
    main()
