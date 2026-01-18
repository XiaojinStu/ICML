"""CLI entry for ANE-TTA v5."""

from __future__ import annotations

import argparse
import torch

from experiment import run_experiment, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="ANE-TTA v5")

    parser.add_argument("--model", default="/home/zhangsy/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B-Instruct") 
    parser.add_argument("--data_path", default="../data/addition_problems_dataset(1-50)(1).json")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--output_dir", default="results_ane")
    parser.add_argument("--no_viz", action="store_true")

    parser.add_argument("--update_target", default="mlp+ln", choices=["mlp", "ln", "mlp+ln", "attn", "attn+ln", "all"])
    parser.add_argument("--num_layers", default="all")
    parser.add_argument("--layer_stride", type=int, default=1)

    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1e6)

    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_rank", type=int, default=1)
    parser.add_argument("--early_stop_prob", type=float, default=None)
    parser.add_argument("--patience", type=int, default=0)

    parser.add_argument("--fast_eval", action="store_true")
    parser.add_argument("--skip_if_correct", action="store_true")
    parser.add_argument("--reset_mode", choices=["per_token", "per_sample", "none"], default="per_token")

    parser.add_argument("--tracked_topk", type=int, default=10)
    parser.add_argument("--num_topk", type=int, default=10)
    parser.add_argument("--snapshot_steps", default="auto")

    parser.add_argument("--loss_float32", action="store_true", default=True)
    parser.add_argument("--no_loss_float32", action="store_false", dest="loss_float32")
    parser.add_argument("--loss_eps", type=float, default=1e-4)
    parser.add_argument("--no_embed_cache", action="store_true")

    parser.add_argument("--allow_prefix_space", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--empty_cache_each", action="store_true")

    parser.add_argument("--eval_ar", action="store_true", default=True)
    parser.add_argument("--no_eval_ar", action="store_false", dest="eval_ar")
    parser.add_argument("--eval_ar_tta", action="store_true", default=True)
    parser.add_argument("--no_eval_ar_tta", action="store_false", dest="eval_ar_tta")

    parser.add_argument("--topk_list", default="1,5,10")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--backup_on_cpu", action="store_true", default=None)
    parser.add_argument("--no_backup_on_cpu", action="store_false", dest="backup_on_cpu")

    return parser.parse_args()


def parse_snapshot_steps(value: str, steps: int) -> list[int]:
    if value == "auto":
        mid = max(1, steps // 2)
        return [0, mid, steps]
    items = [int(x.strip()) for x in value.split(",") if x.strip()]
    if 0 not in items:
        items.insert(0, 0)
    if steps not in items:
        items.append(steps)
    return sorted(set(items))


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.dtype == "bf16":
        args.dtype = torch.bfloat16
    elif args.dtype == "fp16":
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    args.snapshot_steps = parse_snapshot_steps(args.snapshot_steps, args.steps)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_experiment(args)


if __name__ == "__main__":
    main()
