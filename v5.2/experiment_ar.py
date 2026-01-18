"""v5.2: ANE test-time adaptation experiments (supports TF/AR).

本版本用于快速验证 1B 模型在不同可训练参数子集下的效果与效率。
默认使用 autoregressive (AR) evaluation，更贴近 test-time 生成；也支持 teacher forcing (TF)。

输出：每个实验一个 JSON（summary + per-sample 结果），并可选生成可视化。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple
from datasets import load_dataset
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from losses import AngularEntropyLoss
from numerical_utils import (
    build_id_to_pos,
    build_num_mask,
    compute_num_prob_rank,
    get_numerical_tokens,
    mask_logits_to_num,
)
from tta_engine import TTAEngine, backup_params, configure_trainable_params, restore_params

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise calculator. Output ONLY the numerical answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 12 and 34?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

46<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 567 and 890?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1457<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def set_seed(seed: int) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return f"<{token_id}>"


def parse_topk_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _cat_prompt_and_tokens(prompt_ids: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
    if not token_ids:
        return prompt_ids
    device = prompt_ids.device
    suffix = torch.tensor([token_ids], dtype=torch.long, device=device)
    return torch.cat([prompt_ids, suffix], dim=1)


def _forward_num_argmax(model, input_ids: torch.Tensor, num_mask: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits[0, -1, :]
    return torch.argmax(mask_logits_to_num(logits, num_mask))


def evaluate_one_config(
    *,
    config_name: str,
    model_name: str,
    model,
    tokenizer,
    data: List[Dict],
    num_mask: torch.Tensor,
    num_idx: List[int],
    num_idx_tensor: torch.Tensor,
    id_to_pos: torch.Tensor,
    steps: int,
    lr: float,
    optimizer: str,
    grad_clip: float,
    max_grad_norm: float,
    update_target: str,
    num_layers: str,
    layer_stride: int,
    eval_mode: str,
    topk_list: List[int],
    backup_on_cpu: bool,
    no_viz: bool,
    output_dir: str,
) -> Dict:
    start = time.time()

    # loss (ANE)
    loss_fn = AngularEntropyLoss(num_idx, model.get_input_embeddings(), use_float32=True, eps=1e-4, cache_embeddings=True).to(model.device)

    params, train_stats = configure_trainable_params(model, update_target, num_layers, layer_stride)

    tta_engine = TTAEngine(
        model=model,
        params=params,
        num_mask=num_mask,
        num_idx_tensor=num_idx_tensor,
        id_to_pos=id_to_pos,
        loss_fn=loss_fn,
        steps=steps,
        lr=lr,
        grad_clip=grad_clip,
        max_grad_norm=max_grad_norm,
        optimizer=optimizer,
        snapshot_steps=[0, steps],
        num_topk=10,
    )

    token_total = 0
    token_correct_before = 0
    token_correct_after = 0

    topk_correct_before = {k: 0 for k in topk_list}
    topk_correct_after = {k: 0 for k in topk_list}

    seq_total = len(data)
    seq_correct_before = 0
    seq_correct_after = 0

    flipped_cases = []
    results = []

    for item in tqdm(data, desc=f"{eval_mode}-{config_name}"):
        question = item["question"]
        answer = item["answer"]
        answer_str = str(answer)
        gt_tokens = tokenizer.encode(answer_str, add_special_tokens=False)

        prompt = PROMPT_TEMPLATE.format(question=question)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

        # AR: maintain separate prefixes for before/after.
        gen_before: List[int] = []
        gen_after: List[int] = []

        all_before = True
        all_after = True

        token_results = []

        for pos, target_idx in enumerate(gt_tokens):
            if eval_mode == "tf":
                # Teacher forcing: prefix uses ground-truth tokens.
                prefix_tokens = gt_tokens[:pos]
                input_ids = _cat_prompt_and_tokens(prompt_ids, prefix_tokens)

                backup = backup_params(params, to_cpu=backup_on_cpu) if params else []
                pred0, pred_after, metrics = tta_engine.run_token(input_ids, target_idx)
                if params:
                    restore_params(params, backup, from_cpu=backup_on_cpu)

                pred_before = pred0
                rank_before = int(metrics["target_rank"][0])
                rank_after = int(metrics["target_rank"][-1])
                prob_before = float(metrics["target_prob"][0])
                prob_after = float(metrics["target_prob"][-1])

            elif eval_mode == "ar":
                # Baseline AR (no adaptation)
                input_ids_before = _cat_prompt_and_tokens(prompt_ids, gen_before)
                model.eval()
                with torch.no_grad():
                    logits_before = model(input_ids_before).logits[0, -1, :]
                    pred_before = int(torch.argmax(mask_logits_to_num(logits_before, num_mask)).item())
                    prob_before, rank_before = compute_num_prob_rank(logits_before, target_idx, num_idx_tensor, id_to_pos)
                gen_before.append(pred_before)

                # TTA AR (adaptation, then generate)
                input_ids_after = _cat_prompt_and_tokens(prompt_ids, gen_after)
                backup = backup_params(params, to_cpu=backup_on_cpu) if params else []
                _, pred_after, metrics = tta_engine.run_token(input_ids_after, target_idx)
                if params:
                    restore_params(params, backup, from_cpu=backup_on_cpu)

                rank_after = int(metrics["target_rank"][-1])
                prob_after = float(metrics["target_prob"][-1])
                gen_after.append(pred_after)

            else:
                raise ValueError(f"Unknown eval_mode: {eval_mode}")

            correct_before = pred_before == target_idx
            correct_after = pred_after == target_idx

            token_total += 1
            token_correct_before += int(correct_before)
            token_correct_after += int(correct_after)

            for k in topk_list:
                topk_correct_before[k] += int(rank_before <= k)
                topk_correct_after[k] += int(rank_after <= k)

            all_before = all_before and bool(correct_before)
            all_after = all_after and bool(correct_after)

            if (not correct_before) and correct_after:
                flipped_cases.append(
                    {
                        "question": question,
                        "answer": answer,
                        "position": pos,
                        "target_token": decode_token(tokenizer, target_idx),
                        "pred_before": decode_token(tokenizer, pred_before),
                        "pred_after": decode_token(tokenizer, pred_after),
                    }
                )

            token_results.append(
                {
                    "position": pos,
                    "answer_len": len(gt_tokens),
                    "target_id": int(target_idx),
                    "target_token": decode_token(tokenizer, target_idx),
                    "pred_before_id": int(pred_before),
                    "pred_before": decode_token(tokenizer, pred_before),
                    "pred_after_id": int(pred_after),
                    "pred_after": decode_token(tokenizer, pred_after),
                    "correct_before": bool(correct_before),
                    "correct_after": bool(correct_after),
                    "rank_before": int(rank_before),
                    "rank_after": int(rank_after),
                    "prob_before": float(prob_before),
                    "prob_after": float(prob_after),
                    "metrics": metrics,
                }
            )

        seq_correct_before += int(all_before)
        seq_correct_after += int(all_after)

        results.append(
            {
                "question": question,
                "answer": answer,
                "eval_mode": eval_mode,
                "pred_before_str": tokenizer.decode(gen_before).strip() if eval_mode == "ar" else None,
                "pred_after_str": tokenizer.decode(gen_after).strip() if eval_mode == "ar" else None,
                "tokens": token_results,
            }
        )

    elapsed_min = (time.time() - start) / 60.0

    summary = {
        "method": config_name,
        "algo": "ane_tta",
        "eval_mode": eval_mode,
        "model": model_name,
        "steps": steps,
        "lr": lr,
        "update_target": update_target,
        "num_layers": num_layers,
        "layer_stride": layer_stride,
        "optimizer": optimizer,
        "token_total": token_total,
        "token_correct_before": token_correct_before,
        "token_correct_after": token_correct_after,
        "token_acc_before": token_correct_before / token_total if token_total else 0.0,
        "token_acc_after": token_correct_after / token_total if token_total else 0.0,
        "token_topk_correct_before": {str(k): int(topk_correct_before[k]) for k in topk_list},
        "token_topk_correct_after": {str(k): int(topk_correct_after[k]) for k in topk_list},
        "token_topk_acc_before": {str(k): topk_correct_before[k] / token_total for k in topk_list},
        "token_topk_acc_after": {str(k): topk_correct_after[k] / token_total for k in topk_list},
        "seq_total": seq_total,
        "seq_correct_before": seq_correct_before,
        "seq_correct_after": seq_correct_after,
        "seq_acc_before": seq_correct_before / seq_total if seq_total else 0.0,
        "seq_acc_after": seq_correct_after / seq_total if seq_total else 0.0,
        "flipped_count": len(flipped_cases),
        "elapsed_minutes": elapsed_min,
        "trainable": {
            "trainable_params": train_stats.trainable_params,
            "trainable_pct": train_stats.trainable_pct,
            "layer_count": train_stats.layer_count,
            "total_layers": train_stats.total_layers,
        },
    }

    output = {"summary": summary, "results": results, "flipped_cases": flipped_cases}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{config_name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if not no_viz:
        from visualization import visualize_all

        visualize_all(output, output_dir, config_name)

    return output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v5.2 ANE TTA experiments (TF/AR)")
    p.add_argument("--model", default="/home/zhangsy/llm_model/Llama-3.2-1B-Instruct")
    p.add_argument("--data_path", default="../data/addition_problems_dataset(1-50)(1).json")
    p.add_argument("--num_samples", type=int, default=20)

    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1e6)

    p.add_argument(
        "--update_target",
        default="mlp+ln",
        help="mlp|ln|mlp+ln|attn|attn+ln|all|all+lm_head (suffix +lm_head supported)",
    )
    p.add_argument("--num_layers", default="all")
    p.add_argument("--layer_stride", type=int, default=1)

    p.add_argument("--eval_mode", choices=["tf", "ar"], default="ar")
    p.add_argument("--topk_list", default="5")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--config_name", required=True)

    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", default="cuda:1")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--allow_prefix_space", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--empty_cache_each", action="store_true")
    p.add_argument("--backup_on_cpu", action="store_true")

    p.add_argument("--no_viz", action="store_true")

    return p


def main():
    args = build_parser().parse_args()

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    set_seed(args.seed)

    with open(args.data_path, "r") as f:
        data = json.load(f)
    data = data[: args.num_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device_map)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    vocab_size = model.get_input_embeddings().num_embeddings
    num_idx, _, _ = get_numerical_tokens(tokenizer, allow_prefix_space=args.allow_prefix_space, vocab_size=vocab_size)

    num_mask = build_num_mask(vocab_size, num_idx, model.device)
    num_idx_tensor = torch.tensor(num_idx, dtype=torch.long, device=model.device)
    id_to_pos = build_id_to_pos(num_idx, vocab_size, model.device)

    topk_list = parse_topk_list(args.topk_list)

    evaluate_one_config(
        config_name=args.config_name,
        model_name=os.path.basename(args.model),
        model=model,
        tokenizer=tokenizer,
        data=data,
        num_mask=num_mask,
        num_idx=num_idx,
        num_idx_tensor=num_idx_tensor,
        id_to_pos=id_to_pos,
        steps=args.steps,
        lr=args.lr,
        optimizer=args.optimizer,
        grad_clip=args.grad_clip,
        max_grad_norm=args.max_grad_norm,
        update_target=args.update_target,
        num_layers=args.num_layers,
        layer_stride=args.layer_stride,
        eval_mode=args.eval_mode,
        topk_list=topk_list,
        backup_on_cpu=args.backup_on_cpu,
        no_viz=args.no_viz,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
