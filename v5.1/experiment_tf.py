"""Teacher-forcing experiments for v5.1 ablations (3B, steps=5 by default).

本版本重点：
- rank/prob 统一在“数值子词表”上计算（更符合数值输出任务）；
- 支持 3 个 ablation：real-distance loss、mean-embedding one-shot、lm_head-only；
- 结果里记录 correct/total，并生成跨方法柱状图（见 aggregate.py）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from losses import AngularEntropyLoss, RealDistanceEntropyLoss, mean_embedding_decode
from numerical_utils import (
    build_id_to_pos,
    build_num_mask,
    compute_num_prob_rank,
    get_numerical_tokens,
    mask_logits_to_num,
    numerical_softmax,
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


def pad_list(values: List, target_len: int) -> List:
    if not values:
        return values
    if len(values) >= target_len:
        return values[:target_len]
    values.extend([values[-1]] * (target_len - len(values)))
    return values


def make_constant_metrics(
    baseline_logits: torch.Tensor,
    target_idx: int,
    steps: int,
    num_idx_tensor: torch.Tensor,
    id_to_pos: torch.Tensor,
    num_topk: int = 10,
) -> Dict:
    """用于 mean-embed 这种不做优化的方法：把 step 维度填满，便于统一可视化。"""

    prob0, rank0 = compute_num_prob_rank(baseline_logits, target_idx, num_idx_tensor, id_to_pos)
    num_probs = numerical_softmax(baseline_logits, num_idx_tensor)
    topk_probs, topk_pos = torch.topk(num_probs, min(num_topk, num_probs.shape[0]))
    topk = [{"idx": int(num_idx_tensor[p].item()), "prob": float(prob)} for prob, p in zip(topk_probs.tolist(), topk_pos.tolist())]

    metrics = {
        "loss": [0.0],
        "target_prob": [prob0],
        "target_rank": [rank0],
        "num_topk_probs": [topk_probs.tolist()],
        "num_topk_snapshots": [{"step": 0, "topk": topk}],
        "status": ["no_opt"],
    }

    total_len = steps + 1
    for k in ["loss", "target_prob", "target_rank", "status", "num_topk_probs"]:
        pad_list(metrics[k], total_len)
    if metrics["num_topk_snapshots"][-1]["step"] != steps:
        metrics["num_topk_snapshots"].append({"step": steps, "topk": topk})

    return metrics


def evaluate_method(args) -> Dict:
    start = time.time()

    with open(args.data_path, "r") as f:
        data = json.load(f)
    data = data[: args.num_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.dtype,
        device_map=args.device_map,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    vocab_size = model.get_input_embeddings().weight.shape[0]

    num_idx, num_tok, num_val = get_numerical_tokens(
        tokenizer,
        allow_prefix_space=args.allow_prefix_space,
        vocab_size=vocab_size,
    )
    if not num_idx:
        raise RuntimeError("No numerical tokens found")

    num_mask = build_num_mask(vocab_size, num_idx, model.device)
    num_idx_tensor = torch.tensor(num_idx, device=model.device, dtype=torch.long)
    id_to_pos = build_id_to_pos(num_idx, vocab_size, model.device)
    id_to_str = {idx: tok for idx, tok in zip(num_idx, num_tok)}

    # Loss / decode
    loss_fn = None
    if args.method == "ane_tta":
        loss_fn = AngularEntropyLoss(num_idx, model.get_input_embeddings(), use_float32=True, eps=1e-4, cache_embeddings=True).to(model.device)
    elif args.method == "real_tta":
        loss_fn = RealDistanceEntropyLoss(num_idx, num_val, normalize=True).to(model.device)
    elif args.method == "ane_tta_lm_head":
        loss_fn = AngularEntropyLoss(num_idx, model.get_input_embeddings(), use_float32=True, eps=1e-4, cache_embeddings=True).to(model.device)
    elif args.method == "mean_embed":
        loss_fn = None
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Trainable params
    params: List[torch.nn.Parameter] = []
    train_stats = None
    if args.method in ["ane_tta", "real_tta"]:
        params, train_stats = configure_trainable_params(model, args.update_target, args.num_layers, args.layer_stride)
    elif args.method == "ane_tta_lm_head":
        params, train_stats = configure_trainable_params(model, "lm_head", "all", 1)

    # TTA engine (only for gradient-based methods)
    tta_engine = None
    if args.method in ["ane_tta", "real_tta", "ane_tta_lm_head"]:
        tta_engine = TTAEngine(
            model=model,
            params=params,
            num_mask=num_mask,
            num_idx_tensor=num_idx_tensor,
            id_to_pos=id_to_pos,
            loss_fn=loss_fn,
            steps=args.steps,
            lr=args.lr,
            grad_clip=args.grad_clip,
            max_grad_norm=args.max_grad_norm,
            optimizer=args.optimizer,
            snapshot_steps=[0, args.steps],
            num_topk=10,
        )

    token_total = 0
    token_correct_before = 0
    token_correct_after = 0

    topk_list = parse_topk_list(args.topk_list)
    topk_correct_before = {k: 0 for k in topk_list}
    topk_correct_after = {k: 0 for k in topk_list}

    seq_total = len(data)
    seq_correct_before = 0
    seq_correct_after = 0

    results = []
    flipped_cases = []

    backup_on_cpu = args.backup_on_cpu

    for item in tqdm(data, desc=f"TF-{args.method}"):
        question = item["question"]
        answer = item["answer"]
        answer_str = str(answer)
        gt_tokens = tokenizer.encode(answer_str, add_special_tokens=False)

        prompt = PROMPT_TEMPLATE.format(question=question)
        input_encoding = tokenizer(prompt, return_tensors="pt").to(model.device)

        token_results = []
        all_before = True
        all_after = True

        for pos, target_idx in enumerate(gt_tokens):
            if pos == 0:
                input_ids = input_encoding["input_ids"]
            else:
                prefix = torch.tensor([gt_tokens[:pos]], device=model.device)
                input_ids = torch.cat([input_encoding["input_ids"], prefix], dim=1)

            if args.method == "mean_embed":
                model.eval()
                with torch.no_grad():
                    baseline_logits = model(input_ids).logits[0, -1, :]
                    pred_before = int(torch.argmax(mask_logits_to_num(baseline_logits, num_mask)).item())
                    prob_before, rank_before = compute_num_prob_rank(baseline_logits, target_idx, num_idx_tensor, id_to_pos)

                pred_after = mean_embedding_decode(baseline_logits, num_idx_tensor, model.get_input_embeddings())
                metrics = make_constant_metrics(baseline_logits, target_idx, args.steps, num_idx_tensor, id_to_pos, num_topk=10)
            else:
                backup = backup_params(params, to_cpu=backup_on_cpu) if params else []
                pred_before, pred_after, metrics = tta_engine.run_token(input_ids, target_idx)
                if params:
                    restore_params(params, backup, from_cpu=backup_on_cpu)

                prob_before = float(metrics["target_prob"][0])
                rank_before = int(metrics["target_rank"][0])

            prob_after = float(metrics["target_prob"][-1])
            rank_after = int(metrics["target_rank"][-1])

            correct_before = pred_before == target_idx
            correct_after = pred_after == target_idx

            token_total += 1
            token_correct_before += int(correct_before)
            token_correct_after += int(correct_after)

            for k in topk_list:
                topk_correct_before[k] += int(rank_before <= k)
                topk_correct_after[k] += int(rank_after <= k)

            all_before = all_before and correct_before
            all_after = all_after and correct_after

            if (not correct_before) and correct_after:
                flipped_cases.append(
                    {
                        "question": question,
                        "answer": answer,
                        "position": pos,
                        "target_token": decode_token(tokenizer, target_idx),
                        "pred_before": decode_token(tokenizer, pred_before),
                        "pred_after": decode_token(tokenizer, pred_after),
                        "metrics": metrics,
                    }
                )

            # add token strings to snapshots
            for snap in metrics.get("num_topk_snapshots", []):
                for row in snap.get("topk", []):
                    idx = row.get("idx")
                    if idx in id_to_str:
                        row["token"] = id_to_str[idx]

            token_results.append(
                {
                    "position": pos,
                    "answer_len": len(gt_tokens),
                    "target_id": target_idx,
                    "target_token": decode_token(tokenizer, target_idx),
                    "pred_before_id": pred_before,
                    "pred_before": decode_token(tokenizer, pred_before),
                    "pred_after_id": pred_after,
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

        results.append({"question": question, "answer": answer, "tokens": token_results})

        if args.empty_cache_each and torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_min = (time.time() - start) / 60.0

    summary = {
        "method": args.method,
        "model": os.path.basename(args.model),
        "steps": args.steps,
        "lr": args.lr,
        "update_target": args.update_target,
        "num_layers": args.num_layers,
        "layer_stride": args.layer_stride,
        "optimizer": args.optimizer,
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
        "trainable": None
        if train_stats is None
        else {
            "trainable_params": train_stats.trainable_params,
            "trainable_pct": train_stats.trainable_pct,
            "layer_count": train_stats.layer_count,
            "total_layers": train_stats.total_layers,
        },
    }

    output = {"summary": summary, "results": results, "flipped_cases": flipped_cases}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.exp_name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if not args.no_viz:
        from visualization import visualize_all

        visualize_all(output, args.output_dir, args.exp_name)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v5.1 TF experiments")
    p.add_argument("--method", required=True, choices=["ane_tta", "real_tta", "mean_embed", "ane_tta_lm_head"])
    p.add_argument("--model", default="/home/jinsk/Models/Llama-3.2-3B-Instruct")
    p.add_argument("--data_path", default="../data/addition_problems_dataset(1-50)(1).json")
    p.add_argument("--num_samples", type=int, default=50)

    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1e6)

    p.add_argument("--update_target", choices=["mlp", "ln", "mlp+ln", "attn", "attn+ln", "all"], default="mlp+ln")
    p.add_argument("--num_layers", default="all")
    p.add_argument("--layer_stride", type=int, default=1)

    p.add_argument("--allow_prefix_space", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--empty_cache_each", action="store_true")

    p.add_argument("--topk_list", default="5")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--no_viz", action="store_true")

    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--backup_on_cpu", action="store_true")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dtype == "bf16":
        args.dtype = torch.bfloat16
    elif args.dtype == "fp16":
        args.dtype = torch.float16
    else:
        args.dtype = torch.float32

    set_seed(args.seed)
    evaluate_method(args)


if __name__ == "__main__":
    main()
