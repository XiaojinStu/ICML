"""v7 main experiment runner (Greedy vs Greedy+ANE).

Supported datasets (for now):
- addition_50 (local JSON)
- gsm8k (local JSON)

Supported eval modes:
- tf: teacher forcing (gold prefix)
- ar: autoregressive (generated prefix)

Baseline is always greedy decoding on the numerical sub-vocab (step=0).
Method is Greedy+ANE: minimize ANE for `steps` and decode greedily.
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

from dataset_loader import load_dataset
from losses import AngularEntropyLoss
from metrics import digit_accuracy, encode_int_answer, normalize_int_answer
from numerical_utils import build_id_to_pos, build_num_mask, compute_num_prob_rank, get_numerical_tokens, mask_logits_to_num
from prompts import build_prompt
from tta_engine import TTAEngine, backup_params, configure_trainable_params, restore_params


def set_seed(seed: int) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_topk_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _cat_prompt_and_tokens(prompt_ids: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
    if not token_ids:
        return prompt_ids
    suffix = torch.tensor([token_ids], dtype=torch.long, device=prompt_ids.device)
    return torch.cat([prompt_ids, suffix], dim=1)


def _decode_tokens(tokenizer, token_ids: List[int]) -> List[str]:
    return [tokenizer.decode([i]) for i in token_ids]


def evaluate(
    *,
    dataset: str,
    eval_mode: str,
    model_name: str,
    model,
    tokenizer,
    data: List[Dict],
    num_mask: torch.Tensor,
    num_idx: List[int],
    num_idx_tensor: torch.Tensor,
    num_tokens: List[str],
    id_to_pos: torch.Tensor,
    steps: int,
    lr: float,
    optimizer: str,
    momentum: float,
    grad_clip: float,
    max_grad_norm: float,
    update_target: str,
    num_layers: str,
    layer_stride: int,
    tta_reset: str,
    topk_list: List[int],
    snapshot_stride: int,
    num_topk: int,
    tracked_topk: int,
    backup_on_cpu: bool,
    save_prompts: bool,
) -> Dict:
    start = time.time()

    loss_fn = AngularEntropyLoss(num_idx, model.get_input_embeddings(), use_float32=True, eps=1e-4, cache_embeddings=True).to(model.device)

    params, train_stats = configure_trainable_params(model, update_target, num_layers, layer_stride)
    base_backup = backup_params(params, to_cpu=backup_on_cpu) if params else []

    engine = TTAEngine(
        model=model,
        tokenizer=tokenizer,
        params=params,
        num_mask=num_mask,
        num_idx_tensor=num_idx_tensor,
        num_tokens=num_tokens,
        id_to_pos=id_to_pos,
        loss_fn=loss_fn,
        steps=steps,
        lr=lr,
        optimizer=optimizer,
        momentum=momentum,
        grad_clip=grad_clip,
        max_grad_norm=max_grad_norm,
        num_topk=num_topk,
        tracked_topk=tracked_topk,
        snapshot_stride=snapshot_stride,
    )

    token_total = 0
    token_correct_before = 0
    token_correct_after = 0
    topk_correct_before = {k: 0 for k in topk_list}
    topk_correct_after = {k: 0 for k in topk_list}
    rank_sum_before = 0.0
    rank_sum_after = 0.0
    prob_sum_before = 0.0
    prob_sum_after = 0.0

    seq_total = 0
    seq_em_before = 0
    seq_em_after = 0
    digit_sum_before = 0.0
    digit_sum_after = 0.0

    flipped_cases = []
    results = []

    for item in tqdm(data, desc=f"{dataset}-{eval_mode}"):
        q = item["question"]
        gold = normalize_int_answer(item["answer"])

        prompt = build_prompt(dataset, q)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

        gt_ids = encode_int_answer(tokenizer, gold)
        if not gt_ids:
            continue

        # Evaluate only positions whose gold token is in the numerical sub-vocab.
        numeric_positions = [pos for pos, tid in enumerate(gt_ids) if int(id_to_pos[tid].item()) >= 0]
        if not numeric_positions:
            continue

        if params and tta_reset in {"sample"}:
            restore_params(params, base_backup, from_cpu=backup_on_cpu)

        pred_before_by_pos: Dict[int, int] = {}
        pred_after_by_pos: Dict[int, int] = {}
        token_results = []

        # AR prefixes
        gen_before: List[int] = []
        gen_after: List[int] = []

        for pos, target_idx in enumerate(gt_ids):
            # Non-numeric tokens are kept as gold (mainly '-' sign).
            if int(id_to_pos[target_idx].item()) < 0:
                if eval_mode == "ar":
                    gen_before.append(int(target_idx))
                    gen_after.append(int(target_idx))
                continue

            if eval_mode == "tf":
                prefix_tokens = gt_ids[:pos]
                input_ids = _cat_prompt_and_tokens(prompt_ids, prefix_tokens)

                if params and tta_reset == "token":
                    restore_params(params, base_backup, from_cpu=backup_on_cpu)

                pred_before, pred_after, metrics = engine.run_token(input_ids, int(target_idx))

                if params and tta_reset == "token":
                    restore_params(params, base_backup, from_cpu=backup_on_cpu)

                rank_before = int(metrics["target_rank"][0])
                rank_after = int(metrics["target_rank"][-1])
                prob_before = float(metrics["target_prob"][0])
                prob_after = float(metrics["target_prob"][-1])

            elif eval_mode == "ar":
                # Baseline greedy (no adaptation)
                model.eval()
                with torch.no_grad():
                    logits_before = model(_cat_prompt_and_tokens(prompt_ids, gen_before)).logits[0, -1, :]
                    pred_before = int(torch.argmax(mask_logits_to_num(logits_before, num_mask)).item())
                    prob_before, rank_before = compute_num_prob_rank(logits_before, int(target_idx), num_idx_tensor, id_to_pos)
                gen_before.append(pred_before)

                # Greedy+ANE (adaptation then greedy decode)
                if params and tta_reset == "token":
                    restore_params(params, base_backup, from_cpu=backup_on_cpu)
                pred0, pred_after, metrics = engine.run_token(_cat_prompt_and_tokens(prompt_ids, gen_after), int(target_idx))
                if params and tta_reset == "token":
                    restore_params(params, base_backup, from_cpu=backup_on_cpu)

                rank_after = int(metrics["target_rank"][-1])
                prob_after = float(metrics["target_prob"][-1])
                gen_after.append(int(pred_after))

            else:
                raise ValueError(f"Unknown eval_mode: {eval_mode}")

            pred_before_by_pos[pos] = int(pred_before)
            pred_after_by_pos[pos] = int(pred_after)

            correct_before = int(pred_before) == int(target_idx)
            correct_after = int(pred_after) == int(target_idx)

            token_total += 1
            token_correct_before += int(correct_before)
            token_correct_after += int(correct_after)
            rank_sum_before += float(rank_before)
            rank_sum_after += float(rank_after)
            prob_sum_before += float(prob_before)
            prob_sum_after += float(prob_after)

            for k in topk_list:
                topk_correct_before[k] += int(rank_before <= k)
                topk_correct_after[k] += int(rank_after <= k)

            if (not correct_before) and correct_after:
                flipped_cases.append(
                    {
                        "id": item.get("id"),
                        "position": int(pos),
                        "target_token": tokenizer.decode([int(target_idx)]),
                        "pred_before": tokenizer.decode([int(pred_before)]),
                        "pred_after": tokenizer.decode([int(pred_after)]),
                        "question": q,
                        "answer": gold,
                    }
                )

            token_results.append(
                {
                    "position": int(pos),
                    "answer_len": int(len(gt_ids)),
                    "target_id": int(target_idx),
                    "target_token": tokenizer.decode([int(target_idx)]),
                    "pred_before_id": int(pred_before),
                    "pred_before": tokenizer.decode([int(pred_before)]),
                    "pred_after_id": int(pred_after),
                    "pred_after": tokenizer.decode([int(pred_after)]),
                    "correct_before": bool(correct_before),
                    "correct_after": bool(correct_after),
                    "rank_before": int(rank_before),
                    "rank_after": int(rank_after),
                    "prob_before": float(prob_before),
                    "prob_after": float(prob_after),
                    "metrics": metrics,
                }
            )

        if params and tta_reset == "sample":
            restore_params(params, base_backup, from_cpu=backup_on_cpu)

        # Build predicted answer strings by replacing only numeric positions.
        before_ids = list(gt_ids)
        after_ids = list(gt_ids)
        for pos in numeric_positions:
            before_ids[pos] = pred_before_by_pos.get(pos, before_ids[pos])
            after_ids[pos] = pred_after_by_pos.get(pos, after_ids[pos])

        if eval_mode == "ar":
            pred_before_str = normalize_int_answer(tokenizer.decode(gen_before))
            pred_after_str = normalize_int_answer(tokenizer.decode(gen_after))
        else:
            pred_before_str = normalize_int_answer(tokenizer.decode(before_ids))
            pred_after_str = normalize_int_answer(tokenizer.decode(after_ids))

        em_before = bool(pred_before_str == gold)
        em_after = bool(pred_after_str == gold)
        d_before = float(digit_accuracy(pred_before_str, gold))
        d_after = float(digit_accuracy(pred_after_str, gold))

        seq_total += 1
        seq_em_before += int(em_before)
        seq_em_after += int(em_after)
        digit_sum_before += d_before
        digit_sum_after += d_after

        results.append(
            {
                "id": item.get("id"),
                "question": q,
                "answer": gold,
                "eval_mode": eval_mode,
                "gold_token_ids": gt_ids,
                "gold_tokens": _decode_tokens(tokenizer, gt_ids),
                "numeric_positions": numeric_positions,
                "pred_before_answer": pred_before_str,
                "pred_after_answer": pred_after_str,
                "em_before": em_before,
                "em_after": em_after,
                "digit_acc_before": d_before,
                "digit_acc_after": d_after,
                "tokens": token_results,
                **({"prompt": prompt} if save_prompts else {}),
            }
        )

    elapsed = time.time() - start

    summary = {
        "version": "v7",
        "dataset": dataset,
        "eval_mode": eval_mode,
        "algo": "ane_tta",
        "model": model_name,
        "steps": int(steps),
        "lr": float(lr),
        "optimizer": optimizer,
        "momentum": float(momentum),
        "update_target": update_target,
        "num_layers": num_layers,
        "layer_stride": int(layer_stride),
        "tta_reset": tta_reset,
        "token_total": int(token_total),
        "token_acc_before": token_correct_before / token_total if token_total else 0.0,
        "token_acc_after": token_correct_after / token_total if token_total else 0.0,
        "pass@k_acc_before": {str(k): topk_correct_before[k] / token_total if token_total else 0.0 for k in topk_list},
        "pass@k_acc_after": {str(k): topk_correct_after[k] / token_total if token_total else 0.0 for k in topk_list},
        "target_rank_avg_before": rank_sum_before / token_total if token_total else 0.0,
        "target_rank_avg_after": rank_sum_after / token_total if token_total else 0.0,
        "target_prob_avg_before": prob_sum_before / token_total if token_total else 0.0,
        "target_prob_avg_after": prob_sum_after / token_total if token_total else 0.0,
        "seq_total": int(seq_total),
        "em_before": seq_em_before / seq_total if seq_total else 0.0,
        "em_after": seq_em_after / seq_total if seq_total else 0.0,
        "digit_acc_before": digit_sum_before / seq_total if seq_total else 0.0,
        "digit_acc_after": digit_sum_after / seq_total if seq_total else 0.0,
        "runtime_seconds": float(elapsed),
        "tokens_per_second": float(token_total / max(1e-9, elapsed)),
        "token_budget": {
            "baseline_forwards_est": int(token_total),
            "tta_forwards_est": int(token_total * (steps + 1)),
        },
        "flipped_count": int(len(flipped_cases)),
        "trainable": {
            "trainable_params": int(train_stats.trainable_params),
            "trainable_pct": float(train_stats.trainable_pct),
            "layer_count": int(train_stats.layer_count),
            "total_layers": int(train_stats.total_layers),
        },
    }

    return {"summary": summary, "results": results, "flipped_cases": flipped_cases}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v7 main runner (Greedy vs Greedy+ANE)")

    p.add_argument("--dataset", default="gsm8k", help="addition_50 | gsm8k (others reserved)")
    p.add_argument("--data_path", default=None, help="Override dataset default local JSON path.")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--eval_mode", choices=["tf", "ar"], default="tf")

    p.add_argument("--model", default="/home/jinsk/Models/Llama-3.2-3B-Instruct")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1e6)

    p.add_argument("--update_target", default="mlp+ln", help="mlp|ln|mlp+ln|attn|attn+ln|all|all+lm_head")
    p.add_argument("--num_layers", default="all")
    p.add_argument("--layer_stride", type=int, default=1)
    p.add_argument("--tta_reset", default="token", choices=["token", "sample", "none"])

    p.add_argument("--topk_list", default="5")
    p.add_argument("--num_topk", type=int, default=10)
    p.add_argument("--tracked_topk", type=int, default=10)
    p.add_argument("--snapshot_stride", type=int, default=1, help="Record subspace snapshots every N steps.")

    p.add_argument("--allow_prefix_space", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--backup_on_cpu", action="store_true")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--config_name", required=True)

    p.add_argument("--no_viz", action="store_true")
    p.add_argument("--save_prompts", action="store_true", help="Include prompt strings in JSON (large).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    set_seed(args.seed)

    data = load_dataset(
        name=args.dataset,
        data_path=args.data_path,
        num_samples=args.num_samples,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device_map)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    vocab_size = model.get_input_embeddings().num_embeddings
    num_idx, num_tokens, _ = get_numerical_tokens(tokenizer, allow_prefix_space=args.allow_prefix_space, vocab_size=vocab_size)
    if not num_idx:
        raise RuntimeError("No numerical tokens found")

    num_mask = build_num_mask(vocab_size, num_idx, model.device)
    num_idx_tensor = torch.tensor(num_idx, dtype=torch.long, device=model.device)
    id_to_pos = build_id_to_pos(num_idx, vocab_size, model.device)

    output = evaluate(
        dataset=args.dataset,
        eval_mode=args.eval_mode,
        model_name=os.path.basename(args.model),
        model=model,
        tokenizer=tokenizer,
        data=data,
        num_mask=num_mask,
        num_idx=num_idx,
        num_idx_tensor=num_idx_tensor,
        num_tokens=num_tokens,
        id_to_pos=id_to_pos,
        steps=args.steps,
        lr=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        grad_clip=args.grad_clip,
        max_grad_norm=args.max_grad_norm,
        update_target=args.update_target,
        num_layers=args.num_layers,
        layer_stride=args.layer_stride,
        tta_reset=args.tta_reset,
        topk_list=parse_topk_list(args.topk_list),
        snapshot_stride=args.snapshot_stride,
        num_topk=args.num_topk,
        tracked_topk=args.tracked_topk,
        backup_on_cpu=args.backup_on_cpu,
        save_prompts=args.save_prompts,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.config_name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if not args.no_viz:
        from visualization import visualize_all

        visualize_all(output, args.output_dir, args.config_name)

    s = output["summary"]
    print(
        f"[v7] dataset={s['dataset']} mode={s['eval_mode']} model={s['model']} "
        f"token_acc: {s['token_acc_before']:.3f}->{s['token_acc_after']:.3f} "
        f"EM: {s['em_before']:.3f}->{s['em_after']:.3f} "
        f"digit: {s['digit_acc_before']:.3f}->{s['digit_acc_after']:.3f} "
        f"runtime={s['runtime_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()

