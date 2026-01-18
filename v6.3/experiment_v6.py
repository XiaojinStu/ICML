"""v6.3: GSM8K decoding-style evaluation (teacher forcing on numeric tokens).

Key idea:
- Use the *gold answer tokenization* to decide which next positions should output a numeric token.
- For those positions only, feed (prompt + gold prefix tokens) and compare the model's
  numeric-subvocab argmax to the gold numeric token.
- This avoids relying on the model to generate a fully formatted answer string.

We still optionally apply ANE test-time adaptation (TTA) before decoding the next numeric token.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k_dataset import load_gsm8k, load_local_json, save_json
from losses import AngularEntropyLoss
from numerical_utils import build_id_to_pos, build_num_mask, get_numerical_tokens
from tta_engine import TTAEngine, backup_params, configure_trainable_params, restore_params


PROMPT_FEWSHOT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise calculator. Output ONLY the final integer answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

If you have 5 apples and buy 7 more, how many apples do you have?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

12<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 1200000 + 34567?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1234567<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def set_seed(seed: int) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cat_prompt_and_tokens(prompt_ids: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
    if not token_ids:
        return prompt_ids
    suffix = torch.tensor([token_ids], dtype=torch.long, device=prompt_ids.device)
    return torch.cat([prompt_ids, suffix], dim=1)


def _normalize_answer(text: str) -> str:
    s = str(text).strip()
    s = s.replace(",", "")
    s = s.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    if s.startswith("+"):
        s = s[1:]
    if s.endswith("."):
        s = s[:-1]
    return s


def _encode_gold_answer(tokenizer, gold_norm: str) -> List[int]:
    """Encode gold answer into token ids, forcing '-' to be a separate token if present."""
    if gold_norm.startswith("-") and len(gold_norm) > 1:
        sign_ids = tokenizer.encode("-", add_special_tokens=False)
        rest_ids = tokenizer.encode(gold_norm[1:], add_special_tokens=False)
        return list(sign_ids) + list(rest_ids)
    return list(tokenizer.encode(gold_norm, add_special_tokens=False))


def _decode_tokens(tokenizer, token_ids: List[int]) -> List[str]:
    return [tokenizer.decode([i]) for i in token_ids]


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
    momentum: float,
    grad_clip: float,
    max_grad_norm: float,
    update_target: str,
    num_layers: str,
    layer_stride: int,
    backup_on_cpu: bool,
    output_dir: str,
    no_viz: bool,
    save_prompts: bool,
) -> Dict:
    start = time.time()

    loss_fn = AngularEntropyLoss(num_idx, model.get_input_embeddings(), use_float32=True, eps=1e-4, cache_embeddings=True).to(model.device)

    params, train_stats = configure_trainable_params(model, update_target, num_layers, layer_stride)
    base_backup = backup_params(params, to_cpu=backup_on_cpu) if params else []

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
        momentum=momentum,
        snapshot_steps=[0, steps],
        num_topk=10,
    )

    token_total = 0
    token_correct_before = 0
    token_correct_after = 0

    seq_total = 0
    seq_correct_before = 0
    seq_correct_after = 0

    flipped_cases = []
    results = []

    for item in tqdm(data, desc=f"gsm8k-tfnum-{config_name}"):
        q = item["question"]
        gold = str(item["answer"])
        gold_norm = _normalize_answer(gold)

        prompt = PROMPT_FEWSHOT.format(question=q)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

        gt_ids = _encode_gold_answer(tokenizer, gold_norm)
        if not gt_ids:
            continue

        numeric_positions = [pos for pos, tid in enumerate(gt_ids) if int(id_to_pos[tid].item()) >= 0]
        if not numeric_positions:
            continue

        pred_before_by_pos: Dict[int, int] = {}
        pred_after_by_pos: Dict[int, int] = {}
        token_results = []

        all_before = True
        all_after = True

        for pos in numeric_positions:
            target_idx = gt_ids[pos]
            prefix_tokens = gt_ids[:pos]
            input_ids = _cat_prompt_and_tokens(prompt_ids, prefix_tokens)

            if params:
                restore_params(params, base_backup, from_cpu=backup_on_cpu)
            pred_before, pred_after, metrics = tta_engine.run_token(input_ids, target_idx)
            if params:
                restore_params(params, base_backup, from_cpu=backup_on_cpu)

            pred_before_by_pos[pos] = int(pred_before)
            pred_after_by_pos[pos] = int(pred_after)

            correct_before = int(pred_before) == int(target_idx)
            correct_after = int(pred_after) == int(target_idx)

            token_total += 1
            token_correct_before += int(correct_before)
            token_correct_after += int(correct_after)

            all_before = all_before and bool(correct_before)
            all_after = all_after and bool(correct_after)

            rank_before = int(metrics["target_rank"][0])
            rank_after = int(metrics["target_rank"][-1])

            if (not correct_before) and correct_after:
                flipped_cases.append(
                    {
                        "id": item.get("id"),
                        "question": q,
                        "answer": gold_norm,
                        "position": pos,
                        "target_token": tokenizer.decode([target_idx]),
                        "pred_before": tokenizer.decode([pred_before]),
                        "pred_after": tokenizer.decode([pred_after]),
                    }
                )

            token_results.append(
                {
                    "position": int(pos),
                    "target_id": int(target_idx),
                    "target_token": tokenizer.decode([target_idx]),
                    "pred_before_id": int(pred_before),
                    "pred_before": tokenizer.decode([pred_before]),
                    "pred_after_id": int(pred_after),
                    "pred_after": tokenizer.decode([pred_after]),
                    "correct_before": bool(correct_before),
                    "correct_after": bool(correct_after),
                    "rank_before": rank_before,
                    "rank_after": rank_after,
                    "metrics": metrics,
                }
            )

        before_ids = list(gt_ids)
        after_ids = list(gt_ids)
        for pos in numeric_positions:
            before_ids[pos] = pred_before_by_pos[pos]
            after_ids[pos] = pred_after_by_pos[pos]

        pred_before_str = _normalize_answer(tokenizer.decode(before_ids))
        pred_after_str = _normalize_answer(tokenizer.decode(after_ids))

        seq_total += 1
        seq_correct_before += int(pred_before_str == gold_norm)
        seq_correct_after += int(pred_after_str == gold_norm)

        results.append(
            {
                "id": item.get("id"),
                "question": q,
                "answer": gold_norm,
                "gold_token_ids": gt_ids,
                "gold_tokens": _decode_tokens(tokenizer, gt_ids),
                "numeric_positions": numeric_positions,
                "pred_before_answer": pred_before_str,
                "pred_after_answer": pred_after_str,
                "all_numeric_correct_before": bool(all_before),
                "all_numeric_correct_after": bool(all_after),
                "tokens": token_results,
                **({"prompt": prompt} if save_prompts else {}),
            }
        )

    elapsed_min = (time.time() - start) / 60.0

    summary = {
        "task": "gsm8k",
        "algo": "ane_tta",
        "eval_mode": "tf_num",
        "model": model_name,
        "steps": steps,
        "lr": lr,
        "optimizer": optimizer,
        "momentum": momentum,
        "update_target": update_target,
        "num_layers": num_layers,
        "layer_stride": layer_stride,
        "token_total": token_total,
        "token_correct_before": token_correct_before,
        "token_correct_after": token_correct_after,
        "token_acc_before": token_correct_before / token_total if token_total else 0.0,
        "token_acc_after": token_correct_after / token_total if token_total else 0.0,
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
    p = argparse.ArgumentParser(description="v6.3 GSM8K TF numeric-token evaluation")

    p.add_argument("--split", default="test")
    p.add_argument("--cache_dir", default=None)

    p.add_argument("--model", default="/home/jinsk/Models/Llama-3.2-1B-Instruct")
    p.add_argument("--data_path", default=None)
    p.add_argument("--save_data_path", default=None)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1e6)

    p.add_argument("--update_target", default="mlp", help="mlp|ln|mlp+ln|attn|attn+ln|all|all+lm_head")
    p.add_argument("--num_layers", default="all")
    p.add_argument("--layer_stride", type=int, default=1)

    p.add_argument("--output_dir", required=True)
    p.add_argument("--config_name", required=True)

    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--allow_prefix_space", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--backup_on_cpu", action="store_true")

    p.add_argument("--no_viz", action="store_true")
    p.add_argument("--save_prompts", action="store_true", help="Include full prompt strings in the output JSON (large).")

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

    if args.data_path:
        data = load_local_json(args.data_path)
        if args.shuffle:
            rng = random.Random(args.seed)
            rng.shuffle(data)
        data = data[: args.num_samples]
    else:
        data = load_gsm8k(split=args.split, num_samples=args.num_samples, seed=args.seed, cache_dir=args.cache_dir)

    if args.save_data_path:
        save_json(args.save_data_path, data)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device_map)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    vocab_size = model.get_input_embeddings().num_embeddings
    num_idx, _, _ = get_numerical_tokens(tokenizer, allow_prefix_space=args.allow_prefix_space, vocab_size=vocab_size)
    num_mask = build_num_mask(vocab_size, num_idx, model.device)
    num_idx_tensor = torch.tensor(num_idx, dtype=torch.long, device=model.device)
    id_to_pos = build_id_to_pos(num_idx, vocab_size, model.device)

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
        momentum=args.momentum,
        grad_clip=args.grad_clip,
        max_grad_norm=args.max_grad_norm,
        update_target=args.update_target,
        num_layers=args.num_layers,
        layer_stride=args.layer_stride,
        backup_on_cpu=args.backup_on_cpu,
        output_dir=args.output_dir,
        no_viz=args.no_viz,
        save_prompts=args.save_prompts,
    )


if __name__ == "__main__":
    main()
