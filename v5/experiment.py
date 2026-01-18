"""Experiment runner for ANE-TTA v5."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ane_core import AngularEntropy
from metrics import TokenRecord, build_metrics
from numerical_utils import build_num_mask, get_numerical_tokens, mask_logits_to_num
from tta_engine import (
    TTAEngine,
    backup_params,
    configure_trainable_params,
    restore_params,
)

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


def get_model_size_b(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e9


def parse_topk_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def load_data(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return f"<{token_id}>"


def compute_rank_prob(logits: torch.Tensor, target_idx: int) -> Tuple[float, int]:
    logits_f = logits.float()
    target_logit = logits_f[target_idx]
    lse = torch.logsumexp(logits_f, dim=-1)
    prob = torch.exp(target_logit - lse).item()
    rank = int((logits_f > target_logit).sum().item() + 1)
    return prob, rank


def evaluate_tf_sample(
    sample: Dict,
    tokenizer,
    model: torch.nn.Module,
    tta_engine: TTAEngine,
    params: List[torch.nn.Parameter],
    args,
    backup_on_cpu: bool,
) -> Tuple[Dict, List[TokenRecord], List[TokenRecord], List[Dict]]:
    question = sample["question"]
    answer = sample["answer"]
    answer_str = str(answer)
    gt_tokens = tokenizer.encode(answer_str, add_special_tokens=False)

    prompt = build_prompt(question)
    input_encoding = tokenizer(prompt, return_tensors="pt").to(model.device)

    tf_tokens = []
    tf_baseline_records: List[TokenRecord] = []
    tf_tta_records: List[TokenRecord] = []
    flipped_cases: List[Dict] = []

    if args.reset_mode == "per_sample":
        sample_backup = backup_params(params, backup_on_cpu)

    for pos, target_idx in enumerate(gt_tokens):
        if pos == 0:
            input_ids = input_encoding["input_ids"]
        else:
            prefix = torch.tensor([gt_tokens[:pos]], device=model.device)
            input_ids = torch.cat([input_encoding["input_ids"], prefix], dim=1)

        if args.reset_mode == "per_token":
            token_backup = backup_params(params, backup_on_cpu)

        pred_before, pred_after, metrics = tta_engine.run_token(input_ids, target_idx)

        baseline_rank = metrics["target_rank"][0]
        baseline_prob = metrics["target_prob"][0]
        tta_rank = metrics["target_rank"][-1]
        tta_prob = metrics["target_prob"][-1]

        correct_before = pred_before == target_idx
        correct_after = pred_after == target_idx

        baseline_record = TokenRecord(correct_before, baseline_rank, baseline_prob)
        tta_record = TokenRecord(correct_after, tta_rank, tta_prob)
        tf_baseline_records.append(baseline_record)
        tf_tta_records.append(tta_record)

        token_result = {
            "position": pos,
            "answer_len": len(gt_tokens),
            "target_id": target_idx,
            "target_token": decode_token(tokenizer, target_idx),
            "baseline": {
                "pred_id": pred_before,
                "pred_token": decode_token(tokenizer, pred_before),
                "rank": baseline_rank,
                "prob": baseline_prob,
                "correct": correct_before,
            },
            "tta": {
                "pred_id": pred_after,
                "pred_token": decode_token(tokenizer, pred_after),
                "rank": tta_rank,
                "prob": tta_prob,
                "correct": correct_after,
            },
            "metrics": metrics,
        }
        tf_tokens.append(token_result)

        if not correct_before and correct_after:
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

        if args.reset_mode == "per_token":
            restore_params(params, token_backup, backup_on_cpu)

    if args.reset_mode == "per_sample":
        restore_params(params, sample_backup, backup_on_cpu)

    return (
        {"tokens": tf_tokens},
        tf_baseline_records,
        tf_tta_records,
        flipped_cases,
    )


def evaluate_ar_sample(
    sample: Dict,
    tokenizer,
    model: torch.nn.Module,
    tta_engine: TTAEngine,
    params: List[torch.nn.Parameter],
    args,
    backup_on_cpu: bool,
) -> Tuple[Dict, List[TokenRecord], List[TokenRecord]]:
    question = sample["question"]
    answer = sample["answer"]
    answer_str = str(answer)
    gt_tokens = tokenizer.encode(answer_str, add_special_tokens=False)

    prompt = build_prompt(question)
    input_encoding = tokenizer(prompt, return_tensors="pt").to(model.device)

    baseline_tokens = []
    tta_tokens = []
    baseline_records: List[TokenRecord] = []
    tta_records: List[TokenRecord] = []

    # Baseline autoregressive
    prefix = input_encoding["input_ids"]
    for pos, target_idx in enumerate(gt_tokens):
        model.eval()
        with torch.no_grad():
            logits = model(prefix).logits[0, -1, :]
            pred_id = torch.argmax(mask_logits_to_num(logits, tta_engine.num_mask)).item()
            prob, rank = compute_rank_prob(logits, target_idx)

        baseline_records.append(TokenRecord(pred_id == target_idx, rank, prob))
        baseline_tokens.append(
            {
                "position": pos,
                "target_id": target_idx,
                "target_token": decode_token(tokenizer, target_idx),
                "pred_id": pred_id,
                "pred_token": decode_token(tokenizer, pred_id),
                "rank": rank,
                "prob": prob,
                "correct": pred_id == target_idx,
            }
        )
        prefix = torch.cat([prefix, torch.tensor([[pred_id]], device=model.device)], dim=1)

    # TTA autoregressive
    if args.eval_ar_tta:
        if args.reset_mode == "per_sample":
            sample_backup = backup_params(params, backup_on_cpu)

        prefix = input_encoding["input_ids"]
        for pos, target_idx in enumerate(gt_tokens):
            if args.reset_mode == "per_token":
                token_backup = backup_params(params, backup_on_cpu)

            pred_before, pred_after, metrics = tta_engine.run_token(prefix, target_idx)
            prob = metrics["target_prob"][-1]
            rank = metrics["target_rank"][-1]

            tta_records.append(TokenRecord(pred_after == target_idx, rank, prob))
            tta_tokens.append(
                {
                    "position": pos,
                    "target_id": target_idx,
                    "target_token": decode_token(tokenizer, target_idx),
                    "pred_id": pred_after,
                    "pred_token": decode_token(tokenizer, pred_after),
                    "rank": rank,
                    "prob": prob,
                    "correct": pred_after == target_idx,
                }
            )

            prefix = torch.cat([prefix, torch.tensor([[pred_after]], device=model.device)], dim=1)

            if args.reset_mode == "per_token":
                restore_params(params, token_backup, backup_on_cpu)

        if args.reset_mode == "per_sample":
            restore_params(params, sample_backup, backup_on_cpu)

    return (
        {"baseline_tokens": baseline_tokens, "tta_tokens": tta_tokens},
        baseline_records,
        tta_records,
    )


@dataclass
class ExperimentSummary:
    model: str
    metrics: Dict
    config: Dict
    runtime_minutes: float
    total_tokens: int
    sample_count: int
    flipped_count: int


def run_experiment(args) -> Dict:
    start_time = time.time()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = load_data(args.data_path)
    data = data[: args.num_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.dtype,
        device_map=args.device_map,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model_size = get_model_size_b(model)

    num_idx, num_tokens = get_numerical_tokens(tokenizer, allow_prefix_space=args.allow_prefix_space)
    vocab_size = model.get_input_embeddings().weight.shape[0]
    filtered = [(idx, tok) for idx, tok in zip(num_idx, num_tokens) if idx < vocab_size]
    num_idx = [idx for idx, _ in filtered]
    num_tokens = [tok for _, tok in filtered]
    if not num_idx:
        raise RuntimeError("No numerical tokens found in tokenizer vocabulary")
    num_mask = build_num_mask(num_idx, vocab_size, model.device)
    num_idx_tensor = torch.tensor(num_idx, device=model.device, dtype=torch.long)

    params, train_stats = configure_trainable_params(
        model,
        args.update_target,
        args.num_layers,
        args.layer_stride,
    )

    backup_on_cpu = args.backup_on_cpu
    if backup_on_cpu is None:
        backup_on_cpu = (model_size >= 6.0) or (train_stats.trainable_params > 200_000_000)

    ane_loss = AngularEntropy(
        num_idx,
        model.get_input_embeddings(),
        use_float32=args.loss_float32,
        eps=args.loss_eps,
        cache_embeddings=not args.no_embed_cache,
    )
    ane_loss.to(model.device)

    tta_engine = TTAEngine(
        model,
        tokenizer,
        num_mask,
        num_idx_tensor,
        num_tokens,
        params,
        ane_loss,
        args,
    )

    tf_baseline_samples: List[List[TokenRecord]] = []
    tf_tta_samples: List[List[TokenRecord]] = []
    ar_baseline_samples: List[List[TokenRecord]] = []
    ar_tta_samples: List[List[TokenRecord]] = []

    results = []
    flipped_cases: List[Dict] = []

    for sample in tqdm(data, desc="Samples"):
        tf_result, tf_base_records, tf_tta_records, flips = evaluate_tf_sample(
            sample,
            tokenizer,
            model,
            tta_engine,
            params,
            args,
            backup_on_cpu,
        )
        tf_baseline_samples.append(tf_base_records)
        tf_tta_samples.append(tf_tta_records)
        flipped_cases.extend(flips)

        ar_result = {}
        if args.eval_ar:
            ar_result, ar_base_records, ar_tta_records = evaluate_ar_sample(
                sample,
                tokenizer,
                model,
                tta_engine,
                params,
                args,
                backup_on_cpu,
            )
            ar_baseline_samples.append(ar_base_records)
            if args.eval_ar_tta:
                ar_tta_samples.append(ar_tta_records)

        results.append(
            {
                "question": sample["question"],
                "answer": sample["answer"],
                "answer_len": len(str(sample["answer"])),
                "tf": tf_result,
                "ar": ar_result,
            }
        )

        if args.empty_cache_each and torch.cuda.is_available():
            torch.cuda.empty_cache()

    topk_list = parse_topk_list(args.topk_list)
    metrics = {
        "tf": build_metrics(tf_baseline_samples, topk_list),
        "tf_tta": build_metrics(tf_tta_samples, topk_list),
    }

    if args.eval_ar:
        metrics["ar"] = build_metrics(ar_baseline_samples, topk_list)
        if args.eval_ar_tta:
            metrics["ar_tta"] = build_metrics(ar_tta_samples, topk_list)

    runtime = (time.time() - start_time) / 60.0

    summary = ExperimentSummary(
        model=os.path.basename(args.model),
        metrics=metrics,
        config={
            "update_target": args.update_target,
            "num_layers": args.num_layers,
            "layer_stride": args.layer_stride,
            "steps": args.steps,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "reset_mode": args.reset_mode,
            "fast_eval": args.fast_eval,
            "skip_if_correct": args.skip_if_correct,
            "eval_ar": args.eval_ar,
            "eval_ar_tta": args.eval_ar_tta,
            "topk_list": args.topk_list,
            "trainable_params": train_stats.trainable_params,
            "trainable_pct": train_stats.trainable_pct,
            "layer_count": train_stats.layer_count,
        },
        runtime_minutes=runtime,
        total_tokens=sum(len(s) for s in tf_baseline_samples),
        sample_count=len(data),
        flipped_count=len(flipped_cases),
    )

    output = {
        "summary": asdict(summary),
        "results": results,
        "flipped_cases": flipped_cases,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.exp_name}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if not args.no_viz:
        from visualization import visualize_all

        visualize_all(output, args.output_dir, args.exp_name)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output
