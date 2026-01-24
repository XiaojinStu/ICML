#!/usr/bin/env python3
"""Build a smaller NUPA dataset by filtering too-hard / too-easy samples.

Procedure (as requested):
1) Llama-8B with TTA (steps=5, lr=0.01): find samples whose digit-only answer tokens
   are still ALL wrong after TTA; delete 70% of them.
2) Llama-1B baseline (no TTA): find samples whose digit-only answer tokens are ALL
   correct before TTA; delete 30% of them.

This script is designed to be memory-safe (one model at a time).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "v9"))

from dataset_loader import load_standard_json  # noqa: E402
from losses import AngularEntropyLoss  # noqa: E402
from metrics import encode_answer, normalize_answer  # noqa: E402
from numerical_utils import build_id_to_pos, build_num_mask, get_numerical_tokens, mask_logits_to_num  # noqa: E402
from prompts import build_prompt_ids  # noqa: E402
from tta_engine import TTAEngine, backup_params, configure_trainable_params, restore_params  # noqa: E402


def _dtype_from_arg(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _cat_prompt_and_tokens(prompt_ids: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
    if not token_ids:
        return prompt_ids
    suffix = torch.tensor([token_ids], dtype=torch.long, device=prompt_ids.device)
    return torch.cat([prompt_ids, suffix], dim=1)


def _load_model(model_path: str, *, device: str, dtype: torch.dtype):
    trust = bool("qwen" in os.path.basename(model_path).lower())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=trust, low_cpu_mem_usage=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.to(device)
    return model, tokenizer


def _build_num_tables(model, tokenizer) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[str], torch.Tensor]:
    vocab_size = model.get_input_embeddings().num_embeddings
    num_idx, num_tokens, _ = get_numerical_tokens(tokenizer, allow_prefix_space=False, vocab_size=vocab_size)
    if not num_idx:
        raise RuntimeError("No numerical tokens found")
    num_mask = build_num_mask(vocab_size, num_idx, model.device)
    num_idx_tensor = torch.tensor(num_idx, dtype=torch.long, device=model.device)
    id_to_pos = build_id_to_pos(num_idx, vocab_size, model.device)
    return num_mask, num_idx, num_idx_tensor, num_tokens, id_to_pos


def _find_hard_ids_with_tta(
    *,
    data: List[Dict],
    model_path: str,
    device: str,
    dtype: torch.dtype,
    seed: int,
    steps: int,
    lr: float,
) -> Set[str]:
    """Hard = for this sample, ALL digit-only answer tokens are wrong after TTA (at final step)."""

    model, tokenizer = _load_model(model_path, device=device, dtype=dtype)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    num_mask, num_idx, num_idx_tensor, num_tokens, id_to_pos = _build_num_tables(model, tokenizer)

    loss_fn = AngularEntropyLoss(
        num_idx,
        model.get_input_embeddings(),
        use_float32=True,
        eps=1e-4,
        cache_embeddings=True,
        distance_mode="angle",
    ).to(model.device)

    params, _ = configure_trainable_params(model, target="mlp+ln", num_layers="all", layer_stride=1)
    base_backup = backup_params(params, to_cpu=False) if params else []

    engine = TTAEngine(
        model=model,
        tokenizer=tokenizer,
        params=params,
        num_mask=num_mask,
        num_idx_tensor=num_idx_tensor,
        num_tokens=num_tokens,
        id_to_pos=id_to_pos,
        loss_fn=loss_fn,
        steps=int(steps),
        lr=float(lr),
        lr_schedule="constant",
        lr_min=1e-4,
        lr_norm="none",
        optimizer="sgd",
        momentum=0.0,
        grad_clip=1.0,
        max_grad_norm=1e6,
        num_topk=10,
        tracked_topk=0,
        snapshot_stride=max(1, int(steps)),
        anchor_log="none",
    )

    hard_ids: Set[str] = set()
    for item in tqdm(data, desc="NUPA-hard (Llama-8B, TTA steps=5)"):
        sid = str(item.get("id", ""))
        q = str(item["question"])
        gold = normalize_answer(item["answer"])

        prompt_ids = build_prompt_ids(tokenizer, "NUPA", q).to(model.device)
        gt_ids = encode_answer(tokenizer, gold)
        if not gt_ids:
            continue

        numeric_positions = [pos for pos, tid in enumerate(gt_ids) if int(id_to_pos[tid].item()) >= 0]
        if not numeric_positions:
            continue

        all_wrong = True
        for pos in numeric_positions:
            target_idx = int(gt_ids[pos])
            input_ids = _cat_prompt_and_tokens(prompt_ids, gt_ids[:pos])
            if params:
                restore_params(params, base_backup, from_cpu=False)
            _, __, metrics = engine.run_token(input_ids, target_idx)
            if params:
                restore_params(params, base_backup, from_cpu=False)

            ranks = list(metrics.get("target_rank", []))
            if ranks and int(ranks[-1]) == 1:
                all_wrong = False
                break

        if all_wrong:
            hard_ids.add(sid)

    del engine, loss_fn, params, base_backup, num_mask, num_idx_tensor, id_to_pos, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hard_ids


def _find_easy_ids_baseline(
    *,
    data: List[Dict],
    model_path: str,
    device: str,
    dtype: torch.dtype,
) -> Set[str]:
    """Easy = for this sample, ALL digit-only answer tokens are correct before TTA (baseline greedy)."""

    model, tokenizer = _load_model(model_path, device=device, dtype=dtype)
    model.eval()

    num_mask, _, __, ___, id_to_pos = _build_num_tables(model, tokenizer)

    easy_ids: Set[str] = set()
    for item in tqdm(data, desc="NUPA-easy (Llama-1B, baseline)"):
        sid = str(item.get("id", ""))
        q = str(item["question"])
        gold = normalize_answer(item["answer"])

        prompt_ids = build_prompt_ids(tokenizer, "NUPA", q).to(model.device)
        gt_ids = encode_answer(tokenizer, gold)
        if not gt_ids:
            continue

        numeric_positions = [pos for pos, tid in enumerate(gt_ids) if int(id_to_pos[tid].item()) >= 0]
        if not numeric_positions:
            continue

        all_correct = True
        with torch.no_grad():
            for pos in numeric_positions:
                target_idx = int(gt_ids[pos])
                input_ids = _cat_prompt_and_tokens(prompt_ids, gt_ids[:pos])
                logits = model(input_ids).logits[0, -1, :]
                pred = int(torch.argmax(mask_logits_to_num(logits, num_mask)).item())
                if pred != target_idx:
                    all_correct = False
                    break

        if all_correct:
            easy_ids.add(sid)

    del num_mask, id_to_pos, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return easy_ids


def _drop_fraction(ids: List[str], *, frac: float, seed: int) -> Set[str]:
    if not ids or frac <= 0.0:
        return set()
    rng = random.Random(int(seed))
    ids = list(ids)
    rng.shuffle(ids)
    k = int(len(ids) * float(frac))
    return set(ids[:k])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Curate a smaller NUPA dataset (hard/easy filtering).")
    p.add_argument("--input", default="datasets_final_v3/nupa_test440_v1.json")
    p.add_argument("--output", default="datasets_final_v3/nupa_v2.json")
    p.add_argument("--report", default="datasets_final_v3/nupa_v2_report.json")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    p.add_argument("--llama8b", default="/home/jinsk/Models/Llama-3.1-8B-Instruct")
    p.add_argument("--llama1b", default="/home/jinsk/Models/Llama-3.2-1B-Instruct")
    p.add_argument("--device_hard", default="cuda:0")
    p.add_argument("--device_easy", default="cuda:1")

    p.add_argument("--tta_steps", type=int, default=5)
    p.add_argument("--tta_lr", type=float, default=0.01)
    p.add_argument("--hard_drop_frac", type=float, default=0.70)
    p.add_argument("--easy_drop_frac", type=float, default=0.30)
    return p


def main() -> None:
    args = build_parser().parse_args()
    dtype = _dtype_from_arg(args.dtype)

    data = load_standard_json(data_path=str(args.input), num_samples=0, seed=int(args.seed), shuffle=False)
    n0 = len(data)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hard_ids = _find_hard_ids_with_tta(
        data=data,
        model_path=str(args.llama8b),
        device=str(args.device_hard),
        dtype=dtype,
        seed=int(args.seed),
        steps=int(args.tta_steps),
        lr=float(args.tta_lr),
    )
    hard_drop = _drop_fraction(sorted(hard_ids), frac=float(args.hard_drop_frac), seed=int(args.seed))
    data1 = [ex for ex in data if str(ex.get("id", "")) not in hard_drop]
    n1 = len(data1)

    easy_ids = _find_easy_ids_baseline(
        data=data1,
        model_path=str(args.llama1b),
        device=str(args.device_easy),
        dtype=dtype,
    )
    easy_drop = _drop_fraction(sorted(easy_ids), frac=float(args.easy_drop_frac), seed=int(args.seed))
    data2 = [ex for ex in data1 if str(ex.get("id", "")) not in easy_drop]
    n2 = len(data2)

    Path(args.output).write_text(json.dumps(data2, indent=2, ensure_ascii=False))
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": int(args.seed),
        "hard": {
            "model": str(args.llama8b),
            "device": str(args.device_hard),
            "tta_steps": int(args.tta_steps),
            "tta_lr": float(args.tta_lr),
            "hard_ids": len(hard_ids),
            "dropped": len(hard_drop),
        },
        "easy": {
            "model": str(args.llama1b),
            "device": str(args.device_easy),
            "easy_ids": len(easy_ids),
            "dropped": len(easy_drop),
        },
        "counts": {"before": n0, "after_hard_drop": n1, "after_easy_drop": n2},
        "dropped_ids": {"hard": sorted(hard_drop), "easy": sorted(easy_drop)},
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"[nupa-v2] before={n0} -> after_hard_drop={n1} -> after_easy_drop={n2}")
    print(f"[nupa-v2] hard_ids={len(hard_ids)} dropped={len(hard_drop)} (drop_frac={args.hard_drop_frac})")
    print(f"[nupa-v2] easy_ids={len(easy_ids)} dropped={len(easy_drop)} (drop_frac={args.easy_drop_frac})")
    print(f"[nupa-v2] wrote: {args.output}")
    print(f"[nupa-v2] report: {args.report}")


if __name__ == "__main__":
    main()

