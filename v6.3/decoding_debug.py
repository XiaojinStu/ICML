"""Generate a human-readable v6.3 decoding report on GSM8K (default: gsm8k_test_50).

Writes:
- <out_dir>/report.json  (summary + per-sample, including prompt + tokens)
- <out_dir>/report.md    (question + prompt + per-position token comparisons)

All models (1B/3B/8B) use the same prompt and the same TF numeric-token procedure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k_dataset import load_local_json
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
    """Encode gold answer into token ids, forcing '-' to be a separate token if present.

    This avoids the edge case where a tokenizer may merge '-13' into a single token,
    which would otherwise make the answer look "non-numeric" to the digit-token filter.
    """
    if gold_norm.startswith("-") and len(gold_norm) > 1:
        sign_ids = tokenizer.encode("-", add_special_tokens=False)
        rest_ids = tokenizer.encode(gold_norm[1:], add_special_tokens=False)
        return list(sign_ids) + list(rest_ids)
    return list(tokenizer.encode(gold_norm, add_special_tokens=False))


def _cat_prompt_and_tokens(prompt_ids: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
    if not token_ids:
        return prompt_ids
    suffix = torch.tensor([token_ids], dtype=torch.long, device=prompt_ids.device)
    return torch.cat([prompt_ids, suffix], dim=1)


def write_md(path: Path, records: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# v6.3 Decoding Debug (TF numeric tokens)\n\n")
    lines.append("## Summary\n")
    for k in [
        "model",
        "steps",
        "lr",
        "update_target",
        "token_total",
        "token_acc_before",
        "token_acc_after",
        "seq_total",
        "seq_acc_before",
        "seq_acc_after",
    ]:
        lines.append(f"- {k}: `{summary.get(k)}`\n")

    lines.append("\n## Detailed Samples\n")
    for i, r in enumerate(records, start=1):
        lines.append(f"\n### Sample {i} | id={r.get('id')}\n")
        lines.append(f"- gold_answer: `{r['answer']}`\n")
        lines.append(f"- pred_before_answer: `{r['pred_before_answer']}`\n")
        lines.append(f"- pred_after_answer: `{r['pred_after_answer']}`\n")
        lines.append(f"- numeric_positions: `{r['numeric_positions']}`\n")

        lines.append("\n**Question**\n")
        lines.append("```text\n" + r["question"].strip() + "\n```\n")

        lines.append("\n**Prompt**\n")
        lines.append("```text\n" + r["prompt"].strip() + "\n```\n")

        lines.append("\n**Gold Tokens**\n")
        lines.append("```text\n" + str(r["gold_token_ids"]) + "\n" + "|".join(r["gold_tokens"]) + "\n```\n")

        lines.append("\n**Per-position comparisons (numeric positions only)**\n")
        for t in r["tokens"]:
            lines.append(
                f"- pos={t['position']} target={t['target_token']!r} "
                f"before={t['pred_before']!r} after={t['pred_after']!r} "
                f"rank_before={t['rank_before']} rank_after={t['rank_after']}\n"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v6.3 decoding debug report")
    p.add_argument("--model", default="/home/jinsk/Models/Llama-3.2-1B-Instruct")
    p.add_argument("--data_path", default="data/gsm8k/gsm8k_test_50.json")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--update_target", default="mlp", choices=["mlp", "ln", "mlp+ln", "attn", "attn+ln", "all", "all+lm_head"])
    p.add_argument("--num_layers", default="all")
    p.add_argument("--layer_stride", type=int, default=1)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--backup_on_cpu", action="store_true")
    p.add_argument("--allow_prefix_space", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--out_dir", default="v6.3/results_decoding_debug/llama3.2-1b")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_local_json(args.data_path)[: args.num_samples]

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

    loss_fn = AngularEntropyLoss(num_idx, model.get_input_embeddings(), use_float32=True, eps=1e-4, cache_embeddings=True).to(model.device)
    params, train_stats = configure_trainable_params(model, args.update_target, args.num_layers, args.layer_stride)
    base_backup = backup_params(params, to_cpu=args.backup_on_cpu) if params else []

    engine = TTAEngine(
        model=model,
        params=params,
        num_mask=num_mask,
        num_idx_tensor=num_idx_tensor,
        id_to_pos=id_to_pos,
        loss_fn=loss_fn,
        steps=args.steps,
        lr=args.lr,
        grad_clip=1.0,
        max_grad_norm=1e6,
        optimizer=args.optimizer,
        momentum=args.momentum,
        snapshot_steps=[0, args.steps],
        num_topk=10,
    )

    records = []
    token_total = 0
    token_correct_before = 0
    token_correct_after = 0
    seq_total = 0
    seq_correct_before = 0
    seq_correct_after = 0

    for item in data:
        q = item["question"]
        gold = _normalize_answer(item["answer"])
        prompt = PROMPT_FEWSHOT.format(question=q)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

        gt_ids = _encode_gold_answer(tokenizer, gold)
        if not gt_ids:
            continue

        numeric_positions = [pos for pos, tid in enumerate(gt_ids) if int(id_to_pos[tid].item()) >= 0]
        if not numeric_positions:
            continue

        pred_before_by_pos = {}
        pred_after_by_pos = {}
        token_rows = []

        for pos in numeric_positions:
            target_idx = gt_ids[pos]
            input_ids = _cat_prompt_and_tokens(prompt_ids, gt_ids[:pos])

            if params:
                restore_params(params, base_backup, from_cpu=args.backup_on_cpu)
            pred_before, pred_after, metrics = engine.run_token(input_ids, target_idx)
            if params:
                restore_params(params, base_backup, from_cpu=args.backup_on_cpu)

            pred_before_by_pos[pos] = int(pred_before)
            pred_after_by_pos[pos] = int(pred_after)

            cb = int(pred_before) == int(target_idx)
            ca = int(pred_after) == int(target_idx)
            token_total += 1
            token_correct_before += int(cb)
            token_correct_after += int(ca)

            token_rows.append(
                {
                    "position": int(pos),
                    "target_id": int(target_idx),
                    "target_token": tokenizer.decode([target_idx]),
                    "pred_before_id": int(pred_before),
                    "pred_before": tokenizer.decode([pred_before]),
                    "pred_after_id": int(pred_after),
                    "pred_after": tokenizer.decode([pred_after]),
                    "rank_before": int(metrics["target_rank"][0]),
                    "rank_after": int(metrics["target_rank"][-1]),
                    "metrics": metrics,
                }
            )

        before_ids = list(gt_ids)
        after_ids = list(gt_ids)
        for pos in numeric_positions:
            before_ids[pos] = pred_before_by_pos[pos]
            after_ids[pos] = pred_after_by_pos[pos]

        pred_before_answer = _normalize_answer(tokenizer.decode(before_ids))
        pred_after_answer = _normalize_answer(tokenizer.decode(after_ids))

        seq_total += 1
        seq_correct_before += int(pred_before_answer == gold)
        seq_correct_after += int(pred_after_answer == gold)

        records.append(
            {
                "id": item.get("id"),
                "question": q,
                "answer": gold,
                "prompt": prompt,
                "gold_token_ids": gt_ids,
                "gold_tokens": [tokenizer.decode([i]) for i in gt_ids],
                "numeric_positions": numeric_positions,
                "pred_before_answer": pred_before_answer,
                "pred_after_answer": pred_after_answer,
                "tokens": token_rows,
            }
        )

    summary = {
        "model": args.model,
        "steps": args.steps,
        "lr": args.lr,
        "update_target": args.update_target,
        "num_layers": args.num_layers,
        "token_total": token_total,
        "token_acc_before": token_correct_before / token_total if token_total else 0.0,
        "token_acc_after": token_correct_after / token_total if token_total else 0.0,
        "seq_total": seq_total,
        "seq_acc_before": seq_correct_before / seq_total if seq_total else 0.0,
        "seq_acc_after": seq_correct_after / seq_total if seq_total else 0.0,
        "trainable": {
            "trainable_params": train_stats.trainable_params,
            "trainable_pct": train_stats.trainable_pct,
            "layer_count": train_stats.layer_count,
            "total_layers": train_stats.total_layers,
        },
    }

    out_json = out_dir / "report.json"
    out_json.write_text(json.dumps({"summary": summary, "results": records}, indent=2, ensure_ascii=False))

    out_md = out_dir / "report.md"
    write_md(out_md, records, summary)

    print("Wrote:")
    print(" -", out_json)
    print(" -", out_md)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
