"""Prompt/debug runner for GSM8K.

Goal: verify that the prompt makes the model output ONLY an integer (format compliance),
and record everything needed to inspect I/O clearly:
- input question
- full prompt string
- generated token ids + per-token decoded strings
- raw generated text + normalized numeric extraction

This uses `model.generate` (unmasked) on purpose to test instruction following.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k_dataset import load_local_json


PROMPT_V6 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a careful math solver. For the question, output ONLY the final answer as a non-negative integer, with no extra text.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


PROMPT_V6_2_ZERO = PROMPT_V6

PROMPT_V6_2_FEWSHOT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise calculator.
For each question, output ONLY the final answer as a non-negative integer.
Do NOT output any words, reasoning, units, commas, equations (like '+', '-', '*', '/', '=') or extra lines.<|eot_id|><|start_header_id|>user<|end_header_id|>

If you have 5 apples and buy 7 more, how many apples do you have?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

12<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 1200000 + 34567?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1234567<|eot_id|><|start_header_id|>user<|end_header_id|>

A store sold 18 candies in the morning and 24 in the afternoon. How many candies did it sell in total?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

42<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def _select_prompt_v6_2(style: str, model_path: str) -> str:
    if style == "zero":
        return PROMPT_V6_2_ZERO
    if style == "fewshot":
        return PROMPT_V6_2_FEWSHOT
    if style == "auto":
        return PROMPT_V6_2_FEWSHOT if "1b" in model_path.lower() else PROMPT_V6_2_ZERO
    raise ValueError(f"Unknown v6.2 style: {style}")


_STRICT_INT_RE = re.compile(r"^\s*[0-9]+\s*$")
_FIRST_INT_RE = re.compile(r"([0-9]+)")


@dataclass(frozen=True)
class GenResult:
    token_ids: List[int]
    tokens: List[str]
    raw_text: str
    clean_text: str
    strict_int: bool
    extracted_int: Optional[str]


def _normalize_answer(text: str) -> str:
    s = str(text).strip()
    s = s.replace(",", "")
    s = s.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
    if s.startswith("+"):
        s = s[1:]
    if s.endswith("."):
        s = s[:-1]
    return s


def _analyze_output(text: str) -> Tuple[bool, Optional[str]]:
    strict = bool(_STRICT_INT_RE.match(text))
    m = _FIRST_INT_RE.search(text)
    extracted = m.group(1) if m else None
    return strict, extracted


def build_prompt(template: str, question: str) -> str:
    return template.format(question=question)


@torch.inference_mode()
def generate_once(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> GenResult:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )[0]

    new_ids = out_ids[input_ids.shape[1] :].tolist()
    new_tokens = [tokenizer.decode([i]) for i in new_ids]

    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    clean = tokenizer.decode(new_ids, skip_special_tokens=True)

    strict, extracted = _analyze_output(clean)
    return GenResult(
        token_ids=new_ids,
        tokens=new_tokens,
        raw_text=raw,
        clean_text=clean,
        strict_int=strict,
        extracted_int=extracted,
    )


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    strict = sum(1 for r in records if r["gen"]["strict_int"])
    correct_strict = sum(1 for r in records if r["gen"]["strict_int"] and r["correct"])
    extracted_correct = sum(1 for r in records if r["extracted_correct"])
    return {
        "total": n,
        "strict_int_rate": strict / n if n else 0.0,
        "strict_and_correct_rate": correct_strict / n if n else 0.0,
        "extracted_correct_rate": extracted_correct / n if n else 0.0,
    }


def write_markdown(
    *,
    path: Path,
    model_name: str,
    data_path: str,
    max_new_tokens: int,
    v6_2_style: str,
    results: Dict[str, Any],
) -> None:
    v6 = results["v6"]
    v62 = results["v6_2"]
    lines = []
    lines.append(f"# GSM8K Prompt Debug\n")
    lines.append(f"- model: `{model_name}`\n")
    lines.append(f"- data: `{data_path}`\n")
    lines.append(f"- max_new_tokens: `{max_new_tokens}`\n")
    lines.append(f"- v6.2 style: `{v6_2_style}`\n")
    lines.append("\n## Summary\n")
    lines.append("### v6\n")
    lines.append(f"- total: {v6['summary']['total']}\n")
    lines.append(f"- strict_int_rate: {v6['summary']['strict_int_rate']:.3f}\n")
    lines.append(f"- extracted_correct_rate: {v6['summary']['extracted_correct_rate']:.3f}\n")
    lines.append("\n### v6.2\n")
    lines.append(f"- total: {v62['summary']['total']}\n")
    lines.append(f"- strict_int_rate: {v62['summary']['strict_int_rate']:.3f}\n")
    lines.append(f"- extracted_correct_rate: {v62['summary']['extracted_correct_rate']:.3f}\n")

    def _one_block(title: str, rec: Dict[str, Any]) -> None:
        q = rec["question"]
        gold = rec["gold_answer"]
        gen = rec["gen"]
        lines.append(f"\n#### {title} | id={rec.get('id')}\n")
        lines.append(f"- gold: `{gold}`\n")
        lines.append(f"- output_clean: `{gen['clean_text']}`\n")
        lines.append(f"- strict_int: `{gen['strict_int']}`\n")
        lines.append(f"- extracted_int: `{gen['extracted_int']}`\n")
        lines.append(f"- extracted_correct: `{rec['extracted_correct']}`\n")
        lines.append("\n**Question**\n")
        lines.append("```text\n" + q.strip() + "\n```\n")
        lines.append("\n**Prompt**\n")
        lines.append("```text\n" + rec["prompt"].strip() + "\n```\n")
        lines.append("\n**Generated Token IDs**\n")
        lines.append("```text\n" + str(gen["token_ids"]) + "\n```\n")
        lines.append("\n**Generated Tokens**\n")
        lines.append("```text\n" + "|".join(gen["tokens"]) + "\n```\n")
        lines.append("\n**Generated Raw Text**\n")
        lines.append("```text\n" + gen["raw_text"] + "\n```\n")

    lines.append("\n## Detailed Results (all samples)\n")
    for i, pair in enumerate(zip(v6["records"], v62["records"]), start=1):
        rec6, rec62 = pair
        lines.append(f"\n### Sample {i}\n")
        _one_block("v6", rec6)
        _one_block("v6.2", rec62)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GSM8K prompt I/O debug for v6 vs v6.2")
    p.add_argument("--model", default="/home/jinsk/Models/Llama-3.2-1B-Instruct")
    p.add_argument("--data_path", default="data/gsm8k/gsm8k_test_50.json")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--out_dir", default="v6.2/results_prompt_debug")
    p.add_argument("--v6_2_style", choices=["auto", "zero", "fewshot"], default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_local_json(args.data_path)[: args.num_samples]

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device_map)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = True

    def _run(template: str) -> List[Dict[str, Any]]:
        records = []
        for item in data:
            q = item["question"]
            gold = str(item["answer"])
            prompt = build_prompt(template, q)
            gen = generate_once(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=args.max_new_tokens)
            gold_n = _normalize_answer(gold)
            extracted_n = _normalize_answer(gen.extracted_int) if gen.extracted_int is not None else None

            extracted_correct = extracted_n == gold_n if extracted_n is not None else False
            correct = gen.strict_int and extracted_correct

            records.append(
                {
                    "id": item.get("id"),
                    "question": q,
                    "gold_answer": gold,
                    "prompt": prompt,
                    "gen": {
                        "token_ids": gen.token_ids,
                        "tokens": gen.tokens,
                        "raw_text": gen.raw_text,
                        "clean_text": gen.clean_text,
                        "strict_int": gen.strict_int,
                        "extracted_int": gen.extracted_int,
                    },
                    "extracted_correct": extracted_correct,
                    "correct": correct,
                }
            )
        return records

    v6_records = _run(PROMPT_V6)
    v62_template = _select_prompt_v6_2(args.v6_2_style, args.model)
    v62_records = _run(v62_template)

    results = {
        "v6": {"summary": _summarize(v6_records), "records": v6_records},
        "v6_2": {"summary": _summarize(v62_records), "records": v62_records},
    }

    json_path = out_dir / "prompt_debug.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    md_path = out_dir / "prompt_debug.md"
    write_markdown(
        path=md_path,
        model_name=args.model,
        data_path=args.data_path,
        max_new_tokens=args.max_new_tokens,
        v6_2_style=args.v6_2_style,
        results=results,
    )

    print("Wrote:")
    print(" -", json_path)
    print(" -", md_path)
    print("\nSummary:")
    print("v6   ", results["v6"]["summary"])
    print("v6.2 ", results["v6_2"]["summary"])


if __name__ == "__main__":
    main()
