"""Prompt templates for v9.

If the tokenizer supports `chat_template` (e.g. Qwen), we use
`tokenizer.apply_chat_template(...)` to build prompt ids.
Otherwise, we fallback to Llama-style text prompts.

v9 decoding only predicts digit-only tokens; separators ('.', '/', 'e', '-') are
teacher-forced during evaluation, but we still provide strict prompts to keep the
model from producing extra text in baseline settings.
"""

from __future__ import annotations

from typing import Any, Dict, List


PROMPT_ADDITION = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise calculator. For the question, output ONLY the final answer, with no extra text.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 12 and 34?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

46<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 567 and 890?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1457<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


PROMPT_GSM8K = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a careful math solver. Output ONLY the final answer as a number, with no extra text. The answer may be an integer (possibly negative).<|eot_id|><|start_header_id|>user<|end_header_id|>

If you have 5 apples and buy 7 more, how many apples do you have?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

12<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 1200000 + 34567?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1234567<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


PROMPT_GENERIC_NUMSTR = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a careful solver. Output ONLY the final answer as a number string, with no extra text and no spaces. The answer may be an integer, decimal, fraction (like 3/7), or scientific notation (like 1.23e5).<|eot_id|><|start_header_id|>user<|end_header_id|>

Output only the number string for: 3 divided by 7.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

3/7<|eot_id|><|start_header_id|>user<|end_header_id|>

Output only the number string for: negative eight point two one.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

-8.21<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def _supports_chat_template(tokenizer) -> bool:
    try:
        return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
    except Exception:
        return False


def _dataset_family(dataset: str) -> str:
    d = dataset.strip().lower()
    if d in {"addition_50", "addition50", "add50"} or d.startswith("addition"):
        return "addition"
    if d == "gsm8k" or "gsm8k" in d:
        return "gsm8k"
    return "generic"


def build_prompt_text(dataset: str, question: str) -> str:
    fam = _dataset_family(dataset)
    if fam == "addition":
        return PROMPT_ADDITION.format(question=question)
    if fam == "gsm8k":
        return PROMPT_GSM8K.format(question=question)
    return PROMPT_GENERIC_NUMSTR.format(question=question)


def _fewshot_messages_addition(question: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a precise calculator. Output ONLY the final answer, with no extra text."},
        {"role": "user", "content": "What is the sum of 12 and 34?"},
        {"role": "assistant", "content": "46"},
        {"role": "user", "content": "What is the sum of 567 and 890?"},
        {"role": "assistant", "content": "1457"},
        {"role": "user", "content": question},
    ]


def _fewshot_messages_gsm8k(question: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a careful math solver. Output ONLY the final answer as a number, with no extra text.",
        },
        {"role": "user", "content": "If you have 5 apples and buy 7 more, how many apples do you have?"},
        {"role": "assistant", "content": "12"},
        {"role": "user", "content": "What is 1200000 + 34567?"},
        {"role": "assistant", "content": "1234567"},
        {"role": "user", "content": question},
    ]


def _fewshot_messages_generic(question: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a careful solver. Output ONLY the final answer as a number string, with no extra text and no spaces. The answer may be an integer, decimal, fraction (like 3/7), or scientific notation (like 1.23e5).",
        },
        {"role": "user", "content": "Output only the number string for: 3 divided by 7."},
        {"role": "assistant", "content": "3/7"},
        {"role": "user", "content": "Output only the number string for: negative eight point two one."},
        {"role": "assistant", "content": "-8.21"},
        {"role": "user", "content": question},
    ]


def build_prompt_ids(tokenizer, dataset: str, question: str):
    """Return prompt input_ids (1, L)."""

    if _supports_chat_template(tokenizer):
        fam = _dataset_family(dataset)
        if fam == "addition":
            messages = _fewshot_messages_addition(question)
        elif fam == "gsm8k":
            messages = _fewshot_messages_gsm8k(question)
        else:
            messages = _fewshot_messages_generic(question)
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    return tokenizer(build_prompt_text(dataset, question), return_tensors="pt")["input_ids"]

