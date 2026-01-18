"""Prompt templates for v7.

Design goals:
- Keep output format strict: ONLY an integer, no extra text.
- Provide light few-shot examples to stabilize smaller models.
"""

from __future__ import annotations


PROMPT_ADDITION = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise calculator. For the question, output ONLY the final integer answer, with no extra text.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 12 and 34?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

46<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the sum of 567 and 890?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1457<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


PROMPT_GSM8K = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a careful math solver. Output ONLY the final answer as a (possibly negative) integer, with no extra text.<|eot_id|><|start_header_id|>user<|end_header_id|>

If you have 5 apples and buy 7 more, how many apples do you have?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

12<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 1200000 + 34567?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

1234567<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def build_prompt(dataset: str, question: str) -> str:
    dataset = dataset.strip().lower()
    if dataset in {"addition_50", "addition50", "add50"}:
        return PROMPT_ADDITION.format(question=question)
    if dataset in {"gsm8k"}:
        return PROMPT_GSM8K.format(question=question)
    raise ValueError(f"Unknown dataset for prompts: {dataset}")

