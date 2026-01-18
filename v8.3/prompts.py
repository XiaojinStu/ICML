"""Prompt templates for v8.3.

新增：对支持 chat_template 的模型（如 Qwen）使用 apply_chat_template 构造 prompt ids。
对不支持 chat_template 的模型，保持 v7 的 Llama-style prompt 文本不变。
"""

from __future__ import annotations

from typing import List, Dict, Any


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

# Generic prompt for integer-only answer formatting (dataset-agnostic).
PROMPT_GENERIC_INT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a careful solver. Output ONLY the final answer as an integer (possibly negative), with no extra text.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please output the integer 123. Output only the integer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

123<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


PROMPT_BIGBENCH_CRT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a careful math solver. For the question, output ONLY the final answer as an integer, with no extra text.<|eot_id|><|start_header_id|>user<|end_header_id|>

There is a basket of no more than 100 bananas. If we divide them equally among 3 horses, we have 2 left; if we divide them equally among 5 horses, we have 3 left; if we divide them equally among 7 horses, we have 2 left. How many bananas are in the basket?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

23<|eot_id|><|start_header_id|>user<|end_header_id|>

There is a basket of no more than 200 bananas. If we divide them equally among 8 horses, we have 1 left; if we divide them equally among 9 horses, we have 2 left; if we divide them equally among 11 horses, we have 5 left. How many bananas are in the basket?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

137<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def build_prompt_text(dataset: str, question: str) -> str:
    dataset = dataset.strip().lower()
    if dataset in {"addition_50", "addition50", "add50"}:
        return PROMPT_ADDITION.format(question=question)
    if dataset in {"gsm8k"}:
        return PROMPT_GSM8K.format(question=question)
    if dataset in {"bigbench_crt", "bb_crt", "bigbench-chinese-remainder-theorem", "chinese_remainder_theorem", "crt"}:
        return PROMPT_BIGBENCH_CRT.format(question=question)
    if dataset in {"generic_int", "generic", "json_int"}:
        return PROMPT_GENERIC_INT.format(question=question)
    raise ValueError(f"Unknown dataset for prompts: {dataset}")


def _supports_chat_template(tokenizer) -> bool:
    try:
        return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
    except Exception:
        return False


def _fewshot_messages_addition(question: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a precise calculator. For the question, output ONLY the final integer answer, with no extra text.",
        },
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
            "content": "You are a careful math solver. Output ONLY the final answer as a (possibly negative) integer, with no extra text.",
        },
        {"role": "user", "content": "If you have 5 apples and buy 7 more, how many apples do you have?"},
        {"role": "assistant", "content": "12"},
        {"role": "user", "content": "What is 1200000 + 34567?"},
        {"role": "assistant", "content": "1234567"},
        {"role": "user", "content": question},
    ]


def _fewshot_messages_bigbench_crt(question: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a careful math solver. For the question, output ONLY the final answer as an integer, with no extra text.",
        },
        {
            "role": "user",
            "content": "There is a basket of no more than 100 bananas. If we divide them equally among 3 horses, we have 2 left; if we divide them equally among 5 horses, we have 3 left; if we divide them equally among 7 horses, we have 2 left. How many bananas are in the basket?",
        },
        {"role": "assistant", "content": "23"},
        {
            "role": "user",
            "content": "There is a basket of no more than 200 bananas. If we divide them equally among 8 horses, we have 1 left; if we divide them equally among 9 horses, we have 2 left; if we divide them equally among 11 horses, we have 5 left. How many bananas are in the basket?",
        },
        {"role": "assistant", "content": "137"},
        {"role": "user", "content": question},
    ]


def _fewshot_messages_generic_int(question: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a careful solver. Output ONLY the final answer as an integer (possibly negative), with no extra text.",
        },
        {"role": "user", "content": "Please output the integer 123. Output only the integer."},
        {"role": "assistant", "content": "123"},
        {"role": "user", "content": question},
    ]


def build_prompt_ids(tokenizer, dataset: str, question: str):
    """Return prompt input_ids (1, L).

    - 对有 chat_template 的 tokenizer（Qwen 等）：用 apply_chat_template。
    - 否则：回退到 v7 的纯文本 prompt，然后 tokenizer 编码。
    """

    dataset_key = dataset.strip().lower()

    if _supports_chat_template(tokenizer):
        if dataset_key in {"addition_50", "addition50", "add50"}:
            messages = _fewshot_messages_addition(question)
        elif dataset_key in {"gsm8k"}:
            messages = _fewshot_messages_gsm8k(question)
        elif dataset_key in {"bigbench_crt", "bb_crt", "bigbench-chinese-remainder-theorem", "chinese_remainder_theorem", "crt"}:
            messages = _fewshot_messages_bigbench_crt(question)
        elif dataset_key in {"generic_int", "generic", "json_int"}:
            messages = _fewshot_messages_generic_int(question)
        else:
            raise ValueError(f"Unknown dataset for prompts: {dataset}")

        # add_generation_prompt=True 让最后追加 assistant 起始标记，便于后续继续拼接答案 token。
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    # Fallback: Llama-style prompt
    return tokenizer(build_prompt_text(dataset, question), return_tensors="pt")["input_ids"]
