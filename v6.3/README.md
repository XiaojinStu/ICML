# v6.3（GSM8K + 解码式评估）

v6.3 在 GSM8K 上采用一种更“解码增强”（teacher-forcing）的评估方式，而不是让模型自由生成完整答案：

- 先对 *标准答案（gold）* 做分词，判断每一个位置的“下一个 token”是否应当是**数值 token**。
- 只在这些数值 token 的位置做评估/自适应（ANE/TTA）。
- 每次把 `prompt + gold_prefix_tokens` 作为输入，让模型预测下一个数值 token，并与 gold 的数值 token 直接比较。

这样可以尽量避免因为输出格式不稳定（多余文本、换行等）带来的评估噪声。

## 负号支持

支持负数答案：
- 如果 gold 以 `-` 开头，`-` 会保留在 gold 前缀里；
- 评估与对齐从 `-` 之后的数值 token 开始进行。

## Prepare GSM8K locally

```bash
python v6.3/prepare_gsm8k.py --num_samples 50
python v6.3/prepare_gsm8k.py --num_samples 500
```

## Run v6.3 experiment

```bash
python v6.3/experiment_v6.py \
  --data_path data/gsm8k/gsm8k_test_50.json \
  --num_samples 50 \
  --model /home/jinsk/Models/Llama-3.2-1B-Instruct \
  --steps 5 \
  --lr 0.001 \
  --update_target mlp \
  --num_layers all \
  --backup_on_cpu \
  --output_dir v6.3/results/demo \
  --config_name demo_tfnum
```

## One-glance report (includes prompts + tokens)

```bash
python v6.3/decoding_debug.py \
  --model /home/jinsk/Models/Llama-3.2-1B-Instruct \
  --data_path data/gsm8k/gsm8k_test_50.json \
  --num_samples 50 \
  --steps 5 \
  --lr 0.001 \
  --update_target mlp \
  --out_dir v6.3/results_decoding_debug/llama3.2-1b
```
