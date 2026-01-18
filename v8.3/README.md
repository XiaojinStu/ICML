# v8.3（主实验代码：Greedy vs Greedy+ANE）

v8.3 主要用于主实验的“批量跑 + 指标统计 + 学术可视化”。当前已接入：
- `bigbench_crt`：BIG-bench `chinese_remainder_theorem`（本地 `data/Big-bench_chinese_remainder_theorem.json` 或转换后的 `data/bigbench_chinese_remainder_theorem_500.json`）
- `addition_50`：加法小数据集（本地 `data/addition_problems_dataset(1-50)(1).json`）
- `gsm8k`：GSM8K（本地 `data/gsm8k/gsm8k_test_*.json`）

## 数据格式（统一接口）

建议统一为 JSON 列表，每条样本至少包含：
```json
{"id": "optional", "question": "...", "answer": "13"}
```
其中 `answer` 为整数（字符串或数字皆可，支持 `-13`）。

BIG-bench 的原始格式（dict + examples）也支持直接读取。

## 预处理（BIG-bench CRT）

把 BIG-bench 原始 JSON 转换成统一的 JSON 列表：
```bash
python v8.3/preprocess_bigbench_crt.py \
  --input data/Big-bench_chinese_remainder_theorem.json \
  --output data/bigbench_chinese_remainder_theorem_500.json
```

## 单次运行（推荐先跑小规模自检）

### 1) CRT-50（teacher forcing）
```bash
python v8.3/experiment_v8_3.py \
  --dataset bigbench_crt \
  --data_path data/bigbench_chinese_remainder_theorem_500.json \
  --num_samples 50 \
  --eval_mode tf \
  --model /home/jinsk/Models/Llama-3.2-3B-Instruct \
  --steps 15 --lr 0.001 \
  --update_target mlp+ln --num_layers all --tta_reset token \
  --output_dir v8.3/results_demo/bigbench_crt/Llama-3.2-3B-Instruct \
  --config_name demo_crt_tf
```

### 2) CRT-50（autoregressive，用于 EM / Digit Acc）
```bash
python v8.3/experiment_v8_3.py \
  --dataset bigbench_crt \
  --data_path data/bigbench_chinese_remainder_theorem_500.json \
  --num_samples 50 \
  --eval_mode ar \
  --model /home/jinsk/Models/Llama-3.2-3B-Instruct \
  --steps 15 --lr 0.001 \
  --update_target mlp+ln --num_layers all --tta_reset token \
  --output_dir v8.3/results_demo/bigbench_crt/Llama-3.2-3B-Instruct \
  --config_name demo_crt_ar
```

> 如果使用 Qwen（chat_template），单次脚本需要额外加 `--trust_remote_code`。

## 批量跑主实验（网格）

建议网格跑时先 `--no_viz`（避免生成成千上万张图），跑完再用聚合脚本挑“最好”的配置生成可视化。

```bash
python v8.3/run_grid_v8_3.py \
  --datasets bigbench_crt \
  --bigbench_crt_path data/bigbench_chinese_remainder_theorem_500.json \
  --steps_list 5,10,15 \
  --lr_list 0.001,0.0005 \
  --eval_modes tf,ar \
  --topk_list 2,5 \
  --update_target mlp+ln --num_layers all --tta_reset token \
  --output_root v8.3/results_crt_main \
  --no_viz
```

## 结果汇总与可视化

```bash
python v8.3/aggregate_v8_3.py \
  --root v8.3/results_crt_main \
  --out_dir v8.3/summary_crt_main \
  --render_viz_top_n 1
```

输出包含：
- `v8.3/summary_crt_main/summary_runs.csv`：每个 run 一行（含 TF/AR）
- `v8.3/summary_crt_main/table_main.csv`：合并 TF+AR 的“论文表格”字段
- `v8.3/summary_crt_main/plots/*.png`：steps×lr 的网格热力图
- `v8.3/summary_crt_main/selected_viz/**`：每个模型/模式挑 Top-1 的详细可视化（含 heatmap，且每张 heatmap ≤80 tokens）
