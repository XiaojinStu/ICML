# v7（主实验代码：Greedy vs Greedy+ANE）

v7 作为主实验的统一运行版本，整合了：
- `v6.3` 的数据集接入思路（本地 JSON 格式 + GSM8K 适配）；
- `v5.x` 的 token 级别记录与论文风格可视化（heatmap / flipped curves / subspace evolution 等）。

当前版本先接入两套数据集（其余先预留接口）：
- **算数数据集**：`addition_50`（本地 `data/addition_problems_dataset(1-50)(1).json`）
- **数学数据集**：`gsm8k`（本地 `data/gsm8k/gsm8k_test_*.json`）

## 数据格式（未来其它数据集也按此格式接入）

统一为 JSON 列表，每条样本至少包含：
```json
{"id": "optional", "question": "...", "answer": "13"}
```
其中 `answer` 为整数（字符串或数字皆可，支持 `-13`）。

## 默认实验设置（可通过 CLI 覆盖）

- model：`/home/jinsk/Models/Llama-3.2-3B-Instruct`
- eval_mode：`tf`（teacher forcing）
- steps：`5`
- lr：`0.005`
- update_target：`mlp+ln`
- num_layers：`all`

## 运行示例

### 1) Addition-50（快速 sanity check）
```bash
python v7/experiment_v7.py \
  --dataset addition_50 \
  --num_samples 50 \
  --eval_mode tf \
  --output_dir v7/results_demo/addition50 \
  --config_name demo_addition50_tf
```

### 2) GSM8K-50（teacher forcing）
```bash
python v7/experiment_v7.py \
  --dataset gsm8k \
  --data_path data/gsm8k/gsm8k_test_50.json \
  --num_samples 50 \
  --eval_mode tf \
  --output_dir v7/results_demo/gsm8k50 \
  --config_name demo_gsm8k50_tf
```

### 3) Autoregressive（用于 EM 指标）
```bash
python v7/experiment_v7.py \
  --dataset gsm8k \
  --data_path data/gsm8k/gsm8k_test_50.json \
  --num_samples 50 \
  --eval_mode ar \
  --output_dir v7/results_demo/gsm8k50 \
  --config_name demo_gsm8k50_ar
```

运行结束会保存：
- `<output_dir>/<config_name>.json`：summary + per-sample 详细结果
- `<output_dir>/*.png`：可视化图（可用 `--no_viz` 关闭）

