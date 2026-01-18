# Final Datasets (Main Experiments)
本文件夹由 `datasets_final/build_datasets.py` 从 `data/final_raw_data` 生成。
统一格式：JSON 列表，每条样本至少包含 `id/question/answer`，其中 `answer` 为整数（字符串）。

## 目录结构
- `int/`：可直接接入当前 v8.3（整数输出）主实验流水线
- `raw/`：原始数据（包含小数/分数等），需要后续扩展 float/fraction 支持

## 数据集统计（int/）
| dataset_key | filename | n | q_len_mean | q_len_p50 | neg_rate |
|---|---|---:|---:|---:|---:|
| addition50_v1 | `addition50_v1.json` | 50 | 76.0 | 76 | 0.000 |
| bigbench_arithmetic200_v1 | `bigbench_arithmetic200_v1.json` | 200 | 22.7 | 23 | 0.145 |
| bigbench_crt_test500_v1 | `bigbench_crt_test500_v1.json` | 500 | 270.5 | 270 | 0.000 |
| bigbench_mixed_number_string300_v1_seed42 | `bigbench_mixed_number_string300_v1_seed42.json` | 300 | 102.0 | 102 | 0.000 |
| gsm8k_test500_v1 | `gsm8k_test500_v1.json` | 500 | 234.4 | 219 | 0.000 |
| math401_int241_v1 | `math401_int241_v1.json` | 241 | 9.4 | 6 | 0.149 |
| nupa_test440_int160_v1 | `nupa_test440_int160_v1.json` | 160 | 189.6 | 170 | 0.000 |

## 备注
- `math401_all`/`nupa_test_440` 在 raw 里包含大量非整数答案；在 int 里我们保留了可严格视作整数的子集（例如 `3350.0000 -> 3350`）。
- `NumericBench_all` 主要是两位小数输出任务，目前只保留 raw 版本，暂不纳入整数主流水线。
