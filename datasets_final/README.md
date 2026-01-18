# Final Datasets (Main Experiments)
本文件夹由 `datasets_final/build_datasets.py` 从 `data/final_raw_data` 生成。
统一格式：JSON 列表，每条样本至少包含 `id/question/answer`。
其中 `answer` 是“数值字符串”，可能包含小数点/分数/科学计数法。

## 目录结构
- `main/`：主实验直接使用（统一 schema）
- `raw/`：原始数据快照（对齐来源，便于溯源）

## 数据集统计（main/）
| dataset_key | filename | n | q_len_p50 | ans_chars_p50 | ans_digits_p50 | neg | dot | slash | sci |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| addition50_v1 | `addition50_v1.json` | 50 | 76 | 26 | 26 | 0.000 | 0.000 | 0.000 | 0.000 |
| bigbench_arithmetic200_v1 | `bigbench_arithmetic200_v1.json` | 200 | 23 | 3 | 3 | 0.145 | 0.000 | 0.000 | 0.000 |
| bigbench_mixed_number_string300_seed42_v1 | `bigbench_mixed_number_string300_seed42_v1.json` | 300 | 102 | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| gsm8k_test500_v1 | `gsm8k_test500_v1.json` | 500 | 219 | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| math401_all401_v1 | `math401_all401_v1.json` | 401 | 8 | 6 | 5 | 0.150 | 0.499 | 0.000 | 0.000 |
| numericbench_all31200_v1 | `numericbench_all31200_v1.json` | 31200 | 104 | 7 | 6 | 0.497 | 1.000 | 0.000 | 0.000 |
| nupa_test440_v1 | `nupa_test440_v1.json` | 440 | 182 | 21 | 20 | 0.000 | 0.455 | 0.182 | 0.227 |

## 备注
- v9 实验会只在“digit-only token”的位置做 TTA；非数字 token（如 `.`, `/`, `e`）视为分隔符并在评测时 teacher-forcing。
