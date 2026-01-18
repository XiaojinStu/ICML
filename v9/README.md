# v9（主实验版本）— 数值字符串 + ANE Test-Time Adaptation

v9 的核心设定：
- 数据集统一为 `[{id, question, answer}]`，其中 `answer` 是“数值字符串”（可包含负号/小数点/分数/科学计数法）。
- **只在“digit-only token”（纯数字 token）的位置做 TTA 与 token 级指标统计**。
- `answer` 中的分隔符/符号（如 `- . / e`）在评测时**直接 teacher-forcing**，不参与预测与优化。
  - 例：`8.21` 会按 `8`（预测+TTA）→ `.`（teacher-forcing）→ `21`（预测+TTA）来处理。

## 数据集
默认读取 `datasets_final/main/*.json`（由 `datasets_final/build_datasets.py` 生成），可用 key：
- `addition50_v1`
- `gsm8k_test500_v1`
- `bigbench_arithmetic200_v1`
- `bigbench_mixed_number_string300_seed42_v1`
- `math401_all401_v1`
- `nupa_test440_v1`
- `numericbench_all31200_v1`

也支持别名：`addition_50`, `gsm8k`, `math401`, `nupa`, `numericbench` 等（见 `v9/dataset_loader.py`）。

## 单次运行（生成结果 JSON + 可视化）
Teacher Forcing（推荐用于稳定评估 token-level 指标）：
```bash
python v9/experiment_v9.py \
  --dataset gsm8k \
  --eval_mode tf \
  --model /home/jinsk/Models/Llama-3.2-3B-Instruct \
  --steps 15 --lr 0.005 --optimizer sgd --update_target mlp+ln --num_layers all \
  --output_dir runs/v9_demo/gsm8k/Llama-3.2-3B-Instruct \
  --config_name tf_steps15_lr0.005_sgd_mlp+ln_all
```

Auto-regressive（数字 token 位置自回归，但分隔符仍 teacher-forcing）：
```bash
python v9/experiment_v9.py \
  --dataset gsm8k \
  --eval_mode ar \
  --steps 15 --lr 0.005 \
  --output_dir runs/v9_demo/gsm8k/Llama-3.2-3B-Instruct \
  --config_name ar_steps15_lr0.005
```

## 网格运行（多数据集/多设置）
```bash
python v9/run_grid_v9.py \
  --datasets addition50_v1,gsm8k_test500_v1 \
  --models /home/jinsk/Models/Llama-3.2-1B-Instruct,/home/jinsk/Models/Llama-3.2-3B-Instruct \
  --steps_list 5,15 \
  --lrs 0.005 \
  --eval_modes tf,ar \
  --output_root runs/v9_grid
```

## 结果汇总与主表/仪表盘
```bash
python v9/aggregate_v9.py --results_root runs/v9_grid --out_dir runs/v9_grid_summary
```

输出包含：
- `table_main.csv`：用于论文主表的精简字段（去掉不重要/不常变动的超参）。
- 每个 dataset/model 的多指标 dashboard（高分辨率、全框、少留白）。

## Anchor（期望 embedding）轨迹记录与可视化
默认只在 flipped token（wrong→correct）上记录 anchor 轨迹，避免 JSON 过大：
- `--anchor_log flipped|all|none`
- `--anchor_trace_max N`：最多保留 N 个 token 的 anchor_trace（按 flipped 与 rank 改善优先）

对应可视化图会生成在同一输出目录下：
- `*_anchor_traces.png`：包含
  1) `cos(anchor,target)` 随 step 变化（并标出 nearest token 变化点）
  2) embedding 空间（PCA-2D）中的 anchor 轨迹 + target/pred/topk token 对比
  3) top-k 概率分布从 step0→final 的变化

