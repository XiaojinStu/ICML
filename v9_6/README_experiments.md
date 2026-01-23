## v9.6 主实验设置

ANE 的 test-time adaptation：每个答案里的“数字 token”在 logits 上做少量梯度步，让数值分布更确定，从而提高数值输出准确率。实验默认跑 teacher forcing，并记录 step=0/2/5/10 的 token-level 准确率。

### 默认设置

下面记录默认 setting，对照不同数据集/模型套用。

```yaml
common:
  eval_mode: tf
  steps: 10
  track_steps: [0, 2, 5, 10]
  tta_reset: token
  update_target: mlp+ln
  num_layers: all
  optimizer: sgd
  ane_metric: cosine

llama_non_gsm:
  lr: 0.01
  lr_schedule: constant
  lr_norm: none

llama_mixed_robust:
  dataset: bigbench_mixed_number_string300_seed42_v1
  lr: 0.01
  lr_schedule: cosine
  lr_min: 1e-4
  lr_norm: grad_norm

qwen_non_gsm:
  lr: 0.002
  update_target: ln
  lr_schedule: constant
  lr_norm: none
  gradient_checkpointing: true

gsm8k_all_models:
  lr: 1e-4
  lr_schedule: constant
  lr_norm: none
  note: 8B 可以额外试 2e-4
```

### 还可以怎么继续涨分

- 对加法和算术类：step 常在 2 或 5 就接近饱和，优先把 sweep 放在 lr 上。
- 对 mixed-number-string：默认更容易负迁移，先用 `cosine + grad_norm`，不稳就把 `update_target` 降到 `ln`。
- 对 GSM8K：目前基本不动，可能是推理瓶颈，先别投入太多网格，最多对 8B 小范围试 `lr=2e-4`。
