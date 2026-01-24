## v9.7 默认 setting

目标：在不改模型结构的前提下，对“答案中的数字 token 子空间”做少量 test-time adaptation（ANE-TTA），提升数值输出的确定性与准确率。

### 指标记录

- `token_acc@{0,1,2,5,10,20}`：teacher forcing 下，digit-only token 在指定 step 的 top-1 准确率  
- `pass@2 / pass@5`：digit-only token 在 top-k 里命中目标的比例  
- `EM`：最终答案字符串完全一致（该脚本默认跑 TF，EM 仅作参考）  

### 默认超参（主实验）

```yaml
common:
  eval_mode: tf
  steps_total: 30
  report_steps: [0, 1, 2, 5, 10, 15, 20, 25, 30]
  tta_reset: token
  optimizer: sgd
  topk_list: [2, 5]
  ane_metric: angle
  update_target_llama: mlp+ln
  update_target_qwen: ln

llama:
  1B/3B:
    lr: 0.01
    lr_schedule: constant
    lr_norm: none
  8B:
    lr: 0.005
    lr_schedule: constant
    lr_norm: none
  bigbench_mixed_number_string:
    lr_schedule: cosine
    lr_min: 1e-4
    lr_norm: grad_norm

qwen:
  addition50: 0.004
  bigbench_arithmetic: 0.001
  bigbench_mixed_number_string: 0.004
  math401: 0.002
  numericbench: 0.002
  nupa: 0.002
  gsm8k: 5e-05
```
