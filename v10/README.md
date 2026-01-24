# v10 实验代码（主实验 + 消融 + 敏感性）

## 结论性建议（std）

主实验可以用 **跨随机种子重复运行** 来报均值±std。`3` 次是 ICML 常见的最小配置；如果方差明显或 reviewer 盯稳定性，建议补到 `5` 次。

## 实验清单（精炼版）

主实验（Main）
- 任务：所有数据集 × 所有模型，`steps=30`，记录 `token_acc@{0,1,2,5,10,15,20,25,30}`、`pass@{2,5}`、`EM`、`digit_acc`、runtime
- 设置：`ane_metric=angle`，并额外跑一套 `cosine` 对比
- 重复：`seed={0,1,2}`（可改）
- 汇总：输出 per-run 表 + 按 seed 聚合后的 mean/std 表

消融（Ablation，建议用 Llama-3.2-3B）
- `update_target`: `ln` vs `mlp` vs `mlp+ln`
- `lr_norm`: `none` vs `grad_norm`
- `lr_schedule`: `constant` vs `cosine`
- `ane_metric`: `angle` vs `cosine`

敏感性（Sensitivity，建议用 Llama-3.2-3B）
- `lr` sweep：围绕默认值做 3~5 个点
- `steps` sweep：`10/20/30`（并结合 best-step 观察是否“过优化”）
- `num_layers`：`all` vs `8` vs `4`（最后 N 层）

## 运行方式

主实验：
- `bash v10/run_main_v10.sh`

消融：
- `bash v10/run_ablation_v10.sh`

敏感性：
- `bash v10/run_sensitivity_v10.sh`

汇总（主实验/消融/敏感性都可用）：
- `python v10/aggregate_v10.py --root <results_root> --out_dir <summary_dir>`

