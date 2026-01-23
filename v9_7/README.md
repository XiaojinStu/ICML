# v9.7 主实验

`v9_7/run_main_v9_7.sh` 是主实验脚本：按模型/数据集使用默认超参跑 `TF`，自动保存结果、日志、表格和可视化。

- 运行：`bash v9_7/run_main_v9_7.sh`
- 汇总：`python v9_7/aggregate_v9_7.py --root runs/v9.7_main_final/results --out_dir runs/v9.7_main_final/summary`

默认输出路径：
- 结果：`runs/v9.7_main_final/results/**`
- 日志：`runs/v9.7_main_final/logs/**`
- 表格/图：`runs/v9.7_main_final/summary/**`

数据集路径：`datasets_final_v3`

