ANE-TTA v5

Overview
- Full pipeline for ANE test-time adaptation with teacher forcing (TF) and autoregressive (AR) evaluation
- Batch runner with JSON configs, plus summary tables and plots
- Publication-grade visualizations (heatmaps, case evolution, statistics)

Key entry points
- Single run: `python ane_tta.py --exp_name demo`
- Batch run: `python batch_run.py --config configs/main_experiments.json --output_root results_batch`
- Summary: `python summary.py --input_dir results_batch --output_dir results_batch/summary`

Scripts
- `run_all.sh`: main experiments (1B/3B/8B) + summary
- `run_sweep_3b.sh`: 3B parameter sweep + summary
- `run_efficiency_sweep.sh`: efficiency-focused sweep (steps/lr/targets)

Metrics (per mode)
- token_acc, token_acc@k (rank <= k)
- seq_acc (all tokens correct), seq_acc@k
- mean rank, median rank, MRR, mean target prob

Modes
- tf: teacher forcing baseline
- tf_tta: teacher forcing + TTA
- ar: autoregressive baseline
- ar_tta: autoregressive + TTA (optional)

Notes
- For 8B, consider `layer_stride=2`, `reset_mode=per_sample`, `gradient_checkpointing=true`
- `fast_eval` and `early_stop` can improve throughput with modest impact on quality
