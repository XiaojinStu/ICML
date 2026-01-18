# v6 (GSM8K)

This folder contains a GSM8K adapter + v6 experiment runner for ANE test-time adaptation.

## 1) Prepare GSM8K (local JSON)

Creates `data/gsm8k/gsm8k_test_500.json` (integer-only answers extracted from `#### ...`):

```bash
python v6/prepare_gsm8k.py --num_samples 500 --seed 42
```

## 2) Run a single experiment

Example (3B, all MLP, 500 samples, steps=10):

```bash
python v6/experiment_v6.py \
  --data_path data/gsm8k/gsm8k_test_500.json \
  --num_samples 500 \
  --model /home/jinsk/Models/Llama-3.2-3B-Instruct \
  --update_target mlp \
  --num_layers all \
  --steps 10 \
  --lr 0.001 \
  --eval_mode ar \
  --tta_reset sample \
  --backup_on_cpu \
  --output_dir v6/results/gsm8k500/llama3.2-3b/demo \
  --config_name gsm8k500_llama3.2-3b_mlp_all_steps10_lr1e-3
```

## 3) Run the full grid (LR sweep + steps sweep)

Runs:
- models: 1B / 3B / 8B
- update targets: `mlp`, `ln` (all layers)
- steps: `5,10,30`
- LR candidates (search at steps=10): `1e-4,3e-4,1e-3,3e-3`

```bash
python v6/run_gsm8k_grid.py
```

Outputs are written to `v6/results/gsm8k500/`, and an aggregate summary to `v6/results/gsm8k500/summary/`.

