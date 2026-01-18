#!/bin/bash
# run_nae_experiments.sh

mkdir -p results_nae

echo "======================================"
echo "NAE Test-Time Adaptation Experiments"
echo "======================================"

# Baseline
python nae_tta.py --exp_name baseline_mlp2 \
    --update_target mlp --num_layers 2 --num_samples 50

# Ablations
python nae_tta.py --exp_name abl_layers4 \
    --update_target mlp --num_layers 4 --num_samples 50

python nae_tta.py --exp_name abl_mlp_ln \
    --update_target mlp+ln --num_layers 2 --num_samples 50

python nae_tta.py --exp_name abl_scheduler \
    --update_target mlp --num_layers 2 --num_samples 50 --scheduler

echo "✓ All experiments completed"