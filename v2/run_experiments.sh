#!/bin/bash
# NAE-TTA v2 Experiment Runner
# Run various ablation experiments

set -e  # Exit on error

# Base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
MODEL="/home/jinsk/Models/Llama-3.1-8B-Instruct"
DATA="../data/addition_problems_dataset(1-50)(1).json"
OUTPUT_DIR="results_nae"
NUM_SAMPLES=50

echo "=================================================="
echo "NAE-TTA v2 Experiments"
echo "=================================================="
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "=================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Experiment 1: Baseline - MLP, 2 layers
# ============================================================
echo ""
echo "[1/5] Running baseline experiment (MLP, 2 layers)..."
python nae_tta.py \
    --models "$MODEL" \
    --data_path "$DATA" \
    --num_samples "$NUM_SAMPLES" \
    --update_target mlp \
    --num_layers 'all' \
    --steps 20 \
    --lr 1e-3 \
    --output_dir "$OUTPUT_DIR" \
    --exp_name baseline_mlp2

# ============================================================
# Experiment 2: Ablation - 4 layers
# ============================================================
echo ""
echo "[2/5] Running ablation: 4 layers..."
python nae_tta.py \
    --models "$MODEL" \
    --data_path "$DATA" \
    --num_samples "$NUM_SAMPLES" \
    --update_target mlp \
    --num_layers 4 \
    --steps 20 \
    --lr 1e-3 \
    --output_dir "$OUTPUT_DIR" \
    --exp_name abl_layers4

# ============================================================
# Experiment 3: Ablation - MLP + LayerNorm
# ============================================================
echo ""
echo "[3/5] Running ablation: MLP + LayerNorm..."
python nae_tta.py \
    --models "$MODEL" \
    --data_path "$DATA" \
    --num_samples "$NUM_SAMPLES" \
    --update_target mlp+ln \
    --num_layers 2 \
    --steps 20 \
    --lr 1e-3 \
    --output_dir "$OUTPUT_DIR" \
    --exp_name abl_mlp_ln

# ============================================================
# Experiment 4: Ablation - With scheduler
# ============================================================
echo ""
echo "[4/5] Running ablation: With cosine scheduler..."
python nae_tta.py \
    --models "$MODEL" \
    --data_path "$DATA" \
    --num_samples "$NUM_SAMPLES" \
    --update_target mlp \
    --num_layers 2 \
    --steps 20 \
    --lr 1e-3 \
    --scheduler \
    --output_dir "$OUTPUT_DIR" \
    --exp_name abl_scheduler

# ============================================================
# Experiment 5: Ablation - More steps
# ============================================================
echo ""
echo "[5/5] Running ablation: 40 steps..."
python nae_tta.py \
    --models "$MODEL" \
    --data_path "$DATA" \
    --num_samples "$NUM_SAMPLES" \
    --update_target mlp \
    --num_layers 2 \
    --steps 40 \
    --lr 1e-3 \
    --output_dir "$OUTPUT_DIR" \
    --exp_name abl_steps40

# ============================================================
# Summary
# ============================================================
echo ""
echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No JSON files found"
echo ""
echo "Generated visualizations:"
ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || echo "No PNG files found"
echo "=================================================="
