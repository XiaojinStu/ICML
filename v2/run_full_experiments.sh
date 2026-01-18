#!/bin/bash
# NAE-TTA v2.1 Full Experiment Suite
# Tests 3 models x 3 step configurations = 9 experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Models
MODEL_1B="/home/jinsk/Models/Llama-3.2-1B-Instruct"
MODEL_3B="/home/jinsk/Models/Llama-3.2-3B-Instruct"
MODEL_8B="/home/jinsk/Models/Llama-3.1-8B-Instruct"

# Data
DATA="../data/addition_problems_dataset(1-50)(1).json"
OUTPUT_DIR="results_nae"
NUM_SAMPLES=50

echo "=================================================="
echo "NAE-TTA v2.1 Full Experiment Suite"
echo "=================================================="
echo "Models: 1B, 3B, 8B"
echo "Steps: 5, 20, 50"
echo "Samples: $NUM_SAMPLES"
echo "Total experiments: 9"
echo "=================================================="

mkdir -p "$OUTPUT_DIR"

# Function to run experiment
run_exp() {
    local model=$1
    local model_name=$2
    local steps=$3
    local exp_name="${model_name}_steps${steps}"

    echo ""
    echo "=================================================="
    echo "Running: $exp_name"
    echo "Model: $model"
    echo "Steps: $steps"
    echo "=================================================="

    python nae_tta.py \
        --models "$model" \
        --data_path "$DATA" \
        --num_samples "$NUM_SAMPLES" \
        --update_target mlp \
        --num_layers 2 \
        --steps "$steps" \
        --lr 1e-3 \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "$exp_name"

    echo "Completed: $exp_name"
}

# ============================================================
# 1B Model Experiments
# ============================================================
echo ""
echo "========== Llama-3.2-1B-Instruct =========="
run_exp "$MODEL_1B" "llama1b" 5
run_exp "$MODEL_1B" "llama1b" 20
run_exp "$MODEL_1B" "llama1b" 50

# ============================================================
# 3B Model Experiments
# ============================================================
echo ""
echo "========== Llama-3.2-3B-Instruct =========="
run_exp "$MODEL_3B" "llama3b" 5
run_exp "$MODEL_3B" "llama3b" 20
run_exp "$MODEL_3B" "llama3b" 50

# ============================================================
# 8B Model Experiments
# ============================================================
echo ""
echo "========== Llama-3.1-8B-Instruct =========="
run_exp "$MODEL_8B" "llama8b" 5
run_exp "$MODEL_8B" "llama8b" 20
run_exp "$MODEL_8B" "llama8b" 50

# ============================================================
# Summary
# ============================================================
echo ""
echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
echo ""
echo "Results summary:"
for f in "$OUTPUT_DIR"/*.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .json)
        acc_before=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['summary']['accuracy_before']*100:.2f}%\")")
        acc_after=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['summary']['accuracy_after']*100:.2f}%\")")
        improvement=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['summary']['improvement']*100:+.2f}%\")")
        echo "  $name: Before=$acc_before, After=$acc_after, Change=$improvement"
    fi
done
echo "=================================================="
