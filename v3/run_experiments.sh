#!/bin/bash
# ANE v3 Production Experiments
# Full experiments with optimal configuration

cd "$(dirname "$0")"

DATA_PATH="../data/addition_problems_dataset(1-50)(1).json"
OUTPUT_DIR="results_ane"
SAMPLES=50

# Optimal configuration (based on exploration)
# Best: attn+ln with lr=0.0005 showed +8.64% improvement
UPDATE_TARGET="attn+ln"
LR=0.0005
OPTIMIZER="sgd"

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "ANE v3 Production Experiments"
echo "=========================================="
echo "Samples: $SAMPLES"
echo "Update Target: $UPDATE_TARGET"
echo "Learning Rate: $LR"
echo "Optimizer: $OPTIMIZER"
echo "Output: $OUTPUT_DIR"
echo ""

# ===========================================
# Llama-3.2-1B Experiments
# ===========================================
echo "=== Llama-3.2-1B Experiments ==="
MODEL_1B="/home/jinsk/Models/Llama-3.2-1B-Instruct"

for steps in 5 30; do
    echo ""
    echo ">>> 1B Model, $steps steps"
    python ane_tta.py \
        --models "$MODEL_1B" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target $UPDATE_TARGET \
        --num_layers all \
        --steps $steps \
        --lr $LR \
        --optimizer $OPTIMIZER \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "llama1b_steps${steps}"
done

# ===========================================
# Llama-3.2-3B Experiments
# ===========================================
echo ""
echo "=== Llama-3.2-3B Experiments ==="
MODEL_3B="/home/jinsk/Models/Llama-3.2-3B-Instruct"

for steps in 5 30; do
    echo ""
    echo ">>> 3B Model, $steps steps"
    python ane_tta.py \
        --models "$MODEL_3B" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target $UPDATE_TARGET \
        --num_layers all \
        --steps $steps \
        --lr $LR \
        --optimizer $OPTIMIZER \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "llama3b_steps${steps}"
done

# ===========================================
# Llama-3.1-8B Experiments (memory optimized with 16 layers)
# ===========================================
echo ""
echo "=== Llama-3.1-8B Experiments (16 layers for memory efficiency) ==="
MODEL_8B="/home/jinsk/Models/Llama-3.1-8B-Instruct"

for steps in 5 30; do
    echo ""
    echo ">>> 8B Model, $steps steps (16 layers)"
    python ane_tta.py \
        --models "$MODEL_8B" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target $UPDATE_TARGET \
        --num_layers 16 \
        --steps $steps \
        --lr $LR \
        --optimizer $OPTIMIZER \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "llama8b_steps${steps}"
done

# ===========================================
# Summary
# ===========================================
echo ""
echo "=========================================="
echo "All Production Experiments Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR/"
