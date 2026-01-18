#!/bin/bash
# ANE v3 Parameter Exploration Script
# Uses 3B model with 10 samples for quick validation

cd "$(dirname "$0")"

MODEL="/home/jinsk/Models/Llama-3.2-3B-Instruct"
DATA_PATH="../data/addition_problems_dataset(1-50)(1).json"
OUTPUT_DIR="results_exploration"
SAMPLES=10  # Quick validation
BASE_STEPS=20

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "ANE v3 Parameter Exploration"
echo "=========================================="
echo "Model: $MODEL"
echo "Samples: $SAMPLES"
echo "Output: $OUTPUT_DIR"
echo ""

# ===========================================
# Phase 1: Update Target Comparison
# ===========================================
echo "=== Phase 1: Update Target ==="
echo "Testing: mlp, ln, mlp+ln, attn, attn+ln"

for target in mlp ln mlp+ln attn attn+ln; do
    echo ""
    echo ">>> Running: update_target=$target"
    python ane_tta.py \
        --models "$MODEL" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target $target \
        --num_layers all \
        --steps $BASE_STEPS \
        --lr 0.001 \
        --optimizer sgd \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "explore_target_${target}"
done

# ===========================================
# Phase 2: Learning Rate Sweep
# ===========================================
echo ""
echo "=== Phase 2: Learning Rate Sweep ==="
echo "Testing: 0.0001, 0.0005, 0.001, 0.002, 0.005"

for lr in 0.0001 0.0005 0.001 0.002 0.005; do
    echo ""
    echo ">>> Running: lr=$lr"
    python ane_tta.py \
        --models "$MODEL" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target mlp+ln \
        --num_layers all \
        --steps $BASE_STEPS \
        --lr $lr \
        --optimizer sgd \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "explore_lr_${lr}"
done

# ===========================================
# Phase 3: Optimizer Comparison
# ===========================================
echo ""
echo "=== Phase 3: Optimizer Comparison ==="
echo "Testing: sgd, adamw, adam"

for opt in sgd adamw adam; do
    echo ""
    echo ">>> Running: optimizer=$opt"
    python ane_tta.py \
        --models "$MODEL" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target mlp+ln \
        --num_layers all \
        --steps $BASE_STEPS \
        --lr 0.001 \
        --optimizer $opt \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "explore_opt_${opt}"
done

# ===========================================
# Phase 4: Layer Count Study
# ===========================================
echo ""
echo "=== Phase 4: Layer Count Study ==="
echo "Testing: 4, 8, 14, all layers"

for layers in 4 8 14 all; do
    echo ""
    echo ">>> Running: num_layers=$layers"
    python ane_tta.py \
        --models "$MODEL" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target mlp+ln \
        --num_layers $layers \
        --steps $BASE_STEPS \
        --lr 0.001 \
        --optimizer sgd \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "explore_layers_${layers}"
done

# ===========================================
# Phase 5: Scheduler Comparison
# ===========================================
echo ""
echo "=== Phase 5: Scheduler Comparison ==="
echo "Testing: none, cosine, linear, onecycle"

for sched in none cosine linear onecycle; do
    echo ""
    echo ">>> Running: scheduler=$sched"
    python ane_tta.py \
        --models "$MODEL" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target mlp+ln \
        --num_layers all \
        --steps $BASE_STEPS \
        --lr 0.001 \
        --optimizer sgd \
        --scheduler $sched \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "explore_sched_${sched}"
done

# ===========================================
# Phase 6: Step Count Study
# ===========================================
echo ""
echo "=== Phase 6: Step Count Study ==="
echo "Testing: 5, 10, 20, 30, 50 steps"

for steps in 5 10 20 30 50; do
    echo ""
    echo ">>> Running: steps=$steps"
    python ane_tta.py \
        --models "$MODEL" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target mlp+ln \
        --num_layers all \
        --steps $steps \
        --lr 0.001 \
        --optimizer sgd \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "explore_steps_${steps}"
done

# ===========================================
# Summary
# ===========================================
echo ""
echo "=========================================="
echo "Parameter Exploration Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Next: Analyze results and determine optimal configuration"
