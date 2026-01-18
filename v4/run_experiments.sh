#!/bin/bash
# ANE v4 Production Experiments
# Restored v2.1 optimal configuration: mlp+ln, lr=0.001, all layers

cd "$(dirname "$0")"

DATA_PATH="../data/addition_problems_dataset(1-50)(1).json"
OUTPUT_DIR="results_ane"
SAMPLES=50

# v2.1 optimal configuration (proven to work best)
UPDATE_TARGET="mlp+ln"
LR=0.001
NUM_LAYERS="all"

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "ANE v4 Production Experiments"
echo "=========================================="
echo "Config: ${UPDATE_TARGET}, lr=${LR}, layers=${NUM_LAYERS}"
echo "Samples: $SAMPLES"
echo ""

# Llama-3.2-1B
echo "=== Llama-3.2-1B ==="
MODEL_1B="/home/jinsk/Models/Llama-3.2-1B-Instruct"

for steps in 5 30; do
    echo ">>> 1B, $steps steps"
    python ane_tta.py \
        --models "$MODEL_1B" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target $UPDATE_TARGET \
        --num_layers $NUM_LAYERS \
        --steps $steps \
        --lr $LR \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "llama1b_steps${steps}"
done

# Llama-3.2-3B
echo ""
echo "=== Llama-3.2-3B ==="
MODEL_3B="/home/jinsk/Models/Llama-3.2-3B-Instruct"

for steps in 5 30; do
    echo ">>> 3B, $steps steps"
    python ane_tta.py \
        --models "$MODEL_3B" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target $UPDATE_TARGET \
        --num_layers $NUM_LAYERS \
        --steps $steps \
        --lr $LR \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "llama3b_steps${steps}"
done

# Llama-3.1-8B (ALL layers for best performance)
echo ""
echo "=== Llama-3.1-8B (all layers) ==="
MODEL_8B="/home/jinsk/Models/Llama-3.1-8B-Instruct"

for steps in 5 30; do
    echo ">>> 8B, $steps steps (all layers)"
    python ane_tta.py \
        --models "$MODEL_8B" \
        --data_path "$DATA_PATH" \
        --num_samples $SAMPLES \
        --update_target $UPDATE_TARGET \
        --num_layers $NUM_LAYERS \
        --steps $steps \
        --lr $LR \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "llama8b_steps${steps}"
done

echo ""
echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="

# Generate comparison chart
echo "Generating comparison statistics..."
python -c "
import json
import glob
from visualization import visualize_comparison

results = {}
for f in glob.glob('${OUTPUT_DIR}/*.json'):
    name = f.split('/')[-1].replace('.json', '')
    with open(f) as fp:
        results[name] = json.load(fp)

if len(results) >= 2:
    visualize_comparison(results, '${OUTPUT_DIR}')
"

echo "Results saved to: $OUTPUT_DIR/"
