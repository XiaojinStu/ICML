#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1

MODEL="/home/zhangsy/llm_model/Llama-3.2-1B-Instruct"
DATA_PATH="../data/addition_problems_dataset(1-50)(1).json"

OUT_ROOT="results_addition50"
NUM_SAMPLES=30
STEPS=5
LR=0.001
EVAL_MODE="ar"
TOPK_LIST="5"

run_one() {
  local name="$1"
  local update_target="$2"
  local num_layers="$3"

  python experiment_ar.py \
    --model "$MODEL" \
    --data_path "$DATA_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --steps "$STEPS" \
    --lr "$LR" \
    --optimizer "sgd" \
    --update_target "$update_target" \
    --num_layers "$num_layers" \
    --eval_mode "$EVAL_MODE" \
    --topk_list "$TOPK_LIST" \
    --dtype "bf16" \
    --device_map "cuda:7" \
    --output_dir "$OUT_ROOT/$name" \
    --config_name "$name"
}

# Baselines
run_one baseline_mlp_ln_all "mlp+ln" "all"
run_one baseline_attn_ln_all "attn+ln" "all"
run_one baseline_all_all "all" "all"
run_one baseline_all_lm_head_all "all+lm_head" "all"

# LN only (all layers)
run_one ln_all "ln" "all"

# Attention only: all -> last layer (interpolation), no LN
run_one attn_all "attn" "all"
run_one attn_8 "attn" "8"
run_one attn_4 "attn" "4"
run_one attn_2 "attn" "2"
run_one attn_1 "attn" "1"

# MLP only: all -> last layer
run_one mlp_all "mlp" "all"
run_one mlp_8 "mlp" "8"
run_one mlp_4 "mlp" "4"
run_one mlp_2 "mlp" "2"
run_one mlp_1 "mlp" "1"

# Last layer (attn+mlp+ln)
run_one last_layer_all "all" "1"

python aggregate.py --root "$OUT_ROOT" --out_dir "summary"
