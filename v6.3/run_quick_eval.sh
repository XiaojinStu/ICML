#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1

MODEL="${MODEL:-/home/jinsk/Models/Llama-3.2-3B-Instruct}"
OUT_ROOT="${OUT_ROOT:-results}"
DATA_OUT_DIR="${DATA_OUT_DIR:-../data/gsm8k}"

NUM_SAMPLES="${NUM_SAMPLES:-50}"
STEPS="${STEPS:-10}"
LR="${LR:-0.001}"
EVAL_MODE="${EVAL_MODE:-ar}"
TOPK_LIST="${TOPK_LIST:-5}"

UPDATE_TARGET="${UPDATE_TARGET:-mlp}"
NUM_LAYERS="${NUM_LAYERS:-all}"
TTA_RESET="${TTA_RESET:-sample}"

run_one() {
  local data_path="$1"
  local name="$2"

  python experiment_v6.py \
    --model "$MODEL" \
    --data_path "$data_path" \
    --num_samples "$NUM_SAMPLES" \
    --steps "$STEPS" \
    --lr "$LR" \
    --optimizer "sgd" \
    --momentum 0.0 \
    --update_target "$UPDATE_TARGET" \
    --num_layers "$NUM_LAYERS" \
    --eval_mode "$EVAL_MODE" \
    --tta_reset "$TTA_RESET" \
    --topk_list "$TOPK_LIST" \
    --dtype "bf16" \
    --device_map "auto" \
    --backup_on_cpu \
    --no_viz \
    --output_dir "$OUT_ROOT/$name" \
    --config_name "$name"
}

# Prepare a local GSM8K JSON file for a quick test.
python prepare_gsm8k.py --out_dir "$DATA_OUT_DIR" --split "test" --num_samples "$NUM_SAMPLES" --seed 42

GSM_PATH="$DATA_OUT_DIR/gsm8k_test_${NUM_SAMPLES}.json"
run_one "$GSM_PATH" "gsm8k_${UPDATE_TARGET}_${NUM_LAYERS}_steps${STEPS}"
