#!/usr/bin/env bash
set -euo pipefail

# LR sweep after v9.5 main run.
# Sweeps a small LR neighborhood around the default LR choices, per model family.

RUNROOT="${RUNROOT:-runs/v9.5_lr_sweep_curated_cosine_s10}"
CURATED_DIR="${CURATED_DIR:-datasets_final_v2/curated/main}"

OUTROOT="$RUNROOT/results"
LOGDIR="$RUNROOT/logs"
SUMMARYDIR="$RUNROOT/summary"

mkdir -p "$OUTROOT" "$LOGDIR" "$SUMMARYDIR"

LLAMA_MODELS="/home/jinsk/Models/Llama-3.2-1B-Instruct,/home/jinsk/Models/Llama-3.2-3B-Instruct,/home/jinsk/Models/Llama-3.1-8B-Instruct"
QWEN_MODELS="/home/jinsk/Models/Qwen2.5-Math-7B,/home/jinsk/Models/Qwen3-4B-Instruct-2507"

DATASETS_NON_GSM="addition50_v1,bigbench_arithmetic600_seed42_v1,bigbench_mixed_number_string300_seed42_v1,math401_all401_v1,nupa_test440_v1,numericbench_test500_seed2_v1"
DATASETS_GSM="gsm8k_test500_v1"

DATASET_PATHS_NON_GSM="addition50_v1=${CURATED_DIR}/addition50_v1.json,bigbench_arithmetic600_seed42_v1=${CURATED_DIR}/bigbench_arithmetic600_seed42_v1.json,bigbench_mixed_number_string300_seed42_v1=${CURATED_DIR}/bigbench_mixed_number_string300_seed42_v1.json,math401_all401_v1=${CURATED_DIR}/math401_all401_v1.json,nupa_test440_v1=${CURATED_DIR}/nupa_test440_v1.json,numericbench_test500_seed2_v1=${CURATED_DIR}/numericbench_test500_seed2_v1.json"
DATASET_PATHS_GSM="gsm8k_test500_v1=${CURATED_DIR}/gsm8k_test500_v1.json"

COMMON_ARGS=(
  --num_samples 100
  --shuffle
  --seed 42
  --eval_modes tf
  --steps_list 10
  --lr_schedule constant
  --lr_min 1e-4
  --lr_norm none
  --ane_metric cosine
  --optimizer sgd
  --momentum 0.0
  --update_target mlp+ln
  --num_layers all
  --layer_stride 1
  --tta_reset token
  --dtype bf16
  --device_map auto
  --no_viz
)

run_pair() {
  local tag="$1"; shift
  local datasets="$1"; shift
  local dpaths="$1"; shift
  local lr_llama="$1"; shift
  local lr_qwen="$1"; shift

  echo "[v9.5-lr] phase=$tag datasets=$datasets lr_llama=$lr_llama lr_qwen=$lr_qwen"

  CUDA_VISIBLE_DEVICES=0 python v9/run_grid_v9.py \
    --models "$LLAMA_MODELS" \
    --datasets "$datasets" \
    --dataset_paths "$dpaths" \
    --lr_list "$lr_llama" \
    --output_root "$OUTROOT/$tag" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOGDIR/${tag}_llama.log" &

  CUDA_VISIBLE_DEVICES=1 python v9/run_grid_v9.py \
    --models "$QWEN_MODELS" \
    --datasets "$datasets" \
    --dataset_paths "$dpaths" \
    --lr_list "$lr_qwen" \
    --output_root "$OUTROOT/$tag" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOGDIR/${tag}_qwen.log" &

  wait
}

# Small sweep neighborhoods
run_pair "non_gsm" "$DATASETS_NON_GSM" "$DATASET_PATHS_NON_GSM" "0.02,0.01,0.005" "0.004,0.002,0.001"
run_pair "gsm8k" "$DATASETS_GSM" "$DATASET_PATHS_GSM" "0.0002,0.0001,0.00005" "0.0002,0.0001,0.00005"

echo "[v9.5-lr] aggregating..."
python v9_5/aggregate_v9_5.py \
  --root "$OUTROOT" \
  --out_dir "$SUMMARYDIR" \
  --checkpoints "0,2,5,10"

echo "[v9.5-lr] done. Summary in: $SUMMARYDIR"

