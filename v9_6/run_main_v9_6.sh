#!/usr/bin/env bash
set -euo pipefail

# v9.6 final main experiment runner.
#
# Outputs:
# - JSONs under $RUNROOT/results/**
# - Logs under $RUNROOT/logs/**
# - Tables + plots under $RUNROOT/summary/**

RUNROOT="${RUNROOT:-runs/v9.6_main_final}"
DATA_DIR="${DATA_DIR:-datasets_final_v3}"

OUTROOT="$RUNROOT/results"
LOGDIR="$RUNROOT/logs"
SUMMARYDIR="$RUNROOT/summary"

mkdir -p "$OUTROOT" "$LOGDIR" "$SUMMARYDIR"

LLAMA_MODELS="/home/jinsk/Models/Llama-3.2-1B-Instruct,/home/jinsk/Models/Llama-3.2-3B-Instruct,/home/jinsk/Models/Llama-3.1-8B-Instruct"
QWEN_MODELS="/home/jinsk/Models/Qwen2.5-Math-7B,/home/jinsk/Models/Qwen3-4B-Instruct-2507"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[v9.6] missing data dir: $DATA_DIR" >&2
  exit 1
fi

COMMON_ARGS=(
  --num_samples 100
  --shuffle
  --seed 42
  --eval_modes tf
  --steps_list 10
  --lr_min 1e-4
  --ane_metric cosine
  --optimizer sgd
  --momentum 0.0
  --num_layers all
  --layer_stride 1
  --tta_reset token
  --dtype bf16
  --device_map auto
  --no_viz
)

echo "[v9.6] runroot=$RUNROOT"
echo "[v9.6] data_dir=$DATA_DIR"

run_grid() {
  local gpu="$1"; shift
  local models="$1"; shift
  local datasets="$1"; shift
  local dataset_paths="$1"; shift
  local lr_list="$1"; shift
  local lr_schedule="$1"; shift
  local lr_norm="$1"; shift
  local update_target="$1"; shift
  local out_root="$1"; shift
  local log_path="$1"; shift
  # Remaining args are passed through (e.g. --gradient_checkpointing).

  CUDA_VISIBLE_DEVICES="$gpu" python v9/run_grid_v9.py \
    --models "$models" \
    --datasets "$datasets" \
    --dataset_paths "$dataset_paths" \
    --lr_list "$lr_list" \
    --lr_schedule "$lr_schedule" \
    --lr_norm "$lr_norm" \
    --update_target "$update_target" \
    --output_root "$out_root" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$log_path"
}

run_phase() {
  local tag="$1"; shift
  local datasets="$1"; shift
  local dataset_paths="$1"; shift
  local lr_llama="$1"; shift
  local lr_qwen="$1"; shift
  local lr_schedule="$1"; shift
  local lr_norm="$1"; shift
  local tgt_llama="$1"; shift
  local tgt_qwen="$1"; shift

  echo "[v9.6] phase=$tag schedule=$lr_schedule lr_norm=$lr_norm lr_llama=$lr_llama lr_qwen=$lr_qwen"

  run_grid 0 "$LLAMA_MODELS" "$datasets" "$dataset_paths" "$lr_llama" "$lr_schedule" "$lr_norm" "$tgt_llama" \
    "$OUTROOT/$tag/llama" "$LOGDIR/${tag}_llama.log" &
  pid_llama=$!

  # Qwen uses LN-only to avoid OOM; enable gradient checkpointing by default.
  run_grid 1 "$QWEN_MODELS" "$datasets" "$dataset_paths" "$lr_qwen" "$lr_schedule" "$lr_norm" "$tgt_qwen" \
    "$OUTROOT/$tag/qwen" "$LOGDIR/${tag}_qwen.log" \
    --gradient_checkpointing &
  pid_qwen=$!

  wait "$pid_llama" "$pid_qwen"
}

# (A) non-GSM default (strong)
run_phase "non_gsm_default" \
  "addition50_v1,bigbench_arithmetic600_seed42_v1,math401_all401_v1,nupa_test440_v1,numericbench_test500_seed2_v1" \
  "addition50_v1=${DATA_DIR}/addition50_v1.json,bigbench_arithmetic600_seed42_v1=${DATA_DIR}/bigbench_arithmetic600_seed42_v1.json,math401_all401_v1=${DATA_DIR}/math401_all401_v1.json,nupa_test440_v1=${DATA_DIR}/nupa_test440_v1.json,numericbench_test500_seed2_v1=${DATA_DIR}/numericbench_test500_seed2_v1.json" \
  "0.01" "0.002" "constant" "none" "mlp+ln" "ln"

# (B) mixed-number-string robust
run_phase "mixed_robust" \
  "bigbench_mixed_number_string300_seed42_v1" \
  "bigbench_mixed_number_string300_seed42_v1=${DATA_DIR}/bigbench_mixed_number_string300_seed42_v1.json" \
  "0.01" "0.002" "cosine" "grad_norm" "mlp+ln" "ln"

# (C) GSM8K (small LR)
run_phase "gsm8k_smalllr" \
  "gsm8k_test500_v1" \
  "gsm8k_test500_v1=${DATA_DIR}/gsm8k_test500_v1.json" \
  "0.0001" "0.0001" "constant" "none" "mlp+ln" "ln"

echo "[v9.6] aggregating..."
python v9_6/aggregate_v9_6.py \
  --root "$OUTROOT" \
  --out_dir "$SUMMARYDIR" \
  --checkpoints "0,2,5,10"

echo "[v9.6] done. Summary in: $SUMMARYDIR"
