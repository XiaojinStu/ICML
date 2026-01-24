#!/usr/bin/env bash
set -euo pipefail

RUNROOT="${RUNROOT:-runs/v9.7_main_final}"
DATA_DIR="${DATA_DIR:-datasets_final_v3}"

OUTROOT="$RUNROOT/results"
LOGDIR="$RUNROOT/logs"
SUMMARYDIR="$RUNROOT/summary"

mkdir -p "$OUTROOT" "$LOGDIR" "$SUMMARYDIR"

LLAMA_1B="/home/jinsk/Models/Llama-3.2-1B-Instruct"
LLAMA_3B="/home/jinsk/Models/Llama-3.2-3B-Instruct"
LLAMA_8B="/home/jinsk/Models/Llama-3.1-8B-Instruct"

QWEN_4B="/home/jinsk/Models/Qwen3-4B-Instruct-2507"
QWEN_25M="/home/jinsk/Models/Qwen2.5-Math-7B"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[v9.7] missing data dir: $DATA_DIR" >&2
  exit 1
fi

COMMON=(
  --num_samples 100
  --shuffle
  --seed 42
  --eval_modes tf
  --steps_list 30
  --lr_min 1e-4
  --optimizer sgd
  --momentum 0.0
  --topk_list 2,5
  --num_layers all
  --layer_stride 1
  --tta_reset token
  --dtype bf16
  --device_map auto
)

echo "[v9.7] runroot=$RUNROOT"
echo "[v9.7] data_dir=$DATA_DIR"

run_grid() {
  local gpu="$1"; shift
  local models="$1"; shift
  local phase="$1"; shift
  local datasets="$1"; shift
  local dataset_paths="$1"; shift
  local lr="$1"; shift
  local lr_schedule="$1"; shift
  local lr_norm="$1"; shift
  local update_target="$1"; shift
  local extra=("$@")

  CUDA_VISIBLE_DEVICES="$gpu" python v9/run_grid_v9.py \
    --models "$models" \
    --datasets "$datasets" \
    --dataset_paths "$dataset_paths" \
    --lr_list "$lr" \
    --lr_schedule "$lr_schedule" \
    --lr_norm "$lr_norm" \
    --update_target "$update_target" \
    --output_root "$OUTROOT/$phase" \
    "${COMMON[@]}" \
    "${extra[@]}"
}

wait_all() { for p in "$@"; do wait "$p"; done; }

NON_GSM_DS="addition50_v1,bigbench_arithmetic600_seed42_v1,math401_all401_v1,nupa_test440_v1,numericbench_test500_seed2_v1"
NON_GSM_PATHS="addition50_v1=${DATA_DIR}/addition50_v1.json,bigbench_arithmetic600_seed42_v1=${DATA_DIR}/bigbench_arithmetic600_seed42_v1.json,math401_all401_v1=${DATA_DIR}/math401_all401_v1.json,nupa_test440_v1=${DATA_DIR}/nupa_test440_v1.json,numericbench_test500_seed2_v1=${DATA_DIR}/numericbench_test500_seed2_v1.json"

MIXED_DS="bigbench_mixed_number_string300_seed42_v1"
MIXED_PATHS="bigbench_mixed_number_string300_seed42_v1=${DATA_DIR}/bigbench_mixed_number_string300_seed42_v1.json"

GSM_DS="gsm8k_test500_v1"
GSM_PATHS="gsm8k_test500_v1=${DATA_DIR}/gsm8k_test500_v1.json"

run_llama_suite() {
  # Llama (GPU0): 1B/3B use lr=0.01; 8B use lr=0.005
  run_grid 0 "$LLAMA_1B,$LLAMA_3B" "llama_non_gsm_lr0.01_angle" "$NON_GSM_DS" "$NON_GSM_PATHS" "0.01" "constant" "none" "mlp+ln" --ane_metric angle 2>&1 | tee "$LOGDIR/llama_non_gsm_lr0.01_angle.log"
  run_grid 0 "$LLAMA_8B"          "llama8b_non_gsm_lr0.005_angle" "$NON_GSM_DS" "$NON_GSM_PATHS" "0.005" "constant" "none" "mlp+ln" --ane_metric angle 2>&1 | tee "$LOGDIR/llama8b_non_gsm_lr0.005_angle.log"

  run_grid 0 "$LLAMA_1B,$LLAMA_3B" "llama_mixed_robust_lr0.01_angle" "$MIXED_DS" "$MIXED_PATHS" "0.01" "cosine" "grad_norm" "mlp+ln" --ane_metric angle 2>&1 | tee "$LOGDIR/llama_mixed_robust_lr0.01_angle.log"
  run_grid 0 "$LLAMA_8B"          "llama8b_mixed_robust_lr0.005_angle" "$MIXED_DS" "$MIXED_PATHS" "0.005" "cosine" "grad_norm" "mlp+ln" --ane_metric angle 2>&1 | tee "$LOGDIR/llama8b_mixed_robust_lr0.005_angle.log"

  run_grid 0 "$LLAMA_1B,$LLAMA_3B" "llama_gsm8k_lr1e-4_angle" "$GSM_DS" "$GSM_PATHS" "0.0001" "constant" "none" "mlp+ln" --ane_metric angle 2>&1 | tee "$LOGDIR/llama_gsm8k_lr1e-4_angle.log"
  run_grid 0 "$LLAMA_8B"          "llama8b_gsm8k_lr1e-4_angle" "$GSM_DS" "$GSM_PATHS" "0.0001" "constant" "none" "mlp+ln" --ane_metric angle 2>&1 | tee "$LOGDIR/llama8b_gsm8k_lr1e-4_angle.log"

  # cosine (comparison)
  run_grid 0 "$LLAMA_1B,$LLAMA_3B" "llama_non_gsm_lr0.01_cosine" "$NON_GSM_DS" "$NON_GSM_PATHS" "0.01" "constant" "none" "mlp+ln" --ane_metric cosine 2>&1 | tee "$LOGDIR/llama_non_gsm_lr0.01_cosine.log"
  run_grid 0 "$LLAMA_8B"          "llama8b_non_gsm_lr0.005_cosine" "$NON_GSM_DS" "$NON_GSM_PATHS" "0.005" "constant" "none" "mlp+ln" --ane_metric cosine 2>&1 | tee "$LOGDIR/llama8b_non_gsm_lr0.005_cosine.log"

  run_grid 0 "$LLAMA_1B,$LLAMA_3B" "llama_mixed_robust_lr0.01_cosine" "$MIXED_DS" "$MIXED_PATHS" "0.01" "cosine" "grad_norm" "mlp+ln" --ane_metric cosine 2>&1 | tee "$LOGDIR/llama_mixed_robust_lr0.01_cosine.log"
  run_grid 0 "$LLAMA_8B"          "llama8b_mixed_robust_lr0.005_cosine" "$MIXED_DS" "$MIXED_PATHS" "0.005" "cosine" "grad_norm" "mlp+ln" --ane_metric cosine 2>&1 | tee "$LOGDIR/llama8b_mixed_robust_lr0.005_cosine.log"

  run_grid 0 "$LLAMA_1B,$LLAMA_3B" "llama_gsm8k_lr1e-4_cosine" "$GSM_DS" "$GSM_PATHS" "0.0001" "constant" "none" "mlp+ln" --ane_metric cosine 2>&1 | tee "$LOGDIR/llama_gsm8k_lr1e-4_cosine.log"
  run_grid 0 "$LLAMA_8B"          "llama8b_gsm8k_lr1e-4_cosine" "$GSM_DS" "$GSM_PATHS" "0.0001" "constant" "none" "mlp+ln" --ane_metric cosine 2>&1 | tee "$LOGDIR/llama8b_gsm8k_lr1e-4_cosine.log"
}

run_qwen_suite() {
  # Qwen (GPU1): per-dataset lr, default LN-only + gradient checkpointing
  QWEN_MODELS="$QWEN_25M,$QWEN_4B"
  QWEN_EXTRA=(--gradient_checkpointing)

  run_grid 1 "$QWEN_MODELS" "qwen_add50_lr0.004_angle" "addition50_v1" "addition50_v1=${DATA_DIR}/addition50_v1.json" "0.004" "constant" "none" "ln" --ane_metric angle "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_add50_lr0.004_angle.log"
  run_grid 1 "$QWEN_MODELS" "qwen_bba_lr0.001_angle" "bigbench_arithmetic600_seed42_v1" "bigbench_arithmetic600_seed42_v1=${DATA_DIR}/bigbench_arithmetic600_seed42_v1.json" "0.001" "constant" "none" "ln" --ane_metric angle "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_bba_lr0.001_angle.log"
  run_grid 1 "$QWEN_MODELS" "qwen_lr0.002_group_angle" "math401_all401_v1,numericbench_test500_seed2_v1,nupa_test440_v1" "math401_all401_v1=${DATA_DIR}/math401_all401_v1.json,numericbench_test500_seed2_v1=${DATA_DIR}/numericbench_test500_seed2_v1.json,nupa_test440_v1=${DATA_DIR}/nupa_test440_v1.json" "0.002" "constant" "none" "ln" --ane_metric angle "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_lr0.002_group_angle.log"

  run_grid 1 "$QWEN_MODELS" "qwen_mixed_robust_lr0.004_angle" "$MIXED_DS" "$MIXED_PATHS" "0.004" "cosine" "grad_norm" "ln" --ane_metric angle "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_mixed_robust_lr0.004_angle.log"
  run_grid 1 "$QWEN_MODELS" "qwen_gsm8k_lr5e-5_angle" "$GSM_DS" "$GSM_PATHS" "0.00005" "constant" "none" "ln" --ane_metric angle "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_gsm8k_lr5e-5_angle.log"

  # cosine (comparison)
  run_grid 1 "$QWEN_MODELS" "qwen_add50_lr0.004_cosine" "addition50_v1" "addition50_v1=${DATA_DIR}/addition50_v1.json" "0.004" "constant" "none" "ln" --ane_metric cosine "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_add50_lr0.004_cosine.log"
  run_grid 1 "$QWEN_MODELS" "qwen_bba_lr0.001_cosine" "bigbench_arithmetic600_seed42_v1" "bigbench_arithmetic600_seed42_v1=${DATA_DIR}/bigbench_arithmetic600_seed42_v1.json" "0.001" "constant" "none" "ln" --ane_metric cosine "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_bba_lr0.001_cosine.log"
  run_grid 1 "$QWEN_MODELS" "qwen_lr0.002_group_cosine" "math401_all401_v1,numericbench_test500_seed2_v1,nupa_test440_v1" "math401_all401_v1=${DATA_DIR}/math401_all401_v1.json,numericbench_test500_seed2_v1=${DATA_DIR}/numericbench_test500_seed2_v1.json,nupa_test440_v1=${DATA_DIR}/nupa_test440_v1.json" "0.002" "constant" "none" "ln" --ane_metric cosine "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_lr0.002_group_cosine.log"

  run_grid 1 "$QWEN_MODELS" "qwen_mixed_robust_lr0.004_cosine" "$MIXED_DS" "$MIXED_PATHS" "0.004" "cosine" "grad_norm" "ln" --ane_metric cosine "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_mixed_robust_lr0.004_cosine.log"
  run_grid 1 "$QWEN_MODELS" "qwen_gsm8k_lr5e-5_cosine" "$GSM_DS" "$GSM_PATHS" "0.00005" "constant" "none" "ln" --ane_metric cosine "${QWEN_EXTRA[@]}" 2>&1 | tee "$LOGDIR/qwen_gsm8k_lr5e-5_cosine.log"
}

run_llama_suite & pid_llama=$!
run_qwen_suite & pid_qwen=$!
wait_all "$pid_llama" "$pid_qwen"

echo "[v9.7] aggregating..."
python v9_7/aggregate_v9_7.py \
  --root "$OUTROOT" \
  --out_dir "$SUMMARYDIR" \
  --checkpoints "0,1,2,5,10,15,20,25,30"

echo "[v9.7] done. Summary in: $SUMMARYDIR"
