#!/usr/bin/env bash
set -euo pipefail

# Ablation: Llama-3.2-3B-Instruct, compare:
# - constant LR + no lr_norm  (baseline)
# - cosine LR + no lr_norm
# - constant LR + grad_norm
# - cosine LR + grad_norm
#
# Datasets: curated (100 samples each), lr policy:
# - gsm8k: lr=1e-4
# - others: lr=1e-2
#
# Produces: per-run JSONs + aggregated CSV + plots.

RUNROOT="${RUNROOT:-runs/v9.5_ablate_llama3b_sched_gradnorm}"
CURATED_DIR="${CURATED_DIR:-datasets_final_v2/curated/main}"

OUTROOT="$RUNROOT/results"
LOGDIR="$RUNROOT/logs"
SUMMARYDIR="$RUNROOT/summary"
mkdir -p "$OUTROOT" "$LOGDIR" "$SUMMARYDIR"

MODEL="/home/jinsk/Models/Llama-3.2-3B-Instruct"

DATASETS_NON_GSM="addition50_v1,bigbench_arithmetic600_seed42_v1,bigbench_mixed_number_string300_seed42_v1,math401_all401_v1,nupa_test440_v1,numericbench_test500_seed2_v1"
DATASETS_GSM="gsm8k_test500_v1"

DATASET_PATHS_NON_GSM="addition50_v1=${CURATED_DIR}/addition50_v1.json,bigbench_arithmetic600_seed42_v1=${CURATED_DIR}/bigbench_arithmetic600_seed42_v1.json,bigbench_mixed_number_string300_seed42_v1=${CURATED_DIR}/bigbench_mixed_number_string300_seed42_v1.json,math401_all401_v1=${CURATED_DIR}/math401_all401_v1.json,nupa_test440_v1=${CURATED_DIR}/nupa_test440_v1.json,numericbench_test500_seed2_v1=${CURATED_DIR}/numericbench_test500_seed2_v1.json"
DATASET_PATHS_GSM="gsm8k_test500_v1=${CURATED_DIR}/gsm8k_test500_v1.json"

COMMON_ARGS=(
  --models "$MODEL"
  --num_samples 100
  --shuffle
  --seed 42
  --eval_modes tf
  --steps_list 10
  --lr_min 1e-4
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

run_variant() {
  local variant="$1"; shift
  local lr_schedule="$1"; shift
  local lr_norm="$1"; shift
  local gpu="$1"; shift

  echo "[ablate] variant=$variant schedule=$lr_schedule lr_norm=$lr_norm gpu=$gpu"

  CUDA_VISIBLE_DEVICES="$gpu" python v9/run_grid_v9.py \
    "${COMMON_ARGS[@]}" \
    --datasets "$DATASETS_NON_GSM" \
    --dataset_paths "$DATASET_PATHS_NON_GSM" \
    --lr_list 0.01 \
    --lr_schedule "$lr_schedule" \
    --lr_norm "$lr_norm" \
    --output_root "$OUTROOT/$variant/non_gsm" \
    2>&1 | tee "$LOGDIR/${variant}_non_gsm.log"

  CUDA_VISIBLE_DEVICES="$gpu" python v9/run_grid_v9.py \
    "${COMMON_ARGS[@]}" \
    --datasets "$DATASETS_GSM" \
    --dataset_paths "$DATASET_PATHS_GSM" \
    --lr_list 0.0001 \
    --lr_schedule "$lr_schedule" \
    --lr_norm "$lr_norm" \
    --output_root "$OUTROOT/$variant/gsm8k" \
    2>&1 | tee "$LOGDIR/${variant}_gsm8k.log"
}

echo "[ablate] runroot=$RUNROOT"
echo "[ablate] curated_dir=$CURATED_DIR"

# Wave 1: one variant per GPU
run_variant "const_none" "constant" "none" 0 &
PID0=$!
run_variant "cos_gradnorm" "cosine" "grad_norm" 1 &
PID1=$!
wait "$PID0" "$PID1"

# Wave 2
run_variant "const_gradnorm" "constant" "grad_norm" 0 &
PID0=$!
run_variant "cos_none" "cosine" "none" 1 &
PID1=$!
wait "$PID0" "$PID1"

echo "[ablate] aggregating..."
python v9_5/aggregate_ablation_sched_gradnorm.py \
  --root "$OUTROOT" \
  --out_dir "$SUMMARYDIR" \
  --checkpoints "0,2,5,10"

echo "[ablate] done. Summary in: $SUMMARYDIR"

