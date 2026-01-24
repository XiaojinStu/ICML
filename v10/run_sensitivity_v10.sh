#!/usr/bin/env bash
set -euo pipefail

RUNROOT="${RUNROOT:-runs/v10_sensitivity}"
OUTROOT="$RUNROOT/results"
LOGDIR="$RUNROOT/logs"
SUMMARYDIR="$RUNROOT/summary"

mkdir -p "$OUTROOT" "$LOGDIR" "$SUMMARYDIR"

SEED="${SEED:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
SHUFFLE="${SHUFFLE:-1}"
NO_VIZ="${NO_VIZ:-1}"

MODEL="/home/jinsk/Models/Llama-3.2-3B-Instruct"

NON_GSM_DS="addition50_v1,bigbench_arithmetic600_seed42_v1,math401_all401_v1,nupa_test440_v1,numericbench_test500_seed2_v1"
NON_GSM_PATHS="addition50_v1=datasets_final_v3/addition50_v1.json,bigbench_arithmetic600_seed42_v1=datasets_final_v3/bigbench_arithmetic600_seed42_v1.json,math401_all401_v1=datasets_final_v3/math401_all401_v1.json,nupa_test440_v1=datasets_final_v3/nupa_test440_v1.json,numericbench_test500_seed2_v1=datasets_final_v3/numericbench_test500_seed2_v1.json"

MIXED_DS="bigbench_mixed_number_string300_seed42_v1"
MIXED_PATHS="bigbench_mixed_number_string300_seed42_v1=datasets_final_v3/bigbench_mixed_number_string300_seed42_v1.json"

GSM_DS="gsm8k_test500_v1"
GSM_PATHS="gsm8k_test500_v1=datasets_final_v3/gsm8k_test500_v1.json"

common=(
  --models "$MODEL"
  --seed "$SEED"
  --num_samples "$NUM_SAMPLES"
  --eval_modes tf
  --update_target mlp+ln
  --num_layers all
  --layer_stride 1
  --tta_reset token
  --optimizer sgd
  --momentum 0.0
  --topk_list 2,5
  --ane_metric angle
  --dtype bf16
  --device_map auto
)

if [[ "$SHUFFLE" == "1" ]]; then
  common+=(--shuffle)
fi
if [[ "$NO_VIZ" == "1" ]]; then
  common+=(--no_viz)
fi

echo "[v10-sens] runroot=$RUNROOT seed=$SEED num_samples=$NUM_SAMPLES steps=30"

# (1) LR sensitivity (non-GSM)
CUDA_VISIBLE_DEVICES=0 python v9/run_grid_v9.py \
  --datasets "$NON_GSM_DS" \
  --dataset_paths "$NON_GSM_PATHS" \
  --steps_list "30" \
  --lr_list "0.005,0.01,0.02" \
  --lr_schedule constant \
  --lr_norm none \
  --output_root "$OUTROOT/lr_sweep_non_gsm" \
  "${common[@]}" \
  2>&1 | tee "$LOGDIR/lr_sweep_non_gsm.log"

# (2) LR sensitivity (mixed, robust)
CUDA_VISIBLE_DEVICES=0 python v9/run_grid_v9.py \
  --datasets "$MIXED_DS" \
  --dataset_paths "$MIXED_PATHS" \
  --steps_list "30" \
  --lr_list "0.005,0.01,0.02" \
  --lr_schedule cosine \
  --lr_min 1e-4 \
  --lr_norm grad_norm \
  --output_root "$OUTROOT/lr_sweep_mixed" \
  "${common[@]}" \
  2>&1 | tee "$LOGDIR/lr_sweep_mixed.log"

# (3) LR sensitivity (GSM8K)
CUDA_VISIBLE_DEVICES=0 python v9/run_grid_v9.py \
  --datasets "$GSM_DS" \
  --dataset_paths "$GSM_PATHS" \
  --steps_list "30" \
  --lr_list "2e-05,5e-05,1e-04" \
  --lr_schedule constant \
  --lr_norm none \
  --output_root "$OUTROOT/lr_sweep_gsm8k" \
  "${common[@]}" \
  2>&1 | tee "$LOGDIR/lr_sweep_gsm8k.log"

# (4) Steps sensitivity (non-GSM)
CUDA_VISIBLE_DEVICES=0 python v9/run_grid_v9.py \
  --datasets "$NON_GSM_DS" \
  --dataset_paths "$NON_GSM_PATHS" \
  --steps_list "10,20,30" \
  --lr_list "0.01" \
  --lr_schedule constant \
  --lr_norm none \
  --output_root "$OUTROOT/steps_sweep_non_gsm" \
  "${common[@]}" \
  2>&1 | tee "$LOGDIR/steps_sweep_non_gsm.log"

echo "[v10-sens] aggregating..."
python v10/aggregate_v10.py --root "$OUTROOT" --out_dir "$SUMMARYDIR" --checkpoints "0,1,2,5,10,15,20,25,30"
echo "[v10-sens] done. Summary in: $SUMMARYDIR"

