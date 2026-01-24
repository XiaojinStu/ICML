#!/usr/bin/env bash
set -euo pipefail

RUNROOT="${RUNROOT:-runs/v10_main}"
OUTROOT="$RUNROOT/results"
LOGDIR="$RUNROOT/logs"
SUMMARYDIR="$RUNROOT/summary"

mkdir -p "$OUTROOT" "$LOGDIR" "$SUMMARYDIR"

SEEDS_CSV="${SEEDS:-0,1,2}"
NUM_SAMPLES="${NUM_SAMPLES:-0}" # <=0 means all
SHUFFLE="${SHUFFLE:-1}"
RUN_COSINE="${RUN_COSINE:-1}"

LLAMA_SMALL="/home/jinsk/Models/Llama-3.2-1B-Instruct,/home/jinsk/Models/Llama-3.2-3B-Instruct"
LLAMA_8B="/home/jinsk/Models/Llama-3.1-8B-Instruct"
QWEN_MODELS="/home/jinsk/Models/Qwen2.5-Math-7B,/home/jinsk/Models/Qwen3-4B-Instruct-2507"

SUITE_LLAMA_SMALL="v10/suites/main_llama_small.json"
SUITE_LLAMA_8B="v10/suites/main_llama8b.json"
SUITE_QWEN="v10/suites/main_qwen.json"

common_args=(
  --eval_modes tf
  --steps 30
  --num_samples "$NUM_SAMPLES"
  --topk_list 2,5
  --num_layers all
  --layer_stride 1
  --tta_reset token
  --snapshot_steps "0,1,2,5,10,15,20,25,30"
  --dtype bf16
  --device_map auto
)

if [[ "$SHUFFLE" == "1" ]]; then
  common_args+=(--shuffle)
fi

run_llama() {
  local seed="$1"; shift
  local metric="$1"; shift
  local out="$1"; shift

  CUDA_VISIBLE_DEVICES=0 python v10/run_suite_v10.py \
    --models "$LLAMA_SMALL" \
    --suite "$SUITE_LLAMA_SMALL" \
    --seed "$seed" \
    --ane_metric "$metric" \
    --update_target "mlp+ln" \
    --output_root "$out" \
    "${common_args[@]}" \
    2>&1 | tee "$LOGDIR/main_${metric}_seed${seed}_llama_small.log"

  CUDA_VISIBLE_DEVICES=0 python v10/run_suite_v10.py \
    --models "$LLAMA_8B" \
    --suite "$SUITE_LLAMA_8B" \
    --seed "$seed" \
    --ane_metric "$metric" \
    --update_target "mlp+ln" \
    --output_root "$out" \
    "${common_args[@]}" \
    2>&1 | tee "$LOGDIR/main_${metric}_seed${seed}_llama8b.log"
}

run_qwen() {
  local seed="$1"; shift
  local metric="$1"; shift
  local out="$1"; shift

  CUDA_VISIBLE_DEVICES=1 python v10/run_suite_v10.py \
    --models "$QWEN_MODELS" \
    --suite "$SUITE_QWEN" \
    --seed "$seed" \
    --ane_metric "$metric" \
    --update_target "ln" \
    --gradient_checkpointing \
    --output_root "$out" \
    "${common_args[@]}" \
    2>&1 | tee "$LOGDIR/main_${metric}_seed${seed}_qwen.log"
}

IFS=',' read -r -a SEEDS_ARR <<<"$SEEDS_CSV"

echo "[v10-main] runroot=$RUNROOT"
echo "[v10-main] seeds=$SEEDS_CSV num_samples=$NUM_SAMPLES steps=30"

for seed in "${SEEDS_ARR[@]}"; do
  seed="$(echo "$seed" | xargs)"
  [[ -n "$seed" ]] || continue

  out_angle="$OUTROOT/main_angle/seed${seed}"
  mkdir -p "$out_angle"

  echo "[v10-main] seed=$seed metric=angle"
  (run_llama "$seed" "angle" "$out_angle") & pid0=$!
  (run_qwen "$seed" "angle" "$out_angle") & pid1=$!
  wait "$pid0" "$pid1"

  if [[ "$RUN_COSINE" == "1" ]]; then
    out_cos="$OUTROOT/main_cosine/seed${seed}"
    mkdir -p "$out_cos"
    echo "[v10-main] seed=$seed metric=cosine"
    (run_llama "$seed" "cosine" "$out_cos") & pid2=$!
    (run_qwen "$seed" "cosine" "$out_cos") & pid3=$!
    wait "$pid2" "$pid3"
  fi
done

echo "[v10-main] aggregating..."
python v10/aggregate_v10.py --root "$OUTROOT" --out_dir "$SUMMARYDIR" --checkpoints "0,1,2,5,10,15,20,25,30"
echo "[v10-main] done. Summary in: $SUMMARYDIR"
