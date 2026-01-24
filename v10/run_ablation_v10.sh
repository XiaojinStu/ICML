#!/usr/bin/env bash
set -euo pipefail

RUNROOT="${RUNROOT:-runs/v10_ablation}"
OUTROOT="$RUNROOT/results"
LOGDIR="$RUNROOT/logs"
SUMMARYDIR="$RUNROOT/summary"

mkdir -p "$OUTROOT" "$LOGDIR" "$SUMMARYDIR"

SEEDS_CSV="${SEEDS:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
SHUFFLE="${SHUFFLE:-1}"

MODEL="/home/jinsk/Models/Llama-3.2-3B-Instruct"

SUITE_BASE="v10/suites/main_llama_small.json"
SUITE_MIXED_SCHED_CONST="v10/suites/ablate_mixed_schedule_constant.json"
SUITE_MIXED_LRNORM_NONE="v10/suites/ablate_mixed_lrnorm_none.json"

common_args=(
  --models "$MODEL"
  --eval_modes tf
  --steps 30
  --num_samples "$NUM_SAMPLES"
  --topk_list 2,5
  --num_layers all
  --layer_stride 1
  --tta_reset token
  --dtype bf16
  --device_map auto
  --ane_metric angle
)

if [[ "$SHUFFLE" == "1" ]]; then
  common_args+=(--shuffle)
fi

run_one() {
  local seed="$1"; shift
  local tag="$1"; shift
  local suite="$1"; shift
  local update_target="$1"; shift
  local metric="$1"; shift

  local out="$OUTROOT/$tag/seed${seed}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 python v10/run_suite_v10.py \
    --suite "$suite" \
    --seed "$seed" \
    --update_target "$update_target" \
    --ane_metric "$metric" \
    --output_root "$out" \
    "${common_args[@]}" \
    2>&1 | tee "$LOGDIR/${tag}_seed${seed}.log"
}

IFS=',' read -r -a SEEDS_ARR <<<"$SEEDS_CSV"

echo "[v10-ablation] runroot=$RUNROOT"
echo "[v10-ablation] seeds=$SEEDS_CSV num_samples=$NUM_SAMPLES steps=30"

for seed in "${SEEDS_ARR[@]}"; do
  seed="$(echo "$seed" | xargs)"
  [[ -n "$seed" ]] || continue

  # update_target ablation (angle)
  run_one "$seed" "ablate_update_target_ln" "$SUITE_BASE" "ln" "angle"
  run_one "$seed" "ablate_update_target_mlp" "$SUITE_BASE" "mlp" "angle"
  run_one "$seed" "ablate_update_target_mlp_ln" "$SUITE_BASE" "mlp+ln" "angle"

  # ane_metric ablation on the same suite (cosine vs angle)
  run_one "$seed" "ablate_metric_cosine" "$SUITE_BASE" "mlp+ln" "cosine"

  # mixed schedule ablation (keep grad_norm; switch schedule)
  run_one "$seed" "ablate_mixed_schedule_constant" "$SUITE_MIXED_SCHED_CONST" "mlp+ln" "angle"

  # mixed lr_norm ablation (keep cosine schedule; switch lr_norm)
  run_one "$seed" "ablate_mixed_lrnorm_none" "$SUITE_MIXED_LRNORM_NONE" "mlp+ln" "angle"
done

echo "[v10-ablation] aggregating..."
python v10/aggregate_v10.py --root "$OUTROOT" --out_dir "$SUMMARYDIR" --checkpoints "0,1,2,5,10,15,20,25,30"
echo "[v10-ablation] done. Summary in: $SUMMARYDIR"

