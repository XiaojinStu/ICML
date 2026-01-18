#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONDONTWRITEBYTECODE=1

MODEL="/home/jinsk/Models/Llama-3.2-3B-Instruct"
DATA_PATH="../data/addition_problems_dataset(1-50)(1).json"

STEPS=5
TOPK_LIST="5"

# Baseline: ANE-TTA (current)
python experiment_tf.py \
  --method ane_tta \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --num_samples 50 \
  --steps $STEPS \
  --lr 0.001 \
  --update_target mlp+ln \
  --num_layers all \
  --output_dir baseline_ane \
  --exp_name tf_ane_tta \
  --topk_list "$TOPK_LIST" \

# Idea 1: real-number distance loss
python experiment_tf.py \
  --method real_tta \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --num_samples 50 \
  --steps $STEPS \
  --lr 0.001 \
  --update_target mlp+ln \
  --num_layers all \
  --output_dir idea1_real_distance \
  --exp_name tf_real_tta \
  --topk_list "$TOPK_LIST" \

# Idea 2: mean-embedding one-shot decode (no optimization)
python experiment_tf.py \
  --method mean_embed \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --num_samples 50 \
  --steps $STEPS \
  --output_dir idea2_mean_embedding \
  --exp_name tf_mean_embed \
  --topk_list "$TOPK_LIST"

# Idea 3: only optimize lm_head projection
python experiment_tf.py \
  --method ane_tta_lm_head \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --num_samples 50 \
  --steps $STEPS \
  --lr 0.001 \
  --output_dir idea3_lm_head_only \
  --exp_name tf_ane_lm_head \
  --topk_list "$TOPK_LIST" \

# Aggregate summary
python aggregate.py --root . --out_dir summary
