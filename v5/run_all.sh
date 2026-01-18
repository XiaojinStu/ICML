#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

OUTPUT_ROOT="results_batch"
CONFIG="configs/main_experiments.json"

python batch_run.py --config "$CONFIG" --output_root "$OUTPUT_ROOT"
python summary.py --input_dir "$OUTPUT_ROOT" --output_dir "$OUTPUT_ROOT/summary"
