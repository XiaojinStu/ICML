# v6.2 (GSM8K + prompt compliance)

v6.2 focuses on **prompt format compliance** for GSM8K: making the model output **only** the final integer answer, especially for small models (1B).

## What changed vs v6

- Adds `--prompt_style auto|zero|fewshot` (default `auto`)
  - `auto`: uses few-shot for `*1B*` model paths, zero-shot otherwise
  - `zero`: single-turn instruction prompt (works well for 3B/8B)
  - `fewshot`: multi-turn few-shot prompt (helps 1B follow output format)
- Adds `prompt_debug.py` to log question/prompt/generated tokens and verify output format.

## Prepare GSM8K locally

This writes local JSON files under `data/gsm8k/` (ignored by git):

```bash
python v6.2/prepare_gsm8k.py --num_samples 50
python v6.2/prepare_gsm8k.py --num_samples 500
```

## Prompt I/O debug (recommended before running big experiments)

Generates a full per-sample report including:
question, full prompt, generated token ids, generated tokens, raw text, and summary stats.

```bash
python v6.2/prompt_debug.py \
  --model /home/jinsk/Models/Llama-3.2-1B-Instruct \
  --data_path data/gsm8k/gsm8k_test_50.json \
  --num_samples 50 \
  --out_dir v6.2/results_prompt_debug/llama3.2-1b
```

Outputs:
- `v6.2/results_prompt_debug/<model>/prompt_debug.md`
- `v6.2/results_prompt_debug/<model>/prompt_debug.json`

## Run a single experiment

Example (3B, all MLP, 500 samples, steps=10):

```bash
python v6.2/experiment_v6.py \
  --data_path data/gsm8k/gsm8k_test_500.json \
  --num_samples 500 \
  --model /home/jinsk/Models/Llama-3.2-3B-Instruct \
  --prompt_style auto \
  --update_target mlp \
  --num_layers all \
  --steps 10 \
  --lr 0.001 \
  --eval_mode ar \
  --tta_reset sample \
  --backup_on_cpu \
  --output_dir v6.2/results/gsm8k500/llama3.2-3b/demo \
  --config_name gsm8k500_llama3.2-3b_mlp_all_steps10_lr1e-3
```

## Run the full grid (LR sweep + steps sweep)

```bash
python v6.2/run_gsm8k_grid.py --only_lr_search
python v6.2/run_gsm8k_grid.py --only_grid
```

Aggregated plots/tables are written to `v6.2/results/gsm8k500/summary/`.

