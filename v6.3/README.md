# v6.3 (GSM8K + decoding-style evaluation)

v6.3 implements a **decoding-style** (teacher-forcing) evaluation for GSM8K:

- Use the *gold answer tokenization* to decide whether the **next position** should output a **numeric token**.
- Only evaluate / adapt at those numeric-token positions.
- Feed `prompt + gold_prefix_tokens` and compare the model's numeric-subvocab prediction with the gold numeric token.

This avoids depending on strict answer formatting during free generation.

## Negative answers

Negative answers are supported:
- If the gold answer begins with `-`, that token stays in the gold prefix.
- The procedure evaluates numeric tokens that follow the sign.

## Prepare GSM8K locally

```bash
python v6.3/prepare_gsm8k.py --num_samples 50
python v6.3/prepare_gsm8k.py --num_samples 500
```

## Run v6.3 experiment

```bash
python v6.3/experiment_v6.py \
  --data_path data/gsm8k/gsm8k_test_50.json \
  --num_samples 50 \
  --model /home/jinsk/Models/Llama-3.2-1B-Instruct \
  --steps 5 \
  --lr 0.001 \
  --update_target mlp \
  --num_layers all \
  --backup_on_cpu \
  --output_dir v6.3/results/demo \
  --config_name demo_tfnum
```

## One-glance report (includes prompts + tokens)

```bash
python v6.3/decoding_debug.py \
  --model /home/jinsk/Models/Llama-3.2-1B-Instruct \
  --data_path data/gsm8k/gsm8k_test_50.json \
  --num_samples 50 \
  --steps 5 \
  --lr 0.001 \
  --update_target mlp \
  --out_dir v6.3/results_decoding_debug/llama3.2-1b
```
