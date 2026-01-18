This folder is for locally-prepared GSM8K JSON files.

We intentionally do NOT commit GSM8K contents to git. Generate files locally via:

- 50-sample quick debug: `python v6.2/prepare_gsm8k.py --num_samples 50`
- 500-sample eval: `python v6.2/prepare_gsm8k.py --num_samples 500`
- full test split: `python v6.2/prepare_gsm8k.py --num_samples 0`

Files are written as `gsm8k_<split>_<count>.json` and ignored by `.gitignore`.
