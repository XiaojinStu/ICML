"""Download & preprocess SVAMP into a local JSON file.

This script loads the local SVAMP parquet files (if present under `data/SVAMP`)
and writes a filtered JSON with integer-only answers suitable for the repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from svamp_dataset import load_svamp
from gsm8k_dataset import save_json


DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" 


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare SVAMP JSON file.")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Output directory for the processed JSON file.")
    p.add_argument("--split", default="test")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default=None, help="Local SVAMP dataset dir (optional).")
    p.add_argument("--use_de", action="store_true", help="Prefer German translated fields if available.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_svamp(split=args.split, num_samples=args.num_samples, seed=args.seed, data_dir=args.data_dir, use_de=args.use_de)
    count_tag = "all" if args.num_samples <= 0 else str(args.num_samples)
    out_path = out_dir / f"svamp_{args.split}_{count_tag}.json"
    save_json(out_path, data)
    print(f"[svamp] wrote {len(data)} samples to {out_path}")


if __name__ == "__main__":
    main()
