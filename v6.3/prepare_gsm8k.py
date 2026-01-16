"""Download & preprocess GSM8K into a local JSON file for v6."""

from __future__ import annotations

import argparse
from pathlib import Path

from gsm8k_dataset import load_gsm8k, save_json


DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gsm8k"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare GSM8K (integer-only) JSON file for v6.")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Output directory for the processed JSON file.")
    p.add_argument("--split", default="test")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", default=None, help="HF datasets cache dir (optional).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_gsm8k(split=args.split, num_samples=args.num_samples, seed=args.seed, cache_dir=args.cache_dir)
    count_tag = "all" if args.num_samples <= 0 else str(args.num_samples)
    out_path = out_dir / f"gsm8k_{args.split}_{count_tag}.json"
    save_json(out_path, data)
    print(f"[gsm8k] wrote {len(data)} samples to {out_path}")


if __name__ == "__main__":
    main()
