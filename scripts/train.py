#!/usr/bin/env python3
"""Simple CLI wrapper for Trainer in src/clara_prototype.train"""

import argparse

from clara_prototype.train import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="HF slug or local base path")
    parser.add_argument("--data", required=True, help="Local json or HF dataset id")
    parser.add_argument("--mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--out", default="models/ft", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--register", action="store_true", help="Register trained model in registry.json")
    parser.add_argument("--prune", type=int, default=0, help="Prune registry to keep N entries (0=no prune)")
    args = parser.parse_args()

    trainer = Trainer(base_model=args.base, data=args.data, mode=args.mode, out_dir=args.out, seed=args.seed)
    meta = trainer.train()

    if args.register:
        trainer.register(meta)

    if args.prune and args.prune > 0:
        try:
            from scripts.prune_models import prune
            prune(keep=args.prune)
        except Exception as e:
            print(f"Prune failed: {e}")

    print("Done. Metadata:", meta)


if __name__ == "__main__":
    main()
