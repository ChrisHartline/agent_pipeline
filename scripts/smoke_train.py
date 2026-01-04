#!/usr/bin/env python3
"""Lightweight smoke training script.

This simulates a training run by invoking `Trainer.train()`, creating a small
placeholder adapter file (to emulate a LoRA adapter), and registering the run
in the registry. It is intended for quick verification and does not perform
any GPU-based training.
"""
import argparse
import json
import os
from pathlib import Path

from clara_prototype.train import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="HF slug or local base path")
    parser.add_argument("--data", required=True, help="Local json or HF dataset id")
    parser.add_argument("--out", default="models/ft/smoke_run", help="Output directory")
    parser.add_argument("--mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--register", action="store_true", help="Register model in registry.json")
    parser.add_argument("--prune", type=int, default=0, help="Prune registry to keep N entries (0=no prune)")
    args = parser.parse_args()

    trainer = Trainer(base_model=args.base, data=args.data, mode=args.mode, out_dir=args.out)
    meta = trainer.train()

    # Create a placeholder "adapter" file to emulate a LoRA adapter
    os.makedirs(args.out, exist_ok=True)
    adapter_path = Path(args.out) / "adapter.placeholder"
    adapter_path.write_text(json.dumps({"note": "placeholder adapter", "base": args.base}))
    print(f"Created placeholder adapter at {adapter_path}")

    # Update metadata to include adapter
    meta_file = Path(args.out) / "metadata.json"
    meta_data = json.loads(meta_file.read_text())
    meta_data["adapter"] = str(adapter_path)
    meta_file.write_text(json.dumps(meta_data, indent=2))

    if args.register:
        try:
            import subprocess
            cmd = ["python", "scripts/register_model.py", "--path", args.out, "--name", f"smoke_{meta.get('run_id')}", "--base", args.base]
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Registration failed: {e}")

    if args.prune and args.prune > 0:
        try:
            import subprocess
            cmd = ["python", "scripts/prune_models.py", "--keep", str(args.prune)]
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Prune failed: {e}")

    print("Smoke run complete.")


if __name__ == "__main__":
    main()
