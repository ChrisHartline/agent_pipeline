#!/usr/bin/env python3
"""Run Mergekit with a YAML/JSON config.

This script wraps the `mergekit-yaml` CLI used by the pipeline. It supports a
`--dry-run` mode that prints the command but does not execute it (useful for
CI or testing). Optionally registers the resulting model in `models/registry.json`
using `scripts/register_model.py` and can upload to Weights & Biases if requested.
"""
import argparse
import subprocess
import os
import sys
import yaml
from pathlib import Path


def build_command(config_file: Path, output_path: Path, copy_tokenizer: bool = True, use_cuda: bool = None):
    cmd = ["mergekit-yaml", str(config_file), str(output_path)]
    if copy_tokenizer:
        cmd.append("--copy-tokenizer")

    # If caller did not specify use_cuda, try to auto-detect
    if use_cuda is None:
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

    if use_cuda:
        cmd.append("--cuda")

    return cmd


def run_merge(config_path: str, out_dir: str = None, copy_tokenizer: bool = True, use_cuda: bool = None, dry_run: bool = False, register: bool = False, wandb_upload: bool = False):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    # Default output path
    if out_dir:
        output_path = Path(out_dir)
    else:
        merge_name = config_file.stem.replace("merge_", "")
        output_path = Path("models") / f"clara_{merge_name}"

    cmd = build_command(config_file, output_path, copy_tokenizer=copy_tokenizer, use_cuda=use_cuda)

    print("Merge command:")
    print("  "+" ".join(cmd))

    if dry_run:
        print("Dry-run mode: not executing merge")
        return 0

    # Execute the merge
    try:
        print("Executing merge... (this may take a while)")
        subprocess.run(cmd, check=True)
        print("Merge completed")

        # Basic sanity check on output
        if not output_path.exists():
            print(f"Warning: expected output path does not exist: {output_path}")
        else:
            total_mb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024 * 1024)
            print(f"Merged model size: {total_mb:.1f} MB")

        if register:
            try:
                # Import register function from scripts/register_model.py
                from scripts.register_model import register
                register({
                    "path": str(output_path),
                    "name": output_path.name,
                    "base": None
                })
            except Exception as e:
                print(f"Failed to register model: {e}")

        if wandb_upload:
            try:
                import wandb
                api = wandb.Api()
                print("Uploading artifact to W&B is not implemented in this script; please use W&B CLI or the monitoring helpers.")
            except Exception as e:
                print(f"W&B upload skipped: {e}")

        return 0

    except FileNotFoundError:
        print("mergekit not found! Install with: pip install mergekit")
        return 2
    except subprocess.CalledProcessError as e:
        print(f"Merge failed: {e}")
        return e.returncode


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to mergekit yaml/json config")
    parser.add_argument("--out", required=False, help="Output path for merged model")
    parser.add_argument("--no-copy-tokenizer", dest="copy_tokenizer", action="store_false", help="Do not copy tokenizer")
    parser.add_argument("--cuda", dest="cuda", action="store_true", help="Force use of CUDA")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="Force no CUDA")
    parser.add_argument("--dry-run", action="store_true", help="Print command but do not execute")
    parser.add_argument("--register", action="store_true", help="Register result in models/registry.json (registry entry only)")
    parser.add_argument("--wandb-upload", action="store_true", help="Attempt to upload merged artifact to W&B (placeholder)")

    args = parser.parse_args(argv)

    return run_merge(
        config_path=args.config,
        out_dir=args.out,
        copy_tokenizer=args.copy_tokenizer,
        use_cuda=args.cuda if args.cuda is not None else None,
        dry_run=args.dry_run,
        register=args.register,
        wandb_upload=args.wandb_upload
    )


if __name__ == "__main__":
    raise SystemExit(main())
