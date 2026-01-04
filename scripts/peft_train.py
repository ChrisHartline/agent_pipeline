"""CLI entrypoint for PEFT training (LoRA / QLoRA / 8-bit defaults).

This script is a thin wrapper around `src.peft_trainer.PeftTrainer` and implements
argument parsing, basic validation, and orchestration for optional pushes to GCS
and Artifact Registry.

This initial implementation provides a working smoke-mode (no-GPU) flow and
hooks for full training implementation later.
"""
import argparse
import json
import os
import sys
from datetime import datetime

from pathlib import Path

# Local import; keep relative to repo
from src.peft_trainer import PeftTrainer
from scripts.prepare_dataset import validate_or_prepare_dataset


def build_parser():
    parser = argparse.ArgumentParser(description="PEFT training CLI")
    parser.add_argument("--base", required=True, help="Base model HF slug or local path")
    parser.add_argument("--data", required=True, help="Path to dataset (jsonl), HF dataset id, or gs:// path")
    parser.add_argument("--out", default=None, help="Output path for adapters/checkpoints")

    parser.add_argument("--mode", choices=["lora", "qlora", "full"], default="lora")
    parser.add_argument("--bnb-bit", type=int, choices=[4, 8], default=8,
                        help="bitsandbytes quantization bits (default: 8)")
    parser.add_argument("--bnb-quant-type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--bnb-double-quant", action="store_true")
    parser.add_argument("--bnb-compute-dtype", choices=["bf16", "fp16", "float32"], default="fp16")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--accumulate-grad", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-id", default=None)

    parser.add_argument("--push-to-gcs", action="store_true")
    parser.add_argument("--gcs-bucket", default=os.environ.get("GCS_BUCKET", "agent_models"))

    parser.add_argument("--push-image-to-ar", action="store_true")
    parser.add_argument("--artifact-repo", default=os.environ.get("ARTIFACT_REPO", "clara-training-repo"))

    parser.add_argument("--register", action="store_true", help="Register run in models/registry.json")
    parser.add_argument("--prune", type=int, default=None, help="Prune registry to keep N entries after register")

    parser.add_argument("--export-gguf", action="store_true")

    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke training (useful for CI)")
    parser.add_argument("--conservative", action="store_true", help="Use conservative defaults for memory")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Basic dataset validation / normalization
    dataset_path = validate_or_prepare_dataset(args.data)

    # Out folder default
    out = args.out
    if out is None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out = f"models/ft/{Path(args.base).name}_{timestamp}"

    os.makedirs(out, exist_ok=True)

    trainer = PeftTrainer(
        base=args.base,
        dataset=dataset_path,
        out_dir=out,
        mode=args.mode,
        bnb_bit=args.bnb_bit,
        bnb_quant_type=args.bnb_quant_type,
        bnb_double_quant=args.bnb_double_quant,
        bnb_compute_dtype=args.bnb_compute_dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        per_device_batch_size=args.per_device_batch_size,
        accumulate_grad=args.accumulate_grad,
        lr=args.lr,
        epochs=args.epochs,
        max_tokens=args.max_tokens,
        fp16=args.fp16,
        bf16=args.bf16,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        push_to_gcs=args.push_to_gcs,
        gcs_bucket=args.gcs_bucket,
        push_image_to_ar=args.push_image_to_ar,
        artifact_repo=args.artifact_repo,
        register=args.register,
        prune=args.prune,
        export_gguf=args.export_gguf,
        smoke=args.smoke,
        conservative=args.conservative,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    metadata = trainer.run()

    # Write metadata.json
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Run complete. Metadata written to {os.path.join(out, 'metadata.json')}")


if __name__ == "__main__":
    main()
