#!/usr/bin/env python3
"""Train TinyLlama with warmth personality using LoRA.

This script fine-tunes TinyLlama-1.1B-Chat on warmth personality data
using LoRA for efficient training.

Usage:
    python scripts/train_tinyllama_warmth.py

    # With custom output directory
    python scripts/train_tinyllama_warmth.py --output-dir ./output/warmth-v1

    # Smoke test (no actual training)
    python scripts/train_tinyllama_warmth.py --smoke

Requirements:
    pip install torch transformers peft datasets accelerate bitsandbytes
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train TinyLlama with warmth personality")
    parser.add_argument("--output-dir", default="./output/tinyllama-warmth", help="Output directory")
    parser.add_argument("--dataset", default="./data/sft/warmth_sft.jsonl", help="Training data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only")
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    args = parser.parse_args()

    # Handle GCS paths - download if needed
    dataset_path = args.dataset
    if dataset_path.startswith("gs://"):
        logger.info(f"Downloading from GCS: {dataset_path}")
        try:
            from google.cloud import storage
            import tempfile

            # Parse GCS path
            gcs_parts = dataset_path.replace("gs://", "").split("/", 1)
            bucket_name, blob_name = gcs_parts[0], gcs_parts[1]

            # Download to temp file
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            local_path = Path(tempfile.gettempdir()) / Path(blob_name).name
            blob.download_to_filename(str(local_path))
            dataset_path = str(local_path)
            logger.info(f"Downloaded to: {dataset_path}")
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            sys.exit(1)
    elif not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Run: python scripts/convert_to_sft.py")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("TinyLlama Warmth Training")
    logger.info("=" * 60)

    # Import here to fail fast if deps missing
    try:
        from clara_prototype import PeftTrainer, lora_training_config
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install torch transformers peft datasets accelerate")
        sys.exit(1)

    # Create configuration
    config = lora_training_config(
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dataset=dataset_path,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,  # Common practice: alpha = 2 * r
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_length,
        bf16=True,  # TinyLlama works well with bf16
        wandb_project=args.wandb_project,
        smoke_test=args.smoke,
    )

    logger.info(f"Config:")
    logger.info(f"  Base model: {config.base_model}")
    logger.info(f"  Dataset: {config.dataset}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  LoRA rank: {config.lora.r}")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")

    # Create trainer and run
    trainer = PeftTrainer(config)
    result = trainer.train()

    if result.success:
        logger.info("=" * 60)
        logger.info("✓ Training completed successfully!")
        logger.info(f"  Output: {result.output_dir}")
        if "train_loss" in result.metadata:
            logger.info(f"  Final loss: {result.metadata['train_loss']:.4f}")
        logger.info("=" * 60)
    else:
        logger.error(f"Training failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
