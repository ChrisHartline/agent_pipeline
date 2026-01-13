#!/usr/bin/env python3
"""Submit a custom training job to Vertex AI.

Usage:
    python gcp_pipeline/submit_training_job.py
    python gcp_pipeline/submit_training_job.py --dry-run
    python gcp_pipeline/submit_training_job.py --data gs://training_datasets_ai/custom/data.json
"""
import argparse
from datetime import datetime

from config import (
    PROJECT_ID,
    REGION,
    TRAINING_IMAGE,
    TRAINING_DATA_PATH,
    MODEL_OUTPUT_PATH,
    MACHINE_TYPE,
    ACCELERATOR_TYPE,
    ACCELERATOR_COUNT,
    MODELS_BUCKET,
)


def submit_job(
    data_path: str,
    output_path: str,
    dry_run: bool = False,
    epochs: int = 3,
    batch_size: int = 4,
):
    """Submit training job to Vertex AI."""

    from google.cloud import aiplatform

    # Initialize Vertex AI with staging bucket for temp files
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{MODELS_BUCKET}",
    )

    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"clara-tinyllama-train-{timestamp}"

    # Container args
    container_args = [
        "--dataset", data_path,
        "--output-dir", f"{output_path}/{timestamp}",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    print(f"Job Configuration:")
    print(f"  Name: {job_name}")
    print(f"  Image: {TRAINING_IMAGE}")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_path}/{timestamp}")
    print(f"  Machine: {MACHINE_TYPE} + {ACCELERATOR_COUNT}x {ACCELERATOR_TYPE}")
    print(f"  Args: {container_args}")

    if dry_run:
        print("\n[DRY RUN] Would submit job with above configuration.")
        return None

    # Create custom job (training only, no model registration)
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": MACHINE_TYPE,
                    "accelerator_type": ACCELERATOR_TYPE,
                    "accelerator_count": ACCELERATOR_COUNT,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": TRAINING_IMAGE,
                    "args": container_args,
                },
            }
        ],
        base_output_dir=output_path,
    )

    print(f"\nSubmitting job...")

    job.run(sync=False)  # Don't wait for completion

    print(f"\nJob submitted successfully!")
    print(f"Job name: {job_name}")
    print(f"\nMonitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")

    return job


def main():
    parser = argparse.ArgumentParser(description="Submit Vertex AI training job")
    parser.add_argument("--data", default=TRAINING_DATA_PATH, help="GCS path to training data")
    parser.add_argument("--output", default=MODEL_OUTPUT_PATH, help="GCS path for output")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Print config without submitting")
    args = parser.parse_args()

    submit_job(
        data_path=args.data,
        output_path=args.output,
        dry_run=args.dry_run,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
