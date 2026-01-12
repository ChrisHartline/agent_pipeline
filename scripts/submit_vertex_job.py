#!/usr/bin/env python3
"""Submit a training job to Vertex AI.

This script handles:
1. Uploading training data to GCS
2. Submitting a CustomJob to Vertex AI
3. Monitoring job status

Usage:
    # Submit with default settings (T4, TinyLlama, warmth data)
    python scripts/submit_vertex_job.py

    # Submit with custom data
    python scripts/submit_vertex_job.py --data data/sft/casual_train.jsonl

    # Dry run (show what would be submitted)
    python scripts/submit_vertex_job.py --dry-run

Requirements:
    - gcloud auth application-default login
    - Artifact Registry repo exists: clara-training-repo
    - Docker image built: gcloud builds submit --config cloudbuild.yaml
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Project configuration
PROJECT_ID = "celestinecircle"
REGION = "us-east1"
REPOSITORY = "clara-training-repo"
IMAGE_NAME = "clara-train"
BUCKET_NAME = "training_datasets_ai"  # Existing bucket with training data

# Image URI
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:latest"


def upload_to_gcs(local_path: str, gcs_path: str, bucket: str) -> str:
    """Upload a file to GCS."""
    try:
        from google.cloud import storage
    except ImportError:
        print("ERROR: google-cloud-storage not installed")
        print("Run: pip install google-cloud-storage")
        sys.exit(1)

    client = storage.Client(project=PROJECT_ID)
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(gcs_path)

    print(f"Uploading {local_path} to gs://{bucket}/{gcs_path}...")
    blob.upload_from_filename(local_path)
    return f"gs://{bucket}/{gcs_path}"


def submit_job(
    data_gcs_path: str,
    output_gcs_path: str,
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    epochs: int = 3,
    batch_size: int = 4,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    dry_run: bool = False,
) -> dict:
    """Submit a training job to Vertex AI."""
    try:
        from google.cloud import aiplatform
    except ImportError:
        print("ERROR: google-cloud-aiplatform not installed")
        print("Run: pip install google-cloud-aiplatform")
        sys.exit(1)

    # Build job spec
    job_name = f"clara-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    container_args = [
        "--dataset", data_gcs_path,
        "--output-dir", "/gcs/output",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    worker_pool_spec = {
        "machine_spec": {
            "machine_type": machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": IMAGE_URI,
            "args": container_args,
        },
    }

    job_spec = {
        "display_name": job_name,
        "job_spec": {
            "worker_pool_specs": [worker_pool_spec],
            "base_output_directory": {
                "output_uri_prefix": output_gcs_path,
            },
        },
    }

    print("\n" + "=" * 60)
    print("VERTEX AI JOB CONFIGURATION")
    print("=" * 60)
    print(f"Job name: {job_name}")
    print(f"Image: {IMAGE_URI}")
    print(f"Machine: {machine_type} + {accelerator_count}x {accelerator_type}")
    print(f"Data: {data_gcs_path}")
    print(f"Output: {output_gcs_path}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] Would submit job with spec:")
        print(json.dumps(job_spec, indent=2))
        return {"dry_run": True, "job_spec": job_spec}

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Create and submit job
    print("\nSubmitting job to Vertex AI...")
    custom_job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[worker_pool_spec],
        base_output_dir=output_gcs_path,
    )

    custom_job.submit()

    print(f"\n✓ Job submitted!")
    print(f"  Job name: {custom_job.display_name}")
    print(f"  Resource name: {custom_job.resource_name}")
    print(f"\nMonitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")

    return {
        "job_name": job_name,
        "resource_name": custom_job.resource_name,
        "status": "submitted",
    }


def main():
    parser = argparse.ArgumentParser(description="Submit Vertex AI training job")
    parser.add_argument("--data", default="data/sft/casual_train.jsonl",
                        help="Local path to training data")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--machine-type", default="n1-standard-4",
                        help="GCE machine type")
    parser.add_argument("--accelerator", default="NVIDIA_TESLA_T4",
                        choices=["NVIDIA_TESLA_T4", "NVIDIA_TESLA_A100"],
                        help="GPU type")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="GCS bucket for data")
    parser.add_argument("--dry-run", action="store_true", help="Show config without submitting")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip data upload (assume already in GCS)")
    args = parser.parse_args()

    # Check data exists
    if not args.skip_upload and not Path(args.data).exists():
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    # Generate paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_gcs_path = f"gs://{args.bucket}/data/{Path(args.data).name}"
    output_gcs_path = f"gs://{args.bucket}/output/run_{timestamp}/"

    # Upload data to GCS
    if not args.skip_upload and not args.dry_run:
        data_gcs_path = upload_to_gcs(
            args.data,
            f"data/{Path(args.data).name}",
            args.bucket,
        )

    # Submit job
    result = submit_job(
        data_gcs_path=data_gcs_path,
        output_gcs_path=output_gcs_path,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print("\n" + "=" * 60)
        print("COST ESTIMATE")
        print("=" * 60)
        if args.accelerator == "NVIDIA_TESLA_T4":
            print("T4 GPU: ~$0.35/hour")
            print("n1-standard-4: ~$0.19/hour")
            print("Total: ~$0.54/hour")
            print(f"For {args.epochs} epochs on TinyLlama: expect ~1-2 hours")
            print("Estimated cost: $0.50 - $1.00")
        else:
            print("A100 GPU: ~$3.50/hour")
            print("Estimated cost: $3-7 for full training")


if __name__ == "__main__":
    main()
