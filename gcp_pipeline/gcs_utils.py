#!/usr/bin/env python3
"""GCS utility functions for data movement."""
from pathlib import Path
from typing import Optional

from google.cloud import storage

from config import TRAINING_DATA_BUCKET, MODELS_BUCKET


def upload_file(local_path: str, bucket_name: str, destination_blob: str) -> str:
    """Upload a file to GCS.

    Returns:
        GCS URI (gs://bucket/path)
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(local_path)

    gcs_uri = f"gs://{bucket_name}/{destination_blob}"
    print(f"Uploaded {local_path} -> {gcs_uri}")
    return gcs_uri


def download_file(gcs_uri: str, local_path: str) -> str:
    """Download a file from GCS.

    Args:
        gcs_uri: Full GCS path (gs://bucket/path)
        local_path: Local destination path

    Returns:
        Local path
    """
    # Parse gs:// URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)

    print(f"Downloaded {gcs_uri} -> {local_path}")
    return local_path


def list_bucket(bucket_name: str, prefix: Optional[str] = None) -> list:
    """List files in a GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    files = []
    for blob in blobs:
        files.append({
            "name": blob.name,
            "size": blob.size,
            "updated": blob.updated,
            "uri": f"gs://{bucket_name}/{blob.name}",
        })
    return files


def upload_training_data(local_path: str, destination_name: Optional[str] = None) -> str:
    """Upload training data to the training bucket.

    Args:
        local_path: Path to local training file
        destination_name: Optional name in bucket (defaults to filename)

    Returns:
        GCS URI
    """
    if destination_name is None:
        destination_name = Path(local_path).name

    return upload_file(local_path, TRAINING_DATA_BUCKET, destination_name)


def upload_model(local_dir: str, model_name: str) -> str:
    """Upload model files to the models bucket.

    Args:
        local_dir: Local directory containing model files
        model_name: Name for the model in GCS

    Returns:
        GCS URI prefix
    """
    local_path = Path(local_dir)
    if not local_path.is_dir():
        raise ValueError(f"Not a directory: {local_dir}")

    uploaded = []
    for file in local_path.glob("**/*"):
        if file.is_file():
            relative = file.relative_to(local_path)
            destination = f"{model_name}/{relative}"
            upload_file(str(file), MODELS_BUCKET, destination)
            uploaded.append(destination)

    gcs_prefix = f"gs://{MODELS_BUCKET}/{model_name}"
    print(f"Uploaded {len(uploaded)} files to {gcs_prefix}")
    return gcs_prefix


if __name__ == "__main__":
    # Quick test - list buckets
    print("Training data bucket:")
    for f in list_bucket(TRAINING_DATA_BUCKET)[:5]:
        print(f"  {f['name']} ({f['size']} bytes)")

    print("\nModels bucket:")
    for f in list_bucket(MODELS_BUCKET)[:5]:
        print(f"  {f['name']} ({f['size']} bytes)")
