"""GCS helpers for uploading/downloading artifacts used by the Clara pipeline.

Requires: google-cloud-storage

Usage:
    from cloud.gcs_helpers import upload_file, download_file, upload_dir

    upload_file('my-bucket', 'local/path/model.pt', 'models/model.pt')

Notes:
- Ensure GOOGLE_APPLICATION_CREDENTIALS points to a service account JSON with the
  needed roles (Storage Object Admin or similar).
"""
from typing import List
import os
from pathlib import Path

try:
    from google.cloud import storage
except Exception:  # pragma: no cover - import errors happen in test envs
    storage = None


def _get_client():
    if storage is None:
        raise RuntimeError("google-cloud-storage is not installed. Add it to requirements and install it.")
    return storage.Client()


def upload_file(bucket_name: str, source_file: str, dest_blob: str):
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(source_file)
    return f"gs://{bucket_name}/{dest_blob}"


def download_file(bucket_name: str, blob_name: str, dest_path: str):
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    blob.download_to_filename(dest_path)
    return dest_path


def upload_dir(bucket_name: str, source_dir: str, dest_prefix: str = "") -> List[str]:
    """Recursively upload a directory to GCS under dest_prefix and return list of GCS URIs."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    uploaded = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            local_path = os.path.join(root, f)
            rel = os.path.relpath(local_path, source_dir)
            dest_blob = os.path.join(dest_prefix, rel).replace("\\", "/")
            blob = bucket.blob(dest_blob)
            blob.upload_from_filename(local_path)
            uploaded.append(f"gs://{bucket_name}/{dest_blob}")
    return uploaded


def list_bucket(bucket_name: str, prefix: str = ""):
    client = _get_client()
    bucket = client.bucket(bucket_name)
    return [blob.name for blob in client.list_blobs(bucket, prefix=prefix)]
