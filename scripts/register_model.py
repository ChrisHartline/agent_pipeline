#!/usr/bin/env python3
"""Register a fine-tuned model into models/registry.json"""
import argparse
import json
from pathlib import Path
from datetime import datetime

REGISTRY_PATH = Path("models") / "registry.json"


def load_registry():
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(entries):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _current_iso_ts():
    return datetime.utcnow().isoformat()


def register(entry: dict):
    """Register or update a model entry in the registry.

    The function fills missing metadata (timestamp, git_sha) and updates an
    existing entry if the `path` matches; otherwise it appends a new entry.
    """
    entries = load_registry()

    # Fill metadata defaults
    if "timestamp" not in entry or not entry.get("timestamp"):
        entry["timestamp"] = _current_iso_ts()

    if "git_sha" not in entry:
        # Attempt to set git sha; import locally to avoid global dep
        try:
            import subprocess
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
            entry["git_sha"] = sha
        except Exception:
            entry["git_sha"] = None

    # Deduplicate by path
    path = entry.get("path")
    if path:
        for i, e in enumerate(entries):
            if e.get("path") == path:
                entries[i] = {**e, **entry}  # update
                save_registry(entries)
                print(f"Updated registry entry for: {entry.get('name') or path}")
                return

    entries.append(entry)
    save_registry(entries)
    print(f"Registered model: {entry.get('name') or entry.get('run_id')}")


def upload_registry_to_gcs(bucket_name: str, dest_path: str = "registry.json"):
    """Upload the local registry.json to the specified GCS bucket.

    Uses `google.cloud.storage`. If the package is not installed, raises ImportError.
    """
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Local registry not found at {REGISTRY_PATH}")

    try:
        from google.cloud import storage
    except Exception as e:
        raise ImportError(
            "google-cloud-storage is required to upload registry to GCS. "
            "Install it with `pip install google-cloud-storage` and ensure GCP creds are set." ) from e

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(str(REGISTRY_PATH))
    print(f"Uploaded registry to gs://{bucket_name}/{dest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to model directory")
    parser.add_argument("--name", required=False, help="Optional friendly name")
    parser.add_argument("--base", required=False, help="Base model slug")
    args = parser.parse_args()

    entry = {
        "path": args.path,
        "name": args.name,
        "base": args.base,
    }
    register(entry)


if __name__ == "__main__":
    main()
