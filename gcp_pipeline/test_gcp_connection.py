#!/usr/bin/env python3
"""Test GCP connectivity and list resources.

Usage:
    python gcp_pipeline/test_gcp_connection.py
"""
from config import (
    PROJECT_ID,
    REGION,
    TRAINING_DATA_BUCKET,
    MODELS_BUCKET,
    ARTIFACT_REGISTRY,
)


def test_auth():
    """Test GCP authentication."""
    print("1. Testing GCP Authentication...")
    try:
        from google.auth import default
        credentials, project = default()
        print(f"   ✓ Authenticated")
        print(f"   Project: {project}")
        return True
    except Exception as e:
        print(f"   ✗ Auth failed: {e}")
        print("   Run: gcloud auth application-default login")
        return False


def test_storage():
    """Test Cloud Storage access."""
    print("\n2. Testing Cloud Storage...")
    try:
        from google.cloud import storage
        client = storage.Client()

        # Test training data bucket
        bucket = client.bucket(TRAINING_DATA_BUCKET)
        blobs = list(bucket.list_blobs(max_results=5))
        print(f"   ✓ Bucket '{TRAINING_DATA_BUCKET}' accessible")
        print(f"   Files: {[b.name for b in blobs]}")

        # Test models bucket
        bucket = client.bucket(MODELS_BUCKET)
        blobs = list(bucket.list_blobs(max_results=5))
        print(f"   ✓ Bucket '{MODELS_BUCKET}' accessible")
        print(f"   Files: {[b.name for b in blobs]}")

        return True
    except Exception as e:
        print(f"   ✗ Storage test failed: {e}")
        return False


def test_vertex():
    """Test Vertex AI access."""
    print("\n3. Testing Vertex AI...")
    try:
        from google.cloud import aiplatform
        aiplatform.init(project=PROJECT_ID, location=REGION)

        # List recent jobs (no filter to avoid syntax issues)
        jobs = aiplatform.CustomJob.list()
        print(f"   ✓ Vertex AI accessible")
        print(f"   Total custom jobs: {len(jobs)}")

        return True
    except Exception as e:
        print(f"   ✗ Vertex AI test failed: {e}")
        print("   Enable API: gcloud services enable aiplatform.googleapis.com")
        return False


def test_artifact_registry():
    """Test Artifact Registry access."""
    print("\n4. Testing Artifact Registry...")
    try:
        from google.cloud import artifactregistry_v1
        client = artifactregistry_v1.ArtifactRegistryClient()

        parent = f"projects/{PROJECT_ID}/locations/{REGION}/repositories/clara-training-repo"
        # Try to get repo info
        repo = client.get_repository(name=parent)
        print(f"   ✓ Artifact Registry accessible")
        print(f"   Repo: {repo.name}")

        return True
    except Exception as e:
        print(f"   ✗ Artifact Registry test failed: {e}")
        return False


def main():
    print("=" * 50)
    print("GCP Connection Test")
    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print("=" * 50)

    results = {
        "auth": test_auth(),
        "storage": test_storage(),
        "vertex": test_vertex(),
        "artifact_registry": test_artifact_registry(),
    }

    print("\n" + "=" * 50)
    print("Summary:")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("=" * 50)

    if all_passed:
        print("All tests passed! Ready to submit training jobs.")
    else:
        print("Some tests failed. Fix issues above before proceeding.")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
