# Cloud Integration (GCS & Vertex AI)

This document explains how to set up and use the Google Cloud helpers in
`cloud/gcs_helpers.py` and `cloud/vertex_helpers.py`.

Setup

1. Create a GCP service account with the following roles:
   - Storage Admin / Storage Object Admin (for uploading model artifacts)
   - Vertex AI Admin (if you will submit jobs)

2. Download the service account JSON and set an environment variable:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   ```

3. Create a GCS bucket to host artifacts:

   ```bash
   gsutil mb -l us-central1 gs://my-clara-bucket
   ```

Using the GCS helpers

- Upload a file:

```python
from cloud.gcs_helpers import upload_file
upload_file('my-clara-bucket', 'local/model/safetensors', 'models/tinyllama/weights.safetensors')
```

- Upload a directory:

```python
from cloud.gcs_helpers import upload_dir
uris = upload_dir('my-clara-bucket', 'models/ft/tinyllama_v1', 'artifacts/tinyllama_v1')
```

Vertex AI

- Build a CustomJob spec using `cloud.vertex_helpers.build_custom_job_spec()` and submit using `submit_custom_job()` (the latter requires `google-cloud-aiplatform` and appropriate IAM).

- Example (pseudo):

```python
from cloud.vertex_helpers import build_custom_job_spec, submit_custom_job
spec = build_custom_job_spec(
    image_uri='gcr.io/my-project/clara-train:latest',
    worker_pool_specs=[{...}],
    base_output_dir='gs://my-clara-bucket/clara-outputs/'
)
submit_custom_job('my-gcp-project', 'us-central1', spec)
```

Notes & recommendations

- Use GCS for canonical artifact storage (model checkpoints, tokenizers, datasets). Keep local copies for fast experiments, but push final artifacts to GCS and/or HF/W&B for long-term persistence.
- For Vertex jobs use container images with all dependencies installed and rely on GCS mounted paths in the job container.
- Keep secrets & service-account JSON out of the repo. Use CI/CD secrets for automated submissions.
