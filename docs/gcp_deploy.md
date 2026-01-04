# GCP Deployment & Artifact Registry

This document contains commands and snippets to create the Artifact Registry `clara-training-repo` and the GCS bucket `agent_models`.

Quick `gcloud` commands (replace PROJECT_ID and REGION):

```bash
# Enable APIs
gcloud services enable artifactregistry.googleapis.com storage.googleapis.com

# Create Artifact Registry repo
gcloud artifacts repositories create clara-training-repo --repository-format=docker --location=us-central1 --description="Clara training images"

# Create GCS bucket
gsutil mb -l us-central1 gs://agent_models
```

You can also use Terraform as part of infra provisioning; include a service account with `roles/storage.admin` and `roles/artifactregistry.writer` to CI and Vertex.
