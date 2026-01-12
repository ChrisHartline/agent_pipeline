# GCP Training Quickstart

This guide walks through running Clara personality fine-tuning on Vertex AI.

## Prerequisites

1. **GCP Project**: `celestinecircle`
2. **Artifact Registry**: `clara-training-repo` (us-east1)
3. **GCS Bucket**: `training_datasets_ai` (existing bucket)

## Setup

### 1. Authenticate with GCP

```bash
# Login (opens browser)
gcloud auth application-default login

# Set project
gcloud config set project celestinecircle

# Verify
gcloud auth list
```

### 2. GCS Bucket

Bucket `training_datasets_ai` already exists with training data:
```bash
# List contents
gsutil ls gs://training_datasets_ai/
```

### 3. Artifact Registry (already exists)

Repository `clara-training-repo` is in us-east1:
```bash
# List repositories
gcloud artifacts repositories list --location=us-east1
```

### 4. Enable Required APIs

```bash
gcloud services enable \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com
```

## Build and Push Docker Image

```bash
# Build with Cloud Build (recommended)
gcloud builds submit --config cloudbuild.yaml

# Or build locally and push
docker build -t us-east1-docker.pkg.dev/celestinecircle/clara-training-repo/clara-train:latest .
docker push us-east1-docker.pkg.dev/celestinecircle/clara-training-repo/clara-train:latest
```

## Run Training

### Option A: Submit to Vertex AI

```bash
# Dry run first (shows what would be submitted)
python scripts/submit_vertex_job.py --dry-run

# Submit for real
python scripts/submit_vertex_job.py \
    --data data/sft/casual_train.jsonl \
    --epochs 3 \
    --batch-size 4

# Monitor at:
# https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=celestinecircle
```

### Option B: Run Locally (requires GPU)

```bash
# Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes

# Run training
python scripts/train_tinyllama_warmth.py \
    --dataset data/sft/casual_train.jsonl \
    --output-dir ./output/tinyllama-warmth \
    --epochs 3
```

## Cost Estimate

| Resource | Hourly Cost | Expected Duration | Total |
|----------|-------------|-------------------|-------|
| T4 GPU (n1-standard-4) | ~$0.54/hr | 1-2 hours | $0.50-$1.00 |
| A100 GPU | ~$4.00/hr | 30 min | ~$2.00 |

For TinyLlama (1.1B params) with 6700 examples, expect:
- T4: ~1-2 hours
- A100: ~30 minutes

**Your $20 budget**: ~35 T4-hours or ~5 A100-hours

## Training Data

| Dataset | Examples | Description |
|---------|----------|-------------|
| `casual_train.jsonl` | 6,705 | Warm/casual conversation style |
| `casual_test.jsonl` | 745 | Held-out test set |
| `warmth_sft.jsonl` | 2,000 | Explicit warmth transformations |
| `all_personality_sft.jsonl` | 8,000 | All personality dimensions |

## Output

After training, find outputs in:
- **Vertex AI**: `gs://training_datasets_ai/output/run_<timestamp>/`
- **Local**: `./output/tinyllama-warmth/`

Output includes:
- `adapter_model.safetensors` - LoRA weights
- `adapter_config.json` - LoRA configuration
- `training_metadata.json` - Run metrics

## Troubleshooting

### "Permission denied" on GCS
```bash
gcloud auth application-default login
```

### "Image not found" on Vertex AI
```bash
gcloud builds submit --config cloudbuild.yaml
```

### "Quota exceeded"
Check your quota at: https://console.cloud.google.com/iam-admin/quotas?project=celestinecircle

Filter for "NVIDIA T4" and request increase if needed.
