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

## GCP Model Garden Integration

Model Garden is Google Cloud's curated repository of pre-trained models available through Vertex AI.
For models like Phi-3, there are two distinct use cases:

### Option 1: Model from GCS Storage (Training/Fine-tuning)

Use this approach when you need to **fine-tune** or **customize** the model.

```
Download from HF → Store in GCS → Fine-tune with Vertex AI CustomJob → Save adapters to GCS
```

**Workflow:**
1. Download model weights from Hugging Face (e.g., `microsoft/Phi-3-mini-4k-instruct`)
2. Upload to GCS bucket for durable storage
3. Submit a PEFT training job via Vertex AI CustomJob
4. Save LoRA adapters back to GCS

**Supported models:** Any model from `configs/model_manifest.yaml` with `trainable: true`

### Option 2: Model from Model Garden (Inference/Deployment Only)

Use this approach when you need **fast deployment** of a pre-trained model **without customization**.

```
Model Garden → Deploy to Vertex AI Endpoint → Serve predictions
```

**Workflow:**
1. Select model from Model Garden (e.g., `microsoft/phi3`)
2. Deploy to Vertex AI Endpoint using vLLM or HexLLM serving
3. Send inference requests to the endpoint

**Phi-3 variants available in Model Garden:**
| Variant | Context | GPU Requirement |
|---------|---------|-----------------|
| Phi-3-mini-4k-instruct | 4K | 1x L4 |
| Phi-3-mini-128k-instruct | 128K | 1x L4 |
| Phi-3-small-8k-instruct | 8K | 4x L4 |
| Phi-3-small-128k-instruct | 128K | 4x L4 |
| Phi-3-medium-4k-instruct | 4K | 4x L4 |
| Phi-3-medium-128k-instruct | 128K | 8x L4 |
| Phi-3.5-mini-instruct | 128K | 1x L4 |
| Phi-3.5-MoE-instruct | 128K | 8x L4 |

**Limitations:**
- Model Garden does NOT support fine-tuning for Phi-3 (as of Jan 2025)
- For fine-tuning Phi-3, use Option 1 (GCS storage approach) or Azure AI
- Llama 3 and Gemma have fine-tuning support in Model Garden; Phi-3 does not

### When to Use Each Approach

| Use Case | Recommended Approach |
|----------|---------------------|
| Fine-tune Phi-3 for Clara personality | Option 1 (GCS + CustomJob) |
| Quick inference endpoint for testing | Option 2 (Model Garden) |
| Production deployment of fine-tuned model | Option 1 (merge adapters, deploy via Model Registry) |
| Baseline comparison without tuning | Option 2 (Model Garden) |

### Model Garden Deployment Example

See the official notebook: [model_garden_phi3_deployment.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_phi3_deployment.ipynb)

```python
# Pseudo-code for Model Garden deployment
from google.cloud import aiplatform

# Initialize
aiplatform.init(project="my-project", location="us-central1")

# Deploy from Model Garden (inference only)
endpoint = aiplatform.Endpoint.create(display_name="phi3-endpoint")
model = aiplatform.Model.upload(
    display_name="phi3-mini",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:latest",
    serving_container_environment_variables={
        "MODEL_ID": "microsoft/Phi-3-mini-4k-instruct",
        "DEPLOY_SOURCE": "model_garden",
    },
)
model.deploy(endpoint=endpoint, machine_type="g2-standard-8", accelerator_type="NVIDIA_L4")
```

Notes & recommendations

- Use GCS for canonical artifact storage (model checkpoints, tokenizers, datasets). Keep local copies for fast experiments, but push final artifacts to GCS and/or HF/W&B for long-term persistence.
- For Vertex jobs use container images with all dependencies installed and rely on GCS mounted paths in the job container.
- Keep secrets & service-account JSON out of the repo. Use CI/CD secrets for automated submissions.
