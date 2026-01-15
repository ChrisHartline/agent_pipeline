"""GCP Pipeline Configuration."""

# Project settings
PROJECT_ID = "celestinecircle"
REGION = "us-east1"

# Cloud Storage buckets
TRAINING_DATA_BUCKET = "training_datasets_ai"
MODELS_BUCKET = "agent_models"

# Artifact Registry
ARTIFACT_REGISTRY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/clara-training-repo"
TRAINING_IMAGE = f"{ARTIFACT_REGISTRY}/clara-train:latest"
SERVING_IMAGE = f"{ARTIFACT_REGISTRY}/clara-serve:latest"

# Default paths
# NOTE: Data must be in SFT format (instruction/input/output), not DPO format (prompt/chosen/rejected)
# Upload SFT data first: gsutil cp data/sft/casual_conversation_sft.jsonl gs://training_datasets_ai/sft/
TRAINING_DATA_PATH = "gs://training_datasets_ai/sft/casual_conversation_sft.jsonl"
MODEL_OUTPUT_PATH = f"gs://{MODELS_BUCKET}/output"
BASE_MODEL_PATH = f"gs://{MODELS_BUCKET}/models"

# Machine specs for Vertex AI (T4 GPU - quota approved)
MACHINE_TYPE = "n1-standard-8"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1
