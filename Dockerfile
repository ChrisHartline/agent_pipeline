# Clara Training Container for Vertex AI
# Optimized for TinyLlama fine-tuning with LoRA on T4 GPU
#
# Build: gcloud builds submit --config cloudbuild.yaml
# Or locally: docker build -t clara-train .

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Clara Team"
LABEL description="Clara personality fine-tuning with LoRA/PEFT"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    datasets>=2.14.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece>=0.1.99 \
    pydantic>=2.0.0 \
    wandb \
    google-cloud-storage>=2.12.0

# Copy application code
COPY . /app

# Install package in editable mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_home

# Default command - can be overridden
ENTRYPOINT ["python", "scripts/train_tinyllama_warmth.py"]

# Default args
CMD ["--help"]
