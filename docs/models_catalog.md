# Models Catalog (short notes)

This file contains short reminders and quick notes about selected HF models (useful for agents & humans).

- google/functiongemma-270m-it
  - Type: small function-oriented model (270M).
  - Use: function synthesis/understanding, fast experimentation.
  - Notes: Good for agent skills that need to synthesize or call functions. Check license & model card.

- google/t5gemma-2-4b-4b
  - Type: T5-style seq2seq (≈4B).
  - Use: text-to-text tasks, strong for instruction->response transformations and structured output.

- unsloth/Z-Image-Turbo-GGUF
  - Type: image model (GGUF).
  - Use: inference-only image model (gguf = llama.cpp/ggml format).
  - Notes: GGUF files are inference-only; keep canonical training artifacts separately.

- mistralai/Ministral-3-8B-Instruct-2512
  - Type: 8B instruction-tuned.
  - Use: general-purpose instruction following; candidate for LoRA-based fine-tuning.
  - Notes: Large; prefer 4-bit/8-bit quantization + LoRA for local runs.

- ibm-granite/granite-docling-258M
  - Type: small doc-focused model (≈258M).
  - Use: document understanding, parsing and domain-specific extraction.

- TheDrummer/Cydonia-24B-v4.3
  - Type: 24B general model.
  - Use: high quality generation, heavy resource requirements.
  - Notes: Typically quantize + run on specialized infra; fine-tuning is expensive.

- HuggingFaceTB/SmolLM3-3B
  - Type: 3B instruction-tuned.
  - Use: fast experiments with reasonable capability for instruction tasks and agents.

- TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - Type: 1.1B chat model.
  - Use: quick local fine-tuning, experiments, toy agents.

- Phi-3 (recommended for agent use)
  - Suggested models:
    - `microsoft/Phi-3-mini-4k-instruct` (3.8B, 4K context)
    - `microsoft/Phi-3-mini-128k-instruct` (3.8B, 128K context)
    - `microsoft/Phi-3.5-mini-instruct` (latest, 128K context)
  - Type: Phi-3 family (mini variant recommended for agents).
  - License: MIT
  - Use: instruction following / agent inner loop where latency and reliability matter.
  - GCP: Available in Model Garden for deployment; fine-tuning requires GCS storage approach.
  - Notes: Phi-3 models are MIT licensed. For larger capacity, consider Phi-3-small or Phi-3-medium variants.

General notes
- GGUF files are inference-only and useful for Ollama/llama.cpp workflows. Keep safetensors/PyTorch checkpoints for training and LoRA adapters.
- Before using any model commercially, verify license and model card on Hugging Face.
- For deployment: merge adapters (if using LoRA) into a full checkpoint, then convert to GGUF via llama.cpp utilities for Ollama/CPU inference.

How to use
- Point agents to this file for short descriptions.
- Use the companion manifest (`configs/model_manifest.yaml`) for automated download scripts.
