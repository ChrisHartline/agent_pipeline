# Data location and policy

Local data directories:

- `data/raw/` - Raw source files (original, unprocessed).
- `data/finetune/` - Processed, ready-to-train datasets (JSONL or HuggingFace dataset exports).
- `data/working/` - Temporary transformed datasets during experiments.

Example sample dataset for quick tests:
- `data/finetune/sample.jsonl` (small file with 3 examples for smoke tests)

Recommendations:
- Keep canonical training datasets (raw + processed snapshots) for reproducibility.
- For production, push datasets to Cloud Storage (GCS) and log dataset artifacts in W&B.
- When working in Colab, mount Google Drive or upload artifacts to GCS/HF to persist them outside ephemeral VMs.

Scripts and flows:
- `scripts/train.py --data data/finetune/sample.jsonl` --example usage for local flow.
- `scripts/download_models.py` can accept a `configs/model_manifest.yaml` to obtain base models from HF.

Privacy & security:
- Do not store sensitive customer data in the repository. Use private GCS buckets and proper IAM policies for customer data and synthetic datasets.
