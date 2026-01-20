# Data Pipeline - Cloud Run Job

Validates, converts, and prepares training data for ML pipeline.

## Features

- **Format Detection**: Automatically detects data format (instruction/response, chat messages, DPO, etc.)
- **Validation**: Checks for empty fields, format consistency, and issues
- **Conversion**: Converts between formats (e.g., chat messages → instruction/response)
- **Thinking Blocks**: Optionally keeps or strips `<think>` reasoning blocks
- **Reports**: Generates JSON reports with validation results and stats

## Quick Start

### 1. Deploy the Cloud Run Job

```bash
# From repository root
gcloud builds submit --config pipeline/cloudbuild.yaml .
```

### 2. Run the Job

```bash
# Basic usage
gcloud run jobs execute data-pipeline \
  --region us-central1 \
  --args="--input,gs://training_datasets_ai/claudeOpusReasoning/claude-opus-4.5-250x.jsonl,--output,gs://training_datasets_ai/sft/claude-opus-converted.jsonl"

# With all options
gcloud run jobs execute data-pipeline \
  --region us-central1 \
  --args="--input,gs://training_datasets_ai/claudeOpusReasoning/claude-opus-4.5-250x.jsonl,--output,gs://training_datasets_ai/sft/claude-opus-converted.jsonl,--target-format,instruction_response,--report,gs://training_datasets_ai/reports/validation-report.json"
```

### 3. Check Job Status

```bash
# View job executions
gcloud run jobs executions list --job data-pipeline --region us-central1

# View logs
gcloud run jobs executions logs <execution-name> --region us-central1
```

## Local Testing

```bash
# Set up credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Run locally
python -m pipeline \
  --input gs://training_datasets_ai/claudeOpusReasoning/claude-opus-4.5-250x.jsonl \
  --output gs://training_datasets_ai/sft/claude-opus-converted.jsonl \
  --report gs://training_datasets_ai/reports/validation-report.json
```

## Supported Formats

| Format | Fields | Use Case |
|--------|--------|----------|
| `instruction_response` | `instruction`, `response` | Standard SFT training |
| `chat_messages` | `messages[]` with `role`, `content` | Chat/conversation format |
| `human_assistant` | `human`, `assistant` | Alternative SFT format |
| `dpo_preference` | `prompt`, `chosen`, `rejected` | DPO preference training |

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input, -i` | Yes | Input GCS URI |
| `--output, -o` | Yes | Output GCS URI |
| `--target-format` | No | Target format (default: instruction_response) |
| `--no-thinking` | No | Remove `<think>` blocks |
| `--report` | No | GCS URI for validation report |
| `--allow-invalid` | No | Continue even if validation fails |
| `--project` | No | GCP project ID |

## Example Output

```
============================================================
STEP 1: Download Input Data
============================================================
📥 Downloading gs://training_datasets_ai/... to /tmp/.../data.jsonl
   Downloaded 8.95 MB

============================================================
STEP 2: Validate Input Data
============================================================

✅ VALID
Format: chat_messages
Examples: 250/250 valid
Issues: 0

============================================================
STEP 4: Convert Format
============================================================
🔄 Converting from chat_messages to instruction_response...
   Converted: 250, Skipped: 0

============================================================
✅ JOB COMPLETED SUCCESSFULLY
============================================================
   Output: gs://training_datasets_ai/sft/claude-opus-converted.jsonl
   Examples: 250
```

## Integration with Training Pipeline

After data is validated and converted, trigger training:

```bash
# Example: Chain data job → training job
gcloud run jobs execute data-pipeline --args="..." --wait

# Then trigger training
gcloud ai custom-jobs create \
  --region us-central1 \
  --display-name "clara-train-$(date +%Y%m%d)" \
  --config training-job-config.yaml
```
