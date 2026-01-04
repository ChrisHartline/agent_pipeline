# PEFT Training (LoRA / QLoRA)

This document explains how to run `scripts/peft_train.py` for LoRA and QLoRA experiments.

Key points:
- Default quantization: 8-bit (`--bnb-bit 8`) to maximize compatibility.
- No automatic push to Hugging Face Hub; use GCS (`--push-to-gcs --gcs-bucket agent_models`) for artifacts.
- Use `--smoke` for a quick, CI-friendly smoke-run that does not require GPUs.

Example (local smoke):

```
python scripts/peft_train.py --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data tests/fixtures/tiny_data.jsonl --smoke
```

For production runs, see `docs/gcp_deploy.md` for Artifact Registry and Vertex job templates.

## SFT prompt format and collator behavior 🔧

- Expected SFT format: `Human: <instruction or context>\nAssistant: <response>`.
- The repo includes `DataCollatorSFT` which:
  - Builds `input_ids` as the concatenation of `Human:` prefix and the assistant response tokens,
  - Produces `labels` where human tokens are `-100` (ignored by loss), and assistant tokens are the actual token ids,
  - Truncates by **preserving assistant tokens** when possible (truncate the human prefix first), and pads using `pad_token` (falls back to `eos_token`).

Example features -> collator output:

- Input feature: `{"human": "How are you?", "assistant": "Fine, thanks."}`
- Collator returns tensors with `input_ids`, `attention_mask`, and `labels` where `labels` is `[-100, -100, ..., token_ids_for_assistant]`.

This ensures training loss is only computed on the assistant response portion of each example.

