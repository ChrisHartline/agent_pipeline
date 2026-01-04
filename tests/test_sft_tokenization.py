from transformers import AutoTokenizer
from clara_prototype.sft_utils import format_sft_prompt, build_sft_input_ids


def test_format_sft_prompt():
    human = "Hello, how are you?"
    assistant = "I'm fine, thanks!"
    s = format_sft_prompt(human, assistant)
    assert "Human:" in s and "Assistant:" in s


def test_build_sft_input_ids_preserve_assistant():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    human = "This is the context that might be long. " * 10
    assistant = "This is the assistant reply that we want preserved."

    # small max_length forces truncation of human while preserving assistant
    out = build_sft_input_ids(tokenizer, human, assistant, max_length=32)
    input_ids = out["input_ids"]
    labels = out["labels"]

    assert len(input_ids) <= 32
    # ensure assistant tokens appear in labels (not all -100)
    assert any(l != -100 for l in labels)
    # human prefix positions are -100
    assert labels[0] == -100


def test_build_sft_input_ids_assistant_truncation():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    human = "hi"
    assistant = "long " * 200

    out = build_sft_input_ids(tokenizer, human, assistant, max_length=16)
    input_ids = out["input_ids"]
    labels = out["labels"]

    assert len(input_ids) <= 16
    # If assistant too long, some assistant tokens must be present in labels
    assert any(l != -100 for l in labels)
