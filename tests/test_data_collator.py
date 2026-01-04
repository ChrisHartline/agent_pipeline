from transformers import AutoTokenizer
import torch
from clara_prototype.data_collator import DataCollatorSFT


def test_data_collator_masks_human_tokens():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorSFT(tokenizer=tokenizer, max_length=32)
    features = [{"human": "Hello", "assistant": "Hi there"}]
    batch = collator(features)

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    assert input_ids.shape == labels.shape
    # there should be at least one label token that is not -100
    assert (labels != -100).any()
    # prefix human tokens should be -100 -- check first token
    assert labels[0, 0].item() == -100


def test_data_collator_padding_and_truncation():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorSFT(tokenizer=tokenizer, max_length=10)
    features = [
        {"human": "one", "assistant": "two"},
        {"human": "short", "assistant": "response that is longer and will be truncated"},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
    # labels must contain -100 padding for human tokens and valid ids for assistant
    assert (batch["labels"] == -100).sum() > 0
    assert (batch["labels"] != -100).sum() > 0
