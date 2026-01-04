from typing import Dict, List


def format_sft_prompt(human: str, assistant: str) -> str:
    """Format SFT prompt with 'Human' and 'Assistant' markers."""
    return f"Human: {human}\nAssistant: {assistant}"


def build_sft_input_ids(tokenizer, human: str, assistant: str, max_length: int):
    """Tokenize human+assistant while ensuring assistant tokens are preserved when truncating.

    Returns dict with 'input_ids' and 'labels' (lists of ints).
    Labels are -100 for human tokens and equal to token ids for assistant tokens.
    """
    # Build prefix (human + assistant prompt marker)
    prefix_text = f"Human: {human}\nAssistant: "
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    # Tokenize assistant and try to include EOS if tokenizer has one
    assistant_ids = tokenizer.encode(assistant, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        assistant_ids = assistant_ids + [tokenizer.eos_token_id]

    total_len = len(prefix_ids) + len(assistant_ids)

    # Truncation strategy: prefer preserving assistant tokens
    if total_len > max_length:
        # If assistant alone exceeds max_length, truncate assistant from the left (keep tail)
        if len(assistant_ids) >= max_length:
            assistant_ids = assistant_ids[-max_length:]
            prefix_ids = []
        else:
            allowed_prefix = max_length - len(assistant_ids)
            # keep the rightmost tokens of the prefix (recent context)
            prefix_ids = prefix_ids[-allowed_prefix:]

    input_ids = prefix_ids + assistant_ids
    labels = [-100] * len(prefix_ids) + assistant_ids.copy()

    return {"input_ids": input_ids, "labels": labels}
