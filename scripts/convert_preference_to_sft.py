"""Convert preference data (prompt/chosen/rejected) to SFT format.

This handles the casual conversation dataset format:
{
    "idx": 0,
    "prompt": "hi, how are you doing?",
    "chosen": "i'm fine. how about yourself?",
    "rejected": "Hello! I'm doing well, thank you. How about yourself?"
}

The 'chosen' responses are warm/casual, 'rejected' are formal.
"""
import json
import random
from pathlib import Path
from typing import List, Dict


def convert_preference_to_sft(
    input_path: str,
    output_path: str,
    use_chosen: bool = True,
) -> int:
    """Convert preference data to SFT format.

    Args:
        input_path: Path to preference data JSON
        output_path: Path to write JSONL output
        use_chosen: If True, use 'chosen' responses; if False, use 'rejected'

    Returns:
        Number of examples written
    """
    with open(input_path) as f:
        data = json.load(f)

    sft_examples = []

    for item in data:
        prompt = item.get("prompt", "")
        response = item.get("chosen" if use_chosen else "rejected", "")

        if not prompt or not response:
            continue

        # Format as instruction/output (Alpaca-style)
        sft_examples.append({
            "instruction": prompt,
            "input": "",
            "output": response,
        })

        # Also add as human/assistant format
        sft_examples.append({
            "human": prompt,
            "assistant": response,
        })

    # Write as JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    return len(sft_examples)


def main():
    input_path = "data/finetune/casual_conv_DP0/casual-conversation-poo.json"
    output_path = "data/sft/casual_conversation_sft.jsonl"

    print(f"Converting {input_path}...")
    count = convert_preference_to_sft(input_path, output_path, use_chosen=True)
    print(f"Wrote {count} examples to {output_path}")

    # Also create a small test set
    with open(output_path) as f:
        lines = f.readlines()

    random.shuffle(lines)
    train_lines = lines[:int(len(lines) * 0.9)]
    test_lines = lines[int(len(lines) * 0.9):]

    train_path = "data/sft/casual_train.jsonl"
    test_path = "data/sft/casual_test.jsonl"

    with open(train_path, "w") as f:
        f.writelines(train_lines)
    with open(test_path, "w") as f:
        f.writelines(test_lines)

    print(f"Split: {len(train_lines)} train, {len(test_lines)} test")


if __name__ == "__main__":
    main()
