"""Convert Clara personality training data to SFT format.

The raw data has format:
{
    "neutral": "Original text",
    "low": "Low-trait version",
    "high": "High-trait version",
    "dimension": "warmth|playful|encouragement|formal"
}

For SFT we want instruction/response pairs that teach the model
to transform neutral → high-trait responses.
"""
import json
import random
from pathlib import Path
from typing import List, Dict


def convert_personality_to_sft(
    input_path: str,
    output_path: str,
    dimension: str,
    include_reverse: bool = False
) -> int:
    """Convert personality training data to SFT format.

    Args:
        input_path: Path to raw personality data JSON
        output_path: Path to write JSONL output
        dimension: Personality dimension name
        include_reverse: Also include high→neutral examples (helps avoid over-fitting)

    Returns:
        Number of examples written
    """
    with open(input_path) as f:
        data = json.load(f)

    sft_examples = []

    # Instruction templates for variety
    instructions = [
        f"Rewrite this response with more {dimension}: {{text}}",
        f"Make this more {dimension}: {{text}}",
        f"Transform this to be warmer and more {dimension}: {{text}}",
        f"Add {dimension} to this response: {{text}}",
        f"Rephrase with a {dimension} tone: {{text}}",
    ]

    for item in data:
        neutral = item.get("neutral", "")
        high = item.get("high", "")

        if not neutral or not high:
            continue

        # Main example: neutral → high
        instruction = random.choice(instructions).format(text=neutral)
        sft_examples.append({
            "instruction": instruction,
            "input": "",
            "output": high,
            "dimension": dimension
        })

        # Also add as human/assistant format for compatibility
        sft_examples.append({
            "human": f"Please rewrite this with more {dimension}: {neutral}",
            "assistant": high,
            "dimension": dimension
        })

    # Write as JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    return len(sft_examples)


def convert_all_personality_data(
    data_dir: str = "data",
    output_dir: str = "data/sft"
) -> Dict[str, int]:
    """Convert all personality training data to SFT format."""

    personality_files = {
        "warmth": "warmth_training.json",
        "playful": "playful_training.json",
        "encouragement": "encouragement_training.json",
        "formal": "formal_training.json",
    }

    results = {}
    all_examples = []

    for dimension, filename in personality_files.items():
        input_path = Path(data_dir) / filename
        output_path = Path(output_dir) / f"{dimension}_sft.jsonl"

        if not input_path.exists():
            print(f"  Skipping {dimension}: {input_path} not found")
            continue

        count = convert_personality_to_sft(
            str(input_path),
            str(output_path),
            dimension
        )
        results[dimension] = count
        print(f"  {dimension}: {count} examples → {output_path}")

        # Also collect for combined file
        with open(output_path) as f:
            for line in f:
                all_examples.append(json.loads(line))

    # Write combined file
    combined_path = Path(output_dir) / "all_personality_sft.jsonl"
    random.shuffle(all_examples)
    with open(combined_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    results["combined"] = len(all_examples)
    print(f"  Combined: {len(all_examples)} examples → {combined_path}")

    return results


if __name__ == "__main__":
    print("Converting personality data to SFT format...")
    results = convert_all_personality_data()
    print(f"\nDone! Total: {sum(results.values())} examples")
