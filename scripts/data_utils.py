"""
Data validation and format utilities for the ML pipeline.

Detects data format, validates structure, and converts between formats.

Supported formats:
- instruction_response: {"instruction": "...", "response": "..."}
- chat_messages: {"messages": [{"role": "...", "content": "..."}]}
- human_assistant: {"human": "...", "assistant": "..."}
- dpo_preference: {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
from enum import Enum


class DataFormat(Enum):
    """Supported training data formats."""
    INSTRUCTION_RESPONSE = "instruction_response"
    CHAT_MESSAGES = "chat_messages"
    HUMAN_ASSISTANT = "human_assistant"
    DPO_PREFERENCE = "dpo_preference"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool
    format: DataFormat
    total_examples: int
    valid_examples: int
    issues: List[str]
    sample: Optional[Dict[str, Any]] = None

    def __str__(self):
        status = "✅ VALID" if self.valid else "❌ INVALID"
        return f"""
{status}
Format: {self.format.value}
Examples: {self.valid_examples}/{self.total_examples} valid
Issues: {len(self.issues)}
{chr(10).join(f'  - {issue}' for issue in self.issues[:5])}
{'  ... and more' if len(self.issues) > 5 else ''}
"""


def detect_format(example: Dict[str, Any]) -> DataFormat:
    """
    Detect the format of a single example.

    Args:
        example: A single training example dict

    Returns:
        DataFormat enum indicating the detected format
    """
    keys = set(example.keys())

    # Check for DPO format
    if {"prompt", "chosen", "rejected"}.issubset(keys):
        return DataFormat.DPO_PREFERENCE

    # Check for instruction/response format
    if {"instruction", "response"}.issubset(keys):
        return DataFormat.INSTRUCTION_RESPONSE

    # Check for human/assistant format
    if {"human", "assistant"}.issubset(keys):
        return DataFormat.HUMAN_ASSISTANT

    # Check for chat messages format
    if "messages" in keys and isinstance(example.get("messages"), list):
        messages = example["messages"]
        if messages and all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
            return DataFormat.CHAT_MESSAGES

    # Check for role-based format (flat array of messages)
    if "role" in keys and "content" in keys:
        return DataFormat.CHAT_MESSAGES

    return DataFormat.UNKNOWN


def load_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Load examples from a JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                yield {"_error": f"Line {line_num}: {e}", "_line": line_num}


def load_json_array(path: str) -> Iterator[Dict[str, Any]]:
    """Load examples from a JSON array file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            yield from data
        else:
            yield data


def load_data(path: str) -> Iterator[Dict[str, Any]]:
    """Load data from JSON or JSONL file."""
    path = Path(path)

    if path.suffix == '.jsonl':
        yield from load_jsonl(str(path))
    elif path.suffix == '.json':
        # Try JSONL first (some .json files are actually JSONL)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == '[':
                    yield from load_json_array(str(path))
                else:
                    yield from load_jsonl(str(path))
        except json.JSONDecodeError:
            yield from load_jsonl(str(path))
    else:
        # Try to auto-detect
        yield from load_jsonl(str(path))


def validate_data(
    path: str,
    expected_format: Optional[DataFormat] = None,
    max_examples: int = 1000
) -> ValidationResult:
    """
    Validate a training data file.

    Args:
        path: Path to the data file
        expected_format: Expected format (auto-detect if None)
        max_examples: Maximum examples to validate

    Returns:
        ValidationResult with details
    """
    issues = []
    valid_count = 0
    total_count = 0
    detected_format = None
    sample = None

    for example in load_data(path):
        total_count += 1

        if total_count > max_examples:
            break

        # Check for load errors
        if "_error" in example:
            issues.append(example["_error"])
            continue

        # Detect format from first valid example
        fmt = detect_format(example)

        if detected_format is None:
            detected_format = fmt
            sample = example

        if fmt == DataFormat.UNKNOWN:
            issues.append(f"Example {total_count}: Unknown format, keys={list(example.keys())}")
            continue

        if expected_format and fmt != expected_format:
            issues.append(f"Example {total_count}: Expected {expected_format.value}, got {fmt.value}")
            continue

        if fmt != detected_format:
            issues.append(f"Example {total_count}: Inconsistent format ({fmt.value} vs {detected_format.value})")

        # Format-specific validation
        format_issues = _validate_example(example, fmt, total_count)
        issues.extend(format_issues)

        if not format_issues:
            valid_count += 1

    return ValidationResult(
        valid=len(issues) == 0 and valid_count > 0,
        format=detected_format or DataFormat.UNKNOWN,
        total_examples=total_count,
        valid_examples=valid_count,
        issues=issues,
        sample=sample
    )


def _validate_example(example: Dict[str, Any], fmt: DataFormat, idx: int) -> List[str]:
    """Validate a single example based on its format."""
    issues = []

    if fmt == DataFormat.INSTRUCTION_RESPONSE:
        if not example.get("instruction"):
            issues.append(f"Example {idx}: Empty instruction")
        if not example.get("response"):
            issues.append(f"Example {idx}: Empty response")

    elif fmt == DataFormat.HUMAN_ASSISTANT:
        if not example.get("human"):
            issues.append(f"Example {idx}: Empty human field")
        if not example.get("assistant"):
            issues.append(f"Example {idx}: Empty assistant field")

    elif fmt == DataFormat.DPO_PREFERENCE:
        if not example.get("prompt"):
            issues.append(f"Example {idx}: Empty prompt")
        if not example.get("chosen"):
            issues.append(f"Example {idx}: Empty chosen response")
        if not example.get("rejected"):
            issues.append(f"Example {idx}: Empty rejected response")
        if example.get("chosen") == example.get("rejected"):
            issues.append(f"Example {idx}: chosen and rejected are identical")

    elif fmt == DataFormat.CHAT_MESSAGES:
        messages = example.get("messages", [])
        if not messages:
            # Check if it's a flat message
            if "role" in example and "content" in example:
                if not example.get("content"):
                    issues.append(f"Example {idx}: Empty content")
            else:
                issues.append(f"Example {idx}: Empty messages array")
        else:
            roles = [m.get("role") for m in messages]
            if "assistant" not in roles:
                issues.append(f"Example {idx}: No assistant response in messages")

    return issues


def convert_to_instruction_response(
    example: Dict[str, Any],
    source_format: DataFormat
) -> Optional[Dict[str, str]]:
    """
    Convert an example to instruction/response format.

    Args:
        example: Source example
        source_format: Format of the source

    Returns:
        Converted example or None if conversion not possible
    """
    if source_format == DataFormat.INSTRUCTION_RESPONSE:
        return {
            "instruction": example["instruction"],
            "response": example["response"]
        }

    elif source_format == DataFormat.HUMAN_ASSISTANT:
        return {
            "instruction": example["human"],
            "response": example["assistant"]
        }

    elif source_format == DataFormat.CHAT_MESSAGES:
        messages = example.get("messages", [])

        # Handle flat message format (array of separate role/content dicts)
        if not messages and "role" in example:
            # This is part of a message array, can't convert single message
            return None

        # Find user and assistant messages
        user_content = []
        assistant_content = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system" and content:
                user_content.insert(0, f"System: {content}")
            elif role == "user":
                user_content.append(content)
            elif role == "assistant":
                assistant_content = content

        if user_content and assistant_content:
            return {
                "instruction": "\n".join(user_content),
                "response": assistant_content
            }

    return None


def convert_file(
    input_path: str,
    output_path: str,
    target_format: DataFormat = DataFormat.INSTRUCTION_RESPONSE,
    include_thinking: bool = True
) -> Tuple[int, int]:
    """
    Convert a data file to a different format.

    Args:
        input_path: Source file path
        output_path: Destination file path
        target_format: Target format
        include_thinking: Whether to include <think> blocks in responses

    Returns:
        Tuple of (converted_count, skipped_count)
    """
    converted = 0
    skipped = 0

    # First, detect source format
    validation = validate_data(input_path, max_examples=10)
    source_format = validation.format

    print(f"Source format: {source_format.value}")
    print(f"Target format: {target_format.value}")

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for example in load_data(input_path):
            if "_error" in example:
                skipped += 1
                continue

            if target_format == DataFormat.INSTRUCTION_RESPONSE:
                converted_example = convert_to_instruction_response(example, source_format)

                if converted_example:
                    # Optionally strip thinking blocks
                    if not include_thinking and "<think>" in converted_example.get("response", ""):
                        response = converted_example["response"]
                        # Remove <think>...</think> blocks
                        import re
                        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
                        converted_example["response"] = response.strip()

                    out_f.write(json.dumps(converted_example, ensure_ascii=False) + "\n")
                    converted += 1
                else:
                    skipped += 1
            else:
                # Other conversions not yet implemented
                skipped += 1

    return converted, skipped


def get_data_stats(path: str, max_examples: int = 10000) -> Dict[str, Any]:
    """
    Get statistics about a training data file.

    Args:
        path: Path to the data file
        max_examples: Maximum examples to analyze

    Returns:
        Dictionary with statistics
    """
    validation = validate_data(path, max_examples=max_examples)

    total_tokens_approx = 0
    instruction_lengths = []
    response_lengths = []

    for example in load_data(path):
        if "_error" in example:
            continue

        # Convert to get text lengths
        converted = convert_to_instruction_response(example, validation.format)
        if converted:
            inst_len = len(converted["instruction"])
            resp_len = len(converted["response"])
            instruction_lengths.append(inst_len)
            response_lengths.append(resp_len)
            # Rough token estimate: ~4 chars per token
            total_tokens_approx += (inst_len + resp_len) // 4

    return {
        "format": validation.format.value,
        "total_examples": validation.total_examples,
        "valid_examples": validation.valid_examples,
        "approx_tokens": total_tokens_approx,
        "avg_instruction_chars": sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
        "avg_response_chars": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
        "issues_count": len(validation.issues),
    }


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate and convert training data")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a data file")
    validate_parser.add_argument("path", help="Path to data file")
    validate_parser.add_argument("--format", choices=[f.value for f in DataFormat], help="Expected format")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert data format")
    convert_parser.add_argument("input", help="Input file path")
    convert_parser.add_argument("output", help="Output file path")
    convert_parser.add_argument("--target", default="instruction_response", help="Target format")
    convert_parser.add_argument("--no-thinking", action="store_true", help="Remove <think> blocks")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get data statistics")
    stats_parser.add_argument("path", help="Path to data file")

    args = parser.parse_args()

    if args.command == "validate":
        expected = DataFormat(args.format) if args.format else None
        result = validate_data(args.path, expected_format=expected)
        print(result)
        if result.sample:
            print("Sample example:")
            print(json.dumps(result.sample, indent=2, ensure_ascii=False)[:500] + "...")

    elif args.command == "convert":
        target = DataFormat(args.target)
        converted, skipped = convert_file(
            args.input, args.output,
            target_format=target,
            include_thinking=not args.no_thinking
        )
        print(f"Converted: {converted}, Skipped: {skipped}")

    elif args.command == "stats":
        stats = get_data_stats(args.path)
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()
