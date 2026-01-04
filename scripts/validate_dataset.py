#!/usr/bin/env python3
"""Validate a fine-tune dataset directory or JSONL file.

Checks:
- Path exists and contains at least one .jsonl file (or is a single .jsonl)
- Each JSON line is parseable JSON
- Each example contains expected fields: one of (instruction+output) or (input+output) or (messages)
- No empty outputs
- Optionally show counts and a small sample
"""
import argparse
import json
from pathlib import Path


def infer_schema(obj):
    if "instruction" in obj and "output" in obj:
        return "instruction-output"
    if "input" in obj and "output" in obj:
        return "io"
    if "messages" in obj:
        return "chat"
    return None


def validate_file(path: Path, max_error_lines:int=10):
    errors = []
    count = 0
    schema = None
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append(f"Line {i}: JSON parse error: {e}")
                if len(errors) >= max_error_lines:
                    break
                continue

            if schema is None:
                schema = infer_schema(obj)
                if schema is None:
                    errors.append(f"Line {i}: Unknown schema fields: {list(obj.keys())}")
                    if len(errors) >= max_error_lines:
                        break
            else:
                # quick check of required fields depending on schema
                if schema == "instruction-output" and (not obj.get("instruction") or not obj.get("output")):
                    errors.append(f"Line {i}: Missing instruction or output")
                if schema == "io" and (not obj.get("input") or not obj.get("output")):
                    errors.append(f"Line {i}: Missing input or output")
                if schema == "chat" and (not obj.get("messages") or not isinstance(obj.get("messages"), list)):
                    errors.append(f"Line {i}: Invalid messages format")

            count += 1
    return {"path": str(path), "count": count, "schema": schema, "errors": errors}


def find_jsonl_files(path: Path):
    if path.is_file() and path.suffix == ".jsonl":
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        return files
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to a .jsonl file or a dataset folder containing .jsonl files")
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"Path not found: {p}")

    files = find_jsonl_files(p)
    if not files:
        raise SystemExit(f"No .jsonl files found in {p}")

    print(f"Found {len(files)} jsonl file(s):")
    overall = {"total_examples":0, "schemas":{}, "errors":0}
    for f in files:
        info = validate_file(f)
        print(f" - {info['path']}: {info['count']} examples, schema={info['schema']} errors={len(info['errors'])}")
        overall['total_examples'] += info['count']
        if info['schema']:
            overall['schemas'][info['schema']] = overall['schemas'].get(info['schema'], 0) + info['count']
        overall['errors'] += len(info['errors'])
        for e in info['errors'][:5]:
            print("    ", e)

    print("\nSummary:")
    print(f"  Total examples: {overall['total_examples']}")
    print(f"  Schemas: {overall['schemas']}")
    print(f"  Total errors: {overall['errors']}")


if __name__ == "__main__":
    main()
