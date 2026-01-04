#!/usr/bin/env python3
"""Convert a JSON dataset (list of objects) into JSONL train/valid/test splits and metadata.

Specifically supports objects with fields 'prompt' and 'chosen' (from your file), mapping to 'input' and 'output'.
"""
import argparse
import json
import random
from pathlib import Path


def convert_and_split(src: Path, out_dir: Path, train_frac=0.9, valid_frac=0.05, seed=42):
    with src.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Expected top-level JSON list')

    # Map items to standard schema
    items = []
    for obj in data:
        inp = obj.get('prompt') or obj.get('input') or obj.get('instruction')
        out = obj.get('chosen') or obj.get('output') or obj.get('response')
        if inp is None or out is None:
            continue
        text = format_sft_prompt(inp, out)
        items.append({'input': inp, 'output': out, 'text': text, 'meta': {'idx': obj.get('idx')}})


def format_sft_prompt(human: str, assistant: str) -> str:
    """Format a Human/Assistant SFT style prompt.

    Example:
        Human: How are you?
        Assistant: I'm fine.
    """
    human = human.strip()
    assistant = assistant.strip()
    return f"Human: {human}\nAssistant: {assistant}"
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    train = items[:n_train]
    valid = items[n_train:n_train + n_valid]
    test = items[n_train + n_valid:]

    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, lst):
        with path.open('w', encoding='utf-8') as f:
            for obj in lst:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    write_jsonl(out_dir / 'train.jsonl', train)
    write_jsonl(out_dir / 'valid.jsonl', valid)
    write_jsonl(out_dir / 'test.jsonl', test)

    metadata = {
        'name': src.stem,
        'src': str(src),
        'n_raw': n,
        'n_train': len(train),
        'n_valid': len(valid),
        'n_test': len(test),
        'schema': 'input/output',
    }
    with (out_dir / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def validate_or_prepare_dataset(path: str) -> str:
    """Return a dataset path ready for training.

    Supports local .jsonl (returned as-is), local .json (converted to train.jsonl), HF dataset ids
    (returned as-is for future handling), and gs:// paths (returned as-is).
    """
    if path.startswith("gs://"):
        return path
    if path.startswith("hf:") or "/" in path and not Path(path).exists():
        # Assume HF dataset id like 'user/dataset'
        return path

    p = Path(path)
    if p.exists():
        if p.is_file() and p.suffix == ".jsonl":
            return str(p)
        if p.is_file() and p.suffix == ".json":
            out_dir = Path("data/finetune") / p.stem
            convert_and_split(p, out_dir)
            return str(out_dir / "train.jsonl")
        if p.is_dir():
            # Prefer train.jsonl inside dir
            train = p / "train.jsonl"
            if train.exists():
                return str(train)
            # Fallback to find any .jsonl
            for f in p.glob("*.jsonl"):
                return str(f)
    # Last resort: return original path and let caller handle error
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Path to source JSON file')
    parser.add_argument('--out', required=True, help='Output folder for prepared dataset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    meta = convert_and_split(Path(args.src), Path(args.out), seed=args.seed)
    print('Wrote dataset to', args.out)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
