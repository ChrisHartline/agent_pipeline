#!/usr/bin/env python3
"""Prune oldest models to enforce a global retention policy"""
import argparse
import json
from pathlib import Path
from datetime import datetime

REGISTRY_PATH = Path("models") / "registry.json"


def load_registry():
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(entries):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def prune(keep: int = 12):
    entries = load_registry()
    if len(entries) <= keep:
        print(f"Nothing to prune ({len(entries)} <= {keep})")
        return

    # Expect entries to have `timestamp` or `date` fields, otherwise use append order
    def entry_time(e):
        for k in ("timestamp", "date"):
            if k in e and e[k]:
                try:
                    return datetime.fromisoformat(e[k])
                except Exception:
                    pass
        return datetime.min

    sorted_entries = sorted(entries, key=entry_time)
    to_remove = sorted_entries[:-keep]
    remaining = sorted_entries[-keep:]

    for r in to_remove:
        path = r.get("path")
        print(f"Pruning entry: {r.get('name') or path}")
        # Optionally delete files on disk (risky) — for now we only remove registry entries

    save_registry(remaining)
    print(f"Pruned registry to {keep} entries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=int, default=12)
    args = parser.parse_args()
    prune(keep=args.keep)


if __name__ == "__main__":
    main()
