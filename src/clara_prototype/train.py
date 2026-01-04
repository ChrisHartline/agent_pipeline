"""Trainer wrapper for Clara prototype.

This provides a small, testable interface for training runs. It intentionally
keeps the implementation minimal and uses existing scripts/notebooks as the
source of truth; expand these functions to reuse and call notebook logic.
"""
import json
import os
from datetime import datetime
from typing import Optional


class Trainer:
    def __init__(self, base_model: str, data: str, mode: str = "lora", out_dir: str = "models/ft", seed: int = 42):
        self.base_model = base_model
        self.data = data
        self.mode = mode
        self.out_dir = out_dir
        self.seed = seed

    def train(self) -> dict:
        """Run a toy training workflow and write metadata.

        Replace this stub with the actual training logic that calls into
        the existing notebooks or scripts (clara_finetune.py etc.).
        """
        os.makedirs(self.out_dir, exist_ok=True)
        now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"run-{now}"
        # Minimal metadata to be extended
        meta = {
            "run_id": run_id,
            "base_model": self.base_model,
            "data": self.data,
            "mode": self.mode,
            "seed": self.seed,
            "out_dir": self.out_dir,
            "timestamp": now,
        }

        # Write metadata to out_dir
        meta_path = os.path.join(self.out_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[Trainer] Wrote metadata to {meta_path}")
        return meta

    def _get_git_sha(self) -> Optional[str]:
        """Return short git SHA for current repo, or None."""
        try:
            import subprocess
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
            return sha
        except Exception:
            return None

    def register(self, metadata: dict) -> None:
        """Register the trained model into the local registry.

        This attempts to call the `scripts.register_model.register` helper.
        If it is not available or fails, we print a warning but do not raise.
        """
        try:
            from scripts.register_model import register as register_fn
        except Exception as e:
            print(f"[Trainer] register helper not available: {e}")
            return

        # Enrich metadata for registry entry
        git_sha = self._get_git_sha()
        entry = {
            "path": str(metadata.get("out_dir")),
            "name": f"{self.base_model}_{metadata.get('run_id')}",
            "base": self.base_model,
            "run_id": metadata.get("run_id"),
            "timestamp": metadata.get("timestamp"),
            "git_sha": git_sha,
        }

        try:
            register_fn(entry)
            print(f"[Trainer] Registered model {entry.get('name')}")
        except Exception as e:
            print(f"[Trainer] Failed to register: {e}")
