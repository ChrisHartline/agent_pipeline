#!/usr/bin/env python3
"""Download models from Hugging Face based on configs/model_manifest.yaml"""
import yaml
import argparse
from huggingface_hub import snapshot_download
import os


def load_manifest(path="configs/model_manifest.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_model(hf_slug, dest_root="models/base"):
    """Download a model snapshot from HF and copy files to `dest_root/<slug>`.

    Additionally detect and copy any `.gguf` files into `models/gguf/<slug>/`.
    """
    dest = os.path.join(dest_root, hf_slug.replace("/", "__"))
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {hf_slug} -> {dest}")

    snapshot_dir = snapshot_download(repo_id=hf_slug)
    print("Downloaded to cache at:", snapshot_dir)

    # Copy all files from snapshot to dest
    try:
        import shutil
        shutil.copytree(snapshot_dir, dest, dirs_exist_ok=True)
        print(f"Copied snapshot files to {dest}")
    except Exception as e:
        print(f"Failed to copy files to {dest}: {e}")

    # Detect gguf files and copy them to models/gguf/<slug>/
    gguf_paths = []
    try:
        from pathlib import Path
        p = Path(snapshot_dir)
        for gg in p.rglob("*.gguf"):
            gg_dest_dir = Path("models") / "gguf" / hf_slug.replace("/", "__")
            gg_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gg, gg_dest_dir / gg.name)
            gguf_paths.append(str(gg_dest_dir / gg.name))
            print(f"Found GGUF: {gg.name} -> {gg_dest_dir}")
    except Exception as e:
        print(f"GGUF detection failed: {e}")

    return dest, gguf_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/model_manifest.yaml")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    for m in manifest.get("models", []):
        hf_slug = m.get("hf_slug")
        if not hf_slug:
            continue
        try:
            download_model(hf_slug)
        except Exception as e:
            print(f"Failed to download {hf_slug}: {e}")


if __name__ == "__main__":
    main()
