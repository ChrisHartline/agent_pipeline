from pathlib import Path
import json

import scripts.register_model as reg_mod
import scripts.prune_models as prune_mod


def test_register_and_prune(tmp_path):
    # Use a temp registry path to avoid touching repo state
    registry_file = tmp_path / "registry.json"
    reg_mod.REGISTRY_PATH = registry_file
    prune_mod.REGISTRY_PATH = registry_file

    # Ensure starting clean
    assert not registry_file.exists()

    entry1 = {"path": "models/ft/tinyllama_v1", "name": "tinyllama_v1", "base": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
    entry2 = {"path": "models/ft/tinyllama_v2", "name": "tinyllama_v2", "base": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}

    reg_mod.register(entry1)
    reg_mod.register(entry2)

    # Verify registry has 2 entries
    entries = json.loads(registry_file.read_text(encoding='utf-8'))
    assert len(entries) == 2

    # Prune to keep=1
    prune_mod.prune(keep=1)
    entries = json.loads(registry_file.read_text(encoding='utf-8'))
    assert len(entries) == 1

    # Remaining entry should be the most recent (entry2)
    assert entries[0]["path"] == "models/ft/tinyllama_v2"
