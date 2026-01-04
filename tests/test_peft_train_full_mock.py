import os
import json
from pathlib import Path
import importlib.util
from pathlib import Path

# Load `src/peft_trainer.py` as a module (avoid relying on package import)
from clara_prototype.peft_trainer import PeftTrainer

import scripts.register_model as register_model
import cloud.gcs_helpers as gcs_helpers


def test_full_run_mocked(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()

    # Dummy model & tokenizer with save_pretrained
    class Dummy:
        def __init__(self, out):
            self._out = out

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "pytorch_model.bin").write_text("dummy")

    # Monkeypatch load/save flows
    monkeypatch.setattr(PeftTrainer, "_load_tokenizer_and_model", lambda self: (Dummy(out), Dummy(out)))
    monkeypatch.setattr(PeftTrainer, "_prepare_datasets", lambda self: {"train": [1, 2, 3]})
    monkeypatch.setattr(PeftTrainer, "_train_with_trainer", lambda self, m, t, ds: {"trained": True})

    uploaded = {}

    def fake_upload_dir(bucket_name, source_dir, dest_prefix=""):
        uploaded['called'] = True
        uploaded['bucket'] = bucket_name
        return [f"gs://{bucket_name}/{dest_prefix}/pytorch_model.bin"]

    monkeypatch.setattr(gcs_helpers, "upload_dir", fake_upload_dir)

    registered = {}

    def fake_register(entry):
        registered['path'] = entry.get('path')
        registered['name'] = entry.get('name')

    monkeypatch.setattr(register_model, "register", fake_register)
    # Also ensure the local name used in the peft_trainer module is replaced
    import clara_prototype.peft_trainer as peft_trainer
    monkeypatch.setattr(peft_trainer, "register", fake_register)

    # Create a fake 'peft' module with required symbols so run_training will proceed
    import sys
    import types

    def _LoraConfig(**kwargs):
        return kwargs

    def _get_peft_model(model, cfg):
        return model

    def _prepare_for_kbit(model):
        return model

    peft_mod = types.SimpleNamespace(LoraConfig=_LoraConfig, get_peft_model=_get_peft_model,
                                     prepare_model_for_kbit_training=_prepare_for_kbit)
    sys.modules['peft'] = peft_mod

    trainer = PeftTrainer(
        base="TinyLlama/local",
        dataset=str(tmp_path / "tiny.jsonl"),
        out_dir=str(out),
        mode="lora",
        bnb_bit=8,
        push_to_gcs=True,
        gcs_bucket="agent_models",
        register=True,
    )

    meta = trainer.run()

    assert meta.get("trained") is not None
    assert uploaded.get('called', False) is True
    assert registered.get('path') == str(out)
