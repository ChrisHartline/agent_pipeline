import os
import json
import shutil
import scripts.peft_train as peft_train


def test_smoke_run_monkeypatched(tmp_path, monkeypatch):
    # Create tiny dataset fixture
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text('\n'.join(['{"input": "hi", "output": "hello"}']))

    out_dir = tmp_path / "out"

    # Monkeypatch validate_or_prepare_dataset to bypass heavy logic
    monkeypatch.setattr(peft_train, 'validate_or_prepare_dataset', lambda x: str(dataset))

    # Run CLI main in smoke mode
    argv = [
        "--base", "TinyLlama/test-tiny",
        "--data", str(dataset),
        "--out", str(out_dir),
        "--smoke",
    ]

    peft_train.main(argv)

    # Check metadata.json and adapter placeholder exist
    meta_path = out_dir / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta.get("smoke") is True

    adapter_path = out_dir / "adapter.pt"
    assert adapter_path.exists()
    assert "placeholder adapter" in adapter_path.read_text()
