from pathlib import Path
import shutil
import tempfile

import scripts.download_models as dm


def test_download_model_copies_files(tmp_path, monkeypatch):
    # Create a fake snapshot dir with files
    fake_snapshot = tmp_path / "hf_cache" / "model__test"
    fake_snapshot.mkdir(parents=True)
    (fake_snapshot / "pytorch_model.safetensors").write_text("fake")
    (fake_snapshot / "model.gguf").write_text("ggufdata")

    def fake_snapshot_download(repo_id):
        return str(fake_snapshot)

    monkeypatch.setattr(dm, 'snapshot_download', fake_snapshot_download)

    dest, ggufs = dm.download_model('owner/model')
    dest = Path(dest)

    assert (dest / "pytorch_model.safetensors").exists()
    assert len(ggufs) == 1
    assert Path(ggufs[0]).exists()
