import tempfile
from pathlib import Path
import yaml
import sys

from scripts.mergekit_run import main as merge_main


def test_mergekit_dry_run(tmp_path, capsys):
    cfg = {
        "dummy": True,
        "sources": []
    }
    cfg_path = tmp_path / "merge_test.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Run dry-run mode
    rc = merge_main(["--config", str(cfg_path), "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Merge command:" in captured.out
    assert "Dry-run mode: not executing merge" in captured.out
