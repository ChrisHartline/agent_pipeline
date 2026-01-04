import importlib.util
from pathlib import Path
import json

from scripts import vertex_submit
vertex_submit = vertex_submit


def test_build_write_spec(tmp_path):
    out = tmp_path / "job.json"
    vertex_submit.main(["--image", "gcr.io/x/y:latest", "--base", "TinyLlama/test", "--data", "gs://agent_models/train.jsonl", "--gcs-output", "gs://agent_models/out/", "--out", str(out)])
    assert out.exists()
    data = json.loads(out.read_text())
    assert data.get("job_spec") is not None
    w = data["job_spec"]["worker_pool_specs"][0]
    assert w["container_spec"]["image_uri"] == "gcr.io/x/y:latest"
    assert "--base" in w["container_spec"]["args"]
