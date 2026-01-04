import importlib.util
from pathlib import Path

from cloud.vertex_helpers import build_peft_custom_job_spec


def test_build_peft_custom_job_spec():
    spec = build_peft_custom_job_spec(
        image_uri="gcr.io/myproj/clara:latest",
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_gs_path="gs://agent_models/train.jsonl",
        gcs_output_dir="gs://agent_models/outputs/",
        extra_args=["--smoke"],
    )
    assert "job_spec" in spec
    w = spec["job_spec"]["worker_pool_specs"][0]
    assert w["container_spec"]["image_uri"] == "gcr.io/myproj/clara:latest"
    assert "--base" in w["container_spec"]["args"]
    assert "--smoke" in w["container_spec"]["args"]