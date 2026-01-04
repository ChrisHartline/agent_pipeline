"""Vertex AI helpers and job spec templates.

This module contains helpers to construct a Vertex CustomJob spec for running
training on A100s and an example of how to submit via `google.cloud.aiplatform`.

Important: The code below uses placeholders; the user must supply project/bucket
and service account details.
"""
from typing import Dict, Any

try:
    from google.cloud import aiplatform
except Exception:  # pragma: no cover - optional dependency
    aiplatform = None


def build_custom_job_spec(
    image_uri: str,
    worker_pool_specs: list,
    base_output_dir: str,
    display_name: str = "clara-finetune-job",
) -> Dict[str, Any]:
    """Return a minimal CustomJob spec dict suitable for Vertex.

    Example worker_pool_specs:
    [
        {
            "machine_spec": {"machine_type": "n1-standard-32", "accelerator_type": "NVIDIA_TESLA_A100", "accelerator_count": 1},
            "replica_count": 1,
            "container_spec": {"image_uri": "gcr.io/my-project/my-image:latest", "command": [], "args": []}
        }
    ]

    base_output_dir: a GCS path e.g. gs://my-bucket/clara-runs/
    """
    return {
        "display_name": display_name,
        "job_spec": {
            "worker_pool_specs": worker_pool_specs,
            "base_output_directory": {"output_uri_prefix": base_output_dir},
        },
    }


# New helper for PEFT specific job specs
def build_peft_custom_job_spec(
    image_uri: str,
    base_model: str,
    data_gs_path: str,
    gcs_output_dir: str,
    machine_type: str = "a2-highgpu-1g",
    accelerator_type: str = "NVIDIA_TESLA_A100",
    accelerator_count: int = 1,
    replica_count: int = 1,
    extra_args: list | None = None,
) -> Dict[str, Any]:
    """Build a CustomJob spec dict tailored for `scripts/peft_train.py`.

    The returned dict is compatible with the Vertex `CustomJob` YAML/JSON schema.
    """
    args = [
        "--base",
        base_model,
        "--data",
        data_gs_path,
        "--out",
        "/gcs/output",
    ]
    if extra_args:
        args.extend(extra_args)

    container_spec = {
        "image_uri": image_uri,
        "args": args,
    }

    worker = {
        "machine_spec": {
            "machine_type": machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
        },
        "replica_count": replica_count,
        "container_spec": container_spec,
    }

    return {
        "display_name": "clara-peft-job",
        "job_spec": {
            "worker_pool_specs": [worker],
            "base_output_directory": {"output_uri_prefix": gcs_output_dir},
        },
    }


def submit_peft_custom_job(project: str, region: str, job_spec: Dict[str, Any]):
    if aiplatform is None:
        raise RuntimeError("google-cloud-aiplatform is not installed. Install it to submit Vertex jobs.")

    aiplatform.init(project=project, location=region)
    custom_job = aiplatform.CustomJob(job_spec=job_spec)
    custom_job.run(sync=False)
    return custom_job

def submit_custom_job(project: str, region: str, job_spec: Dict[str, Any]):
    if aiplatform is None:
        raise RuntimeError("google-cloud-aiplatform is not installed. Install google-cloud-aiplatform to submit jobs.")

    aiplatform.init(project=project, location=region)
    job = aiplatform.CustomJob.from_dict(job_spec)
    custom_job = aiplatform.CustomJob(job_spec=job_spec)
    custom_job.run(sync=False)
    return custom_job


# Example template (documentation usage)
VERTEX_EXAMPLE = """
# Example: Vertex custom job template (YAML-like pseudo-spec)
job:
  display_name: clara-finetune-{{timestamp}}
  worker_pool_specs:
    - replica_count: 1
      machine_spec:
        machine_type: a2-highgpu-1g
        accelerator_type: NVIDIA_TESLA_A100
        accelerator_count: 1
      container_spec:
        image_uri: gcr.io/<project>/clara-train:latest
        command: ["python", "scripts/train.py"]
        args: ["--base", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--data", "gs://my-bucket/data/finetune/sample.jsonl", "--out", "/gcs/output/"]
  base_output_directory:
    output_uri_prefix: gs://my-bucket/clara-outputs/
"""
