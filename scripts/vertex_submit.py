"""CLI to generate Vertex CustomJob spec for PEFT runs and optionally submit it.

Usage examples:

# Generate job spec JSON
python scripts/vertex_submit.py --image gcr.io/myproj/clara:latest --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data gs://agent_models/train.jsonl --gcs-output gs://agent_models/outputs/ --out job.json

# Submit (requires google-cloud-aiplatform installed & authenticated)
python scripts/vertex_submit.py --image gcr.io/myproj/clara:latest --base TinyLlama/... --data gs://... --gcs-output gs://... --submit --project my-project --region us-central1
"""
import argparse
import json
import os
from pathlib import Path

from cloud.vertex_helpers import build_peft_custom_job_spec, submit_peft_custom_job


def build_parser():
    p = argparse.ArgumentParser(description="Build and optionally submit a Vertex CustomJob for PEFT training")
    p.add_argument("--image", required=True, help="Container image URI (Artifact Registry) to run")
    p.add_argument("--base", required=True, help="Base model slug (HF or local path)")
    p.add_argument("--data", required=True, help="gs:// path to dataset for Vertex or HF id")
    p.add_argument("--gcs-output", required=True, help="gs:// path to write outputs")
    p.add_argument("--machine-type", default="a2-highgpu-1g")
    p.add_argument("--accelerator-type", default="NVIDIA_TESLA_A100")
    p.add_argument("--accelerator-count", type=int, default=1)
    p.add_argument("--replica-count", type=int, default=1)
    p.add_argument("--extra-args", nargs="*", help="Extra args to append to the peft_train invocation (e.g. --smoke)")
    p.add_argument("--out", help="Path to write job spec JSON (default: ./vertex_job.json)")

    p.add_argument("--submit", action="store_true", help="Submit the job via google-cloud-aiplatform (requires auth)")
    p.add_argument("--project", help="GCP project id (required for submit)")
    p.add_argument("--region", help="GCP region (required for submit)")

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    spec = build_peft_custom_job_spec(
        image_uri=args.image,
        base_model=args.base,
        data_gs_path=args.data,
        gcs_output_dir=args.gcs_output,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        replica_count=args.replica_count,
        extra_args=args.extra_args,
    )

    out_path = args.out or "vertex_job.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"Wrote job spec to {out_path}")

    if args.submit:
        if not args.project or not args.region:
            raise SystemExit("Submitting requires --project and --region")
        print("Submitting to Vertex...")
        submit_peft_custom_job(project=args.project, region=args.region, job_spec=spec)
        print("Submitted job")


if __name__ == "__main__":
    main()
