#!/usr/bin/env python3
"""List Vertex AI models and endpoints in your GCP project.

Usage:
    python scripts/list_vertex_models.py --project celestinecircle --region us-central1
"""
import argparse
from google.cloud import aiplatform


def list_models(project: str, region: str):
    """List all models in the Vertex AI Model Registry."""
    aiplatform.init(project=project, location=region)

    print("\n" + "=" * 60)
    print("VERTEX AI MODELS (Model Registry)")
    print("=" * 60)

    models = aiplatform.Model.list()
    if not models:
        print("  No models found in registry.")
    else:
        for model in models:
            print(f"\n  Name: {model.display_name}")
            print(f"  Resource: {model.resource_name}")
            print(f"  Created: {model.create_time}")
            if model.deployed_models:
                print(f"  Deployed to: {len(model.deployed_models)} endpoint(s)")


def list_endpoints(project: str, region: str):
    """List all endpoints and their deployed models."""
    aiplatform.init(project=project, location=region)

    print("\n" + "=" * 60)
    print("VERTEX AI ENDPOINTS")
    print("=" * 60)

    endpoints = aiplatform.Endpoint.list()
    if not endpoints:
        print("  No endpoints found.")
    else:
        for endpoint in endpoints:
            print(f"\n  Name: {endpoint.display_name}")
            print(f"  Resource: {endpoint.resource_name}")
            print(f"  >>> ADK model string: {endpoint.resource_name}")

            # List deployed models on this endpoint
            if endpoint.traffic_split:
                print(f"  Traffic split: {endpoint.traffic_split}")


def list_tuned_models(project: str, region: str):
    """List fine-tuned/tuning jobs."""
    aiplatform.init(project=project, location=region)

    print("\n" + "=" * 60)
    print("TUNING JOBS (Fine-tuned Models)")
    print("=" * 60)

    try:
        from google.cloud.aiplatform import TuningJob
        jobs = TuningJob.list()
        if not jobs:
            print("  No tuning jobs found.")
        else:
            for job in jobs:
                print(f"\n  Name: {job.display_name}")
                print(f"  State: {job.state}")
                print(f"  Tuned model: {getattr(job, 'tuned_model_name', 'N/A')}")
    except Exception as e:
        print(f"  Could not list tuning jobs: {e}")


def list_custom_jobs(project: str, region: str):
    """List recent custom training jobs."""
    aiplatform.init(project=project, location=region)

    print("\n" + "=" * 60)
    print("CUSTOM JOBS (Training Runs)")
    print("=" * 60)

    jobs = aiplatform.CustomJob.list(order_by="create_time desc")
    if not jobs:
        print("  No custom jobs found.")
    else:
        for job in jobs[:10]:  # Last 10
            print(f"\n  Name: {job.display_name}")
            print(f"  State: {job.state}")
            print(f"  Created: {job.create_time}")


def main():
    parser = argparse.ArgumentParser(description="List Vertex AI resources")
    parser.add_argument("--project", default="celestinecircle", help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    args = parser.parse_args()

    print(f"\nScanning project: {args.project} in {args.region}")

    list_models(args.project, args.region)
    list_endpoints(args.project, args.region)
    list_tuned_models(args.project, args.region)
    list_custom_jobs(args.project, args.region)

    print("\n" + "=" * 60)
    print("HOW TO USE IN ADK (lily agent)")
    print("=" * 60)
    print("""
In your lily/agent.py, use the endpoint resource string:

    from google.adk.agents import LlmAgent

    agent = LlmAgent(
        model="projects/celestinecircle/locations/us-central1/endpoints/ENDPOINT_ID",
        name="clara",
        instruction="Your system prompt here...",
    )
""")


if __name__ == "__main__":
    main()
