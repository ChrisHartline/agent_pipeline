"""
Cloud Run Job: Data Validation and Conversion

Validates training data, detects format, converts if needed, and uploads to GCS.
Generates validation reports to catch issues before training.

Usage (Cloud Run Job):
    python -m pipeline.data_job \
        --input gs://training_datasets_ai/claudeOpusReasoning/claude-opus-4.5-250x.jsonl \
        --output gs://training_datasets_ai/sft/claude-opus-converted.jsonl \
        --target-format instruction_response

Environment Variables:
    GOOGLE_CLOUD_PROJECT: GCP project ID
    GCS_REPORT_BUCKET: Bucket for validation reports (optional)
"""

import os
import sys
import json
import tempfile
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from google.cloud import storage
from scripts.data_utils import (
    validate_data,
    convert_file,
    get_data_stats,
    DataFormat,
    ValidationResult
)


class DataJob:
    """Cloud Run Job for data validation and conversion."""

    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.storage_client = storage.Client(project=self.project_id)
        self.temp_dir = tempfile.mkdtemp()
        self.job_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    def download_from_gcs(self, gcs_uri: str) -> str:
        """Download a file from GCS to local temp directory."""
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")

        # Parse GCS URI
        parts = gcs_uri[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        # Download
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        local_path = os.path.join(self.temp_dir, os.path.basename(blob_name))

        print(f"📥 Downloading {gcs_uri} to {local_path}")
        blob.download_to_filename(local_path)

        file_size = os.path.getsize(local_path)
        print(f"   Downloaded {file_size / 1024 / 1024:.2f} MB")

        return local_path

    def upload_to_gcs(self, local_path: str, gcs_uri: str) -> str:
        """Upload a local file to GCS."""
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")

        # Parse GCS URI
        parts = gcs_uri[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else os.path.basename(local_path)

        # Upload
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        print(f"📤 Uploading {local_path} to {gcs_uri}")
        blob.upload_from_filename(local_path)

        file_size = os.path.getsize(local_path)
        print(f"   Uploaded {file_size / 1024 / 1024:.2f} MB")

        return gcs_uri

    def run(
        self,
        input_uri: str,
        output_uri: str,
        target_format: str = "instruction_response",
        include_thinking: bool = True,
        report_uri: Optional[str] = None,
        fail_on_invalid: bool = True
    ) -> Dict[str, Any]:
        """
        Run the data validation and conversion job.

        Args:
            input_uri: GCS URI of input data
            output_uri: GCS URI for converted output
            target_format: Target format (instruction_response, etc.)
            include_thinking: Whether to keep <think> blocks
            report_uri: GCS URI for validation report (optional)
            fail_on_invalid: Raise error if validation fails

        Returns:
            Job result dictionary
        """
        result = {
            "job_id": self.job_id,
            "input_uri": input_uri,
            "output_uri": output_uri,
            "target_format": target_format,
            "status": "started",
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Step 1: Download input data
            print("\n" + "=" * 60)
            print("STEP 1: Download Input Data")
            print("=" * 60)
            local_input = self.download_from_gcs(input_uri)
            result["input_downloaded"] = True

            # Step 2: Validate input data
            print("\n" + "=" * 60)
            print("STEP 2: Validate Input Data")
            print("=" * 60)
            validation = validate_data(local_input)
            print(validation)

            result["validation"] = {
                "format_detected": validation.format.value,
                "total_examples": validation.total_examples,
                "valid_examples": validation.valid_examples,
                "issues_count": len(validation.issues),
                "is_valid": validation.valid,
            }

            if not validation.valid and fail_on_invalid:
                result["status"] = "failed"
                result["error"] = f"Validation failed: {len(validation.issues)} issues found"
                print(f"\n❌ {result['error']}")

                # Still generate report even on failure
                if report_uri:
                    self._save_report(result, report_uri)

                return result

            # Step 3: Get statistics
            print("\n" + "=" * 60)
            print("STEP 3: Analyze Data Statistics")
            print("=" * 60)
            stats = get_data_stats(local_input)
            print(json.dumps(stats, indent=2))
            result["stats"] = stats

            # Step 4: Convert format if needed
            print("\n" + "=" * 60)
            print("STEP 4: Convert Format")
            print("=" * 60)

            target_fmt = DataFormat(target_format)
            local_output = os.path.join(self.temp_dir, "converted_output.jsonl")

            if validation.format == target_fmt:
                print(f"✅ Data already in {target_format} format, copying...")
                import shutil
                shutil.copy(local_input, local_output)
                converted, skipped = validation.valid_examples, 0
            else:
                print(f"🔄 Converting from {validation.format.value} to {target_format}...")
                converted, skipped = convert_file(
                    local_input,
                    local_output,
                    target_format=target_fmt,
                    include_thinking=include_thinking
                )

            result["conversion"] = {
                "converted": converted,
                "skipped": skipped,
                "include_thinking": include_thinking,
            }
            print(f"   Converted: {converted}, Skipped: {skipped}")

            if converted == 0:
                result["status"] = "failed"
                result["error"] = "No examples were converted"
                print(f"\n❌ {result['error']}")
                return result

            # Step 5: Validate converted output
            print("\n" + "=" * 60)
            print("STEP 5: Validate Converted Output")
            print("=" * 60)
            output_validation = validate_data(local_output, expected_format=target_fmt)
            print(output_validation)

            result["output_validation"] = {
                "format": output_validation.format.value,
                "total_examples": output_validation.total_examples,
                "valid_examples": output_validation.valid_examples,
                "is_valid": output_validation.valid,
            }

            # Step 6: Upload converted data
            print("\n" + "=" * 60)
            print("STEP 6: Upload Converted Data")
            print("=" * 60)
            self.upload_to_gcs(local_output, output_uri)
            result["output_uploaded"] = True

            # Step 7: Generate and upload report
            if report_uri:
                print("\n" + "=" * 60)
                print("STEP 7: Upload Validation Report")
                print("=" * 60)
                self._save_report(result, report_uri)

            result["status"] = "success"
            print("\n" + "=" * 60)
            print("✅ JOB COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"   Output: {output_uri}")
            print(f"   Examples: {converted}")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"\n❌ Job failed: {e}")
            import traceback
            traceback.print_exc()

            # Try to save report even on error
            if report_uri:
                try:
                    self._save_report(result, report_uri)
                except Exception:
                    pass

        return result

    def _save_report(self, result: Dict[str, Any], report_uri: str):
        """Save job report to GCS."""
        report_path = os.path.join(self.temp_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        self.upload_to_gcs(report_path, report_uri)
        print(f"   Report saved to: {report_uri}")


def main():
    parser = argparse.ArgumentParser(
        description="Data validation and conversion job for ML pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input GCS URI (gs://bucket/path/file.jsonl)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output GCS URI (gs://bucket/path/output.jsonl)"
    )
    parser.add_argument(
        "--target-format",
        default="instruction_response",
        choices=["instruction_response", "chat_messages", "human_assistant"],
        help="Target format for conversion"
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Remove <think> blocks from responses"
    )
    parser.add_argument(
        "--report",
        help="GCS URI for validation report (optional)"
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Continue even if validation fails"
    )
    parser.add_argument(
        "--project",
        help="GCP project ID (or set GOOGLE_CLOUD_PROJECT)"
    )

    args = parser.parse_args()

    job = DataJob(project_id=args.project)
    result = job.run(
        input_uri=args.input,
        output_uri=args.output,
        target_format=args.target_format,
        include_thinking=not args.no_thinking,
        report_uri=args.report,
        fail_on_invalid=not args.allow_invalid
    )

    # Exit with error code if job failed
    if result["status"] != "success":
        sys.exit(1)

    return result


if __name__ == "__main__":
    main()
