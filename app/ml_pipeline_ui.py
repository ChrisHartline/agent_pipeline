"""
ML Pipeline UI - Streamlit app for end-to-end model fine-tuning.

Features:
1. Model Selection - Choose from supported models
2. Data Validation - Validate training data format and compatibility
3. Training - Submit fine-tuning jobs to Vertex AI
4. Status - Monitor job progress
5. Deploy - Deploy trained models to endpoints

Usage:
    streamlit run app/ml_pipeline_ui.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import pipeline modules
from pipeline.models import SUPPORTED_MODELS, get_model_config, list_models
from pipeline.model_validation import validate_for_model, recommend_model
from scripts.data_utils import validate_data, get_data_stats, DataFormat

# Try to import GCP modules
try:
    from google.cloud import storage
    from google.cloud import aiplatform
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="ML Pipeline",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-running { color: #ff9800; }
    .status-success { color: #4caf50; }
    .status-failed { color: #f44336; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "selected_model": None,
        "data_path": "",
        "validation_result": None,
        "model_validation": None,
        "gcp_project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        "gcp_region": os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        "gcs_output": "",
        "training_jobs": [],
        "current_job": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the sidebar with GCP configuration."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        st.subheader("GCP Settings")
        st.session_state.gcp_project = st.text_input(
            "Project ID",
            value=st.session_state.gcp_project,
            help="Your Google Cloud project ID"
        )
        st.session_state.gcp_region = st.selectbox(
            "Region",
            ["us-central1", "us-west1", "us-east1", "europe-west1", "asia-east1"],
            index=0,
            help="Vertex AI region"
        )

        st.divider()

        st.subheader("Storage")
        st.session_state.gcs_output = st.text_input(
            "Output Bucket",
            value=st.session_state.gcs_output or f"gs://agent_models/outputs/{datetime.now().strftime('%Y%m%d')}",
            help="GCS path for training outputs"
        )

        st.divider()

        # Status indicators
        st.subheader("Status")
        if GCP_AVAILABLE:
            st.success("GCP SDK: Connected")
        else:
            st.warning("GCP SDK: Not installed")
            st.caption("Install: `pip install google-cloud-aiplatform google-cloud-storage`")


def render_model_selection():
    """Render the model selection tab."""
    st.header("1️⃣ Select Model")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("Choose a base model for fine-tuning:")

        # Model cards
        for key, config in SUPPORTED_MODELS.items():
            with st.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    selected = st.session_state.selected_model == key
                    if st.button(
                        f"{'✅ ' if selected else ''}{config.name}",
                        key=f"model_{key}",
                        use_container_width=True,
                        type="primary" if selected else "secondary"
                    ):
                        st.session_state.selected_model = key
                        st.rerun()

                    st.caption(f"{config.description}")
                    st.caption(f"Max tokens: {config.max_tokens:,} | LoRA rank: {config.recommended_lora_r}")

                with cols[1]:
                    st.metric("Context", f"{config.max_tokens//1000}K")

                st.divider()

    with col2:
        st.subheader("Selected Model")
        if st.session_state.selected_model:
            config = SUPPORTED_MODELS[st.session_state.selected_model]
            st.info(f"**{config.name}**")
            st.json({
                "hf_id": config.hf_id,
                "max_tokens": config.max_tokens,
                "family": config.family.value,
                "lora_r": config.recommended_lora_r,
                "batch_size": config.recommended_batch_size,
            })
        else:
            st.warning("No model selected")


def render_data_validation():
    """Render the data validation tab."""
    st.header("2️⃣ Validate Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Local File", "GCS Path", "HuggingFace Dataset"],
            horizontal=True
        )

        if data_source == "Local File":
            uploaded_file = st.file_uploader(
                "Upload training data",
                type=["jsonl", "json"],
                help="Upload a JSONL or JSON file with training examples"
            )
            if uploaded_file:
                # Save to temp location
                temp_path = Path("/tmp") / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.session_state.data_path = str(temp_path)

        elif data_source == "GCS Path":
            st.session_state.data_path = st.text_input(
                "GCS Path",
                value=st.session_state.data_path or "gs://training_datasets_ai/claudeOpusReasoning/claude-opus-4.5-250x.jsonl",
                help="Full GCS path to your training data"
            )

        else:  # HuggingFace
            hf_dataset = st.text_input(
                "Dataset ID",
                value="TeichAI/claude-4.5-opus-high-reasoning-250x",
                help="HuggingFace dataset identifier"
            )
            st.session_state.data_path = f"hf://{hf_dataset}"

        # Validation button
        if st.button("🔍 Validate Data", type="primary", use_container_width=True):
            with st.spinner("Validating data..."):
                try:
                    # For local files, run validation
                    if st.session_state.data_path.startswith("/") or st.session_state.data_path.startswith("C:"):
                        result = validate_data(st.session_state.data_path)
                        st.session_state.validation_result = result

                        # Also validate for selected model
                        if st.session_state.selected_model:
                            model_result = validate_for_model(
                                st.session_state.data_path,
                                st.session_state.selected_model
                            )
                            st.session_state.model_validation = model_result
                    else:
                        st.info("GCS/HuggingFace validation requires downloading data first. Click 'Prepare Data' to proceed.")
                        st.session_state.validation_result = None

                except Exception as e:
                    st.error(f"Validation failed: {e}")

    with col2:
        st.subheader("Validation Results")

        if st.session_state.validation_result:
            result = st.session_state.validation_result

            if result.valid:
                st.success("✅ Data Valid")
            else:
                st.error("❌ Data Invalid")

            st.metric("Format", result.format.value)
            st.metric("Examples", f"{result.valid_examples}/{result.total_examples}")

            if result.issues:
                with st.expander(f"Issues ({len(result.issues)})"):
                    for issue in result.issues[:10]:
                        st.caption(f"• {issue}")

            if result.sample:
                with st.expander("Sample Example"):
                    st.json(result.sample)

        # Model-specific validation
        if st.session_state.model_validation:
            st.divider()
            st.subheader("Model Compatibility")
            mv = st.session_state.model_validation

            if mv.get("compatible"):
                st.success(f"✅ Compatible with {st.session_state.selected_model}")
            else:
                st.warning(f"⚠️ {mv.get('message', 'Check token lengths')}")

            if mv.get("stats"):
                st.json(mv["stats"])


def render_training():
    """Render the training tab."""
    st.header("3️⃣ Train Model")

    # Check prerequisites
    if not st.session_state.selected_model:
        st.warning("Please select a model first (Tab 1)")
        return

    if not st.session_state.data_path:
        st.warning("Please configure training data (Tab 2)")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Training Configuration")

        config = SUPPORTED_MODELS[st.session_state.selected_model]

        # Training parameters
        with st.expander("Training Parameters", expanded=True):
            lora_r = st.slider(
                "LoRA Rank",
                min_value=8,
                max_value=64,
                value=config.recommended_lora_r,
                step=8,
                help="Higher = more parameters, better quality, slower training"
            )

            batch_size = st.slider(
                "Batch Size",
                min_value=1,
                max_value=16,
                value=config.recommended_batch_size,
                help="Adjust based on GPU memory"
            )

            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                value=2e-4,
                format_func=lambda x: f"{x:.0e}"
            )

            epochs = st.slider(
                "Epochs",
                min_value=1,
                max_value=10,
                value=3
            )

            max_steps = st.number_input(
                "Max Steps (0 = use epochs)",
                min_value=0,
                max_value=10000,
                value=0,
                step=100
            )

        # Hardware configuration
        with st.expander("Hardware Configuration"):
            machine_type = st.selectbox(
                "Machine Type",
                ["a2-highgpu-1g", "a2-highgpu-2g", "a2-highgpu-4g", "n1-standard-8"],
                help="a2-highgpu-1g has 1 A100 GPU"
            )

            accelerator = st.selectbox(
                "Accelerator",
                ["NVIDIA_TESLA_A100", "NVIDIA_L4", "NVIDIA_TESLA_T4"],
            )

            accelerator_count = st.selectbox(
                "GPU Count",
                [1, 2, 4, 8],
                index=0
            )

        # Submit button
        st.divider()

        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not GCP_AVAILABLE:
                st.error("GCP SDK not available. Install google-cloud-aiplatform to submit jobs.")
                return

            if not st.session_state.gcp_project:
                st.error("Please configure GCP Project in sidebar")
                return

            with st.spinner("Submitting training job..."):
                try:
                    # Build job spec
                    from cloud.vertex_helpers import build_peft_custom_job_spec, submit_peft_custom_job

                    job_spec = build_peft_custom_job_spec(
                        image_uri=f"gcr.io/{st.session_state.gcp_project}/clara-train:latest",
                        base_model=config.hf_id,
                        data_gs_path=st.session_state.data_path,
                        gcs_output_dir=st.session_state.gcs_output,
                        machine_type=machine_type,
                        accelerator_type=accelerator,
                        accelerator_count=accelerator_count,
                        extra_args=[
                            "--lora-r", str(lora_r),
                            "--batch-size", str(batch_size),
                            "--lr", str(learning_rate),
                            "--epochs", str(epochs),
                        ] + (["--max-steps", str(max_steps)] if max_steps > 0 else [])
                    )

                    # Submit job
                    job = submit_peft_custom_job(
                        project=st.session_state.gcp_project,
                        region=st.session_state.gcp_region,
                        job_spec=job_spec
                    )

                    st.session_state.current_job = {
                        "name": job.display_name,
                        "resource_name": job.resource_name,
                        "state": "RUNNING",
                        "submitted_at": datetime.now().isoformat(),
                        "config": {
                            "model": config.name,
                            "lora_r": lora_r,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                        }
                    }
                    st.session_state.training_jobs.append(st.session_state.current_job)

                    st.success(f"✅ Job submitted: {job.display_name}")
                    st.balloons()

                except Exception as e:
                    st.error(f"Failed to submit job: {e}")

    with col2:
        st.subheader("Job Summary")

        config = SUPPORTED_MODELS[st.session_state.selected_model]

        st.markdown(f"""
        **Model:** {config.name}
        **Data:** `{st.session_state.data_path.split('/')[-1] if st.session_state.data_path else 'Not set'}`
        **Output:** `{st.session_state.gcs_output}`
        **Project:** {st.session_state.gcp_project}
        **Region:** {st.session_state.gcp_region}
        """)

        # Cost estimate
        st.divider()
        st.subheader("Cost Estimate")
        st.caption("A100 GPU: ~$3.67/hr on Vertex AI")

        # Recent jobs
        if st.session_state.training_jobs:
            st.divider()
            st.subheader("Recent Jobs")
            for job in reversed(st.session_state.training_jobs[-3:]):
                st.caption(f"• {job['name']} - {job['state']}")


def render_status():
    """Render the status monitoring tab."""
    st.header("4️⃣ Job Status")

    if not st.session_state.training_jobs:
        st.info("No training jobs submitted yet. Go to Tab 3 to start training.")
        return

    # Refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()

    # Job list
    for job in reversed(st.session_state.training_jobs):
        with st.expander(f"📋 {job['name']}", expanded=(job == st.session_state.current_job)):
            cols = st.columns(3)

            with cols[0]:
                state = job.get("state", "UNKNOWN")
                if state == "RUNNING":
                    st.markdown(f"**Status:** <span class='status-running'>⏳ {state}</span>", unsafe_allow_html=True)
                elif state == "SUCCEEDED":
                    st.markdown(f"**Status:** <span class='status-success'>✅ {state}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Status:** <span class='status-failed'>❌ {state}</span>", unsafe_allow_html=True)

            with cols[1]:
                st.markdown(f"**Submitted:** {job.get('submitted_at', 'Unknown')[:19]}")

            with cols[2]:
                st.markdown(f"**Model:** {job.get('config', {}).get('model', 'Unknown')}")

            # Job details
            if job.get("config"):
                st.json(job["config"])

            # Actions
            st.divider()
            bcols = st.columns(4)
            with bcols[0]:
                st.button("View Logs", key=f"logs_{job['name']}", disabled=not GCP_AVAILABLE)
            with bcols[1]:
                st.button("Cancel Job", key=f"cancel_{job['name']}", disabled=state != "RUNNING")
            with bcols[2]:
                st.button("View Output", key=f"output_{job['name']}")


def render_deploy():
    """Render the deployment tab."""
    st.header("5️⃣ Deploy Model")

    st.info("🚧 Deployment functionality coming soon!")

    st.markdown("""
    **Planned features:**
    - List trained models from Model Registry
    - Deploy to Vertex AI Endpoint
    - Configure scaling and machine type
    - Test deployed model
    - Manage endpoints
    """)

    # Placeholder for model registry
    st.subheader("Available Models")
    st.caption("Models will appear here after successful training jobs.")

    # Mock data
    with st.expander("Example: clara-tinyllama-v1"):
        st.json({
            "model_id": "clara-tinyllama-v1",
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "trained_at": "2026-01-20T15:30:00Z",
            "metrics": {
                "train_loss": 0.42,
                "eval_loss": 0.45,
            }
        })

        cols = st.columns(3)
        with cols[0]:
            st.button("Deploy to Endpoint", disabled=True)
        with cols[1]:
            st.button("Download Weights", disabled=True)
        with cols[2]:
            st.button("Test Model", disabled=True)


def main():
    """Main app entry point."""
    init_session_state()

    # Header
    st.title("🚀 ML Pipeline")
    st.caption("End-to-end model fine-tuning on Vertex AI")

    # Sidebar
    render_sidebar()

    # Main tabs
    tabs = st.tabs([
        "1️⃣ Model",
        "2️⃣ Data",
        "3️⃣ Train",
        "4️⃣ Status",
        "5️⃣ Deploy"
    ])

    with tabs[0]:
        render_model_selection()

    with tabs[1]:
        render_data_validation()

    with tabs[2]:
        render_training()

    with tabs[3]:
        render_status()

    with tabs[4]:
        render_deploy()


if __name__ == "__main__":
    main()
