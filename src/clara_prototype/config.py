"""Pydantic configuration models for Clara training pipeline.

Provides validated, typed configuration objects for:
- Training hyperparameters (LoRA, QLoRA, full fine-tuning)
- Model specifications
- HDC memory settings
- GCP/Cloud settings
- Personality specifications
"""
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingMode(str, Enum):
    """Training mode selection."""
    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"


class ComputeDtype(str, Enum):
    """Compute dtype for quantization."""
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class QuantType(str, Enum):
    """Quantization type for bitsandbytes."""
    NF4 = "nf4"
    FP4 = "fp4"


class BitsAndBytesConfig(BaseModel):
    """Configuration for bitsandbytes quantization."""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: QuantType = QuantType.NF4
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_compute_dtype: ComputeDtype = ComputeDtype.BF16

    @model_validator(mode="after")
    def validate_quantization(self) -> "BitsAndBytesConfig":
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")
        return self


class LoraConfig(BaseModel):
    """Configuration for LoRA adapters."""
    r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: Optional[List[str]] = Field(
        default=None,
        description="Target modules for LoRA. None for auto-detection."
    )
    bias: str = Field(default="none", pattern="^(none|all|lora_only)$")
    task_type: str = Field(default="CAUSAL_LM")


class TrainingConfig(BaseModel):
    """Main training configuration."""
    # Model settings
    base_model: str = Field(..., description="HuggingFace model ID or local path")
    dataset: str = Field(..., description="Dataset path (local JSONL, HF dataset ID, or gs:// path)")
    output_dir: str = Field(default="./output", description="Output directory for model artifacts")

    # Training mode
    mode: TrainingMode = Field(default=TrainingMode.LORA)

    # Hyperparameters
    learning_rate: float = Field(default=2e-4, gt=0, le=1.0)
    num_train_epochs: int = Field(default=3, ge=1)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_seq_length: int = Field(default=512, ge=64, le=32768)

    # Precision
    fp16: bool = False
    bf16: bool = False

    # LoRA config (used when mode is lora or qlora)
    lora: LoraConfig = Field(default_factory=LoraConfig)

    # Quantization config
    quantization: BitsAndBytesConfig = Field(default_factory=BitsAndBytesConfig)

    # Reproducibility
    seed: int = Field(default=42, ge=0)

    # Logging
    logging_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=200, ge=1)
    save_total_limit: Optional[int] = Field(default=3, ge=1)

    # W&B integration
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Advanced
    num_workers: int = Field(default=4, ge=0)
    gradient_checkpointing: bool = False
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)

    # Testing
    smoke_test: bool = Field(default=False, description="Run minimal smoke test without actual training")

    @model_validator(mode="after")
    def validate_precision(self) -> "TrainingConfig":
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        return self

    @field_validator("output_dir")
    @classmethod
    def ensure_output_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class GCSConfig(BaseModel):
    """Configuration for Google Cloud Storage."""
    bucket: str = Field(..., description="GCS bucket name")
    prefix: str = Field(default="", description="Prefix path within bucket")

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}".rstrip("/")


class VertexConfig(BaseModel):
    """Configuration for Vertex AI training jobs."""
    project_id: str = Field(..., description="GCP project ID")
    region: str = Field(default="us-central1")
    staging_bucket: str = Field(..., description="GCS bucket for staging")

    # Machine configuration
    machine_type: str = Field(default="n1-standard-8")
    accelerator_type: str = Field(default="NVIDIA_TESLA_T4")
    accelerator_count: int = Field(default=1, ge=1)

    # Container
    container_image_uri: Optional[str] = None

    # Job settings
    replica_count: int = Field(default=1, ge=1)
    timeout_hours: int = Field(default=24, ge=1, le=168)


class CloudConfig(BaseModel):
    """Combined cloud configuration."""
    gcs: Optional[GCSConfig] = None
    vertex: Optional[VertexConfig] = None
    push_to_gcs: bool = False
    push_to_artifact_registry: bool = False
    artifact_registry_repo: Optional[str] = None


class HDCConfig(BaseModel):
    """Configuration for Hyperdimensional Computing memory."""
    dimensions: int = Field(default=10000, ge=1000, le=100000)
    recall_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    memory_boost_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    seed: int = Field(default=42)
    debug: bool = False

    # Memory file persistence
    memory_file: Optional[str] = None


class PersonalityConfig(BaseModel):
    """Configuration for Clara's personality traits."""
    warmth: float = Field(default=0.85, ge=0.0, le=1.0)
    patience: float = Field(default=0.90, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.75, ge=0.0, le=1.0)
    encouragement: float = Field(default=0.85, ge=0.0, le=1.0)
    playful: float = Field(default=0.70, ge=0.0, le=1.0)
    formal: float = Field(default=0.30, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


class ExpertiseConfig(BaseModel):
    """Configuration for Clara's domain expertise weights."""
    medical: float = Field(default=0.7, ge=0.0, le=1.0)
    coding: float = Field(default=0.8, ge=0.0, le=1.0)
    teaching: float = Field(default=0.9, ge=0.0, le=1.0)
    quantum: float = Field(default=0.8, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


class ClaraSpec(BaseModel):
    """Complete Clara specification combining personality and expertise."""
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    expertise: ExpertiseConfig = Field(default_factory=ExpertiseConfig)

    @classmethod
    def from_file(cls, path: str) -> "ClaraSpec":
        """Load Clara spec from JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: str) -> None:
        """Save Clara spec to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration combining all components."""
    training: TrainingConfig
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    hdc: HDCConfig = Field(default_factory=HDCConfig)
    clara: ClaraSpec = Field(default_factory=ClaraSpec)

    @classmethod
    def from_file(cls, path: str) -> "PipelineConfig":
        """Load pipeline config from YAML or JSON file."""
        import json
        from pathlib import Path

        path = Path(path)
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files")
            else:
                data = json.load(f)

        return cls(**data)

    def to_file(self, path: str) -> None:
        """Save pipeline config to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


# Convenience factory functions
def training_config_from_args(**kwargs: Any) -> TrainingConfig:
    """Create TrainingConfig from keyword arguments with sensible defaults."""
    return TrainingConfig(**kwargs)


def lora_training_config(
    base_model: str,
    dataset: str,
    output_dir: str = "./output",
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    **kwargs: Any
) -> TrainingConfig:
    """Create a LoRA training configuration with common defaults."""
    return TrainingConfig(
        base_model=base_model,
        dataset=dataset,
        output_dir=output_dir,
        mode=TrainingMode.LORA,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        lora=LoraConfig(r=lora_r, lora_alpha=lora_alpha),
        **kwargs
    )


def qlora_training_config(
    base_model: str,
    dataset: str,
    output_dir: str = "./output",
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    **kwargs: Any
) -> TrainingConfig:
    """Create a QLoRA training configuration with 4-bit quantization."""
    return TrainingConfig(
        base_model=base_model,
        dataset=dataset,
        output_dir=output_dir,
        mode=TrainingMode.QLORA,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        lora=LoraConfig(r=lora_r, lora_alpha=lora_alpha),
        quantization=BitsAndBytesConfig(load_in_4bit=True),
        bf16=True,
        **kwargs
    )
