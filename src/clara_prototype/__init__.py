"""Clara prototype package.

This package provides the core components for Clara's E2E agent fine-tuning pipeline:

- PeftTrainer: Production-ready LoRA/QLoRA training
- Config models: Pydantic-based validated configuration
- HDCMemory: Hyperdimensional computing memory system
- DataCollatorSFT: SFT-style data collation with label masking

Example:
    >>> from clara_prototype import PeftTrainer, lora_training_config
    >>> config = lora_training_config(
    ...     base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ...     dataset="./data/training.jsonl"
    ... )
    >>> trainer = PeftTrainer(config)
    >>> result = trainer.train()
"""

__version__ = "0.2.0"

# Core trainer
from .peft_trainer import PeftTrainer, TrainingResult
from .train import Trainer

# Configuration
from .config import (
    TrainingConfig,
    TrainingMode,
    LoraConfig,
    BitsAndBytesConfig,
    GCSConfig,
    VertexConfig,
    CloudConfig,
    HDCConfig,
    PersonalityConfig,
    ExpertiseConfig,
    ClaraSpec,
    PipelineConfig,
    lora_training_config,
    qlora_training_config,
)

# Data handling (requires torch)
try:
    from .data_collator import DataCollatorSFT
except ImportError:
    DataCollatorSFT = None  # torch not installed

# HDC Memory
from .hdc_memory import ClaraHDCMemory, Memory

__all__ = [
    # Version
    "__version__",
    # Trainers
    "PeftTrainer",
    "TrainingResult",
    "Trainer",
    # Config models
    "TrainingConfig",
    "TrainingMode",
    "LoraConfig",
    "BitsAndBytesConfig",
    "GCSConfig",
    "VertexConfig",
    "CloudConfig",
    "HDCConfig",
    "PersonalityConfig",
    "ExpertiseConfig",
    "ClaraSpec",
    "PipelineConfig",
    # Config factories
    "lora_training_config",
    "qlora_training_config",
    # Data
    "DataCollatorSFT",
    # HDC Memory
    "ClaraHDCMemory",
    "Memory",
]
