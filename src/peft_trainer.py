"""Backwards-compatible re-export of PeftTrainer.

The canonical implementation is in clara_prototype.peft_trainer.
This module re-exports it for backwards compatibility with existing imports.

DEPRECATED: Import directly from clara_prototype instead:
    from clara_prototype.peft_trainer import PeftTrainer
    from clara_prototype.config import TrainingConfig, lora_training_config
"""
import warnings

# Re-export from canonical location
from clara_prototype.peft_trainer import PeftTrainer, TrainingResult
from clara_prototype.config import (
    TrainingConfig,
    TrainingMode,
    LoraConfig,
    BitsAndBytesConfig,
    lora_training_config,
    qlora_training_config,
)

__all__ = [
    "PeftTrainer",
    "TrainingResult",
    "TrainingConfig",
    "TrainingMode",
    "LoraConfig",
    "BitsAndBytesConfig",
    "lora_training_config",
    "qlora_training_config",
]

# Emit deprecation warning on import
warnings.warn(
    "Importing from src.peft_trainer is deprecated. "
    "Please import from clara_prototype.peft_trainer instead.",
    DeprecationWarning,
    stacklevel=2
)
