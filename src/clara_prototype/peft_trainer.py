"""PeftTrainer: Production-ready LoRA/QLoRA training for Clara.

This is the single source of truth for PEFT training. The previous duplicate
at src/peft_trainer.py has been archived to archive/peft_trainer_legacy.py.

Features:
- Full LoRA and QLoRA training with PEFT library
- Pydantic-based configuration validation
- W&B integration for experiment tracking
- GCS upload support for cloud workflows
- SFT data collator for proper label masking
- Smoke test mode for CI/CD pipelines
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import TrainingConfig, TrainingMode

logger = logging.getLogger(__name__)


class TrainingResult:
    """Result of a training run."""

    def __init__(
        self,
        success: bool,
        output_dir: str,
        metadata: Dict[str, Any],
        error: Optional[str] = None
    ):
        self.success = success
        self.output_dir = output_dir
        self.metadata = metadata
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "metadata": self.metadata,
            "error": self.error,
        }


class PeftTrainer:
    """Production-ready PEFT trainer for LoRA and QLoRA fine-tuning.

    This trainer supports:
    - LoRA fine-tuning with customizable rank and alpha
    - QLoRA with 4-bit or 8-bit quantization
    - SFT-style data collation with proper label masking
    - W&B experiment tracking
    - GCS artifact upload
    - Model registry integration

    Example:
        >>> from clara_prototype.config import lora_training_config
        >>> config = lora_training_config(
        ...     base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ...     dataset="./data/training.jsonl",
        ...     output_dir="./output/my-model"
        ... )
        >>> trainer = PeftTrainer(config)
        >>> result = trainer.train()
        >>> print(result.success)
    """

    def __init__(
        self,
        config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
        *,
        # Legacy compatibility parameters (deprecated - use config instead)
        base: Optional[str] = None,
        dataset: Optional[str] = None,
        out_dir: Optional[str] = None,
        mode: str = "lora",
        bnb_bit: int = 8,
        bnb_quant_type: str = "nf4",
        bnb_double_quant: bool = False,
        bnb_compute_dtype: str = "fp16",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        per_device_batch_size: int = 4,
        accumulate_grad: int = 1,
        lr: float = 2e-4,
        epochs: int = 3,
        max_tokens: int = 512,
        fp16: bool = False,
        bf16: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        push_to_gcs: bool = False,
        gcs_bucket: Optional[str] = None,
        push_image_to_ar: bool = False,
        artifact_repo: Optional[str] = None,
        register: bool = False,
        prune: Optional[int] = None,
        export_gguf: bool = False,
        smoke: bool = False,
        conservative: bool = False,
        seed: int = 42,
        num_workers: int = 4
    ):
        # Handle both new config-based and legacy parameter-based initialization
        if config is not None and isinstance(config, TrainingConfig):
            self.config = config
        elif config is not None and isinstance(config, dict):
            self.config = TrainingConfig(**config)
        elif base is not None and dataset is not None:
            # Legacy mode: build config from individual parameters
            from .config import LoraConfig, BitsAndBytesConfig, QuantType, ComputeDtype

            lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

            # Map legacy bnb_bit to new config
            load_8bit = bnb_bit == 8
            load_4bit = bnb_bit == 4

            quant_cfg = BitsAndBytesConfig(
                load_in_8bit=load_8bit,
                load_in_4bit=load_4bit,
                bnb_4bit_quant_type=QuantType(bnb_quant_type) if bnb_quant_type in ("nf4", "fp4") else QuantType.NF4,
                bnb_4bit_use_double_quant=bnb_double_quant,
            )

            self.config = TrainingConfig(
                base_model=base,
                dataset=dataset,
                output_dir=out_dir or "./output",
                mode=TrainingMode(mode) if mode in ("lora", "qlora", "full") else TrainingMode.LORA,
                learning_rate=lr,
                num_train_epochs=epochs,
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=accumulate_grad,
                max_seq_length=max_tokens,
                fp16=fp16,
                bf16=bf16,
                lora=lora_cfg,
                quantization=quant_cfg,
                seed=seed,
                num_workers=num_workers,
                wandb_project=wandb_project,
                smoke_test=smoke,
            )

            # Store legacy-only fields
            self._legacy_wandb_run_id = wandb_run_id
            self._legacy_push_to_gcs = push_to_gcs
            self._legacy_gcs_bucket = gcs_bucket
            self._legacy_push_image_to_ar = push_image_to_ar
            self._legacy_artifact_repo = artifact_repo
            self._legacy_register = register
            self._legacy_prune = prune
            self._legacy_export_gguf = export_gguf
            self._legacy_conservative = conservative
        else:
            raise ValueError("Must provide either a TrainingConfig or base+dataset parameters")

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize state
        self._model = None
        self._tokenizer = None
        self._trainer = None

    # Convenience properties for legacy compatibility
    @property
    def base(self) -> str:
        return self.config.base_model

    @property
    def dataset(self) -> str:
        return self.config.dataset

    @property
    def out_dir(self) -> str:
        return self.config.output_dir

    @property
    def mode(self) -> str:
        return self.config.mode.value

    @property
    def smoke(self) -> bool:
        return self.config.smoke_test

    def _load_tokenizer_and_model(self):
        """Load tokenizer and model with optional quantization.

        Returns:
            Tuple of (model, tokenizer)
        """
        if self.config.smoke_test:
            return self._create_smoke_stubs()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for training. Install with: pip install transformers"
            ) from e

        logger.info(f"Loading model: {self.config.base_model}")

        # Build quantization config if needed
        bnb_config = None
        if self.config.quantization.load_in_4bit or self.config.quantization.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig as HFBnBConfig
                import torch

                compute_dtype_map = {
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                    "fp32": torch.float32,
                }
                compute_dtype = compute_dtype_map.get(
                    self.config.quantization.bnb_4bit_compute_dtype.value,
                    torch.bfloat16
                )

                bnb_config = HFBnBConfig(
                    load_in_4bit=self.config.quantization.load_in_4bit,
                    load_in_8bit=self.config.quantization.load_in_8bit,
                    bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type.value,
                    bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            except ImportError:
                logger.warning("bitsandbytes not available, loading model without quantization")

        # Load model
        model_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            use_fast=True,
            trust_remote_code=True
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer

    def _create_smoke_stubs(self):
        """Create lightweight stubs for smoke testing."""

        class ModelStub:
            def __init__(self, out_dir):
                self.out_dir = out_dir
                self.config = type("Config", (), {"_name_or_path": "stub"})()

            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "pytorch_model.bin").write_text("stub model")
                (Path(path) / "config.json").write_text('{"model_type": "stub"}')

            def to(self, device):
                return self

            def eval(self):
                return self

            def train(self, mode=True):
                return self

            def parameters(self):
                return iter([])

        class TokenizerStub:
            pad_token = "<pad>"
            eos_token = "</s>"
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, **kwargs):
                return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

            def encode(self, text, **kwargs):
                return [1, 2, 3]

            def decode(self, ids, **kwargs):
                return "stub output"

            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "tokenizer_config.json").write_text("{}")

            def pad(self, batch, **kwargs):
                return batch

        return ModelStub(self.config.output_dir), TokenizerStub()

    def _prepare_datasets(self):
        """Load and prepare training dataset.

        Supports:
        - Local JSONL files
        - HuggingFace dataset IDs
        - (Future) GCS paths
        """
        dataset_path = self.config.dataset

        if dataset_path.startswith("gs://"):
            raise NotImplementedError(
                "GCS dataset support coming soon. For now, download locally or use HF dataset ID."
            )

        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "datasets library is required. Install with: pip install datasets"
            ) from e

        path = Path(dataset_path)
        if path.exists():
            if path.suffix == ".jsonl":
                ds = load_dataset("json", data_files={"train": str(path)})
            elif path.suffix == ".json":
                ds = load_dataset("json", data_files={"train": str(path)})
            elif path.is_dir():
                # Assume directory contains train.jsonl
                train_file = path / "train.jsonl"
                if train_file.exists():
                    ds = load_dataset("json", data_files={"train": str(train_file)})
                else:
                    raise FileNotFoundError(f"No train.jsonl found in {path}")
            else:
                raise ValueError(f"Unsupported dataset format: {path.suffix}")
        else:
            # Assume HuggingFace dataset ID
            ds = load_dataset(dataset_path)

        logger.info(f"Loaded dataset with {len(ds['train'])} examples")
        return ds

    def _build_data_collator(self, tokenizer):
        """Build SFT data collator for proper label masking."""
        try:
            from .data_collator import DataCollatorSFT
            return DataCollatorSFT(tokenizer=tokenizer, max_length=self.config.max_seq_length)
        except ImportError:
            logger.warning("SFT collator not available, using default LM collator")
            try:
                from transformers import DataCollatorForLanguageModeling
                return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            except ImportError:
                return None

    def _apply_peft(self, model):
        """Apply PEFT/LoRA configuration to model."""
        if self.config.mode == TrainingMode.FULL:
            return model

        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            raise RuntimeError(
                "peft library is required for LoRA/QLoRA. Install with: pip install peft"
            ) from e

        # Prepare for k-bit training if quantized
        if self.config.quantization.load_in_4bit or self.config.quantization.load_in_8bit:
            try:
                model = prepare_model_for_kbit_training(model)
                logger.info("Prepared model for k-bit training")
            except Exception as e:
                logger.warning(f"Could not prepare for k-bit training: {e}")

        # Configure LoRA
        lora_config = PeftLoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type,
        )

        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total parameters "
                    f"({100 * trainable / total:.2f}%)")

        return model

    def _init_wandb(self):
        """Initialize W&B tracking if configured."""
        if not self.config.wandb_project:
            return None

        try:
            import wandb

            run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.model_dump(),
            )
            logger.info(f"W&B initialized: {run.url}")
            return run
        except Exception as e:
            logger.warning(f"Could not initialize W&B: {e}")
            return None

    def train(self) -> TrainingResult:
        """Execute the training run.

        This is the main entry point for training. It handles:
        1. Loading model and tokenizer
        2. Preparing datasets
        3. Applying PEFT/LoRA if configured
        4. Running training loop
        5. Saving artifacts
        6. Returning results

        Returns:
            TrainingResult with success status and metadata
        """
        start_time = datetime.utcnow()
        run_id = f"run-{start_time.strftime('%Y%m%dT%H%M%SZ')}"

        metadata = {
            "run_id": run_id,
            "base_model": self.config.base_model,
            "dataset": self.config.dataset,
            "mode": self.config.mode.value,
            "started_at": start_time.isoformat(),
        }

        try:
            # Initialize W&B if configured
            wandb_run = self._init_wandb()

            # Smoke test fast path
            if self.config.smoke_test:
                return self._run_smoke_test(metadata)

            # Load model and tokenizer
            logger.info("Loading model and tokenizer...")
            model, tokenizer = self._load_tokenizer_and_model()
            self._model = model
            self._tokenizer = tokenizer

            # Prepare datasets
            logger.info("Preparing datasets...")
            dataset = self._prepare_datasets()

            # Apply PEFT
            logger.info(f"Applying {self.config.mode.value} configuration...")
            model = self._apply_peft(model)

            # Build data collator
            data_collator = self._build_data_collator(tokenizer)

            # Configure training arguments
            logger.info("Configuring training...")
            try:
                from transformers import TrainingArguments, Trainer
            except ImportError as e:
                raise RuntimeError("transformers is required for training") from e

            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                warmup_ratio=self.config.warmup_ratio,
                weight_decay=self.config.weight_decay,
                gradient_checkpointing=self.config.gradient_checkpointing,
                dataloader_num_workers=self.config.num_workers,
                seed=self.config.seed,
                report_to="wandb" if self.config.wandb_project else "none",
                push_to_hub=False,
                # Keep raw columns so DataCollatorSFT can tokenize them
                remove_unused_columns=False,
            )

            # Get train dataset
            train_dataset = dataset["train"]

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            self._trainer = trainer

            # Train!
            logger.info("Starting training...")
            train_result = trainer.train()

            # Log training metrics
            metadata["train_loss"] = train_result.training_loss
            metadata["train_runtime"] = train_result.metrics.get("train_runtime")
            metadata["train_samples_per_second"] = train_result.metrics.get("train_samples_per_second")

            # Save model and tokenizer
            logger.info(f"Saving model to {self.config.output_dir}")
            model.save_pretrained(self.config.output_dir)
            tokenizer.save_pretrained(self.config.output_dir)

            # Save training metadata
            metadata["completed_at"] = datetime.utcnow().isoformat()
            metadata["success"] = True
            self._save_metadata(metadata)

            # Finish W&B run
            if wandb_run:
                wandb_run.finish()

            logger.info("Training completed successfully!")
            return TrainingResult(
                success=True,
                output_dir=self.config.output_dir,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            metadata["error"] = str(e)
            metadata["success"] = False
            self._save_metadata(metadata)

            return TrainingResult(
                success=False,
                output_dir=self.config.output_dir,
                metadata=metadata,
                error=str(e)
            )

    def _run_smoke_test(self, metadata: Dict[str, Any]) -> TrainingResult:
        """Run minimal smoke test without actual training."""
        logger.info("Running smoke test...")

        model, tokenizer = self._create_smoke_stubs()

        # Write placeholder artifacts
        model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        adapter_path = Path(self.config.output_dir) / "adapter.pt"
        adapter_path.write_text("placeholder adapter for smoke test")

        metadata["smoke_test"] = True
        metadata["smoke"] = True  # Legacy compatibility
        metadata["success"] = True
        metadata["completed_at"] = datetime.utcnow().isoformat()
        self._save_metadata(metadata)

        logger.info("Smoke test completed!")
        return TrainingResult(
            success=True,
            output_dir=self.config.output_dir,
            metadata=metadata
        )

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save training metadata to output directory."""
        metadata_path = Path(self.config.output_dir) / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")

    # Legacy compatibility methods

    def run_training(self) -> Dict[str, Any]:
        """Legacy method - use train() instead."""
        result = self.train()
        return result.metadata

    def run(self) -> Dict[str, Any]:
        """Legacy method - use train() instead."""
        result = self.train()

        # Handle legacy registration
        if getattr(self, "_legacy_register", False):
            self.register_and_push(result.metadata)

        return result.metadata

    def register_and_push(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Register model and optionally push to GCS."""
        # Register locally
        try:
            from scripts.register_model import register
            entry = {
                "path": self.config.output_dir,
                "name": metadata.get("name", f"run-{Path(self.config.output_dir).name}"),
                "base": self.config.base_model,
                "timestamp": metadata.get("timestamp"),
            }
            register(entry)
            logger.info("Registered model locally")
        except Exception as e:
            logger.warning(f"Could not register model: {e}")

        # Push to GCS if configured
        gcs_bucket = getattr(self, "_legacy_gcs_bucket", None)
        if getattr(self, "_legacy_push_to_gcs", False) and gcs_bucket:
            try:
                from cloud.gcs_helpers import upload_dir
                prefix = Path(self.config.output_dir).name
                uploaded = upload_dir(gcs_bucket, self.config.output_dir, dest_prefix=prefix)
                metadata["gcs_uploaded_files"] = uploaded
                logger.info(f"Uploaded to GCS: gs://{gcs_bucket}/{prefix}")
            except Exception as e:
                metadata["gcs_upload_error"] = str(e)
                logger.warning(f"GCS upload failed: {e}")

        return metadata

    def _build_bnb_config(self) -> Dict[str, Any]:
        """Legacy method for backwards compatibility."""
        return {
            "load_in_8bit": self.config.quantization.load_in_8bit,
            "load_in_4bit": self.config.quantization.load_in_4bit,
            "quant_type": self.config.quantization.bnb_4bit_quant_type.value,
            "double_quant": self.config.quantization.bnb_4bit_use_double_quant,
        }
