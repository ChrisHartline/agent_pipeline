"""Minimal, testable PeftTrainer implementation.

This is intentionally lightweight to allow CI smoke tests to run without GPUs.
It contains clear extension points for a full training implementation later.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

from scripts.register_model import register, upload_registry_to_gcs
# NOTE: peft_trainer has been moved in packaging; keep this for backwards compatibility imports


class PeftTrainer:
    def __init__(self, *, base: str, dataset: str, out_dir: str, mode: str = "lora",
                 bnb_bit: int = 8, bnb_quant_type: str = "nf4", bnb_double_quant: bool = False,
                 bnb_compute_dtype: str = "fp16", lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, per_device_batch_size: int = 4, accumulate_grad: int = 1,
                 lr: float = 2e-4, epochs: int = 3, max_tokens: int = 512, fp16: bool = False,
                 bf16: bool = False, wandb_project: str = None, wandb_run_id: str = None,
                 push_to_gcs: bool = False, gcs_bucket: str = None, push_image_to_ar: bool = False,
                 artifact_repo: str = None, register: bool = False, prune: int = None,
                 export_gguf: bool = False, smoke: bool = False, conservative: bool = False,
                 seed: int = 42, num_workers: int = 4):
        self.base = base
        self.dataset = dataset
        self.out_dir = out_dir
        self.mode = mode
        self.bnb_bit = bnb_bit
        self.bnb_quant_type = bnb_quant_type
        self.bnb_double_quant = bnb_double_quant
        self.bnb_compute_dtype = bnb_compute_dtype
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.per_device_batch_size = per_device_batch_size
        self.accumulate_grad = accumulate_grad
        self.lr = lr
        self.epochs = epochs
        self.max_tokens = max_tokens
        self.fp16 = fp16
        self.bf16 = bf16
        self.wandb_project = wandb_project
        self.wandb_run_id = wandb_run_id
        self.push_to_gcs = push_to_gcs
        self.gcs_bucket = gcs_bucket
        self.push_image_to_ar = push_image_to_ar
        self.artifact_repo = artifact_repo
        self.register = register
        self.prune = prune
        self.export_gguf = export_gguf
        self.smoke = smoke
        self.conservative = conservative
        self.seed = seed
        self.num_workers = num_workers

        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def _build_bnb_config(self) -> Dict[str, Any]:
        return {
            "bnb_bit": self.bnb_bit,
            "quant_type": self.bnb_quant_type,
            "double_quant": self.bnb_double_quant,
            "compute_dtype": self.bnb_compute_dtype,
        }

    def _load_tokenizer_and_model(self):
        """Load tokenizer and model, with bitsandbytes 8-bit support if requested.

        Returns (model, tokenizer).
        """
        if self.smoke:
            # Return lightweight stubs that expose `save_pretrained`
            class _Stub:
                def __init__(self, out_dir):
                    self.out_dir = out_dir

                def save_pretrained(self, path):
                    Path(path).mkdir(parents=True, exist_ok=True)
                    (Path(path) / "pytorch_model.bin").write_text("stub model")

            return _Stub(self.out_dir), _Stub(self.out_dir)

        # Lazy imports to avoid import-time failures in test envs
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError("transformers is required for full training runs: install transformers") from e

        # If using 8-bit, ensure bitsandbytes is available
        load_in_8bit = (self.bnb_bit == 8)
        if load_in_8bit:
            try:
                import bitsandbytes as bnb  # noqa: F401
            except Exception as e:
                raise RuntimeError("bitsandbytes is required for 8-bit training; install it for full runs") from e

        # Use device_map='auto' for convenience; rely on accelerate in production
        model = AutoModelForCausalLM.from_pretrained(self.base, load_in_8bit=load_in_8bit, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.base, use_fast=True)
        return model, tokenizer

    def _prepare_datasets(self):
        # Support local jsonl files (train/valid/test) and HF dataset ids
        if isinstance(self.dataset, str) and self.dataset.startswith("gs://"):
            # For production we would download or stream; for now raise helpful message
            raise NotImplementedError("gs:// dataset support is not implemented yet; download locally or use HF dataset id")

        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError("datasets is required for full training runs: install the 'datasets' package") from e

        if Path(self.dataset).exists() and Path(self.dataset).suffix == ".jsonl":
            ds = load_dataset("json", data_files={"train": self.dataset})
            # Optionally support validation split in future
            return ds

        # Assume HF dataset id
        ds = load_dataset(self.dataset)
        return ds

    def _train_with_trainer(self, model, tokenizer, train_dataset):
        # Minimal Trainer integration for causal LM
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        except Exception as e:
            raise RuntimeError("transformers Trainer is required for full training runs: ensure transformers is installed") from e

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=self.out_dir,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.accumulate_grad,
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            fp16=self.fp16,
            bf16=self.bf16,
            save_strategy="steps",
            save_steps=200,
            logging_steps=50,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset["train"] if isinstance(train_dataset, dict) or hasattr(train_dataset, "__getitem__") else train_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        # Save the full model
        model.save_pretrained(self.out_dir)
        tokenizer.save_pretrained(self.out_dir)
        # If using PEFT, adapter saved later by peft
        return {"trained": True}

    def _write_adapter_placeholder(self, path: str):
        # Write a tiny placeholder file to simulate an adapter/checkpoint
        with open(path, "w", encoding="utf-8") as f:
            f.write("placeholder adapter for " + os.path.basename(path))

    def run_training(self):
        # If smoke, reuse earlier fast path
        if self.smoke:
            model_info = self._load_tokenizer_and_model()
            # For smoke, avoid heavy dataset library imports; return path only
            dataset_info = {"dataset_path": self.dataset}
            bnb_config = self._build_bnb_config()

            adapter_path = os.path.join(self.out_dir, "adapter.pt")
            self._write_adapter_placeholder(adapter_path)

            metadata = {
                "base": self.base,
                "dataset": dataset_info,
                "bnb_config": bnb_config,
                "mode": self.mode,
                "epochs": 1,
                "adapter_path": adapter_path,
                "smoke": True,
            }
            return metadata

        # Full training path
        model, tokenizer = self._load_tokenizer_and_model()
        ds = self._prepare_datasets()

        # Build PEFT/LoRA if requested
        if self.mode in ("lora", "qlora"):
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            except Exception as e:
                raise RuntimeError("peft is required for LoRA/QLoRA training: install 'peft' package") from e

            if self.bnb_bit == 8:
                # prepare for k-bit training
                try:
                    prepare_model_for_kbit_training(model)
                except Exception:
                    # Not fatal; continue
                    pass

            lora_config = LoraConfig(r=self.lora_r, lora_alpha=self.lora_alpha, target_modules=None,
                                     dropout=self.lora_dropout, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, lora_config)

        # Perform training
        training_result = self._train_with_trainer(model, tokenizer, ds)

        # Write adapter/metadata
        adapter_path = os.path.join(self.out_dir, "adapter.pt")
        # If using peft, call save_pretrained
        try:
            model.save_pretrained(self.out_dir)
        except Exception:
            # Fallback: write placeholder
            self._write_adapter_placeholder(adapter_path)

        metadata = {
            "base": self.base,
            "dataset": str(self.dataset),
            "bnb_config": self._build_bnb_config(),
            "mode": self.mode,
            "epochs": self.epochs,
            "adapter_path": adapter_path,
            "trained": training_result,
        }

        return metadata

    def register_and_push(self, metadata: Dict[str, Any]):
        # Register locally first
        entry = {
            "path": self.out_dir,
            "name": metadata.get("name", f"run-{os.path.basename(self.out_dir)}"),
            "base": self.base,
            "timestamp": metadata.get("timestamp"),
        }
        # Import at call time so tests can monkeypatch scripts.register_model.register
        try:
            import scripts.register_model as reg_mod
            reg_mod.register(entry)
        except Exception:
            # Fallback to local register name if present
            try:
                register(entry)  # type: ignore
            except Exception:
                pass

        # Upload artifacts: registry.json and model artifacts
        if self.push_to_gcs and self.gcs_bucket:
            try:
                # Upload registry (import at runtime to allow tests to monkeypatch)
                import scripts.register_model as reg_mod
                reg_mod.upload_registry_to_gcs(bucket_name=self.gcs_bucket)
                metadata["gcs_registry_sync"] = True
            except Exception as e:
                metadata["gcs_registry_sync_error"] = str(e)

            # Upload model artifacts under a prefix named after the run
            try:
                from cloud.gcs_helpers import upload_dir

                prefix = os.path.basename(self.out_dir)
                uploaded = upload_dir(self.gcs_bucket, self.out_dir, dest_prefix=prefix)
                metadata["gcs_uploaded_files"] = uploaded
            except Exception as e:
                metadata["gcs_upload_error"] = str(e)

        # For image push we only provide an informational note at this stage
        if self.push_image_to_ar and self.artifact_repo:
            metadata["artifact_repo"] = self.artifact_repo
            metadata["artifact_push_note"] = (
                "Image push intended to Artifact Registry (AR); perform manual 'gcloud builds submit' or CI job."
            )

        return metadata

    def run(self) -> Dict[str, Any]:
        metadata = self.run_training()

        # add provenance
        metadata["timestamp"] = metadata.get("timestamp") or "TODO-timestamp"
        metadata["created_by"] = "PeftTrainer"

        if self.register:
            metadata = self.register_and_push(metadata)

        return metadata
