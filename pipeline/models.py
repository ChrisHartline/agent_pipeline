"""
Model configurations for the ML pipeline.

Defines supported models with their tokenizers, max lengths, and chat templates.
Used for model-aware data validation and conversion.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


class ModelFamily(Enum):
    """Model families with different chat templates."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    TINYLLAMA = "tinyllama"
    QWEN = "qwen"
    LIQUID = "liquid"  # LiquidAI LFM models


@dataclass
class ModelConfig:
    """Configuration for a supported model."""
    name: str
    hf_id: str  # HuggingFace model ID
    family: ModelFamily
    max_tokens: int
    chat_template: str  # Template name or custom template
    description: str
    recommended_lora_r: int = 16
    recommended_batch_size: int = 4


# Supported models for the pipeline dropdown
SUPPORTED_MODELS: Dict[str, ModelConfig] = {
    "tinyllama": ModelConfig(
        name="TinyLlama 1.1B",
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        family=ModelFamily.TINYLLAMA,
        max_tokens=2048,
        chat_template="tinyllama",
        description="Fast, lightweight. Good for experiments and quick iterations.",
        recommended_lora_r=16,
        recommended_batch_size=8,
    ),
    "llama-3-8b": ModelConfig(
        name="Llama 3 8B",
        hf_id="meta-llama/Meta-Llama-3-8B-Instruct",
        family=ModelFamily.LLAMA,
        max_tokens=8192,
        chat_template="llama-3",
        description="Strong general-purpose model. Good balance of capability and size.",
        recommended_lora_r=32,
        recommended_batch_size=2,
    ),
    "llama-3-3b": ModelConfig(
        name="Llama 3.2 3B",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        family=ModelFamily.LLAMA,
        max_tokens=8192,
        chat_template="llama-3",
        description="Smaller Llama 3. Good for resource-constrained environments.",
        recommended_lora_r=16,
        recommended_batch_size=4,
    ),
    "phi-3-mini": ModelConfig(
        name="Phi-3 Mini",
        hf_id="microsoft/Phi-3-mini-4k-instruct",
        family=ModelFamily.PHI,
        max_tokens=4096,
        chat_template="phi-3",
        description="Microsoft's efficient model. Strong reasoning for its size.",
        recommended_lora_r=16,
        recommended_batch_size=4,
    ),
    "mistral-7b": ModelConfig(
        name="Mistral 7B",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        family=ModelFamily.MISTRAL,
        max_tokens=8192,
        chat_template="mistral",
        description="Excellent instruction following. Good for chat applications.",
        recommended_lora_r=32,
        recommended_batch_size=2,
    ),
    "qwen-2.5-3b": ModelConfig(
        name="Qwen 2.5 3B",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        family=ModelFamily.QWEN,
        max_tokens=32768,
        chat_template="qwen",
        description="Very long context. Good for document processing.",
        recommended_lora_r=16,
        recommended_batch_size=4,
    ),
    # LiquidAI models (tentative - verify licensing)
    "lfm-1.2b-base": ModelConfig(
        name="LFM 2.5 1.2B Base",
        hf_id="LiquidAI/LFM2.5-1.2B-Base",
        family=ModelFamily.LIQUID,
        max_tokens=32768,
        chat_template="liquid",
        description="LiquidAI base model. Best for heavy fine-tuning on custom data.",
        recommended_lora_r=16,
        recommended_batch_size=8,
    ),
    "lfm-1.2b-thinking": ModelConfig(
        name="LFM 2.5 1.2B Thinking",
        hf_id="LiquidAI/LFM2.5-1.2B-Thinking",
        family=ModelFamily.LIQUID,
        max_tokens=32768,
        chat_template="liquid-thinking",
        description="Reasoning model with <think> traces. Fast inference, <1GB memory.",
        recommended_lora_r=16,
        recommended_batch_size=8,
    ),
}


# Chat templates for different model families
CHAT_TEMPLATES = {
    "tinyllama": {
        "system": "<|system|>\n{content}</s>\n",
        "user": "<|user|>\n{content}</s>\n",
        "assistant": "<|assistant|>\n{content}</s>\n",
        "generation_prompt": "<|assistant|>\n",
    },
    "llama-3": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "generation_prompt": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "phi-3": {
        "system": "<|system|>\n{content}<|end|>\n",
        "user": "<|user|>\n{content}<|end|>\n",
        "assistant": "<|assistant|>\n{content}<|end|>\n",
        "generation_prompt": "<|assistant|>\n",
    },
    "mistral": {
        "system": "",  # Mistral doesn't use system in the same way
        "user": "[INST] {content} [/INST]",
        "assistant": "{content}</s>",
        "generation_prompt": "",
    },
    "qwen": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "generation_prompt": "<|im_start|>assistant\n",
    },
    # LiquidAI templates (may need adjustment based on actual model card)
    "liquid": {
        "system": "<|system|>\n{content}\n",
        "user": "<|user|>\n{content}\n",
        "assistant": "<|assistant|>\n{content}\n",
        "generation_prompt": "<|assistant|>\n",
    },
    "liquid-thinking": {
        "system": "<|system|>\n{content}\n",
        "user": "<|user|>\n{content}\n",
        "assistant": "<|assistant|>\n<think>{thinking}</think>\n{content}\n",
        "generation_prompt": "<|assistant|>\n<think>",
    },
}


def get_model_config(model_key: str) -> ModelConfig:
    """Get configuration for a model by key."""
    if model_key not in SUPPORTED_MODELS:
        available = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")
    return SUPPORTED_MODELS[model_key]


def list_models() -> List[Dict]:
    """List all supported models for dropdown."""
    return [
        {
            "key": key,
            "name": config.name,
            "description": config.description,
            "max_tokens": config.max_tokens,
        }
        for key, config in SUPPORTED_MODELS.items()
    ]


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation without loading a tokenizer.

    Rule of thumb: ~4 characters per token for English.
    This is approximate but fast for validation.
    """
    return len(text) // 4


def format_for_model(
    model_key: str,
    instruction: str,
    response: str,
    system: Optional[str] = None
) -> str:
    """
    Format an example using the model's chat template.

    Args:
        model_key: Model identifier
        instruction: User instruction/prompt
        response: Assistant response
        system: Optional system prompt

    Returns:
        Formatted string ready for training
    """
    config = get_model_config(model_key)
    template = CHAT_TEMPLATES[config.chat_template]

    parts = []

    if system and template["system"]:
        parts.append(template["system"].format(content=system))

    parts.append(template["user"].format(content=instruction))
    parts.append(template["assistant"].format(content=response))

    return "".join(parts)
