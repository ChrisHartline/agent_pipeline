"""
Model-aware data validation.

Validates that training data is compatible with a specific model:
- Token length checks
- Chat template compatibility
- Provides recommendations for issues
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_utils import load_data, convert_to_instruction_response, detect_format
from pipeline.models import (
    get_model_config,
    estimate_tokens,
    format_for_model,
    ModelConfig,
    SUPPORTED_MODELS,
)


@dataclass
class TokenStats:
    """Statistics about token lengths in the dataset."""
    total_examples: int = 0
    min_tokens: int = 0
    max_tokens: int = 0
    avg_tokens: float = 0
    median_tokens: int = 0

    # Breakdown by length
    within_limit: int = 0
    near_limit: int = 0  # 80-100% of max
    over_limit: int = 0
    severely_over: int = 0  # >150% of max

    # Lists for detailed reporting
    over_limit_indices: List[int] = field(default_factory=list)


@dataclass
class ModelValidationResult:
    """Result of model-aware validation."""
    model_key: str
    model_name: str
    max_tokens: int
    compatible: bool
    token_stats: TokenStats
    recommendations: List[str]
    warnings: List[str]
    sample_formatted: Optional[str] = None

    def __str__(self):
        status = "✅ COMPATIBLE" if self.compatible else "⚠️ ISSUES FOUND"
        return f"""
{status}
Model: {self.model_name} (max {self.max_tokens:,} tokens)

Token Distribution:
  ✅ Within limit: {self.token_stats.within_limit}
  ⚠️  Near limit (80-100%): {self.token_stats.near_limit}
  ❌ Over limit: {self.token_stats.over_limit}
  🚫 Severely over (>150%): {self.token_stats.severely_over}

Stats:
  Min: {self.token_stats.min_tokens:,} tokens
  Max: {self.token_stats.max_tokens:,} tokens
  Avg: {self.token_stats.avg_tokens:,.0f} tokens

{self._format_recommendations()}
{self._format_warnings()}
"""

    def _format_recommendations(self) -> str:
        if not self.recommendations:
            return ""
        recs = "\n".join(f"  → {r}" for r in self.recommendations)
        return f"Recommendations:\n{recs}\n"

    def _format_warnings(self) -> str:
        if not self.warnings:
            return ""
        warns = "\n".join(f"  ⚠️ {w}" for w in self.warnings)
        return f"Warnings:\n{warns}\n"


def validate_for_model(
    data_path: str,
    model_key: str,
    max_examples: int = 10000,
    include_thinking: bool = True
) -> ModelValidationResult:
    """
    Validate a dataset for compatibility with a specific model.

    Args:
        data_path: Path to training data file
        model_key: Model identifier (e.g., 'tinyllama', 'llama-3-8b')
        max_examples: Maximum examples to validate
        include_thinking: Whether <think> blocks will be included

    Returns:
        ModelValidationResult with detailed compatibility info
    """
    config = get_model_config(model_key)
    max_tokens = config.max_tokens

    token_lengths = []
    over_limit_indices = []
    sample_formatted = None

    for idx, example in enumerate(load_data(data_path)):
        if idx >= max_examples:
            break

        if "_error" in example:
            continue

        # Convert to instruction/response for analysis
        fmt = detect_format(example)
        converted = convert_to_instruction_response(example, fmt)

        if not converted:
            continue

        instruction = converted["instruction"]
        response = converted["response"]

        # Optionally strip thinking blocks for token estimation
        if not include_thinking and "<think>" in response:
            import re
            response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)

        # Format with model's chat template and estimate tokens
        formatted = format_for_model(
            model_key,
            instruction,
            response,
            system=None
        )

        tokens = estimate_tokens(formatted)
        token_lengths.append(tokens)

        if tokens > max_tokens:
            over_limit_indices.append(idx)

        # Save first example as sample
        if sample_formatted is None:
            sample_formatted = formatted[:500] + "..." if len(formatted) > 500 else formatted

    # Calculate statistics
    if not token_lengths:
        return ModelValidationResult(
            model_key=model_key,
            model_name=config.name,
            max_tokens=max_tokens,
            compatible=False,
            token_stats=TokenStats(),
            recommendations=["No valid examples found in dataset"],
            warnings=[],
        )

    sorted_lengths = sorted(token_lengths)
    stats = TokenStats(
        total_examples=len(token_lengths),
        min_tokens=min(token_lengths),
        max_tokens=max(token_lengths),
        avg_tokens=sum(token_lengths) / len(token_lengths),
        median_tokens=sorted_lengths[len(sorted_lengths) // 2],
        within_limit=sum(1 for t in token_lengths if t <= max_tokens * 0.8),
        near_limit=sum(1 for t in token_lengths if max_tokens * 0.8 < t <= max_tokens),
        over_limit=sum(1 for t in token_lengths if max_tokens < t <= max_tokens * 1.5),
        severely_over=sum(1 for t in token_lengths if t > max_tokens * 1.5),
        over_limit_indices=over_limit_indices[:20],  # First 20 indices
    )

    # Generate recommendations
    recommendations = []
    warnings = []

    # Check compatibility
    compatible = stats.over_limit == 0 and stats.severely_over == 0

    if stats.severely_over > 0:
        pct = (stats.severely_over / stats.total_examples) * 100
        warnings.append(f"{stats.severely_over} examples ({pct:.1f}%) severely exceed token limit")

        # Suggest solutions
        if not include_thinking:
            recommendations.append("Consider using a model with longer context (e.g., llama-3-8b)")
        else:
            recommendations.append("Use --no-thinking to remove <think> blocks (saves ~40% tokens)")

    if stats.over_limit > 0:
        pct = (stats.over_limit / stats.total_examples) * 100
        warnings.append(f"{stats.over_limit} examples ({pct:.1f}%) exceed token limit")

        if stats.over_limit < stats.total_examples * 0.1:
            recommendations.append("Consider filtering out over-limit examples (only ~10% affected)")
        else:
            # Suggest larger model
            for key, cfg in SUPPORTED_MODELS.items():
                if cfg.max_tokens > max_tokens and stats.max_tokens <= cfg.max_tokens:
                    recommendations.append(f"Consider using {cfg.name} (max {cfg.max_tokens:,} tokens)")
                    break

    if stats.near_limit > stats.total_examples * 0.3:
        warnings.append(f"Many examples ({stats.near_limit}) are near the token limit")

    if stats.avg_tokens < max_tokens * 0.2:
        recommendations.append(f"Data is quite short (avg {stats.avg_tokens:.0f} tokens). Model may underfit.")

    if compatible and not warnings:
        recommendations.append(f"✅ Data is well-suited for {config.name}")

    return ModelValidationResult(
        model_key=model_key,
        model_name=config.name,
        max_tokens=max_tokens,
        compatible=compatible,
        token_stats=stats,
        recommendations=recommendations,
        warnings=warnings,
        sample_formatted=sample_formatted,
    )


def recommend_model(data_path: str, max_examples: int = 1000) -> Dict[str, Any]:
    """
    Analyze data and recommend the best model.

    Args:
        data_path: Path to training data
        max_examples: Max examples to analyze

    Returns:
        Dict with recommendation and analysis
    """
    # Get token stats without model constraint
    token_lengths = []

    for idx, example in enumerate(load_data(data_path)):
        if idx >= max_examples:
            break
        if "_error" in example:
            continue

        fmt = detect_format(example)
        converted = convert_to_instruction_response(example, fmt)
        if converted:
            text = converted["instruction"] + converted["response"]
            token_lengths.append(estimate_tokens(text))

    if not token_lengths:
        return {"error": "No valid examples found"}

    max_in_data = max(token_lengths)
    p95 = sorted(token_lengths)[int(len(token_lengths) * 0.95)]

    # Find best fitting models
    recommendations = []
    for key, config in SUPPORTED_MODELS.items():
        if config.max_tokens >= p95:
            fit_pct = sum(1 for t in token_lengths if t <= config.max_tokens) / len(token_lengths) * 100
            recommendations.append({
                "model": key,
                "name": config.name,
                "max_tokens": config.max_tokens,
                "fit_percentage": fit_pct,
                "description": config.description,
            })

    # Sort by smallest model that fits
    recommendations.sort(key=lambda x: x["max_tokens"])

    return {
        "data_stats": {
            "examples": len(token_lengths),
            "max_tokens": max_in_data,
            "p95_tokens": p95,
            "avg_tokens": sum(token_lengths) / len(token_lengths),
        },
        "recommendations": recommendations,
        "best_fit": recommendations[0] if recommendations else None,
    }


# CLI for testing
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Model-aware data validation")
    parser.add_argument("data_path", help="Path to training data")
    parser.add_argument("--model", "-m", help="Model to validate for")
    parser.add_argument("--no-thinking", action="store_true", help="Exclude <think> blocks")
    parser.add_argument("--recommend", action="store_true", help="Recommend best model")

    args = parser.parse_args()

    if args.recommend:
        result = recommend_model(args.data_path)
        print(json.dumps(result, indent=2))
    elif args.model:
        result = validate_for_model(
            args.data_path,
            args.model,
            include_thinking=not args.no_thinking
        )
        print(result)
    else:
        print("Specify --model or --recommend")
