#!/usr/bin/env python3
"""Basic loop-llm example: iterative JSON generation with MockLLMProvider."""
from __future__ import annotations

from loopllm import LoopConfig, LoopedLLM
from loopllm.evaluators import JSONSchemaEvaluator
from loopllm.providers.mock import MockLLMProvider


def main() -> None:
    provider = MockLLMProvider(
        responses=[
            "not json",
            '{"name": "Alice"}',
            '{"name": "Alice", "age": 30, "email": "alice@example.com"}',
        ]
    )
    config = LoopConfig(max_iterations=5, quality_threshold=0.8)
    loop = LoopedLLM(provider=provider, config=config)
    evaluator = JSONSchemaEvaluator(
        required_fields=["name", "age", "email"],
        field_types={"age": int, "name": str},
    )
    result = loop.refine("Generate a JSON user profile.", evaluator)
    print(f"Converged in {result.metrics.total_iterations} iterations")
    print(f"Score: {result.metrics.best_score:.2f}")
    print(f"Output: {result.output}")
    print(f"Exit reason: {result.metrics.exit_reason.condition}")
    print(f"Score trajectory: {result.metrics.score_trajectory}")


if __name__ == "__main__":
    main()
