"""Shared fixtures for loop-llm tests."""
from __future__ import annotations

import pytest

from loopllm import AdaptivePriors, CallObservation, LoopConfig
from loopllm.providers.mock import MockLLMProvider


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    """MockLLMProvider cycling through 5 responses of increasing quality."""
    return MockLLMProvider(
        responses=[
            "Low quality draft",
            '{"name": "Alice"}',
            '{"name": "Alice", "age": 25}',
            '{"name": "Alice", "age": 25, "email": "alice@example.com"}',
            '{"name": "Alice", "age": 30, "email": "alice@example.com", "role": "engineer"}',
        ]
    )


@pytest.fixture
def basic_config() -> LoopConfig:
    """Standard loop configuration for tests."""
    return LoopConfig(max_iterations=5, quality_threshold=0.8, min_iterations=1)


@pytest.fixture
def fresh_priors() -> AdaptivePriors:
    """AdaptivePriors with no prior data."""
    return AdaptivePriors()


@pytest.fixture
def trained_priors() -> AdaptivePriors:
    """AdaptivePriors pre-populated with 30 observations each for two (task, model) pairs."""
    priors = AdaptivePriors()

    # 30 observations for ("decompose", "gpt-4o")
    for _ in range(30):
        priors.observe(
            CallObservation(
                task_type="decompose",
                model_id="gpt-4o",
                scores=[0.3, 0.55, 0.78, 0.88],
                latencies_ms=[1200, 1100, 1000, 900],
                converged=True,
                total_iterations=4,
                max_iterations=5,
                quality_threshold=0.8,
            )
        )

    # 30 observations for ("resolve", "gpt-4o")
    for _ in range(30):
        priors.observe(
            CallObservation(
                task_type="resolve",
                model_id="gpt-4o",
                scores=[0.5, 0.85],
                latencies_ms=[1500, 1200],
                converged=True,
                total_iterations=2,
                max_iterations=5,
                quality_threshold=0.8,
            )
        )

    return priors
