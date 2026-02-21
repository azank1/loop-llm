"""Integration tests requiring live LLM API access.

These tests are skipped by default.  To run them:

    LOOPLLM_INTEGRATION=1 pytest tests/test_integration.py -v

Or:

    pytest tests/test_integration.py -v -m integration

Requires either:
- Ollama running locally on port 11434, OR
- OPENROUTER_API_KEY environment variable set
"""
from __future__ import annotations

import os

import pytest

from loopllm.elicitation import IntentRefiner
from loopllm.engine import LoopConfig, LoopedLLM
from loopllm.evaluators import JSONSchemaEvaluator, LengthEvaluator

_SKIP_REASON = "Set LOOPLLM_INTEGRATION=1 to run integration tests"
_RUN_INTEGRATION = os.environ.get("LOOPLLM_INTEGRATION", "0") == "1"

pytestmark = pytest.mark.integration


def _has_ollama() -> bool:
    """Check if Ollama is accessible."""
    try:
        import httpx

        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _has_openrouter() -> bool:
    """Check if OpenRouter API key is available."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


@pytest.mark.skipif(not _RUN_INTEGRATION, reason=_SKIP_REASON)
@pytest.mark.skipif(not _has_ollama(), reason="Ollama not available")
class TestOllamaIntegration:
    """Integration tests using a local Ollama instance."""

    def test_basic_refinement(self) -> None:
        from loopllm.providers.ollama import OllamaProvider

        provider = OllamaProvider()
        config = LoopConfig(max_iterations=3, quality_threshold=0.7)
        loop = LoopedLLM(provider=provider, config=config)
        evaluator = LengthEvaluator(min_words=10, max_words=500)

        result = loop.refine(
            "Write a short Python function that reverses a string.",
            evaluator,
            model="llama3.2",
        )

        assert result.metrics.total_iterations >= 1
        assert len(result.output) > 0

    def test_elicitation_flow(self) -> None:
        from loopllm.providers.ollama import OllamaProvider

        provider = OllamaProvider()
        refiner = IntentRefiner(
            provider=provider,
            model="llama3.2",
            max_questions=2,
        )

        questions = refiner.analyze("Build a web scraper")
        assert len(questions) > 0
        assert all(q.text for q in questions)


@pytest.mark.skipif(not _RUN_INTEGRATION, reason=_SKIP_REASON)
@pytest.mark.skipif(not _has_openrouter(), reason="OpenRouter API key not set")
class TestOpenRouterIntegration:
    """Integration tests using the OpenRouter API."""

    def test_basic_refinement(self) -> None:
        from loopllm.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"])
        config = LoopConfig(max_iterations=2, quality_threshold=0.7)
        loop = LoopedLLM(provider=provider, config=config)
        evaluator = LengthEvaluator(min_words=10, max_words=500)

        result = loop.refine(
            "Write a haiku about programming.",
            evaluator,
            model="openai/gpt-4o-mini",
        )

        assert result.metrics.total_iterations >= 1
        assert len(result.output) > 0

    def test_json_generation(self) -> None:
        from loopllm.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"])
        config = LoopConfig(max_iterations=3, quality_threshold=0.8)
        loop = LoopedLLM(provider=provider, config=config)
        evaluator = JSONSchemaEvaluator(
            required_fields=["name", "description"],
            field_types={"name": str, "description": str},
        )

        result = loop.refine(
            'Generate a JSON object describing a software project with "name" and "description" fields.',
            evaluator,
            model="openai/gpt-4o-mini",
        )

        assert result.metrics.best_score > 0.5
