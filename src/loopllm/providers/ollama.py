"""Ollama LLM provider."""
from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from typing import Any

from loopllm.provider import LLMProvider, LLMResponse, LLMUsage

logger = structlog.get_logger(__name__)


@dataclass
class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama instance.

    Args:
        base_url: Base URL for the Ollama API.
    """

    base_url: str = "http://localhost:11434"

    @property
    def name(self) -> str:
        """Provider name."""
        return "ollama"

    def complete(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        """Call the Ollama chat endpoint.

        Args:
            prompt: The user prompt to complete.
            model: Ollama model name (e.g. ``llama3``).
            **kwargs: Extra fields merged into the request body.

        Returns:
            Parsed :class:`LLMResponse` with content and latency.

        Raises:
            RuntimeError: If the API returns a non-200 status code.
            ImportError: If ``httpx`` is not installed.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "httpx is required for OllamaProvider. "
                "Install it with: pip install loopllm[ollama]"
            ) from exc

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            **kwargs,
        }

        t0 = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned {response.status_code}: {response.text}"
            )

        data = response.json()
        content = data["message"]["content"]

        usage_data = data.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        logger.debug(
            "ollama_complete",
            model=model,
            latency_ms=round(latency_ms, 1),
        )

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            latency_ms=latency_ms,
        )
