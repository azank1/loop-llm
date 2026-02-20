"""OpenRouter LLM provider."""
from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from typing import Any

from loopllm.provider import LLMProvider, LLMResponse, LLMUsage

logger = structlog.get_logger(__name__)


@dataclass
class OpenRouterProvider(LLMProvider):
    """LLM provider backed by the OpenRouter API.

    Args:
        api_key: OpenRouter API key.
        base_url: Base URL for the OpenRouter API.
    """

    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"

    @property
    def name(self) -> str:
        """Provider name."""
        return "openrouter"

    def complete(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        """Call the OpenRouter chat completions endpoint.

        Args:
            prompt: The user prompt to complete.
            model: OpenRouter model identifier (e.g. ``openai/gpt-4o-mini``).
            **kwargs: Extra fields merged into the request body.

        Returns:
            Parsed :class:`LLMResponse` with content, usage, and latency.

        Raises:
            RuntimeError: If the API returns a non-200 status code.
            ImportError: If ``httpx`` is not installed.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "httpx is required for OpenRouterProvider. "
                "Install it with: pip install loopllm[openrouter]"
            ) from exc

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/azank1/loop-llm",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }

        t0 = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter returned {response.status_code}: {response.text}"
            )

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage_data = data.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        logger.debug(
            "openrouter_complete",
            model=model,
            latency_ms=round(latency_ms, 1),
            total_tokens=usage.total_tokens,
        )

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            latency_ms=latency_ms,
        )
