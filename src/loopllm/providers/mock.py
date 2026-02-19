"""Mock LLM provider for testing."""
from __future__ import annotations

from dataclasses import dataclass, field

from loopllm.provider import LLMProvider, LLMResponse, LLMUsage


@dataclass
class MockLLMProvider(LLMProvider):
    """LLM provider that returns pre-configured responses. Ideal for testing.

    Args:
        responses: Ordered list of responses to cycle through.
        default_score: Unused; kept for compatibility.
        latency_ms: Simulated latency per call in milliseconds.
    """

    responses: list[str] | None = None
    default_score: float = 0.9
    latency_ms: float = 10.0
    calls: list[dict] = field(default_factory=list, repr=False)
    _index: int = field(default=0, repr=False)

    @property
    def name(self) -> str:
        """Provider name."""
        return "mock"

    @property
    def call_count(self) -> int:
        """Number of calls made so far."""
        return len(self.calls)

    def complete(self, prompt: str, model: str, **kwargs) -> LLMResponse:
        """Return the next mock response.

        Cycles through *responses* if provided, otherwise returns
        ``"Mock response {n}"``.

        Args:
            prompt: The prompt (recorded but not used).
            model: The model name (recorded but not used).
            **kwargs: Extra keyword arguments (recorded).

        Returns:
            :class:`LLMResponse` with fake content and usage.
        """
        self.calls.append({"prompt": prompt, "model": model, **kwargs})

        if self.responses:
            content = self.responses[self._index % len(self.responses)]
        else:
            content = f"Mock response {self._index}"

        self._index += 1

        return LLMResponse(
            content=content,
            model=model,
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            latency_ms=self.latency_ms,
        )
