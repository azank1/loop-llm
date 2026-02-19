"""Base abstractions for LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMUsage:
    """Token usage reported by the LLM API.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens consumed.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Normalised response returned by every LLM provider.

    Attributes:
        content: The text content of the completion.
        model: Model identifier used for the call.
        usage: Token usage statistics.
        latency_ms: Round-trip latency in milliseconds.
    """

    content: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: float = 0.0


class LLMProvider(ABC):
    """Abstract base class that all providers must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    def complete(self, prompt: str, model: str, **kwargs) -> LLMResponse:
        """Send *prompt* to *model* and return a normalised response."""
