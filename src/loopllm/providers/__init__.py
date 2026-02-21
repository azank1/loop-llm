"""Built-in LLM provider implementations."""
from __future__ import annotations

from loopllm.providers.agent import AgentExecutionRequired, AgentPassthroughProvider
from loopllm.providers.mock import MockLLMProvider
from loopllm.providers.ollama import OllamaProvider
from loopllm.providers.openrouter import OpenRouterProvider

__all__ = [
    "AgentExecutionRequired",
    "AgentPassthroughProvider",
    "MockLLMProvider",
    "OllamaProvider",
    "OpenRouterProvider",
]
