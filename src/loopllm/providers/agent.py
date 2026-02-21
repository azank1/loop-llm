"""Agent passthrough provider.

Instead of calling an external LLM, this provider signals that the
calling agent (VS Code Copilot, Cursor, Claude, etc.) should perform
the generation itself.  The MCP tools catch :class:`AgentExecutionRequired`
and return a structured ``agent_prompt`` payload—the connected IDE agent
then executes it directly.

This eliminates the Ollama / OpenRouter dependency entirely and lets the
tool use whatever frontier model the user already has active.
"""
from __future__ import annotations

from typing import Any

from loopllm.provider import LLMProvider, LLMResponse, LLMUsage


class AgentExecutionRequired(Exception):
    """Raised by :class:`AgentPassthroughProvider` instead of calling an LLM.

    Attributes:
        prompt: The prompt that should be executed by the calling agent.
        model: The model hint passed by the caller (informational only).
        kwargs: Any extra keyword arguments forwarded from the call site.
    """

    def __init__(self, prompt: str, model: str, **kwargs: Any) -> None:
        super().__init__(f"Agent execution required for model={model!r}")
        self.prompt = prompt
        self.model = model
        self.kwargs = kwargs


class AgentPassthroughProvider(LLMProvider):
    """LLM provider that delegates generation to the calling IDE agent.

    When :meth:`complete` is called it raises :class:`AgentExecutionRequired`
    instead of contacting any external service.  MCP tool implementations
    catch this exception and return a structured ``agent_prompt`` response
    that instructs the connected agent (Copilot / Claude / Cursor) to
    perform the generation itself.

    Usage::

        loopllm mcp-server --provider agent
    """

    @property
    def name(self) -> str:
        return "agent"

    def complete(self, prompt: str, model: str, **kwargs: Any) -> LLMResponse:
        """Raise :class:`AgentExecutionRequired` — never calls a remote API.

        Args:
            prompt: The prompt to be executed by the calling agent.
            model: Model hint (passed through to the exception).
            **kwargs: Forwarded verbatim.

        Raises:
            AgentExecutionRequired: Always — callers must handle this.
        """
        raise AgentExecutionRequired(prompt, model, **kwargs)
