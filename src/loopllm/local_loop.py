"""LocalModelLoop — closes the loop for local models (Ollama, llama.cpp, etc.).

Instead of the local model calling MCP tools itself, this module wraps any
local LLM call with a scoring middleware layer:

1. Send prompt → local model → get output
2. POST output to loopllm /score endpoint
3. Receive score + weighted prompt rewrite
4. If score < threshold, re-submit rewritten prompt to local model
5. Repeat until score >= threshold or max_retries exhausted

The local model never needs to support tool-calling or MCP.  loopllm acts
purely as a prompt optimizer and quality gate that sits between the caller
and the model.

Usage::

    loop = LocalModelLoop(
        base_url="http://localhost:11434",
        model="llama3.2",
        score_url="http://localhost:8765/score",
        quality_threshold=0.80,
        max_retries=3,
    )
    result = loop.run("Write a Python function to parse JSON safely.")
    print(result.output)
    print(f"Final score: {result.final_score}")
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopIteration:
    """Record of a single local-model loop iteration."""

    iteration: int
    prompt: str
    output: str
    score: float
    passed: bool
    deficiencies: list[str]
    latency_ms: float
    rewrite_used: bool = False


@dataclass
class LocalLoopResult:
    """Final result from a LocalModelLoop run."""

    output: str
    final_score: float
    best_score: float
    total_iterations: int
    converged: bool
    iterations: list[LoopIteration] = field(default_factory=list)


class LocalModelLoop:
    """Wraps any local HTTP LLM (Ollama-compatible) with loopllm scoring.

    Args:
        base_url: Base URL of the local model API (Ollama default: http://localhost:11434).
        model: Model name (e.g. "llama3.2", "qwen2.5:0.5b").
        score_url: URL of the loopllm score endpoint (loopllm serve default: http://localhost:8765/score).
        quality_threshold: Minimum score to accept a response without retrying.
        max_retries: Maximum number of retry iterations.
        timeout: HTTP timeout in seconds for model calls.
        prompt_weight: Weight of prompt score in weighted rewrite (0–1).
        output_weight: Weight of output score in weighted rewrite (0–1).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        score_url: str = "http://localhost:8765/score",
        quality_threshold: float = 0.80,
        max_retries: int = 3,
        timeout: float = 60.0,
        prompt_weight: float = 0.35,
        output_weight: float = 0.65,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.score_url = score_url
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        self.timeout = timeout
        self.prompt_weight = prompt_weight
        self.output_weight = output_weight

    # -- public API ----------------------------------------------------------

    def run(
        self,
        prompt: str,
        system: str | None = None,
        evaluator_type: str = "length",
        min_words: int = 5,
        **kwargs: Any,
    ) -> LocalLoopResult:
        """Run prompt → score → rewrite → retry loop.

        Args:
            prompt: The initial user prompt.
            system: Optional system message.
            evaluator_type: Scoring evaluator type passed to loopllm ('length', 'json', 'regex').
            min_words: Minimum word count evaluator argument.
            **kwargs: Extra keyword args forwarded to the model API.

        Returns:
            :class:`LocalLoopResult` with the best output and scores.
        """
        current_prompt = prompt
        iterations: list[LoopIteration] = []
        best_output = ""
        best_score = -1.0

        for i in range(self.max_retries):
            iter_start = time.perf_counter()
            rewrite_used = i > 0

            # 1. Call local model
            output = self._call_model(current_prompt, system=system, **kwargs)
            latency_ms = (time.perf_counter() - iter_start) * 1000.0

            # 2. Score via loopllm
            score_result = self._score(
                prompt=current_prompt,
                output=output,
                evaluator_type=evaluator_type,
                min_words=min_words,
            )
            score = score_result.get("output_score", 0.5)
            deficiencies = score_result.get("deficiencies", [])
            passed = score >= self.quality_threshold

            record = LoopIteration(
                iteration=i,
                prompt=current_prompt,
                output=output,
                score=score,
                passed=passed,
                deficiencies=deficiencies,
                latency_ms=latency_ms,
                rewrite_used=rewrite_used,
            )
            iterations.append(record)

            if score > best_score:
                best_score = score
                best_output = output

            # 3. Accept if good enough
            if passed:
                break

            # 4. Rewrite prompt with score-weighted feedback
            current_prompt = self._rewrite_prompt(
                original_prompt=prompt,
                previous_output=output,
                score=score,
                deficiencies=deficiencies,
                iteration=i + 1,
            )

        converged = best_score >= self.quality_threshold
        return LocalLoopResult(
            output=best_output,
            final_score=iterations[-1].score if iterations else 0.0,
            best_score=best_score,
            total_iterations=len(iterations),
            converged=converged,
            iterations=iterations,
        )

    # -- private helpers -----------------------------------------------------

    def _call_model(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send prompt to the local model and return the response text."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for LocalModelLoop. "
                "Install with: pip install httpx"
            ) from e

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False, **kwargs},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama /api/chat response
        return data.get("message", {}).get("content", data.get("response", ""))

    def _score(
        self,
        prompt: str,
        output: str,
        evaluator_type: str = "length",
        min_words: int = 5,
    ) -> dict[str, Any]:
        """POST to loopllm /score and return the score dict."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for LocalModelLoop. "
                "Install with: pip install httpx"
            ) from e

        try:
            resp = httpx.post(
                self.score_url,
                json={
                    "prompt": prompt,
                    "output": output,
                    "evaluator_type": evaluator_type,
                    "min_words": min_words,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            # If loopllm serve is unreachable, use a simple word-count fallback
            words = len(output.split())
            score = min(1.0, words / max(min_words, 1))
            return {
                "output_score": round(score, 3),
                "deficiencies": [] if score >= self.quality_threshold else ["output too short"],
            }

    def _rewrite_prompt(
        self,
        original_prompt: str,
        previous_output: str,
        score: float,
        deficiencies: list[str],
        iteration: int,
    ) -> str:
        """Build a score-weighted prompt rewrite for the next iteration."""
        deficiency_str = (
            "\n".join(f"  - {d}" for d in deficiencies)
            if deficiencies
            else "  - Output did not meet quality threshold"
        )
        return (
            f"[LOOPLLM | score={score:.2f} | retry={iteration}/{self.max_retries} | "
            f"threshold={self.quality_threshold:.2f}]\n"
            f"Your previous response scored {score:.2f}/1.0 and did not meet the quality bar.\n"
            f"Issues to fix:\n{deficiency_str}\n\n"
            f"Original task:\n{original_prompt}\n\n"
            f"Previous response (do not repeat this):\n{previous_output[:500]}\n\n"
            f"Please produce an improved response that addresses all issues listed above."
        )
