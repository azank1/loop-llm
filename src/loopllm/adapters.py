"""Framework-agnostic stop hook for agent loops.

`AdaptiveStopper` turns the :class:`AgentLoopController` into a single
``should_continue(state) -> bool`` predicate you can drop into a LangGraph
conditional edge, a CrewAI step callback, an AutoGen termination function, or any
hand-rolled ``while`` loop — so stopping is *enforced by the router*, not merely
advised over MCP.

It reads the latest step result out of a duck-typed ``state`` mapping. If the
state already carries a verified ``score`` it is used directly; otherwise, when an
``output`` artifact is present, the stopper computes a deterministic Channel-A
score locally (regex/JSON/completeness/length via the same evaluators CDV uses) so
you are not stuck trusting the agent's self-grade. No LangGraph/CrewAI dependency.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loopllm.agent_loop import AgentLoopController
from loopllm.priors import AdaptivePriors
from loopllm.step_scorer import build_step_evaluator, score_channel_a


class AdaptiveStopper:
    """Bind one adaptive agent-loop session to a ``should_continue`` predicate.

    Args:
        priors: Shared priors instance; a fresh one is created if omitted.
        goal: What the loop is trying to achieve.
        task_type: Task class used to select learned budget/threshold.
        model_id: Model identifier the agent is using.
        quality_threshold: Override the learned threshold if provided.
        cost_weight: Cost-vs-quality weight for budgeting.
        evaluator_type: Channel-A evaluator for local deterministic scoring
            (``composite``/``regex``/``json``/``completeness``/``length``).
        quality_criteria: Aspects the output must address (for completeness).
        max_wall_ms: Wall-clock guard budget in milliseconds (0 disables).
        max_tokens: Token guard budget (0 disables).
        score_key: State key holding a pre-computed verified score in [0, 1].
        output_key: State key holding the step artifact to score locally.
        tokens_key: State key holding tokens consumed this step.
        **evaluator_kwargs: Passed through to the Channel-A evaluator
            (e.g. ``required_patterns``, ``required_fields``).
    """

    def __init__(
        self,
        *,
        priors: AdaptivePriors | None = None,
        goal: str = "",
        task_type: str = "general",
        model_id: str = "unknown",
        quality_threshold: float | None = None,
        cost_weight: float = 0.5,
        evaluator_type: str = "composite",
        quality_criteria: list[str] | None = None,
        max_wall_ms: float = 300_000.0,
        max_tokens: int = 0,
        score_key: str = "score",
        output_key: str = "output",
        tokens_key: str = "tokens",
        **evaluator_kwargs: Any,
    ) -> None:
        self._controller = AgentLoopController(priors or AdaptivePriors())
        self._goal = goal
        self._task_type = task_type
        self._model_id = model_id
        self._quality_threshold = quality_threshold
        self._cost_weight = cost_weight
        self._evaluator_type = evaluator_type
        self._quality_criteria = list(quality_criteria or ([goal] if goal else []))
        self._max_wall_ms = max_wall_ms
        self._max_tokens = max_tokens
        self._score_key = score_key
        self._output_key = output_key
        self._tokens_key = tokens_key
        self._evaluator_kwargs = evaluator_kwargs
        self._evaluator = build_step_evaluator(
            evaluator_type, self._quality_criteria, **evaluator_kwargs
        )
        self._session_id: str | None = None
        self.last_verdict: dict[str, Any] | None = None
        self.summary: dict[str, Any] | None = None

    # -- core predicate ------------------------------------------------------

    def should_continue(self, state: Mapping[str, Any]) -> bool:
        """Return True to keep looping, False to stop.

        Idempotent after stopping: once a guard has fired and the loop is
        finalised, further calls return ``False`` without recording new steps.
        """
        if self.summary is not None:
            return False

        session_id = self._ensure_session()
        score, output, tokens = self._extract(state)
        verdict = self._controller.step(
            session_id, score, step_output=output, step_tokens=tokens
        )
        self.last_verdict = verdict

        if verdict["decision"] == "stop":
            threshold = self._controller.get_session(session_id).quality_threshold
            self.summary = self._controller.end(
                session_id, converged=score >= threshold
            )
            return False
        return True

    def route(self, state: Mapping[str, Any], on_continue: str, on_stop: str) -> str:
        """LangGraph-style conditional edge: return the next node name."""
        return on_continue if self.should_continue(state) else on_stop

    def __call__(self, state: Mapping[str, Any]) -> bool:
        return self.should_continue(state)

    def reset(self) -> None:
        """Forget the current session so the stopper can drive a new loop."""
        self._session_id = None
        self.last_verdict = None
        self.summary = None

    # -- internals -----------------------------------------------------------

    def _ensure_session(self) -> str:
        if self._session_id is None:
            session = self._controller.start(
                goal=self._goal,
                task_type=self._task_type,
                model_id=self._model_id,
                quality_threshold=self._quality_threshold,
                cost_weight=self._cost_weight,
                evaluator_type=self._evaluator_type,
                quality_criteria=self._quality_criteria,
                max_wall_ms=self._max_wall_ms,
                max_tokens=self._max_tokens,
                **self._evaluator_kwargs,
            )
            self._session_id = session.session_id
        return self._session_id

    def _extract(self, state: Mapping[str, Any]) -> tuple[float, str, int]:
        output = str(state.get(self._output_key, "") or "")
        tokens = int(state.get(self._tokens_key, 0) or 0)
        raw = state.get(self._score_key)
        if raw is not None:
            return max(0.0, min(1.0, float(raw))), output, tokens
        if output:
            # No supplied score — verify locally via the deterministic channel.
            result = score_channel_a(output, self._goal, self._evaluator)
            return result.score, output, tokens
        return 0.0, output, tokens
