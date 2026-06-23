"""Adaptive agent-loop control built on the Bayesian priors layer.

Agent loops use Conservative Dual-Verify (CDV) at the MCP boundary: step
artifacts are scored externally before entering this controller. The controller
applies a composable guard stack and learns optimal depth from verified score
trajectories.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from loopllm.guards import (
    CONVERGENCE_DELTA,
    MAX_STEPS_DEFAULT,
    AgentLoopGuard,
    GuardContext,
    GuardStack,
    default_guard_stack,
)
from loopllm.priors import AdaptivePriors, CallObservation

logger = structlog.get_logger(__name__)

MAX_STEPS = MAX_STEPS_DEFAULT


@dataclass
class AgentLoopSession:
    """Mutable state for a single adaptive agent-loop run."""

    session_id: str
    goal: str
    task_type: str
    model_id: str
    quality_threshold: float
    suggested_budget: int
    cost_weight: float = 0.5
    confidence: float = 0.0
    total_observations: int = 0
    scores: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    last_step_at: float = field(default_factory=time.perf_counter)
    last_decision: str = "continue"
    last_reason: str = ""
    converged: bool | None = None
    closed: bool = False
    # CDV verifier recipe (configured at start)
    evaluator_type: str = "composite"
    evaluator_kwargs: dict[str, Any] = field(default_factory=dict)
    quality_criteria: list[str] = field(default_factory=list)
    max_wall_ms: float = 300_000.0
    max_tokens: int = 0
    step_outputs: list[str] = field(default_factory=list)
    step_fingerprints: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Fields persisted to active_runs for crash/restart recovery. Wall-clock
    # timers (started_at/last_step_at) are intentionally excluded — they reset
    # on restore, so the wall-clock guard budget restarts after a resume.
    _SNAPSHOT_FIELDS = (
        "session_id", "goal", "task_type", "model_id", "quality_threshold",
        "suggested_budget", "cost_weight", "confidence", "total_observations",
        "scores", "latencies_ms", "notes", "last_decision", "last_reason",
        "converged", "closed", "evaluator_type", "evaluator_kwargs",
        "quality_criteria", "max_wall_ms", "max_tokens", "step_outputs",
        "step_fingerprints", "prompt_tokens", "completion_tokens",
    )

    def to_snapshot(self) -> dict[str, Any]:
        """Serialise this session for the ``active_runs`` recovery store."""
        return {name: getattr(self, name) for name in self._SNAPSHOT_FIELDS}

    @classmethod
    def from_snapshot(cls, state: dict[str, Any]) -> AgentLoopSession:
        """Rebuild a session from a snapshot; reset wall-clock timers to now."""
        known = {k: v for k, v in state.items() if k in cls._SNAPSHOT_FIELDS}
        now = time.perf_counter()
        return cls(started_at=now, last_step_at=now, **known)


class AgentLoopController:
    """Advises an agent's multi-step loop on when to stop, and learns.

    Lifecycle: ``start`` → repeated ``step`` → ``end``.
    """

    def __init__(
        self,
        priors: AdaptivePriors,
        guards: AgentLoopGuard | GuardStack | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self._priors = priors
        self._sessions: dict[str, AgentLoopSession] = {}
        if guards is None:
            self._guards = default_guard_stack(priors, max_steps)
        elif isinstance(guards, GuardStack):
            self._guards = guards
        else:
            self._guards = GuardStack([guards])

    def start(
        self,
        goal: str,
        task_type: str = "general",
        model_id: str = "unknown",
        quality_threshold: float | None = None,
        cost_weight: float = 0.5,
        evaluator_type: str = "composite",
        quality_criteria: list[str] | None = None,
        max_wall_ms: float = 300_000.0,
        max_tokens: int = 0,
        **evaluator_kwargs: Any,
    ) -> AgentLoopSession:
        """Begin a new adaptive agent-loop session."""
        suggestion = self._priors.suggest_config(task_type, model_id, cost_weight)
        budget = int(suggestion["max_iterations"])
        threshold = (
            float(quality_threshold)
            if quality_threshold is not None
            else float(suggestion["quality_threshold"])
        )
        meta = suggestion.get("metadata", {})
        criteria = list(quality_criteria or [])
        if not criteria and goal:
            criteria = [goal]

        session = AgentLoopSession(
            session_id=uuid.uuid4().hex[:12],
            goal=goal,
            task_type=task_type,
            model_id=model_id,
            quality_threshold=threshold,
            suggested_budget=max(1, min(budget, MAX_STEPS)),
            cost_weight=cost_weight,
            confidence=float(meta.get("confidence", 0.0)),
            total_observations=int(meta.get("total_observations", 0)),
            evaluator_type=evaluator_type,
            evaluator_kwargs=dict(evaluator_kwargs),
            quality_criteria=criteria,
            max_wall_ms=max_wall_ms,
            max_tokens=max_tokens,
        )
        self._sessions[session.session_id] = session
        logger.info(
            "agent_loop_start",
            session_id=session.session_id,
            task_type=task_type,
            suggested_budget=session.suggested_budget,
            confidence=session.confidence,
            evaluator_type=evaluator_type,
        )
        return session

    def step(
        self,
        session_id: str,
        score: float,
        note: str = "",
        step_output: str = "",
        step_tokens: int = 0,
    ) -> dict[str, Any]:
        """Advance a session with a verified (or legacy) progress score."""
        session = self._require(session_id)
        if session.closed:
            raise ValueError(f"Session already closed: {session_id}")

        score = max(0.0, min(1.0, float(score)))
        now = time.perf_counter()
        session.latencies_ms.append((now - session.last_step_at) * 1000.0)
        session.last_step_at = now
        session.scores.append(score)
        if note:
            session.notes.append(note)
        if step_output:
            session.step_outputs.append(step_output)
        if step_tokens > 0:
            session.completion_tokens += step_tokens
            session.prompt_tokens += max(step_tokens // 4, 1)

        steps_used = len(session.scores)
        expected_delta, uncertainty = self._priors.expected_improvement(
            session.task_type, session.model_id, steps_used
        )

        decision, reason = self._decide(session, score, steps_used, step_output)
        session.last_decision = decision
        session.last_reason = reason

        verdict: dict[str, Any] = {
            "session_id": session.session_id,
            "decision": decision,
            "reason": reason,
            "score": round(score, 4),
            "steps_used": steps_used,
            "suggested_budget": session.suggested_budget,
            "quality_threshold": round(session.quality_threshold, 3),
            "expected_delta": round(expected_delta, 4),
            "uncertainty": round(uncertainty, 4),
            "score_trajectory": [round(s, 4) for s in session.scores],
        }
        logger.debug(
            "agent_loop_step",
            session_id=session.session_id,
            decision=decision,
            steps_used=steps_used,
        )
        return verdict

    def end(self, session_id: str, converged: bool | None = None) -> dict[str, Any]:
        """Finalise a loop and learn from verified score trajectories."""
        session = self._require(session_id)
        if not session.closed:
            if converged is None:
                converged = bool(
                    session.scores and session.scores[-1] >= session.quality_threshold
                )
            observation = CallObservation(
                task_type=session.task_type,
                model_id=session.model_id,
                scores=list(session.scores),
                latencies_ms=list(session.latencies_ms),
                converged=converged,
                total_iterations=len(session.scores),
                max_iterations=session.suggested_budget,
                quality_threshold=session.quality_threshold,
                prompt_tokens=session.prompt_tokens,
                completion_tokens=session.completion_tokens,
            )
            self._priors.observe(observation)
            session.converged = converged
            session.closed = True
            logger.info(
                "agent_loop_end",
                session_id=session.session_id,
                steps_run=len(session.scores),
                converged=converged,
            )

        report = self._priors.report(session.task_type, session.model_id)
        return {
            "session_id": session.session_id,
            "goal": session.goal,
            "task_type": session.task_type,
            "model_id": session.model_id,
            "steps_run": len(session.scores),
            "converged": session.converged,
            "final_score": round(session.scores[-1], 4) if session.scores else 0.0,
            "learned": {
                "optimal_depth": report["optimal_depth"],
                "converge_rate": report["converge_rate"],
                "confidence": report["confidence"],
                "total_observations": report["total_calls"],
            },
        }

    def status(self, session_id: str) -> dict[str, Any]:
        """Return the current state of an active session."""
        session = self._require(session_id)
        return {
            "session_id": session.session_id,
            "goal": session.goal,
            "task_type": session.task_type,
            "model_id": session.model_id,
            "steps_used": len(session.scores),
            "suggested_budget": session.suggested_budget,
            "quality_threshold": round(session.quality_threshold, 3),
            "score_trajectory": [round(s, 4) for s in session.scores],
            "last_decision": session.last_decision,
            "closed": session.closed,
            "converged": session.converged,
            "evaluator_type": session.evaluator_type,
            "quality_criteria": session.quality_criteria,
        }

    def get_session(self, session_id: str) -> AgentLoopSession:
        """Return the raw session object (for MCP CDV wiring)."""
        return self._require(session_id)

    def restore_from_snapshot(self, state: dict[str, Any]) -> str:
        """Rehydrate an in-memory session from a persisted snapshot.

        Args:
            state: A dict produced by :meth:`AgentLoopSession.to_snapshot`.

        Returns:
            The restored ``session_id``.
        """
        session = AgentLoopSession.from_snapshot(state)
        self._sessions[session.session_id] = session
        logger.info(
            "agent_loop_restored",
            session_id=session.session_id,
            steps_used=len(session.scores),
            task_type=session.task_type,
        )
        return session.session_id

    def hydrate_active_loops(self, active_runs: list[dict[str, Any]]) -> int:
        """Restore all persisted agent-loop sessions on server startup.

        Args:
            active_runs: Rows from the ``active_runs`` store (each ``{"run_type",
                "state", ...}``); only ``run_type == "agent_loop"`` rows are used.

        Returns:
            The number of sessions hydrated.
        """
        count = 0
        for run in active_runs:
            if run.get("run_type") != "agent_loop":
                continue
            state = run.get("state") or {}
            if not state.get("session_id") or state.get("closed"):
                continue
            try:
                self.restore_from_snapshot(state)
                count += 1
            except (TypeError, ValueError) as exc:  # malformed snapshot
                logger.warning("agent_loop_restore_failed", error=str(exc))
        return count

    def _decide(
        self,
        session: AgentLoopSession,
        score: float,
        steps_used: int,
        step_output: str = "",
    ) -> tuple[str, str]:
        """Run guard stack; continue if no guard fires."""
        ctx = GuardContext(
            session=session,
            iteration=steps_used,
            current_score=score,
            scores_so_far=list(session.scores),
            step_output=step_output,
        )
        reason = self._guards.evaluate(ctx)
        if reason is not None:
            return "stop", reason.message

        return "continue", (
            f"Keep going: step {steps_used}/{session.suggested_budget}, "
            f"score={score:.3f} below threshold {session.quality_threshold:.2f}"
        )

    def _require(self, session_id: str) -> AgentLoopSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown agent-loop session: {session_id}")
        return self._sessions[session_id]


# Re-export for backward compatibility
__all__ = [
    "AgentLoopController",
    "AgentLoopSession",
    "CONVERGENCE_DELTA",
    "MAX_STEPS",
]
