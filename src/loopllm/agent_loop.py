"""Adaptive agent-loop control built on the Bayesian priors layer.

This generalizes the prompt-refinement loop's Bayesian early-exit (see
:class:`loopllm.adaptive_exit.BayesianExitCondition`) to *arbitrary agent
loops*. The agent runs its own plan/act/observe steps and reports a progress
score after each step; the controller returns a continue/stop verdict using the
same learned priors, then records the completed run so future step budgets
improve for that ``(task_type, model)`` pair.

It deliberately reuses :class:`loopllm.priors.AdaptivePriors` — no new model and
no training data: predictions start from sensible priors and sharpen as loops
are observed.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from loopllm.priors import AdaptivePriors, CallObservation

logger = structlog.get_logger(__name__)

# Hard safety cap on agent-loop steps, independent of the learned budget.
MAX_STEPS = 10
# Plateau threshold: stop if the last two score deltas are both below this.
CONVERGENCE_DELTA = 0.01


@dataclass
class AgentLoopSession:
    """Mutable state for a single adaptive agent-loop run.

    Attributes:
        session_id: Short unique identifier for this loop.
        goal: Human-readable description of what the loop is trying to achieve.
        task_type: Identifier for the task class (drives prior selection).
        model_id: Identifier for the model the agent is using.
        quality_threshold: Progress score at which the loop is considered done.
        suggested_budget: Learned/recommended number of steps for this task.
        cost_weight: Weight given to cost vs. quality when budgeting.
        confidence: Confidence in the suggested budget (0–1).
        total_observations: How many prior loops informed the suggestion.
        scores: Per-step progress scores in [0, 1].
        latencies_ms: Per-step wall-clock latency in milliseconds.
        notes: Optional per-step notes recorded by the agent.
        last_decision: The most recent continue/stop verdict.
        converged: Whether the loop reached its goal (set on close).
        closed: Whether the loop has been finalised and learned from.
    """

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
    converged: bool | None = None
    closed: bool = False


class AgentLoopController:
    """Advises an agent's own multi-step loop on when to stop, and learns.

    Bind one controller to an :class:`AdaptivePriors` instance (the same
    SQLite-backed priors the refinement loop uses). The lifecycle is
    ``start`` → repeated ``step`` → ``end``.

    Args:
        priors: The adaptive priors manager holding learned beliefs.
    """

    def __init__(self, priors: AdaptivePriors) -> None:
        self._priors = priors
        self._sessions: dict[str, AgentLoopSession] = {}

    # -- lifecycle -----------------------------------------------------------

    def start(
        self,
        goal: str,
        task_type: str = "general",
        model_id: str = "unknown",
        quality_threshold: float | None = None,
        cost_weight: float = 0.5,
    ) -> AgentLoopSession:
        """Begin a new adaptive agent-loop session.

        Args:
            goal: What the loop is trying to achieve.
            task_type: Identifier for the task class.
            model_id: Identifier for the model the agent is using.
            quality_threshold: Override the learned threshold if provided.
            cost_weight: Weight given to cost vs. quality (0 = quality only).

        Returns:
            A fresh :class:`AgentLoopSession` with a suggested step budget.
        """
        suggestion = self._priors.suggest_config(task_type, model_id, cost_weight)
        budget = int(suggestion["max_iterations"])
        threshold = (
            float(quality_threshold)
            if quality_threshold is not None
            else float(suggestion["quality_threshold"])
        )
        meta = suggestion.get("metadata", {})
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
        )
        self._sessions[session.session_id] = session
        logger.info(
            "agent_loop_start",
            session_id=session.session_id,
            task_type=task_type,
            suggested_budget=session.suggested_budget,
            confidence=session.confidence,
        )
        return session

    def step(self, session_id: str, score: float, note: str = "") -> dict[str, Any]:
        """Report a completed step and get a continue/stop verdict.

        Args:
            session_id: The session to advance.
            score: Progress score for this step, clamped to [0, 1].
            note: Optional note describing what the step did.

        Returns:
            A verdict dict with ``decision`` ("continue" or "stop"), a
            human-readable ``reason``, and loop diagnostics.

        Raises:
            KeyError: If the session does not exist.
            ValueError: If the session has already been closed.
        """
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

        steps_used = len(session.scores)
        expected_delta, uncertainty = self._priors.expected_improvement(
            session.task_type, session.model_id, steps_used
        )

        decision, reason = self._decide(session, score, steps_used)
        session.last_decision = decision

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
        logger.debug("agent_loop_step", **{k: verdict[k] for k in ("session_id", "decision", "steps_used")})
        return verdict

    def end(self, session_id: str, converged: bool | None = None) -> dict[str, Any]:
        """Finalise a loop and learn from it.

        Builds a :class:`CallObservation` from the recorded steps and updates
        the priors so future budgets for this ``(task_type, model)`` improve.
        Idempotent: calling twice does not double-count the observation.

        Args:
            session_id: The session to close.
            converged: Whether the goal was reached. Inferred from the final
                score vs. threshold when omitted.

        Returns:
            A summary dict including what the system learned.

        Raises:
            KeyError: If the session does not exist.
        """
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
        """Return the current state of an active session.

        Raises:
            KeyError: If the session does not exist.
        """
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
        }

    # -- internals -----------------------------------------------------------

    def _decide(
        self, session: AgentLoopSession, score: float, steps_used: int
    ) -> tuple[str, str]:
        """Decide whether the loop should continue, mirroring engine.py order."""
        # 1. Quality threshold reached.
        if score >= session.quality_threshold:
            return "stop", (
                f"Goal reached: score={score:.3f} >= threshold "
                f"{session.quality_threshold:.2f} at step {steps_used}"
            )

        # 2. Plateau / convergence over the last three steps.
        if len(session.scores) >= 3:
            delta1 = abs(session.scores[-1] - session.scores[-2])
            delta2 = abs(session.scores[-2] - session.scores[-3])
            if delta1 < CONVERGENCE_DELTA and delta2 < CONVERGENCE_DELTA:
                return "stop", (
                    f"Progress plateaued (last deltas {delta1:.4f}, {delta2:.4f} "
                    f"< {CONVERGENCE_DELTA:.4f}); further steps unlikely to help"
                )

        # 3. Bayesian verdict from learned priors (same logic as
        #    BayesianExitCondition): is the remaining gap likely to be bridged?
        should_go = self._priors.should_continue(
            session.task_type,
            session.model_id,
            steps_used,
            score,
            list(session.scores),
            quality_threshold=session.quality_threshold,
        )
        if not should_go:
            expected_delta, uncertainty = self._priors.expected_improvement(
                session.task_type, session.model_id, steps_used
            )
            return "stop", (
                f"Bayesian stop at step {steps_used}: score={score:.3f}, "
                f"E[delta]={expected_delta:.3f}±{uncertainty:.3f}, "
                f"threshold={session.quality_threshold:.2f} (low expected ROI)"
            )

        # 4. Learned step budget exhausted.
        if steps_used >= session.suggested_budget:
            return "stop", (
                f"Step budget exhausted ({steps_used}/{session.suggested_budget}); "
                f"escalate or accept current result"
            )

        # 5. Hard safety cap.
        if steps_used >= MAX_STEPS:
            return "stop", f"Hard step cap reached ({MAX_STEPS})"

        return "continue", (
            f"Keep going: step {steps_used}/{session.suggested_budget}, "
            f"score={score:.3f} below threshold {session.quality_threshold:.2f}"
        )

    def _require(self, session_id: str) -> AgentLoopSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown agent-loop session: {session_id}")
        return self._sessions[session_id]
