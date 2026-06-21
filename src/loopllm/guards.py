"""Composable stop guards for adaptive agent loops."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from loopllm.adaptive_exit import BayesianExitCondition
from loopllm.engine import ExitConditionProtocol, ExitReason
from loopllm.priors import AdaptivePriors

if TYPE_CHECKING:
    from loopllm.agent_loop import AgentLoopSession

CONVERGENCE_DELTA = 0.01
MAX_STEPS_DEFAULT = 10


@dataclass
class GuardContext:
    """Runtime context passed to each agent-loop guard."""

    session: AgentLoopSession
    iteration: int
    current_score: float
    scores_so_far: list[float]
    step_output: str = ""


@runtime_checkable
class AgentLoopGuard(Protocol):
    """Protocol for pluggable agent-loop stop conditions."""

    def should_stop(self, ctx: GuardContext) -> ExitReason | None: ...


class GuardStack:
    """Run guards in order; first stop reason wins (OR semantics)."""

    def __init__(self, guards: list[AgentLoopGuard]) -> None:
        self.guards = guards

    def evaluate(self, ctx: GuardContext) -> ExitReason | None:
        for guard in self.guards:
            reason = guard.should_stop(ctx)
            if reason is not None:
                return reason
        return None


class ExitConditionAdapter:
    """Wrap :class:`ExitConditionProtocol` for agent-loop guards."""

    def __init__(self, condition: ExitConditionProtocol) -> None:
        self._condition = condition

    def should_stop(self, ctx: GuardContext) -> ExitReason | None:
        return self._condition.should_exit(
            ctx.iteration,
            ctx.current_score,
            ctx.scores_so_far,
        )


class ScoreThresholdGuard:
    """Stop when verified score meets the session quality threshold."""

    def should_stop(self, ctx: GuardContext) -> ExitReason | None:
        if ctx.current_score >= ctx.session.quality_threshold:
            return ExitReason(
                condition="quality_threshold",
                message=(
                    f"Goal reached: score={ctx.current_score:.3f} >= threshold "
                    f"{ctx.session.quality_threshold:.2f} at step {ctx.iteration}"
                ),
            )
        return None


class PlateauGuard:
    """Stop when the last three verified scores plateau."""

    def __init__(self, delta: float = CONVERGENCE_DELTA) -> None:
        self.delta = delta

    def should_stop(self, ctx: GuardContext) -> ExitReason | None:
        scores = ctx.scores_so_far
        if len(scores) < 3:
            return None
        delta1 = abs(scores[-1] - scores[-2])
        delta2 = abs(scores[-2] - scores[-3])
        if delta1 < self.delta and delta2 < self.delta:
            return ExitReason(
                condition="plateau",
                message=(
                    f"Progress plateaued (last deltas {delta1:.4f}, {delta2:.4f} "
                    f"< {self.delta:.4f}); further steps unlikely to help"
                ),
            )
        return None


class BayesianGuard:
    """Stop when learned priors predict low ROI on further steps."""

    def __init__(self, priors: AdaptivePriors) -> None:
        self._priors = priors

    def should_stop(self, ctx: GuardContext) -> ExitReason | None:
        session = ctx.session
        should_go = self._priors.should_continue(
            session.task_type,
            session.model_id,
            ctx.iteration,
            ctx.current_score,
            ctx.scores_so_far,
            quality_threshold=session.quality_threshold,
        )
        if not should_go:
            expected_delta, uncertainty = self._priors.expected_improvement(
                session.task_type, session.model_id, ctx.iteration
            )
            return ExitReason(
                condition="adaptive_bayesian",
                message=(
                    f"Bayesian stop at step {ctx.iteration}: "
                    f"score={ctx.current_score:.3f}, "
                    f"E[delta]={expected_delta:.3f}±{uncertainty:.3f}, "
                    f"threshold={session.quality_threshold:.2f} (low expected ROI)"
                ),
            )
        return None


class BudgetExhaustedGuard:
    """Stop when the learned step budget is exhausted."""

    def should_stop(self, ctx: GuardContext) -> ExitReason | None:
        if ctx.iteration >= ctx.session.suggested_budget:
            return ExitReason(
                condition="budget_exhausted",
                message=(
                    f"Step budget exhausted ({ctx.iteration}/"
                    f"{ctx.session.suggested_budget}); escalate or accept current result"
                ),
            )
        return None


class MaxStepsGuard:
    """Hard safety cap on agent-loop steps."""

    def __init__(self, max_steps: int = MAX_STEPS_DEFAULT) -> None:
        self.max_steps = max_steps

    def should_stop(self, ctx: GuardContext) -> ExitReason | None:
        if ctx.iteration >= self.max_steps:
            return ExitReason(
                condition="max_steps",
                message=f"Hard step cap reached ({self.max_steps})",
            )
        return None


class TimeoutGuard:
    """Stop when wall-clock time since session start exceeds max_wall_ms."""

    def should_stop(self, ctx: GuardContext) -> ExitReason | None:
        max_ms = ctx.session.max_wall_ms
        if max_ms <= 0:
            return None
        elapsed_ms = (time.perf_counter() - ctx.session.started_at) * 1000.0
        if elapsed_ms >= max_ms:
            return ExitReason(
                condition="timeout",
                message=(
                    f"Wall-clock timeout: {elapsed_ms:.0f}ms >= {max_ms:.0f}ms"
                ),
            )
        return None


class TokenBudgetGuard:
    """Stop when cumulative session tokens exceed max_tokens."""

    def should_stop(self, ctx: GuardContext) -> ExitReason | None:
        cap = ctx.session.max_tokens
        if cap <= 0:
            return None
        total = ctx.session.prompt_tokens + ctx.session.completion_tokens
        if total >= cap:
            return ExitReason(
                condition="token_budget",
                message=f"Token budget exhausted ({total}/{cap})",
            )
        return None


class OutputRepeatGuard:
    """Stop when the same step artifact repeats within a sliding window."""

    def __init__(self, window: int = 5, min_repeats: int = 2) -> None:
        self.window = window
        self.min_repeats = min_repeats

    def should_stop(self, ctx: GuardContext) -> ExitReason | None:
        if not ctx.step_output:
            return None
        fingerprint = hashlib.sha256(ctx.step_output.strip().encode()).hexdigest()[:16]
        session = ctx.session
        session.step_fingerprints.append(fingerprint)
        recent = session.step_fingerprints[-self.window :]
        count = recent.count(fingerprint)
        if count >= self.min_repeats:
            return ExitReason(
                condition="output_repeat",
                message=(
                    f"Step output repeated {count} times in last {len(recent)} "
                    f"step(s); likely stuck in a loop"
                ),
            )
        return None


def default_guard_stack(priors: AdaptivePriors, max_steps: int = MAX_STEPS_DEFAULT) -> GuardStack:
    """Build the default guard stack mirroring legacy _decide() order."""
    return GuardStack([
        ScoreThresholdGuard(),
        PlateauGuard(),
        BayesianGuard(priors),
        BudgetExhaustedGuard(),
        MaxStepsGuard(max_steps),
        TimeoutGuard(),
        TokenBudgetGuard(),
        OutputRepeatGuard(),
    ])


def bayesian_exit_as_guard(
    priors: AdaptivePriors,
    task_type: str,
    model_id: str,
    quality_threshold: float,
) -> ExitConditionAdapter:
    """Wrap :class:`BayesianExitCondition` as an agent-loop guard."""
    return ExitConditionAdapter(
        BayesianExitCondition(
            priors=priors,
            task_type=task_type,
            model_id=model_id,
            quality_threshold=quality_threshold,
        )
    )
