"""Tests for agent-loop guard stack."""
from __future__ import annotations

import time

from loopllm import AdaptivePriors, AgentLoopController, AgentLoopSession
from loopllm.guards import (
    GuardContext,
    GuardStack,
    MaxStepsGuard,
    OutputRepeatGuard,
    PlateauGuard,
    ScoreThresholdGuard,
    TimeoutGuard,
    TokenBudgetGuard,
)


def _session(**kwargs: object) -> AgentLoopSession:
    defaults = {
        "session_id": "abc",
        "goal": "test goal",
        "task_type": "general",
        "model_id": "m",
        "quality_threshold": 0.8,
        "suggested_budget": 5,
    }
    defaults.update(kwargs)
    return AgentLoopSession(**defaults)  # type: ignore[arg-type]


def test_score_threshold_guard_stops() -> None:
    session = _session()
    ctx = GuardContext(session=session, iteration=2, current_score=0.9, scores_so_far=[0.5, 0.9])
    reason = ScoreThresholdGuard().should_stop(ctx)
    assert reason is not None
    assert reason.condition == "quality_threshold"


def test_plateau_guard_stops() -> None:
    session = _session()
    ctx = GuardContext(
        session=session,
        iteration=3,
        current_score=0.51,
        scores_so_far=[0.50, 0.505, 0.51],
    )
    reason = PlateauGuard().should_stop(ctx)
    assert reason is not None
    assert reason.condition == "plateau"


def test_timeout_guard_stops() -> None:
    session = _session(max_wall_ms=1.0, started_at=time.perf_counter() - 1.0)
    ctx = GuardContext(session=session, iteration=1, current_score=0.3, scores_so_far=[0.3])
    reason = TimeoutGuard().should_stop(ctx)
    assert reason is not None
    assert reason.condition == "timeout"


def test_token_budget_guard_stops() -> None:
    session = _session(max_tokens=100, prompt_tokens=60, completion_tokens=50)
    ctx = GuardContext(session=session, iteration=1, current_score=0.3, scores_so_far=[0.3])
    reason = TokenBudgetGuard().should_stop(ctx)
    assert reason is not None
    assert reason.condition == "token_budget"


def test_output_repeat_guard_stops() -> None:
    session = _session()
    guard = OutputRepeatGuard(window=5, min_repeats=2)
    artifact = "same output every time"
    ctx1 = GuardContext(
        session=session, iteration=1, current_score=0.3,
        scores_so_far=[0.3], step_output=artifact,
    )
    assert guard.should_stop(ctx1) is None
    ctx2 = GuardContext(
        session=session, iteration=2, current_score=0.3,
        scores_so_far=[0.3, 0.3], step_output=artifact,
    )
    reason = guard.should_stop(ctx2)
    assert reason is not None
    assert reason.condition == "output_repeat"


def test_guard_stack_first_wins() -> None:
    session = _session()
    stack = GuardStack([ScoreThresholdGuard(), MaxStepsGuard(max_steps=1)])
    ctx = GuardContext(session=session, iteration=1, current_score=0.9, scores_so_far=[0.9])
    reason = stack.evaluate(ctx)
    assert reason is not None
    assert reason.condition == "quality_threshold"


def test_controller_uses_guard_stack_for_plateau() -> None:
    controller = AgentLoopController(AdaptivePriors())
    session = controller.start("goal", task_type="decompose", model_id="m")
    controller.step(session.session_id, 0.50)
    controller.step(session.session_id, 0.505)
    verdict = controller.step(session.session_id, 0.506)
    assert verdict["decision"] == "stop"
    assert "plateau" in verdict["reason"].lower()


def test_controller_end_populates_token_fields() -> None:
    priors = AdaptivePriors()
    controller = AgentLoopController(priors)
    session = controller.start("goal", task_type="general", model_id="m")
    controller.step(session.session_id, 0.9, step_tokens=100)
    controller.end(session.session_id)
    report = priors.report("general", "m")
    assert report["total_calls"] >= 1
