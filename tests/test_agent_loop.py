"""Tests for the adaptive agent-loop controller."""
from __future__ import annotations

import pytest

from loopllm import AdaptivePriors, AgentLoopController, CallObservation


def _seed(priors: AdaptivePriors, task_type: str, scores: list[float], n: int = 12) -> None:
    """Record *n* identical converging observations for a task type."""
    for _ in range(n):
        priors.observe(
            CallObservation(
                task_type=task_type,
                model_id="m",
                scores=scores,
                latencies_ms=[100.0] * len(scores),
                converged=scores[-1] >= 0.8,
                total_iterations=len(scores),
                max_iterations=5,
                quality_threshold=0.8,
            )
        )


def test_start_cold_start_uses_task_type_default() -> None:
    """With no observations, the budget falls back to the task-type default."""
    controller = AgentLoopController(AdaptivePriors())
    session = controller.start("do a thing", task_type="resolve", model_id="m")
    # 'resolve' default is 2 (see AdaptivePriors.predict_optimal_depth).
    assert session.suggested_budget == 2
    assert session.quality_threshold == pytest.approx(0.8)
    assert session.total_observations == 0


def test_step_stops_when_threshold_met() -> None:
    """A step at or above threshold yields a stop verdict."""
    controller = AgentLoopController(AdaptivePriors())
    session = controller.start("goal", task_type="general", model_id="m")
    verdict = controller.step(session.session_id, 0.95)
    assert verdict["decision"] == "stop"
    assert "Goal reached" in verdict["reason"]


def test_step_stops_when_budget_exhausted() -> None:
    """Sub-threshold steps stop once the suggested budget is reached."""
    controller = AgentLoopController(AdaptivePriors())
    # validate default budget is 1, so the first sub-threshold step exhausts it.
    session = controller.start("goal", task_type="validate", model_id="m")
    assert session.suggested_budget == 1
    verdict = controller.step(session.session_id, 0.3)
    assert verdict["decision"] == "stop"


def test_step_continues_while_below_threshold_within_budget() -> None:
    """A rising, sub-threshold step inside budget keeps the loop going."""
    controller = AgentLoopController(AdaptivePriors())
    session = controller.start("goal", task_type="decompose", model_id="m")
    assert session.suggested_budget >= 2
    verdict = controller.step(session.session_id, 0.4)
    assert verdict["decision"] == "continue"
    assert verdict["steps_used"] == 1


def test_step_stops_on_plateau() -> None:
    """Three near-identical sub-threshold scores trigger a plateau stop."""
    controller = AgentLoopController(AdaptivePriors())
    session = controller.start("goal", task_type="decompose", model_id="m")
    controller.step(session.session_id, 0.50)
    controller.step(session.session_id, 0.505)
    verdict = controller.step(session.session_id, 0.506)
    assert verdict["decision"] == "stop"
    assert "plateau" in verdict["reason"].lower()


def test_end_records_observation_and_learns() -> None:
    """Closing loops updates priors; the learned optimal depth moves toward truth."""
    priors = AdaptivePriors()
    controller = AgentLoopController(priors)

    # Before learning: an unknown task type gets the generic default of 3.
    first = controller.start("goal", task_type="quickjob", model_id="m")
    assert first.suggested_budget == 3

    # Seed many fast-converging (2-step) loops for the task type.
    _seed(priors, "quickjob", scores=[0.6, 0.85], n=12)

    after = controller.start("goal again", task_type="quickjob", model_id="m")
    assert after.total_observations >= 12

    # Run one real loop through the controller and confirm it learns.
    controller.step(after.session_id, 0.6)
    controller.step(after.session_id, 0.85)
    summary = controller.end(after.session_id)
    assert summary["learned"]["total_observations"] >= 13
    # These loops converge on step 2, so the belief should sit well below the
    # generic prior mean of 3.0.
    assert summary["learned"]["optimal_depth"] < 3.0


def test_end_is_idempotent() -> None:
    """Calling end twice does not double-count the observation."""
    priors = AdaptivePriors()
    controller = AgentLoopController(priors)
    session = controller.start("goal", task_type="idem", model_id="m")
    controller.step(session.session_id, 0.9)
    first = controller.end(session.session_id)
    second = controller.end(session.session_id)
    assert first["learned"]["total_observations"] == second["learned"]["total_observations"] == 1
    assert second["converged"] is True


def test_step_after_end_raises() -> None:
    """Stepping a closed session raises ValueError."""
    controller = AgentLoopController(AdaptivePriors())
    session = controller.start("goal", task_type="general", model_id="m")
    controller.step(session.session_id, 0.9)
    controller.end(session.session_id)
    with pytest.raises(ValueError):
        controller.step(session.session_id, 0.5)


def test_unknown_session_raises_keyerror() -> None:
    """Operating on an unknown session id raises KeyError."""
    controller = AgentLoopController(AdaptivePriors())
    with pytest.raises(KeyError):
        controller.step("nope", 0.5)
    with pytest.raises(KeyError):
        controller.status("nope")
    with pytest.raises(KeyError):
        controller.end("nope")


def test_score_is_clamped() -> None:
    """Out-of-range scores are clamped into [0, 1]."""
    controller = AgentLoopController(AdaptivePriors())
    session = controller.start("goal", task_type="general", model_id="m")
    verdict = controller.step(session.session_id, 1.7)
    assert verdict["score"] == 1.0
    assert controller.status(session.session_id)["score_trajectory"][-1] == 1.0


def test_status_reports_trajectory() -> None:
    """Status reflects recorded steps and the latest decision."""
    controller = AgentLoopController(AdaptivePriors())
    session = controller.start("goal", task_type="decompose", model_id="m")
    controller.step(session.session_id, 0.3)
    controller.step(session.session_id, 0.5)
    status = controller.status(session.session_id)
    assert status["steps_used"] == 2
    assert status["score_trajectory"] == [0.3, 0.5]
    assert status["last_decision"] in {"continue", "stop"}
