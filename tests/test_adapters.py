"""Tests for the framework-agnostic AdaptiveStopper."""
from __future__ import annotations

from loopllm import AdaptivePriors, AdaptiveStopper


def test_stops_when_supplied_score_meets_threshold() -> None:
    stopper = AdaptiveStopper(goal="g", task_type="t", quality_threshold=0.8)
    assert stopper.should_continue({"score": 0.4}) is True
    assert stopper.should_continue({"score": 0.95}) is False
    assert stopper.summary is not None
    assert stopper.last_verdict is not None
    assert stopper.last_verdict["decision"] == "stop"


def test_idempotent_after_stop() -> None:
    stopper = AdaptiveStopper(goal="g", task_type="t", quality_threshold=0.8)
    stopper.should_continue({"score": 0.99})
    # Further calls keep returning False without recording new steps.
    assert stopper.should_continue({"score": 0.1}) is False
    assert stopper.summary["steps_run"] == 1


def test_local_channel_a_scoring_from_output() -> None:
    """With no supplied score, the stopper verifies the artifact locally."""
    stopper = AdaptiveStopper(
        goal="make tests pass",
        task_type="bugfix",
        evaluator_type="regex",
        required_patterns=[r"0 failed"],
        quality_threshold=0.8,
    )
    # Artifact missing the pattern -> low verified score -> keep going.
    assert stopper.should_continue({"output": "pytest: 3 failed, 9 passed"}) is True
    assert stopper.last_verdict["score"] < 0.8
    # Artifact satisfies the regex -> verified score 1.0 -> stop.
    assert stopper.should_continue({"output": "pytest: 42 passed, 0 failed"}) is False


def test_route_returns_node_names() -> None:
    stopper = AdaptiveStopper(goal="g", task_type="t", quality_threshold=0.8)
    assert stopper.route({"score": 0.3}, "agent", "END") == "agent"
    assert stopper.route({"score": 0.9}, "agent", "END") == "END"


def test_token_guard_stops_loop() -> None:
    stopper = AdaptiveStopper(
        goal="g", task_type="t", quality_threshold=0.99, max_tokens=1000
    )
    # Sub-threshold scores, but tokens blow the cap -> token guard stops it.
    stopper.should_continue({"score": 0.2, "tokens": 600})
    stopped = stopper.should_continue({"score": 0.2, "tokens": 600})
    assert stopped is False
    assert "token" in stopper.last_verdict["reason"].lower()


def test_reset_allows_new_loop() -> None:
    priors = AdaptivePriors()
    stopper = AdaptiveStopper(priors=priors, goal="g", task_type="t", quality_threshold=0.8)
    stopper.should_continue({"score": 0.9})
    assert stopper.summary is not None
    stopper.reset()
    assert stopper.summary is None
    assert stopper.should_continue({"score": 0.4}) is True


def test_callable_alias() -> None:
    stopper = AdaptiveStopper(goal="g", task_type="t", quality_threshold=0.8)
    assert stopper({"score": 0.4}) is True
