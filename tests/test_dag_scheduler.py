"""Tests for DAG virtual sub-agent scheduler."""
from __future__ import annotations

import pytest

from loopllm.dag_scheduler import DagScheduler
from loopllm.episodes import EpisodicStore
from loopllm.store import LoopStore


@pytest.fixture
def scheduler(store: LoopStore) -> DagScheduler:
    return DagScheduler(EpisodicStore(store))


def test_compile_ready_submit_merge(scheduler: DagScheduler) -> None:
    run = scheduler.compile(
        "refactor module and pass tests",
        [
            {
                "id": "n1",
                "role": "implementer",
                "description": "Refactor the module",
                "dependencies": [],
            },
            {
                "id": "n2",
                "role": "test_runner",
                "description": "Run pytest",
                "dependencies": ["n1"],
                "required_patterns": ["passed"],
            },
        ],
        task_type="bugfix",
    )
    frontier = scheduler.ready(run.run_id)
    assert len(frontier) == 1
    assert frontier[0]["node_id"] == "n1"

    r1 = scheduler.submit(
        run.run_id,
        "n1",
        "Refactored module with clean interfaces and docstrings.",
    )
    assert r1["accepted"] is True

    frontier2 = scheduler.ready(run.run_id)
    assert len(frontier2) == 1
    assert frontier2[0]["node_id"] == "n2"

    r2 = scheduler.submit(
        run.run_id,
        "n2",
        "pytest: 12 passed, 0 failed",
    )
    assert r2["accepted"] is True
    assert r2["dag_complete"] is True

    merged = scheduler.merge(run.run_id)
    assert "n1" in merged["merged_output"]
    assert "n2" in merged["merged_output"]


def test_cycle_detection(scheduler: DagScheduler) -> None:
    run = scheduler.compile(
        "bad graph",
        [
            {"id": "a", "role": "x", "description": "a", "dependencies": ["b"]},
            {"id": "b", "role": "x", "description": "b", "dependencies": ["a"]},
        ],
    )
    with pytest.raises(ValueError, match="Cycle"):
        run.execution_order()


def test_submit_rejects_unready_node(scheduler: DagScheduler) -> None:
    run = scheduler.compile(
        "goal",
        [
            {"id": "n1", "role": "a", "description": "first", "dependencies": []},
            {"id": "n2", "role": "b", "description": "second", "dependencies": ["n1"]},
        ],
    )
    result = scheduler.submit(run.run_id, "n2", "too early")
    assert result["accepted"] is False


def test_restore_run(scheduler: DagScheduler) -> None:
    run = scheduler.compile(
        "goal",
        [{"id": "n1", "role": "a", "description": "work", "dependencies": []}],
    )
    state = scheduler.to_dict(run.run_id)
    scheduler2 = DagScheduler()
    scheduler2.restore(state)
    assert scheduler2.status(run.run_id)["run_id"] == run.run_id
