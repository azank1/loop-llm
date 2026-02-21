"""Tests for the SQLite-backed store."""
from __future__ import annotations

from pathlib import Path

import pytest

from loopllm.priors import (
    BetaPrior,
    CallObservation,
    IterationProfile,
    NormalPrior,
    TaskModelPrior,
)
from loopllm.store import LoopStore, SQLiteBackedPriors


@pytest.fixture()
def store() -> LoopStore:
    """Create an in-memory store for testing."""
    return LoopStore(db_path=":memory:")


@pytest.fixture()
def disk_store(tmp_path: Path) -> LoopStore:
    """Create a disk-backed store for persistence tests."""
    return LoopStore(db_path=tmp_path / "test.db")


# -- schema ------------------------------------------------------------------


class TestSchema:
    def test_creates_tables(self, store: LoopStore) -> None:
        with store._connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        names = {row["name"] for row in tables}
        assert "priors" in names
        assert "observations" in names
        assert "questions" in names
        assert "sessions" in names
        assert "tasks" in names
        assert "schema_version" in names
        assert "prompt_history" in names

    def test_schema_version(self, store: LoopStore) -> None:
        with store._connection() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row is not None
        assert row["version"] == 3


# -- priors CRUD -------------------------------------------------------------


class TestPriorsCRUD:
    def _make_prior(self, task_type: str = "test", model_id: str = "m1") -> TaskModelPrior:
        return TaskModelPrior(
            task_type=task_type,
            model_id=model_id,
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
            total_calls=5,
            iterations={
                0: IterationProfile(
                    score=NormalPrior(mean=0.4, variance=0.1, n_observations=5),
                    score_delta=NormalPrior(mean=0.12, variance=0.05, n_observations=4),
                ),
            },
            optimal_depth=NormalPrior(mean=3.5, variance=1.5, n_observations=5),
            overall_converge_rate=BetaPrior(alpha=4.0, beta=2.0),
            first_call_quality=NormalPrior(mean=0.35, variance=0.08, n_observations=5),
        )

    def test_save_and_load(self, store: LoopStore) -> None:
        prior = self._make_prior()
        store.save_prior("test::m1", prior)

        loaded = store.load_prior("test::m1")
        assert loaded is not None
        assert loaded.task_type == "test"
        assert loaded.model_id == "m1"
        assert loaded.total_calls == 5
        assert loaded.optimal_depth.mean == pytest.approx(3.5)
        assert loaded.overall_converge_rate.alpha == pytest.approx(4.0)

    def test_load_nonexistent(self, store: LoopStore) -> None:
        assert store.load_prior("nonexistent") is None

    def test_upsert(self, store: LoopStore) -> None:
        prior = self._make_prior()
        store.save_prior("test::m1", prior)

        prior.total_calls = 10
        store.save_prior("test::m1", prior)

        loaded = store.load_prior("test::m1")
        assert loaded is not None
        assert loaded.total_calls == 10

    def test_load_all(self, store: LoopStore) -> None:
        store.save_prior("a::m1", self._make_prior("a", "m1"))
        store.save_prior("b::m2", self._make_prior("b", "m2"))

        all_priors = store.load_all_priors()
        assert len(all_priors) == 2
        assert "a::m1" in all_priors
        assert "b::m2" in all_priors

    def test_delete(self, store: LoopStore) -> None:
        store.save_prior("test::m1", self._make_prior())
        assert store.delete_prior("test::m1") is True
        assert store.load_prior("test::m1") is None
        assert store.delete_prior("test::m1") is False

    def test_iteration_roundtrip(self, store: LoopStore) -> None:
        """Verify per-iteration profiles survive serialization."""
        prior = self._make_prior()
        store.save_prior("test::m1", prior)
        loaded = store.load_prior("test::m1")
        assert loaded is not None
        assert 0 in loaded.iterations
        profile = loaded.iterations[0]
        assert profile.score.mean == pytest.approx(0.4)
        assert profile.score_delta.mean == pytest.approx(0.12)


# -- observations ------------------------------------------------------------


class TestObservations:
    def test_record_and_query(self, store: LoopStore) -> None:
        obs = CallObservation(
            task_type="code_gen",
            model_id="gpt-4o-mini",
            scores=[0.3, 0.6, 0.85],
            latencies_ms=[100.0, 120.0, 110.0],
            converged=True,
            total_iterations=3,
        )
        row_id = store.record_observation(obs)
        assert row_id > 0

        results = store.get_observations(task_type="code_gen")
        assert len(results) == 1
        assert results[0].scores == [0.3, 0.6, 0.85]
        assert results[0].converged is True

    def test_filter_by_model(self, store: LoopStore) -> None:
        for model in ["gpt-4o-mini", "gpt-4o-mini", "llama3"]:
            store.record_observation(
                CallObservation(
                    task_type="test", model_id=model,
                    scores=[0.5], latencies_ms=[100.0],
                    converged=False, total_iterations=1,
                )
            )

        assert store.count_observations(model_id="gpt-4o-mini") == 2
        assert store.count_observations(model_id="llama3") == 1

    def test_count_all(self, store: LoopStore) -> None:
        for _ in range(5):
            store.record_observation(
                CallObservation(
                    task_type="t", model_id="m",
                    scores=[0.5], latencies_ms=[100.0],
                    converged=False, total_iterations=1,
                )
            )
        assert store.count_observations() == 5


# -- question stats ----------------------------------------------------------


class TestQuestionStats:
    def test_update_and_retrieve(self, store: LoopStore) -> None:
        store.update_question_stats("scope", "code_gen", positive=True, info_gain=0.8)
        store.update_question_stats("scope", "code_gen", positive=True, info_gain=0.6)
        store.update_question_stats("scope", "code_gen", positive=False, info_gain=0.2)

        stats = store.get_question_stats(task_type="code_gen")
        assert len(stats) == 1
        s = stats[0]
        assert s["question_type"] == "scope"
        assert s["asked_count"] == 3
        assert s["positive_impact"] == 2
        assert s["negative_impact"] == 1
        assert s["effectiveness"] == pytest.approx(2 / 3)


# -- sessions ----------------------------------------------------------------


class TestSessions:
    def test_create_and_get(self, store: LoopStore) -> None:
        store.create_session("s1", "Write a function", task_type="code_gen")
        session = store.get_session("s1")
        assert session is not None
        assert session["original_prompt"] == "Write a function"
        assert session["task_type"] == "code_gen"
        assert session["questions"] == []
        assert session["answers"] == {}

    def test_update_session(self, store: LoopStore) -> None:
        store.create_session("s2", "prompt")
        store.update_session(
            "s2",
            questions=[{"type": "scope", "text": "what scope?"}],
            answers={"scope": "full module"},
            final_score=0.9,
        )
        session = store.get_session("s2")
        assert session is not None
        assert len(session["questions"]) == 1
        assert session["answers"]["scope"] == "full module"
        assert session["final_score"] == pytest.approx(0.9)
        assert session["completed_at"] is not None

    def test_get_nonexistent(self, store: LoopStore) -> None:
        assert store.get_session("nope") is None


# -- tasks -------------------------------------------------------------------


class TestTasks:
    def test_save_and_get(self, store: LoopStore) -> None:
        store.save_task({
            "id": "t1",
            "title": "Write function",
            "description": "Implement sorting",
            "state": "pending",
        })
        task = store.get_task("t1")
        assert task is not None
        assert task["title"] == "Write function"
        assert task["state"] == "pending"

    def test_update_state(self, store: LoopStore) -> None:
        store.save_task({"id": "t2", "title": "Test"})
        store.update_task_state("t2", "in_progress")
        task = store.get_task("t2")
        assert task is not None
        assert task["state"] == "in_progress"

    def test_query_by_state(self, store: LoopStore) -> None:
        store.save_task({"id": "t3", "title": "A", "state": "pending"})
        store.save_task({"id": "t4", "title": "B", "state": "completed"})
        store.save_task({"id": "t5", "title": "C", "state": "pending"})

        pending = store.get_tasks(state="pending")
        assert len(pending) == 2
        completed = store.get_tasks(state="completed")
        assert len(completed) == 1


# -- SQLiteBackedPriors integration ------------------------------------------


class TestSQLiteBackedPriors:
    def test_observe_persists(self, store: LoopStore) -> None:
        priors = SQLiteBackedPriors(store)
        obs = CallObservation(
            task_type="test",
            model_id="m1",
            scores=[0.3, 0.6, 0.85],
            latencies_ms=[100.0, 120.0, 110.0],
            converged=True,
            total_iterations=3,
            quality_threshold=0.8,
        )
        priors.observe(obs)

        # Verify in-memory
        report = priors.report("test", "m1")
        assert report["total_calls"] == 1

        # Verify persisted in DB
        loaded = store.load_prior("test::m1")
        assert loaded is not None
        assert loaded.total_calls == 1

        # Verify observation logged
        assert store.count_observations(task_type="test") == 1

    def test_loads_from_store(self, store: LoopStore) -> None:
        """Priors saved by one instance should be loadable by another."""
        p1 = SQLiteBackedPriors(store)
        for _ in range(3):
            p1.observe(CallObservation(
                task_type="load_test", model_id="m1",
                scores=[0.5, 0.8], latencies_ms=[100.0, 120.0],
                converged=True, total_iterations=2, quality_threshold=0.8,
            ))

        # New instance loads from same store
        p2 = SQLiteBackedPriors(store)
        report = p2.report("load_test", "m1")
        assert report["total_calls"] == 3

    def test_disk_persistence(self, disk_store: LoopStore) -> None:
        """Verify data survives store close + reopen."""
        p = SQLiteBackedPriors(disk_store)
        p.observe(CallObservation(
            task_type="persist", model_id="m1",
            scores=[0.4, 0.7], latencies_ms=[100.0, 110.0],
            converged=False, total_iterations=2, quality_threshold=0.8,
        ))
        db_path = disk_store.db_path
        disk_store.close()

        # Reopen
        store2 = LoopStore(db_path=db_path)
        p2 = SQLiteBackedPriors(store2)
        report = p2.report("persist", "m1")
        assert report["total_calls"] == 1
        store2.close()
