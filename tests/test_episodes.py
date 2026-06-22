"""Tests for episodic memory (schema v5)."""
from __future__ import annotations

from loopllm.episodes import EpisodicStore, extract_tags, summarize_artifacts
from loopllm.store import LoopStore, SCHEMA_VERSION


def test_schema_v5_migration(store: LoopStore) -> None:
    with store._connection() as conn:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == SCHEMA_VERSION
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "episodes" in tables
    assert "active_runs" in tables


def test_record_and_recall_episode(store: LoopStore) -> None:
    episodic = EpisodicStore(store)
    episodic.record_episode(
        episode_type="agent_loop",
        goal="fix flaky auth pytest",
        task_type="bugfix",
        model_id="demo",
        summary="Stopped at step 2 with tests passing",
        score_final=0.9,
        steps_used=2,
        stop_reason="goal reached",
    )
    episodic.record_episode(
        episode_type="agent_loop",
        goal="write readme docs",
        task_type="docs",
        model_id="demo",
        summary="Documentation update",
        score_final=0.85,
        steps_used=1,
    )
    hits = episodic.recall("flaky auth pytest", task_type="bugfix", k=3)
    assert len(hits) >= 1
    assert "auth" in hits[0]["goal"].lower() or "pytest" in hits[0]["summary"].lower()


def test_active_run_lifecycle(store: LoopStore, tmp_path) -> None:
    mirror = tmp_path / "active_run.json"
    episodic = EpisodicStore(store, mirror_path=mirror)
    episodic.upsert_active_run("sess1", "agent_loop", {"goal": "test"})
    assert episodic.get_active_run("sess1") is not None
    assert mirror.exists()
    episodic.clear_active_run("sess1")
    assert episodic.get_active_run("sess1") is None


def test_summarize_artifacts() -> None:
    text = summarize_artifacts(
        "fix tests",
        step_outputs=["pytest: 12 passed"],
        stop_reason="goal reached",
        score_final=0.95,
    )
    assert "fix tests" in text
    assert "0.95" in text


def test_extract_tags() -> None:
    tags = extract_tags("refactor auth module and run pytest", "bugfix")
    assert "bugfix" in tags
    assert "pytest" in tags
    assert "auth" in tags
