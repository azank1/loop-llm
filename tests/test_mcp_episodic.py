"""Integration tests for episodic recall + recovery via MCP tool handlers."""
from __future__ import annotations

import json

import pytest

import loopllm.mcp_server as m
from loopllm import AdaptivePriors, AgentLoopController


# -- controller-level recovery contract (pure, no module globals) ------------


def test_controller_snapshot_restore_roundtrip() -> None:
    priors = AdaptivePriors()
    c1 = AgentLoopController(priors)
    s = c1.start("fix flaky auth tests", task_type="bugfix", quality_threshold=0.8)
    c1.step(s.session_id, 0.4, step_output="pytest: 3 failed")
    snap = c1.get_session(s.session_id).to_snapshot()

    # New controller = simulated MCP restart.
    c2 = AgentLoopController(priors)
    sid = c2.restore_from_snapshot(snap)
    assert sid == s.session_id
    assert c2.status(sid)["steps_used"] == 1
    # The resumed loop keeps advancing.
    verdict = c2.step(sid, 0.95, step_output="pytest: 42 passed, 0 failed")
    assert verdict["steps_used"] == 2


def test_hydrate_skips_closed_and_non_loop_runs() -> None:
    priors = AdaptivePriors()
    src = AgentLoopController(priors)
    snap = src.get_session(
        src.start("g", task_type="bugfix").session_id
    ).to_snapshot()

    target = AgentLoopController(priors)
    runs = [
        {"run_type": "agent_loop", "state": snap},
        {"run_type": "agent_loop", "state": {**snap, "session_id": "x", "closed": True}},
        {"run_type": "plan", "state": {"plan_id": "p"}},
    ]
    assert target.hydrate_active_loops(runs) == 1


# -- tool-level tests with isolated module state -----------------------------


@pytest.fixture
def mcp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOPLLM_DB", str(tmp_path / "store.db"))
    monkeypatch.setenv("LOOPLLM_PROVIDER", "mock")
    for g in (
        "_store", "_priors", "_provider", "_episodic", "_agent_loop",
        "_status_path", "_history_path",
    ):
        if hasattr(m, g):
            setattr(m, g, None)
    yield m
    for g in ("_store", "_priors", "_provider", "_episodic", "_agent_loop"):
        if hasattr(m, g):
            setattr(m, g, None)


def _checkpointed_step(env, sid: str, score: float, output: str) -> None:
    ctrl = env._get_agent_loop()
    ctrl.step(sid, score, step_output=output)
    env._get_episodic().upsert_active_run(
        sid, "agent_loop", ctrl.get_session(sid).to_snapshot()
    )


def test_loop_resume_after_restart(mcp_env) -> None:
    env = mcp_env
    sid = json.loads(env._tool_loop_start("fix flaky auth tests", task_type="bugfix"))["session_id"]
    _checkpointed_step(env, sid, 0.4, "pytest: 3 failed")

    # Simulate MCP restart: drop the in-memory controller, re-hydrate from store.
    env._agent_loop = None
    restored = env._get_agent_loop().hydrate_active_loops(
        env._get_episodic().list_active_runs(run_type="agent_loop")
    )
    assert restored == 1

    res = json.loads(env._tool_loop_resume(sid))
    assert res["resumed"] is True
    assert res["status"]["steps_used"] == 1
    # Continue stepping post-resume.
    env._get_agent_loop().step(sid, 0.95, step_output="pytest: 42 passed, 0 failed")
    assert env._get_agent_loop().status(sid)["steps_used"] == 2


def test_loop_resume_no_session_id_resumes_single(mcp_env) -> None:
    env = mcp_env
    sid = json.loads(env._tool_loop_start("migrate sql schema", task_type="refactor"))["session_id"]
    _checkpointed_step(env, sid, 0.3, "in progress")
    env._agent_loop = None
    env._get_agent_loop().hydrate_active_loops(
        env._get_episodic().list_active_runs(run_type="agent_loop")
    )
    res = json.loads(env._tool_loop_resume())
    assert res["resumed"] is True and res["status"]["session_id"] == sid


def test_run_status_lists_active_loop(mcp_env) -> None:
    env = mcp_env
    sid = json.loads(env._tool_loop_start("g", task_type="bugfix"))["session_id"]
    status = json.loads(env._tool_run_status())
    ids = [loop.get("session_id") for loop in status["active_agent_loops"]]
    assert sid in ids


def test_loop_start_injects_similar_episode(mcp_env) -> None:
    env = mcp_env
    # Record an episode by finishing a loop.
    sid = json.loads(env._tool_loop_start("fix flaky auth tests", task_type="bugfix"))["session_id"]
    env._get_agent_loop().step(sid, 0.95, step_output="pytest: 42 passed, 0 failed")
    env._tool_loop_end(sid)

    # A new similar loop should surface it.
    start2 = json.loads(env._tool_loop_start("fix flaky auth tests again", task_type="bugfix"))
    assert start2["similar_episodes"]
    assert "auth" in start2["similar_episodes"][0]["goal"].lower()


def test_recall_and_resume_unknown(mcp_env) -> None:
    env = mcp_env
    assert json.loads(env._tool_loop_resume("nope"))["error"]
    assert json.loads(env._tool_recall("anything"))["count"] == 0


def test_server_registers_expected_tool_count(mcp_env) -> None:
    """31 MCP tools are registered (keeps the README banner honest)."""
    import asyncio

    server = mcp_env.create_mcp_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert len(names) == 31
    assert {"loopllm_recall", "loopllm_run_status", "loopllm_loop_resume"} <= names
