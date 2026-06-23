#!/usr/bin/env python3
"""Example: episodic memory — record past outcomes, then recall them next time.

Layer-3 agent loops record a compact *episode* when they finish (goal, summary,
score, stop reason, tags). The next time a similar task comes up, `loopllm_recall`
(and `loopllm_loop_start`, which auto-injects `similar_episodes`) surfaces what
worked before — so the agent doesn't start cold.

This demo uses the library API directly. Run it::

    python examples/episodic_recall_loop.py
"""
from __future__ import annotations

import logging

import structlog

from loopllm import AdaptivePriors, AgentLoopController, EpisodicStore, LoopStore
from loopllm.episodes import summarize_artifacts

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def run_and_record(controller: AgentLoopController, episodic: EpisodicStore,
                   goal: str, task_type: str, trajectory: list[str], scores: list[float]) -> None:
    """Drive one loop, then record an episode of how it went."""
    session = controller.start(goal, task_type=task_type, quality_threshold=0.8)
    last = ""
    for artifact, score in zip(trajectory, scores):
        verdict = controller.step(session.session_id, score, step_output=artifact)
        last = artifact
        if verdict["decision"] == "stop":
            break
    summary = controller.end(session.session_id)
    episodic.record_episode(
        episode_type="agent_loop",
        goal=goal,
        task_type=task_type,
        model_id="demo",
        summary=summarize_artifacts(goal, output=last,
                                    stop_reason=str(summary.get("converged")),
                                    score_final=summary.get("final_score")),
        score_final=summary.get("final_score"),
        steps_used=summary.get("steps_run"),
        stop_reason="converged" if summary.get("converged") else "stopped",
    )
    print(f"  recorded episode: {goal!r} (steps={summary.get('steps_run')})")


def main() -> None:
    store = LoopStore(db_path=":memory:")
    episodic = EpisodicStore(store)
    controller = AgentLoopController(AdaptivePriors())

    print("1) Run a few loops and record what happened:")
    run_and_record(controller, episodic, "fix flaky auth login tests", "bugfix",
                   ["pytest: 3 failed", "pytest: 42 passed, 0 failed"], [0.4, 0.9])
    run_and_record(controller, episodic, "migrate users table to v5 schema", "refactor",
                   ["alembic: 1 pending", "alembic: head, 0 pending"], [0.5, 0.88])
    run_and_record(controller, episodic, "write API reference docs", "docs",
                   ["draft outline", "full reference, examples included"], [0.6, 0.85])

    print("\n2) A new, similar task — recall what worked before:")
    query = "auth tests failing"
    hits = episodic.recall(query, k=3)
    print(f"  loopllm_recall({query!r}) -> {len(hits)} hit(s)")
    for h in hits:
        print(f"    - {h['goal']}  (score={h['score_final']}, {h['stop_reason']})")

    top = hits[0] if hits else None
    print(
        "\n3) loopllm_loop_start would inject this as `similar_episodes`, so the "
        f"agent begins with context: {top['goal'] if top else '(none yet)'}"
    )


if __name__ == "__main__":
    main()
