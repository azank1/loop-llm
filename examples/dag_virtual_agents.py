#!/usr/bin/env python3
"""Example: DAG virtual sub-agents (no parallel IDE sessions).

loopllm schedules dependency-ordered nodes; one worker (your agent or this
script) executes each frontier node and submits artifacts for verification.
"""
from __future__ import annotations

from loopllm.dag_scheduler import DagScheduler
from loopllm.episodes import EpisodicStore
from loopllm.store import LoopStore


def main() -> None:
    store = LoopStore(":memory:")
    scheduler = DagScheduler(EpisodicStore(store))

    run = scheduler.compile(
        "Add retry logic and verify tests pass",
        [
            {
                "id": "implement",
                "role": "implementer",
                "description": "Add exponential backoff retry to download()",
                "dependencies": [],
            },
            {
                "id": "test",
                "role": "test_runner",
                "description": "Run pytest on download module",
                "dependencies": ["implement"],
                "required_patterns": ["passed"],
            },
        ],
        task_type="bugfix",
    )
    print(f"DAG run_id={run.run_id}, nodes={list(run.nodes)}")

    for frontier in iter(lambda: scheduler.ready(run.run_id), []):
        if not frontier:
            break
        node = frontier[0]
        print(f"\n--- Executing {node['node_id']} ({node['role']}) ---")
        artifact = {
            "implement": "Added retry with backoff; download() raises after 3 tries.",
            "test": "pytest tests/test_download.py: 4 passed",
        }[node["node_id"]]
        result = scheduler.submit(run.run_id, node["node_id"], artifact)
        print(f"  accepted={result['accepted']} score={result.get('score')}")

    merged = scheduler.merge(run.run_id)
    print("\n=== Merged output ===")
    print(merged["merged_output"])


if __name__ == "__main__":
    main()
