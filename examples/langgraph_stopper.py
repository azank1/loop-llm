#!/usr/bin/env python3
"""Example: enforced adaptive stopping in a framework loop via AdaptiveStopper.

`AdaptiveStopper.should_continue(state)` drops straight into a LangGraph
conditional edge, a CrewAI step callback, an AutoGen termination function, or a
plain `while` loop. Here we simulate a tiny agent graph with no external
dependency: a node produces an artifact each step, and the stopper decides
whether to route back to the node or to END — scoring the artifact locally with a
deterministic Channel-A evaluator (no agent self-grading).

Run it::

    python examples/langgraph_stopper.py
"""
from __future__ import annotations

import logging

import structlog

from loopllm import AdaptiveStopper

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


# A scripted "agent node": each step emits an artifact that gets closer to a
# passing test run. In a real graph this is your model/tool call.
ARTIFACTS = [
    "pytest: 8 failed, 4 passed",
    "pytest: 3 failed, 9 passed",
    "pytest: 42 passed, 0 failed",   # crosses the regex bar -> stop
    "pytest: 42 passed, 0 failed",   # never reached
]


def main() -> None:
    stopper = AdaptiveStopper(
        goal="make the failing tests pass",
        task_type="bugfix",
        evaluator_type="regex",
        required_patterns=[r"0 failed"],   # Channel-A: artifact must show 0 failures
        quality_threshold=0.8,
    )

    print("Adaptive stopper driving a (simulated) agent graph\n")
    step = 0
    while step < len(ARTIFACTS):
        artifact = ARTIFACTS[step]
        state = {"output": artifact, "tokens": 1200}

        # This is the line you'd put on a LangGraph conditional edge:
        keep_going = stopper.should_continue(state)
        v = stopper.last_verdict or {}
        print(
            f"  node step {v.get('steps_used', step + 1)}: {artifact!r}\n"
            f"    -> verified score {v.get('score')}, "
            f"route = {'NODE (loop)' if keep_going else 'END'} "
            f"({v.get('reason', '')})"
        )
        if not keep_going:
            break
        step += 1

    print("\nLoop finished. What the stopper learned:")
    summary = stopper.summary or {}
    learned = summary.get("learned", {})
    print(
        f"  steps_run={summary.get('steps_run')} converged={summary.get('converged')} "
        f"optimal_depth={learned.get('optimal_depth')} obs={learned.get('total_observations')}"
    )


if __name__ == "__main__":
    main()
