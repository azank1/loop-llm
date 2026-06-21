#!/usr/bin/env python3
"""Example: adaptive agent loops with AgentLoopController.

Most agent loops stop on a fixed ``max_iterations`` or "let the LLM decide".
loop-llm instead advises an agent's own plan/act/observe loop on *when to stop*
using learned Bayesian priors — and learns the optimal depth per task type from
the loops it sees. No training data required.

Run it::

    python examples/agent_loop.py
"""
from __future__ import annotations

import logging

import structlog

from loopllm import AdaptivePriors, AgentLoopController, CallObservation

# Keep the demo output clean — silence the library's info/debug logs.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def simulate_loop(controller: AgentLoopController, scores: list[float], task_type: str) -> None:
    """Drive one agent loop, reporting each step's progress score."""
    session = controller.start(
        goal="Refactor module and make the test suite pass",
        task_type=task_type,
        model_id="demo-model",
    )
    print(f"\n=== Loop {session.session_id} (task_type={task_type}) ===")
    print(
        f"Suggested budget: {session.suggested_budget} step(s) | "
        f"threshold {session.quality_threshold:.2f} | "
        f"confidence {session.confidence:.2f} "
        f"(from {session.total_observations} past loops)"
    )

    for score in scores:
        verdict = controller.step(session.session_id, score, note=f"step scored {score}")
        bar = "#" * int(score * 20)
        print(
            f"  step {verdict['steps_used']:>2} | {score:.2f} |{bar:<20}| "
            f"-> {verdict['decision'].upper()}: {verdict['reason']}"
        )
        if verdict["decision"] == "stop":
            break

    summary = controller.end(session.session_id)
    learned = summary["learned"]
    print(
        f"  learned: optimal_depth={learned['optimal_depth']} "
        f"converge_rate={learned['converge_rate']} "
        f"(obs={learned['total_observations']})"
    )


def main() -> None:
    priors = AdaptivePriors()
    controller = AgentLoopController(priors)

    print("PromptLoop — adaptive agent loops")
    print("=" * 60)
    print(
        "Cold start: no history yet, so the controller falls back to a\n"
        "task-type default and stops the loop the moment progress is good\n"
        "enough or plateaus."
    )

    # A loop that converges quickly (good progress on step 2).
    simulate_loop(controller, scores=[0.45, 0.85], task_type="bugfix")

    # A loop that stalls — the plateau guard stops it instead of burning budget.
    simulate_loop(controller, scores=[0.50, 0.505, 0.506], task_type="bugfix")

    # Seed history so the system *learns* how deep bugfix loops really need to go.
    print("\n" + "=" * 60)
    print("Feeding 15 historical bugfix loops that all converge by step 2...")
    for _ in range(15):
        priors.observe(
            CallObservation(
                task_type="bugfix",
                model_id="demo-model",
                scores=[0.5, 0.86],
                latencies_ms=[800.0, 750.0],
                converged=True,
                total_iterations=2,
                max_iterations=5,
                quality_threshold=0.8,
            )
        )

    print("Now the controller's beliefs are informed by real data:")
    report = priors.report("bugfix", "demo-model")
    print(
        f"  optimal_depth={report['optimal_depth']} "
        f"converge_rate={report['converge_rate']} "
        f"confidence={report['confidence']} "
        f"first_call_quality={report['first_call_quality']}"
    )

    # A fresh loop now starts with a learned, confident budget.
    simulate_loop(controller, scores=[0.55, 0.88], task_type="bugfix")


if __name__ == "__main__":
    main()
