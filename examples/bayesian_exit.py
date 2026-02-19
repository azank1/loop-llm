#!/usr/bin/env python3
"""Example: Bayesian adaptive exit with AdaptivePriors.

Demonstrates how loop-llm learns from historical observations to predict
optimal loop depth and make early-exit decisions.
"""
from __future__ import annotations

from loopllm import AdaptivePriors, CallObservation, LoopConfig, LoopedLLM
from loopllm.adaptive_exit import BayesianExitCondition
from loopllm.evaluators import ThresholdEvaluator
from loopllm.providers.mock import MockLLMProvider


def main() -> None:
    # 1. Create priors and simulate training observations
    priors = AdaptivePriors()

    print("=== Training phase: 20 observations ===")
    for i in range(20):
        priors.observe(
            CallObservation(
                task_type="decompose",
                model_id="gpt-4o",
                scores=[0.3, 0.55, 0.78, 0.88],
                latencies_ms=[1200, 1100, 1000, 900],
                converged=True,
                total_iterations=4,
                max_iterations=5,
                quality_threshold=0.8,
            )
        )

    # 2. Report learned beliefs
    report = priors.report("decompose", "gpt-4o")
    print(f"\nLearned beliefs after {report['total_calls']} observations:")
    print(f"  Optimal depth: {report['optimal_depth']}")
    print(f"  Converge rate: {report['converge_rate']}")
    print(f"  First-call quality: {report['first_call_quality']}")

    # 3. Get suggested config
    config_suggestion = priors.suggest_config("decompose", "gpt-4o")
    print(f"\nSuggested config: {config_suggestion}")

    # 4. Run a loop with Bayesian exit condition
    provider = MockLLMProvider(
        responses=["Low quality", "Medium quality", "Good quality", "Excellent"]
    )

    # Scores track improvement
    call_scores = iter([0.3, 0.55, 0.78, 0.88])

    def scorer(output: str, context: dict) -> float:
        return next(call_scores, 0.9)

    evaluator = ThresholdEvaluator(scorer=scorer, threshold=0.8)
    exit_condition = BayesianExitCondition(
        priors=priors,
        task_type="decompose",
        model_id="gpt-4o",
        quality_threshold=0.8,
    )

    config = LoopConfig(max_iterations=10, quality_threshold=0.8)
    loop = LoopedLLM(provider=provider, config=config)
    loop.add_exit_condition(exit_condition)

    print("\n=== Running loop with Bayesian exit ===")
    result = loop.refine("Decompose this complex task.", evaluator)

    print(f"Iterations: {result.metrics.total_iterations}")
    print(f"Exit reason: {result.metrics.exit_reason.condition}")
    print(f"Exit message: {result.metrics.exit_reason.message}")
    print(f"Best score: {result.metrics.best_score:.2f}")
    print(f"Output: {result.output}")


if __name__ == "__main__":
    main()
