"""Core iterative refinement engine."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from loopllm.provider import LLMProvider

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating an LLM output.

    Attributes:
        score: Quality score in [0.0, 1.0].
        passed: Whether the output meets the quality bar.
        deficiencies: List of issues found in the output.
        sub_scores: Named component scores.
        feedback: Human-readable summary; auto-generated from deficiencies if not provided.
    """

    score: float
    passed: bool
    deficiencies: list[str] = field(default_factory=list)
    sub_scores: dict[str, float] = field(default_factory=dict)
    feedback: str = ""

    def __post_init__(self) -> None:
        if not self.feedback and self.deficiencies:
            self.feedback = "Issues found: " + "; ".join(self.deficiencies)


@dataclass
class ExitReason:
    """Describes why the refinement loop terminated.

    Attributes:
        condition: Name of the exit condition that triggered.
        message: Human-readable explanation.
    """

    condition: str
    message: str


@dataclass
class IterationRecord:
    """Record of a single loop iteration.

    Attributes:
        iteration: Zero-based iteration index.
        prompt: The prompt sent to the LLM.
        output: The LLM's response content.
        score: Evaluation score for this iteration.
        passed: Whether the evaluation passed.
        deficiencies: Issues identified in this iteration.
        latency_ms: Time taken for this iteration in milliseconds.
    """

    iteration: int
    prompt: str
    output: str
    score: float
    passed: bool
    deficiencies: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class LoopMetrics:
    """Aggregate metrics for the entire refinement run.

    Attributes:
        total_iterations: Number of iterations executed.
        best_score: Highest score achieved across all iterations.
        final_score: Score of the last iteration.
        converged: Whether the loop converged to a passing result.
        exit_reason: Why the loop terminated.
        total_latency_ms: Total time spent in milliseconds.
        score_trajectory: Ordered list of scores per iteration.
    """

    total_iterations: int
    best_score: float
    final_score: float
    converged: bool
    exit_reason: ExitReason
    total_latency_ms: float
    score_trajectory: list[float] = field(default_factory=list)


@dataclass
class RefinementResult:
    """Final result of a refinement run.

    Attributes:
        output: Best (or final) output across all iterations.
        metrics: Aggregate loop metrics.
        iterations: Full history of iteration records.
    """

    output: str
    metrics: LoopMetrics
    iterations: list[IterationRecord] = field(default_factory=list)


@dataclass
class LoopConfig:
    """Configuration for the refinement loop.

    Attributes:
        max_iterations: Maximum number of refinement iterations.
        quality_threshold: Minimum score to consider output acceptable.
        min_iterations: Minimum number of iterations before early exit.
        convergence_delta: Exit if improvement < this for 2 consecutive iters.
        timeout_ms: Maximum total wall-clock time in milliseconds.
        best_of: If True, return the best output; otherwise return the last.
    """

    max_iterations: int = 5
    quality_threshold: float = 0.8
    min_iterations: int = 1
    convergence_delta: float = 0.01
    timeout_ms: float = 30_000
    best_of: bool = True


class CompositeEvaluator:
    """Combines multiple evaluators via weighted average.

    Args:
        evaluators: List of evaluator objects, each with an ``evaluate`` method.
        weights: Optional per-evaluator weights. Defaults to equal weights.
    """

    def __init__(self, evaluators: list[Any], weights: list[float] | None = None) -> None:
        self.evaluators = evaluators
        if weights is None:
            self.weights = [1.0 / len(evaluators)] * len(evaluators)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def evaluate(self, output: str, context: dict[str, Any] | None = None) -> EvaluationResult:
        """Evaluate *output* using all child evaluators and return weighted result.

        Args:
            output: The text to evaluate.
            context: Optional context dict passed to each evaluator.

        Returns:
            A single :class:`EvaluationResult` with merged scores and deficiencies.
        """
        ctx = context or {}
        all_deficiencies: list[str] = []
        all_sub_scores: dict[str, float] = {}
        weighted_score = 0.0

        for evaluator, weight in zip(self.evaluators, self.weights):
            result = evaluator.evaluate(output, ctx)
            weighted_score += result.score * weight
            all_deficiencies.extend(result.deficiencies)
            all_sub_scores.update(result.sub_scores)

        passed = weighted_score >= 0.5 and not all_deficiencies
        return EvaluationResult(
            score=weighted_score,
            passed=passed,
            deficiencies=all_deficiencies,
            sub_scores=all_sub_scores,
        )


class LoopedLLM:
    """Iterative refinement engine that loops an LLM call with evaluation.

    Args:
        provider: The LLM provider to use for completions.
        config: Loop configuration. Uses defaults if not provided.
    """

    def __init__(self, provider: LLMProvider, config: LoopConfig | None = None) -> None:
        self.provider = provider
        self.config = config or LoopConfig()
        self._exit_conditions: list[Any] = []

    def add_exit_condition(self, condition: Any) -> None:
        """Register an additional exit condition (e.g. :class:`BayesianExitCondition`).

        Args:
            condition: An object with a ``should_exit`` method.
        """
        self._exit_conditions.append(condition)

    def refine(
        self,
        initial_prompt: str,
        evaluator: Any,
        context: dict[str, Any] | None = None,
        model: str = "gpt-4o-mini",
    ) -> RefinementResult:
        """Run the iterative refinement loop.

        Args:
            initial_prompt: The initial prompt to send to the LLM.
            evaluator: An evaluator with an ``evaluate(output, context)`` method.
            context: Optional context dict passed to the evaluator.
            model: Model identifier for the LLM provider.

        Returns:
            :class:`RefinementResult` containing the best/final output and metrics.
        """
        ctx = context or {}
        iterations: list[IterationRecord] = []
        scores: list[float] = []
        prompt = initial_prompt
        best_output = ""
        best_score = -1.0
        exit_reason: ExitReason | None = None
        loop_start = time.perf_counter()

        for i in range(self.config.max_iterations):
            iter_start = time.perf_counter()

            # 1. Call provider
            response = self.provider.complete(prompt, model)
            output = response.content

            # 2. Evaluate
            result = evaluator.evaluate(output, ctx)
            iter_latency = (time.perf_counter() - iter_start) * 1000.0

            # Track
            scores.append(result.score)
            record = IterationRecord(
                iteration=i,
                prompt=prompt,
                output=output,
                score=result.score,
                passed=result.passed,
                deficiencies=list(result.deficiencies),
                latency_ms=iter_latency,
            )
            iterations.append(record)

            if result.score > best_score:
                best_score = result.score
                best_output = output

            logger.debug(
                "loop_iteration",
                iteration=i,
                score=result.score,
                passed=result.passed,
                deficiencies=result.deficiencies,
            )

            # 3. Check exit conditions (only after min_iterations)
            if i + 1 >= self.config.min_iterations:
                # Quality threshold
                if result.score >= self.config.quality_threshold:
                    exit_reason = ExitReason(
                        "quality_threshold",
                        f"Score {result.score:.2f} >= threshold {self.config.quality_threshold:.2f}"
                        f" at iteration {i + 1}",
                    )
                    break

                # Convergence delta
                if len(scores) >= 3:
                    delta1 = abs(scores[-1] - scores[-2])
                    delta2 = abs(scores[-2] - scores[-3])
                    if delta1 < self.config.convergence_delta and delta2 < self.config.convergence_delta:
                        exit_reason = ExitReason(
                            "convergence",
                            f"Score plateaued (last deltas: {delta1:.4f}, {delta2:.4f}"
                            f" < {self.config.convergence_delta:.4f})",
                        )
                        break

                # Bayesian / custom exit conditions
                for cond in self._exit_conditions:
                    reason = cond.should_exit(i + 1, result.score, list(scores))
                    if reason is not None:
                        exit_reason = reason
                        break
                if exit_reason is not None:
                    break

                # Timeout
                elapsed = (time.perf_counter() - loop_start) * 1000.0
                if elapsed >= self.config.timeout_ms:
                    exit_reason = ExitReason(
                        "timeout",
                        f"Elapsed {elapsed:.0f}ms >= timeout {self.config.timeout_ms:.0f}ms",
                    )
                    break

            # 4. Build feedback prompt for next iteration
            if result.deficiencies:
                deficiency_lines = "\n".join(f"- {d}" for d in result.deficiencies)
                prompt = (
                    f"{initial_prompt}\n\n"
                    f"Previous attempt scored {result.score:.2f}/1.0. Issues to fix:\n"
                    f"{deficiency_lines}\n\n"
                    f"Please address all issues in your next response."
                )
            else:
                prompt = initial_prompt

        # If we exhausted iterations without another exit reason
        if exit_reason is None:
            exit_reason = ExitReason(
                "max_iterations",
                f"Reached maximum of {self.config.max_iterations} iterations",
            )

        total_latency = (time.perf_counter() - loop_start) * 1000.0
        final_output = best_output if self.config.best_of else iterations[-1].output

        metrics = LoopMetrics(
            total_iterations=len(iterations),
            best_score=best_score,
            final_score=scores[-1] if scores else 0.0,
            converged=best_score >= self.config.quality_threshold,
            exit_reason=exit_reason,
            total_latency_ms=total_latency,
            score_trajectory=scores,
        )

        logger.info(
            "refinement_complete",
            total_iterations=metrics.total_iterations,
            best_score=metrics.best_score,
            exit_reason=exit_reason.condition,
        )

        return RefinementResult(
            output=final_output,
            metrics=metrics,
            iterations=iterations,
        )
