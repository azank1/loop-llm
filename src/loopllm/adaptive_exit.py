"""Bayesian adaptive exit condition for the refinement loop."""
from __future__ import annotations

from dataclasses import dataclass

from loopllm.engine import ExitReason
from loopllm.priors import AdaptivePriors


@dataclass
class BayesianExitCondition:
    """Exit condition that uses learned priors to decide when to stop looping.

    Integrates with :class:`AdaptivePriors` to make statistically-informed
    stopping decisions based on historical observations.

    Attributes:
        priors: The adaptive priors manager holding learned beliefs.
        task_type: Identifier for the task class.
        model_id: Identifier for the LLM model.
        quality_threshold: Target quality level.
        continue_probability_threshold: Minimum probability of improvement to continue.
        min_iterations: Minimum iterations before this condition can fire.
    """

    priors: AdaptivePriors
    task_type: str = "unknown"
    model_id: str = "unknown"
    quality_threshold: float = 0.8
    continue_probability_threshold: float = 0.3
    min_iterations: int = 1

    def should_exit(
        self,
        iteration: int,
        current_score: float,
        scores_so_far: list[float],
    ) -> ExitReason | None:
        """Determine whether the loop should exit based on Bayesian analysis.

        Args:
            iteration: Current iteration number (1-based).
            current_score: Score from the most recent evaluation.
            scores_so_far: All scores observed so far in this run.

        Returns:
            An :class:`ExitReason` if the loop should stop, or ``None`` to continue.
        """
        if iteration < self.min_iterations:
            return None

        if not scores_so_far:
            return None

        should_go = self.priors.should_continue(
            self.task_type,
            self.model_id,
            iteration,
            current_score,
            scores_so_far,
            quality_threshold=self.quality_threshold,
        )

        if not should_go:
            expected_delta, uncertainty = self.priors.expected_improvement(
                self.task_type, self.model_id, iteration
            )
            return ExitReason(
                condition="adaptive_bayesian",
                message=(
                    f"Bayesian exit at iteration {iteration}: "
                    f"score={current_score:.3f}, "
                    f"E[delta]={expected_delta:.3f}±{uncertainty:.3f}, "
                    f"threshold={self.quality_threshold:.2f}"
                ),
            )

        return None
