"""Bayesian meta-learning layer for adaptive loop depth prediction."""
from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Primitive prior distributions
# ---------------------------------------------------------------------------


@dataclass
class BetaPrior:
    """Beta-distributed prior for binary outcomes.

    Attributes:
        alpha: Pseudo-count of successes.
        beta: Pseudo-count of failures.
    """

    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        """Expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def confidence(self) -> float:
        """Confidence level based on number of observations.

        Returns 0.0 if no observations have been recorded.  Otherwise applies
        a sigmoid-like mapping of ``(alpha + beta - 2)`` over ``n + 10``.
        """
        n = self.alpha + self.beta - 2  # prior pseudo-counts subtracted
        if n <= 0:
            return 0.0
        return 1 / (1 + math.exp(-n / (n + 10)))

    def update(self, success: bool) -> None:
        """Record an observation.

        Args:
            success: Whether the outcome was a success.
        """
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def prob_above(self, threshold: float) -> float:
        """Approximate P(X > threshold) using a normal approximation to the Beta CDF.

        Args:
            threshold: The threshold to compare against.

        Returns:
            Approximate probability that a sample exceeds *threshold*.
        """
        mu = self.mean
        std = math.sqrt(max(self.variance, 1e-10))
        z = (threshold - mu) / std
        # P(X > threshold) ≈ 1 - Φ(z) using erf
        return 0.5 * (1 - math.erf(z / math.sqrt(2)))


@dataclass
class NormalPrior:
    """Normal-distributed prior with optional exponential decay, updated via Welford's algorithm.

    Attributes:
        mean: Current mean estimate.
        variance: Current variance estimate.
        n_observations: Number of observations incorporated.
        _m2: Running sum of squared differences (Welford internal state).
        decay: Exponential decay factor; 1.0 disables decay.
    """

    mean: float = 0.0
    variance: float = 1.0
    n_observations: int = 0
    _m2: float = 0.0
    decay: float = 1.0

    @property
    def confidence(self) -> float:
        """Confidence level: ``n / (n + 10)``, adjusted for decay."""
        n = self.n_observations
        if n == 0:
            return 0.0
        base = n / (n + 10)
        if self.decay < 1.0:
            return base * self.decay
        return base

    @property
    def std(self) -> float:
        """Standard deviation (floored to avoid numerical issues)."""
        return math.sqrt(max(self.variance, 1e-10))

    def update(self, value: float) -> None:
        """Incorporate a new observation.

        Uses Welford's online algorithm when ``decay == 1.0``, otherwise an
        exponential moving average.

        Args:
            value: The observed value.
        """
        if self.decay < 1.0:
            # Exponential moving average
            self.n_observations += 1
            alpha = 1 - self.decay
            self.mean = self.decay * self.mean + alpha * value
            diff = value - self.mean
            self.variance = self.decay * self.variance + alpha * diff * diff
        else:
            # Welford's online algorithm
            self.n_observations += 1
            delta = value - self.mean
            self.mean += delta / self.n_observations
            delta2 = value - self.mean
            self._m2 += delta * delta2
            if self.n_observations >= 2:
                self.variance = self._m2 / (self.n_observations - 1)


# ---------------------------------------------------------------------------
# Per-iteration and per-task profiles
# ---------------------------------------------------------------------------


@dataclass
class IterationProfile:
    """Statistical profile for a specific iteration depth.

    Attributes:
        score: Expected quality score at this iteration.
        score_delta: Expected improvement from the previous iteration.
        converge_prob: Probability of having converged by this iteration.
        latency_ms: Expected wall-clock time for this iteration.
    """

    score: NormalPrior = field(default_factory=lambda: NormalPrior(mean=0.3, variance=0.1))
    score_delta: NormalPrior = field(default_factory=lambda: NormalPrior(mean=0.1, variance=0.05))
    converge_prob: BetaPrior = field(default_factory=BetaPrior)
    latency_ms: NormalPrior = field(default_factory=lambda: NormalPrior(mean=2000, variance=500_000))


@dataclass
class TaskModelPrior:
    """Collected beliefs about a (task_type, model_id) pair.

    Attributes:
        task_type: Identifier for the class of task.
        model_id: Identifier for the LLM model.
        created_at: ISO timestamp of first observation.
        updated_at: ISO timestamp of most recent observation.
        total_calls: Total number of refinement runs observed.
        iterations: Per-iteration statistical profiles.
        optimal_depth: Estimated optimal number of iterations.
        overall_converge_rate: Overall probability of convergence.
        first_call_quality: Expected score of the first LLM call.
    """

    task_type: str = ""
    model_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    total_calls: int = 0
    iterations: dict[int, IterationProfile] = field(default_factory=dict)
    optimal_depth: NormalPrior = field(
        default_factory=lambda: NormalPrior(mean=3.0, variance=2.0)
    )
    overall_converge_rate: BetaPrior = field(default_factory=BetaPrior)
    first_call_quality: NormalPrior = field(
        default_factory=lambda: NormalPrior(mean=0.4, variance=0.1)
    )

    def get_iteration(self, k: int) -> IterationProfile:
        """Return the :class:`IterationProfile` for iteration *k*, creating if missing.

        New profiles use diminishing-returns priors:
        - score mean = ``min(0.3 + 0.15*k, 0.9)``
        - score_delta mean = ``max(0.15 - 0.03*k, 0.01)``

        Args:
            k: Zero-based iteration index.

        Returns:
            The iteration profile for depth *k*.
        """
        if k not in self.iterations:
            self.iterations[k] = IterationProfile(
                score=NormalPrior(mean=min(0.3 + 0.15 * k, 0.9), variance=0.1),
                score_delta=NormalPrior(mean=max(0.15 - 0.03 * k, 0.01), variance=0.05),
                converge_prob=BetaPrior(),
                latency_ms=NormalPrior(mean=2000, variance=500_000),
            )
        return self.iterations[k]


@dataclass
class CallObservation:
    """Observation recorded after a refinement run.

    Attributes:
        task_type: Identifier for the task class.
        model_id: Identifier for the LLM model.
        scores: Per-iteration quality scores.
        latencies_ms: Per-iteration latencies in milliseconds.
        converged: Whether the loop converged to an acceptable result.
        total_iterations: Number of iterations executed.
        max_iterations: Maximum iterations configured.
        quality_threshold: Quality threshold configured.
        prompt_tokens: Total prompt tokens consumed.
        completion_tokens: Total completion tokens consumed.
        metadata: Arbitrary extra data.
    """

    task_type: str = ""
    model_id: str = ""
    scores: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    converged: bool = False
    total_iterations: int = 0
    max_iterations: int = 5
    quality_threshold: float = 0.8
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main adaptive priors manager
# ---------------------------------------------------------------------------


class AdaptivePriors:
    """Bayesian meta-learning manager that learns optimal loop depth from observations.

    Args:
        store_path: Optional filesystem path for JSON persistence.
    """

    def __init__(self, store_path: Path | None = None) -> None:
        self.store_path = store_path
        self._priors: dict[str, TaskModelPrior] = {}
        if store_path and store_path.exists():
            self._load()

    # -- key helpers ---------------------------------------------------------

    @staticmethod
    def _key(task_type: str, model_id: str) -> str:
        return f"{task_type}::{model_id}"

    def _get_or_create(self, task_type: str, model_id: str) -> TaskModelPrior:
        key = self._key(task_type, model_id)
        if key not in self._priors:
            now = datetime.now(timezone.utc).isoformat()
            self._priors[key] = TaskModelPrior(
                task_type=task_type,
                model_id=model_id,
                created_at=now,
                updated_at=now,
            )
        return self._priors[key]

    # -- public API ----------------------------------------------------------

    def predict_optimal_depth(
        self, task_type: str, model_id: str, cost_weight: float = 0.5
    ) -> int:
        """Predict the optimal number of refinement iterations.

        Returns task-type defaults when fewer than 5 observations exist.
        Otherwise uses expected improvement analysis.

        Args:
            task_type: Identifier for the task class.
            model_id: Identifier for the LLM model.
            cost_weight: Weight given to cost vs. quality (0 = quality only).

        Returns:
            Recommended number of iterations (clamped to [1, 10]).
        """
        prior = self._get_or_create(task_type, model_id)

        if prior.total_calls < 5:
            defaults: dict[str, int] = {
                "decompose": 4,
                "resolve": 2,
                "assemble": 3,
                "validate": 1,
            }
            return defaults.get(task_type, 3)

        best_k = 1
        for k in range(1, 11):
            profile = prior.get_iteration(k)
            expected_delta = profile.score_delta.mean
            confidence = profile.score_delta.confidence
            benefit = expected_delta + (1 - confidence) * 0.1
            cost = 0.02 + cost_weight * 0.08
            if benefit > cost and expected_delta > 0.01:
                best_k = k
            else:
                break

        return max(1, min(best_k, 10))

    def should_continue(
        self,
        task_type: str,
        model_id: str,
        current_iteration: int,
        current_score: float,
        scores_so_far: list[float],
        quality_threshold: float = 0.8,
    ) -> bool:
        """Decide whether the loop should continue beyond the current iteration.

        Falls back to a simple threshold check during cold start (< 3 observations).

        Args:
            task_type: Identifier for the task class.
            model_id: Identifier for the LLM model.
            current_iteration: Current iteration number (1-based).
            current_score: Score of the current iteration.
            scores_so_far: All scores observed so far in this run.
            quality_threshold: Target quality level.

        Returns:
            True if continuing is recommended.
        """
        prior = self._get_or_create(task_type, model_id)

        # Cold start
        if prior.total_calls < 3:
            return current_score < quality_threshold

        # Compute cumulative expected improvement over remaining iterations
        gap = quality_threshold - current_score
        if gap <= 0:
            return False

        # Sum expected deltas over remaining plausible iterations
        cumulative_delta = 0.0
        cumulative_var = 0.0
        max_remaining = 10 - current_iteration
        for k in range(current_iteration, current_iteration + max(max_remaining, 1)):
            p = prior.get_iteration(k)
            cumulative_delta += p.score_delta.mean
            cumulative_var += p.score_delta.variance
            if cumulative_delta >= gap:
                break

        std = math.sqrt(max(cumulative_var, 1e-10))
        z = (gap - cumulative_delta) / std
        p_bridge_gap = 0.5 * (1 - math.erf(z / math.sqrt(2)))

        return p_bridge_gap > 0.3

    def expected_improvement(
        self, task_type: str, model_id: str, at_iteration: int
    ) -> tuple[float, float]:
        """Return the expected score improvement and its uncertainty at a given iteration.

        Args:
            task_type: Identifier for the task class.
            model_id: Identifier for the LLM model.
            at_iteration: Iteration depth to query.

        Returns:
            ``(mean_delta, std_delta)`` tuple.
        """
        prior = self._get_or_create(task_type, model_id)
        profile = prior.get_iteration(at_iteration)
        return profile.score_delta.mean, profile.score_delta.std

    def suggest_config(
        self, task_type: str, model_id: str, cost_weight: float = 0.5
    ) -> dict[str, Any]:
        """Suggest a :class:`LoopConfig`-compatible dict based on learned beliefs.

        Args:
            task_type: Identifier for the task class.
            model_id: Identifier for the LLM model.
            cost_weight: Weight given to cost vs. quality.

        Returns:
            Dict with ``max_iterations``, ``quality_threshold``, and ``metadata``.
        """
        prior = self._get_or_create(task_type, model_id)
        depth = self.predict_optimal_depth(task_type, model_id, cost_weight)

        if prior.total_calls >= 5:
            profile = prior.get_iteration(depth)
            threshold = min(profile.score.mean + 0.5 * profile.score.std, 0.95)
        else:
            threshold = 0.8

        return {
            "max_iterations": depth,
            "quality_threshold": round(threshold, 3),
            "metadata": {
                "source": "adaptive_priors",
                "confidence": round(prior.optimal_depth.confidence, 3),
                "total_observations": prior.total_calls,
            },
        }

    def observe(self, observation: CallObservation) -> None:
        """Record a completed refinement run and update all priors.

        Args:
            observation: The observation to incorporate.
        """
        prior = self._get_or_create(observation.task_type, observation.model_id)
        prior.total_calls += 1
        prior.updated_at = datetime.now(timezone.utc).isoformat()

        # Overall convergence
        prior.overall_converge_rate.update(observation.converged)

        # First-call quality
        if observation.scores:
            prior.first_call_quality.update(observation.scores[0])

        # Per-iteration updates
        for k, score in enumerate(observation.scores):
            profile = prior.get_iteration(k)
            profile.score.update(score)
            if k > 0:
                delta = score - observation.scores[k - 1]
                profile.score_delta.update(delta)
            if k < len(observation.latencies_ms):
                profile.latency_ms.update(observation.latencies_ms[k])
            converged_at_k = score >= observation.quality_threshold
            profile.converge_prob.update(converged_at_k)

        # Optimal depth: first iteration where score >= threshold
        opt_depth = observation.total_iterations
        for k, score in enumerate(observation.scores):
            if score >= observation.quality_threshold:
                opt_depth = k + 1
                break
        prior.optimal_depth.update(float(opt_depth))

        # Auto-save
        if self.store_path and prior.total_calls % 10 == 0:
            self._save()

    def report(self, task_type: str, model_id: str) -> dict[str, Any]:
        """Generate a human-readable summary of beliefs for a (task, model) pair.

        Args:
            task_type: Identifier for the task class.
            model_id: Identifier for the LLM model.

        Returns:
            Dict with summary statistics.
        """
        prior = self._get_or_create(task_type, model_id)
        iteration_summaries: dict[str, Any] = {}
        for k in sorted(prior.iterations.keys()):
            p = prior.iterations[k]
            iteration_summaries[f"iter_{k}"] = {
                "expected_score": round(p.score.mean, 3),
                "expected_delta": round(p.score_delta.mean, 3),
                "converge_prob": round(p.converge_prob.mean, 3),
                "latency_ms": round(p.latency_ms.mean, 1),
            }
        return {
            "task_type": task_type,
            "model_id": model_id,
            "total_calls": prior.total_calls,
            "optimal_depth": round(prior.optimal_depth.mean, 2),
            "converge_rate": round(prior.overall_converge_rate.mean, 3),
            "first_call_quality": round(prior.first_call_quality.mean, 3),
            "confidence": round(prior.optimal_depth.confidence, 3),
            "iterations": iteration_summaries,
        }

    def report_all(self) -> list[dict[str, Any]]:
        """Generate summaries for all tracked (task, model) combinations.

        Returns:
            List of report dicts.
        """
        results: list[dict[str, Any]] = []
        for key in sorted(self._priors.keys()):
            p = self._priors[key]
            results.append(self.report(p.task_type, p.model_id))
        return results

    # -- persistence ---------------------------------------------------------

    def _save(self) -> None:
        """Atomically persist all priors to *store_path* as JSON."""
        if not self.store_path:
            return

        data: dict[str, Any] = {}
        for key, prior in self._priors.items():
            iterations_data: dict[str, Any] = {}
            for k, profile in prior.iterations.items():
                iterations_data[str(k)] = {
                    "score": self._serialize_normal(profile.score),
                    "score_delta": self._serialize_normal(profile.score_delta),
                    "converge_prob": self._serialize_beta(profile.converge_prob),
                    "latency_ms": self._serialize_normal(profile.latency_ms),
                }
            data[key] = {
                "task_type": prior.task_type,
                "model_id": prior.model_id,
                "created_at": prior.created_at,
                "updated_at": prior.updated_at,
                "total_calls": prior.total_calls,
                "iterations": iterations_data,
                "optimal_depth": self._serialize_normal(prior.optimal_depth),
                "overall_converge_rate": self._serialize_beta(prior.overall_converge_rate),
                "first_call_quality": self._serialize_normal(prior.first_call_quality),
            }

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self.store_path.parent, suffix=".tmp"
        )
        try:
            with open(fd, "w") as f:
                json.dump(data, f, indent=2)
            Path(tmp_path).replace(self.store_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def _load(self) -> None:
        """Load priors from *store_path*."""
        if not self.store_path or not self.store_path.exists():
            return

        with open(self.store_path) as f:
            data = json.load(f)

        for key, pdata in data.items():
            iterations: dict[int, IterationProfile] = {}
            for k_str, idata in pdata.get("iterations", {}).items():
                iterations[int(k_str)] = IterationProfile(
                    score=self._deserialize_normal(idata["score"]),
                    score_delta=self._deserialize_normal(idata["score_delta"]),
                    converge_prob=self._deserialize_beta(idata["converge_prob"]),
                    latency_ms=self._deserialize_normal(idata["latency_ms"]),
                )
            self._priors[key] = TaskModelPrior(
                task_type=pdata["task_type"],
                model_id=pdata["model_id"],
                created_at=pdata["created_at"],
                updated_at=pdata["updated_at"],
                total_calls=pdata["total_calls"],
                iterations=iterations,
                optimal_depth=self._deserialize_normal(pdata["optimal_depth"]),
                overall_converge_rate=self._deserialize_beta(pdata["overall_converge_rate"]),
                first_call_quality=self._deserialize_normal(pdata["first_call_quality"]),
            )

    @staticmethod
    def _serialize_normal(p: NormalPrior) -> dict[str, Any]:
        return {
            "mean": p.mean,
            "variance": p.variance,
            "n_observations": p.n_observations,
            "_m2": p._m2,
            "decay": p.decay,
        }

    @staticmethod
    def _deserialize_normal(d: dict[str, Any]) -> NormalPrior:
        return NormalPrior(**d)

    @staticmethod
    def _serialize_beta(p: BetaPrior) -> dict[str, Any]:
        return {"alpha": p.alpha, "beta": p.beta}

    @staticmethod
    def _deserialize_beta(d: dict[str, Any]) -> BetaPrior:
        return BetaPrior(**d)
