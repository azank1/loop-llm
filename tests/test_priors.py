"""Tests for Bayesian priors and adaptive meta-learning."""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

from loopllm import AdaptivePriors, CallObservation
from loopllm.priors import BetaPrior, NormalPrior


def test_beta_prior_uniform_mean_and_confidence() -> None:
    """Uniform BetaPrior(1,1) has mean=0.5 and confidence=0.0."""
    prior = BetaPrior()
    assert abs(prior.mean - 0.5) < 1e-6
    assert prior.confidence == 0.0


def test_beta_prior_update_moves_mean() -> None:
    """Updating a BetaPrior moves mean toward observed proportion."""
    prior = BetaPrior()
    for _ in range(80):
        prior.update(True)
    for _ in range(20):
        prior.update(False)
    # Expected close to 81/(81+21) ≈ 0.794
    assert 0.75 < prior.mean < 0.85


def test_beta_prior_prob_above_strongly_skewed() -> None:
    """Beta(90,10) should give prob_above(0.5) > 0.99."""
    prior = BetaPrior(alpha=90, beta=10)
    p = prior.prob_above(0.5)
    assert p > 0.99


def test_normal_prior_welford_matches_exact() -> None:
    """Welford's algorithm matches exact mean and variance for 1000 samples."""
    random.seed(42)
    samples = [random.gauss(5.0, 2.0) for _ in range(1000)]

    prior = NormalPrior(mean=0.0, variance=1.0)
    for s in samples:
        prior.update(s)

    exact_mean = sum(samples) / len(samples)
    exact_var = sum((x - exact_mean) ** 2 for x in samples) / (len(samples) - 1)

    assert abs(prior.mean - exact_mean) < 1e-6
    assert abs(prior.variance - exact_var) < 1e-4


def test_predict_optimal_depth_defaults_no_data(fresh_priors: AdaptivePriors) -> None:
    """Returns task-type defaults with no data."""
    assert fresh_priors.predict_optimal_depth("decompose", "gpt-4o") == 4
    assert fresh_priors.predict_optimal_depth("resolve", "gpt-4o") == 2
    assert fresh_priors.predict_optimal_depth("assemble", "gpt-4o") == 3
    assert fresh_priors.predict_optimal_depth("validate", "gpt-4o") == 1
    assert fresh_priors.predict_optimal_depth("custom", "gpt-4o") == 3


def test_predict_optimal_depth_trained(trained_priors: AdaptivePriors) -> None:
    """After 20+ observations, predict_optimal_depth is in [2, 5]."""
    depth = trained_priors.predict_optimal_depth("decompose", "gpt-4o")
    assert 2 <= depth <= 5


def test_should_continue_low_score(trained_priors: AdaptivePriors) -> None:
    """should_continue returns True at low score with sufficient training data."""
    result = trained_priors.should_continue(
        "decompose", "gpt-4o",
        current_iteration=1,
        current_score=0.3,
        scores_so_far=[0.3],
        quality_threshold=0.8,
    )
    assert result is True


def test_should_continue_high_score_plateau(trained_priors: AdaptivePriors) -> None:
    """should_continue returns False at high score after plateau training data."""
    result = trained_priors.should_continue(
        "resolve", "gpt-4o",
        current_iteration=3,
        current_score=0.85,
        scores_so_far=[0.5, 0.85, 0.85],
        quality_threshold=0.8,
    )
    assert result is False


def test_suggest_config_structure(trained_priors: AdaptivePriors) -> None:
    """suggest_config returns dict with correct keys and valid ranges."""
    config = trained_priors.suggest_config("decompose", "gpt-4o")
    assert "max_iterations" in config
    assert "quality_threshold" in config
    assert "metadata" in config
    assert 1 <= config["max_iterations"] <= 10
    assert 0.0 < config["quality_threshold"] <= 0.95
    assert config["metadata"]["source"] == "adaptive_priors"


def test_json_persistence_roundtrip() -> None:
    """JSON persistence round-trip preserves total_calls and convergence rate."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "priors.json"

        priors = AdaptivePriors(store_path=path)
        for _ in range(10):
            priors.observe(
                CallObservation(
                    task_type="test_task",
                    model_id="test_model",
                    scores=[0.4, 0.7, 0.85],
                    latencies_ms=[100, 100, 100],
                    converged=True,
                    total_iterations=3,
                    quality_threshold=0.8,
                )
            )
        priors._save()

        # Load fresh
        loaded = AdaptivePriors(store_path=path)
        key = AdaptivePriors._key("test_task", "test_model")
        original_prior = priors._priors[key]
        loaded_prior = loaded._priors[key]

        assert loaded_prior.total_calls == original_prior.total_calls
        assert abs(
            loaded_prior.overall_converge_rate.mean - original_prior.overall_converge_rate.mean
        ) < 1e-6
