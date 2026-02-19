"""Tests for the core engine module."""
from __future__ import annotations

from loopllm import LoopConfig, LoopedLLM
from loopllm.engine import CompositeEvaluator, EvaluationResult
from loopllm.providers.mock import MockLLMProvider


class _PassEvaluator:
    """Always passes with a high score."""

    def evaluate(self, output: str, context: dict | None = None) -> EvaluationResult:
        return EvaluationResult(score=0.95, passed=True)


class _FailEvaluator:
    """Always fails with a low but slightly varying score and a deficiency."""

    def __init__(self) -> None:
        self._call = 0

    def evaluate(self, output: str, context: dict | None = None) -> EvaluationResult:
        self._call += 1
        # Vary score enough to avoid convergence-delta exit
        score = 0.3 + self._call * 0.05
        return EvaluationResult(
            score=score, passed=False, deficiencies=["Not good enough"]
        )


class _ImprovingEvaluator:
    """Returns increasing scores, passing once threshold is reached."""

    def __init__(self, scores: list[float], threshold: float = 0.8) -> None:
        self._scores = scores
        self._idx = 0
        self._threshold = threshold

    def evaluate(self, output: str, context: dict | None = None) -> EvaluationResult:
        score = self._scores[min(self._idx, len(self._scores) - 1)]
        self._idx += 1
        passed = score >= self._threshold
        deficiencies = [] if passed else [f"Score {score:.2f} below threshold"]
        return EvaluationResult(score=score, passed=passed, deficiencies=deficiencies)


class _PlateauEvaluator:
    """Returns scores that plateau quickly."""

    def __init__(self) -> None:
        self._scores = [0.5, 0.51, 0.511, 0.512, 0.512]
        self._idx = 0

    def evaluate(self, output: str, context: dict | None = None) -> EvaluationResult:
        score = self._scores[min(self._idx, len(self._scores) - 1)]
        self._idx += 1
        return EvaluationResult(
            score=score, passed=False, deficiencies=["Needs improvement"]
        )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_single_iteration_when_first_output_passes(mock_provider: MockLLMProvider) -> None:
    """Exit immediately if the first output passes the quality threshold."""
    config = LoopConfig(max_iterations=5, quality_threshold=0.5)
    loop = LoopedLLM(provider=mock_provider, config=config)
    result = loop.refine("test", _PassEvaluator())
    assert result.metrics.total_iterations == 1
    assert result.metrics.exit_reason.condition == "quality_threshold"


def test_multiple_iterations_until_quality_met(mock_provider: MockLLMProvider) -> None:
    """Loop until the evaluator returns a passing score."""
    evaluator = _ImprovingEvaluator([0.3, 0.5, 0.7, 0.85])
    config = LoopConfig(max_iterations=10, quality_threshold=0.8)
    loop = LoopedLLM(provider=mock_provider, config=config)
    result = loop.refine("test", evaluator)
    assert result.metrics.total_iterations == 4
    assert result.metrics.exit_reason.condition == "quality_threshold"


def test_exit_on_max_iterations() -> None:
    """Exit when max iterations reached without converging."""
    provider = MockLLMProvider()
    config = LoopConfig(max_iterations=3, quality_threshold=0.99)
    loop = LoopedLLM(provider=provider, config=config)
    result = loop.refine("test", _FailEvaluator())
    assert result.metrics.total_iterations == 3
    assert result.metrics.exit_reason.condition == "max_iterations"
    assert not result.metrics.converged


def test_exit_on_convergence_delta() -> None:
    """Exit when score plateaus (convergence delta)."""
    provider = MockLLMProvider()
    config = LoopConfig(max_iterations=10, quality_threshold=0.99, convergence_delta=0.02)
    loop = LoopedLLM(provider=provider, config=config)
    result = loop.refine("test", _PlateauEvaluator())
    assert result.metrics.exit_reason.condition == "convergence"


def test_best_of_true_returns_highest_scoring_output() -> None:
    """best_of=True returns the output with the highest score, not the last."""
    provider = MockLLMProvider(responses=["bad", "best", "ok"])
    evaluator = _ImprovingEvaluator([0.3, 0.9, 0.5])
    config = LoopConfig(max_iterations=3, quality_threshold=0.95, best_of=True)
    loop = LoopedLLM(provider=provider, config=config)
    result = loop.refine("test", evaluator)
    assert result.output == "best"


def test_best_of_false_returns_last_output() -> None:
    """best_of=False returns the last output regardless of score."""
    provider = MockLLMProvider(responses=["bad", "best", "ok"])
    evaluator = _ImprovingEvaluator([0.3, 0.9, 0.5])
    config = LoopConfig(max_iterations=3, quality_threshold=0.95, best_of=False)
    loop = LoopedLLM(provider=provider, config=config)
    result = loop.refine("test", evaluator)
    assert result.output == "ok"


def test_feedback_prompt_contains_deficiencies() -> None:
    """The feedback prompt includes deficiency messages."""
    provider = MockLLMProvider()
    evaluator = _ImprovingEvaluator([0.3, 0.9])
    config = LoopConfig(max_iterations=5, quality_threshold=0.8)
    loop = LoopedLLM(provider=provider, config=config)
    result = loop.refine("Generate something", evaluator)
    # Second iteration should have a feedback prompt with deficiencies
    assert result.metrics.total_iterations == 2
    second_prompt = result.iterations[1].prompt
    assert "Issues to fix" in second_prompt
    assert "Score 0.30 below threshold" in second_prompt


def test_loop_metrics_score_trajectory() -> None:
    """LoopMetrics.score_trajectory tracks all scores in order."""
    provider = MockLLMProvider()
    scores = [0.2, 0.4, 0.6, 0.85]
    evaluator = _ImprovingEvaluator(scores)
    config = LoopConfig(max_iterations=10, quality_threshold=0.8)
    loop = LoopedLLM(provider=provider, config=config)
    result = loop.refine("test", evaluator)
    assert result.metrics.score_trajectory == scores


def test_composite_evaluator_weighted_average() -> None:
    """CompositeEvaluator computes correct weighted average."""

    class _FixedEvaluator:
        def __init__(self, score: float) -> None:
            self._score = score

        def evaluate(self, output: str, context: dict | None = None) -> EvaluationResult:
            return EvaluationResult(score=self._score, passed=True)

    composite = CompositeEvaluator(
        evaluators=[_FixedEvaluator(1.0), _FixedEvaluator(0.0)],
        weights=[3.0, 1.0],
    )
    result = composite.evaluate("test")
    assert abs(result.score - 0.75) < 1e-6


def test_composite_evaluator_merges_deficiencies() -> None:
    """CompositeEvaluator merges deficiencies from all child evaluators."""

    class _DeficiencyEvaluator:
        def __init__(self, deficiency: str) -> None:
            self._deficiency = deficiency

        def evaluate(self, output: str, context: dict | None = None) -> EvaluationResult:
            return EvaluationResult(
                score=0.5, passed=False, deficiencies=[self._deficiency]
            )

    composite = CompositeEvaluator(
        evaluators=[_DeficiencyEvaluator("Issue A"), _DeficiencyEvaluator("Issue B")],
    )
    result = composite.evaluate("test")
    assert "Issue A" in result.deficiencies
    assert "Issue B" in result.deficiencies
    assert len(result.deficiencies) == 2
