"""Tests for built-in evaluators."""
from __future__ import annotations

from loopllm.evaluators import (
    JSONSchemaEvaluator,
    LengthEvaluator,
    RegexEvaluator,
    ThresholdEvaluator,
)


def test_threshold_evaluator_passes_when_above() -> None:
    """ThresholdEvaluator passes when scorer returns >= threshold."""
    evaluator = ThresholdEvaluator(scorer=lambda o, c: 0.9, threshold=0.7)
    result = evaluator.evaluate("hello")
    assert result.passed
    assert result.score >= 0.7
    assert not result.deficiencies


def test_threshold_evaluator_fails_when_below() -> None:
    """ThresholdEvaluator fails when scorer returns < threshold."""
    evaluator = ThresholdEvaluator(scorer=lambda o, c: 0.3, threshold=0.7)
    result = evaluator.evaluate("hello")
    assert not result.passed
    assert result.score < 0.7
    assert len(result.deficiencies) == 1


def test_regex_evaluator_passes_all_required() -> None:
    """RegexEvaluator passes when all required patterns are present."""
    evaluator = RegexEvaluator(required=[r"hello", r"world"])
    result = evaluator.evaluate("hello world!")
    assert result.passed
    assert result.score == 1.0


def test_regex_evaluator_fails_missing_required() -> None:
    """RegexEvaluator fails when a required pattern is missing."""
    evaluator = RegexEvaluator(required=[r"hello", r"world"])
    result = evaluator.evaluate("hello there!")
    assert not result.passed
    assert result.score < 1.0
    assert any("world" in d for d in result.deficiencies)


def test_regex_evaluator_fails_forbidden_present() -> None:
    """RegexEvaluator fails when a forbidden pattern is found."""
    evaluator = RegexEvaluator(forbidden=[r"error", r"fail"])
    result = evaluator.evaluate("This has an error")
    assert not result.passed
    assert any("error" in d for d in result.deficiencies)


def test_json_schema_evaluator_invalid_json() -> None:
    """JSONSchemaEvaluator returns score 0.0 for invalid JSON."""
    evaluator = JSONSchemaEvaluator(required_fields=["name"])
    result = evaluator.evaluate("not json at all")
    assert result.score == 0.0
    assert not result.passed
    assert "Invalid JSON" in result.deficiencies


def test_json_schema_evaluator_full_score() -> None:
    """JSONSchemaEvaluator gives full score for valid JSON with all fields and correct types."""
    evaluator = JSONSchemaEvaluator(
        required_fields=["name", "age"],
        field_types={"name": str, "age": int},
    )
    result = evaluator.evaluate('{"name": "Alice", "age": 30}')
    assert result.passed
    assert result.score == 1.0
    assert not result.deficiencies


def test_length_evaluator_fails_on_bounds() -> None:
    """LengthEvaluator fails on too-short and too-long output."""
    # Too short
    short_eval = LengthEvaluator(min_chars=100)
    result = short_eval.evaluate("hi")
    assert not result.passed
    assert any("Too few characters" in d for d in result.deficiencies)

    # Too long
    long_eval = LengthEvaluator(max_chars=5)
    result = long_eval.evaluate("this is too long")
    assert not result.passed
    assert any("Too many characters" in d for d in result.deficiencies)
