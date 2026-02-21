"""Tests for prompt quality scoring, intercept, stats, and feedback tools."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopllm.mcp_server import (
    _classify_task_type,
    _estimate_complexity,
    _score_prompt_quality,
    _tool_feedback,
    _tool_intercept,
    _tool_prompt_stats,
)
from loopllm.store import LoopStore


# ---------------------------------------------------------------------------
# _score_prompt_quality
# ---------------------------------------------------------------------------


class TestScorePromptQuality:
    """Unit tests for the heuristic prompt scorer."""

    def test_vague_prompt_scores_low(self) -> None:
        result = _score_prompt_quality("fix it")
        assert result["quality_score"] < 0.4
        assert result["grade"] in ("D", "F")
        assert len(result["issues"]) > 0

    def test_detailed_prompt_scores_high(self) -> None:
        prompt = (
            "Write a Python function that takes a list of integers and returns "
            "the top 3 largest values as a JSON array. The function must handle "
            "empty lists by returning an empty array. Include type hints."
        )
        result = _score_prompt_quality(prompt)
        assert result["quality_score"] >= 0.6
        assert result["grade"] in ("A", "B")

    def test_gauge_format(self) -> None:
        result = _score_prompt_quality("test prompt")
        gauge = result["gauge"]
        assert "%" in gauge
        assert "[" in gauge and "]" in gauge

    def test_dimensions_present(self) -> None:
        result = _score_prompt_quality("analyze this data")
        dims = result["dimensions"]
        assert "specificity" in dims
        assert "constraint_clarity" in dims
        assert "context_completeness" in dims
        assert "ambiguity" in dims
        assert "format_spec" in dims
        # All scores in [0, 1]
        for key, val in dims.items():
            assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"

    def test_composite_in_range(self) -> None:
        for prompt in ["x", "do something", "hello world",
                       "Write a Python class implementing a binary search tree"]:
            result = _score_prompt_quality(prompt)
            assert 0.0 <= result["quality_score"] <= 1.0

    def test_grade_assignment(self) -> None:
        """Grades should follow A >= 0.85, B >= 0.70, C >= 0.55, D >= 0.40, F < 0.40."""
        result = _score_prompt_quality("fix it")
        assert result["grade"] == "F" or result["quality_score"] >= 0.40

    def test_suggestions_for_bad_prompt(self) -> None:
        result = _score_prompt_quality("do this thing somehow")
        assert len(result["suggestions"]) > 0

    def test_word_count(self) -> None:
        result = _score_prompt_quality("one two three four five")
        assert result["word_count"] == 5

    def test_empty_prompt(self) -> None:
        result = _score_prompt_quality("")
        assert result["quality_score"] < 0.5
        assert result["word_count"] == 0


# ---------------------------------------------------------------------------
# _classify_task_type
# ---------------------------------------------------------------------------


class TestClassifyTaskType:
    def test_code_generation(self) -> None:
        assert _classify_task_type("write a function to sort a list") == "code_generation"

    def test_summarization(self) -> None:
        assert _classify_task_type("summarize this document") == "summarization"

    def test_data_extraction(self) -> None:
        assert _classify_task_type("extract all email addresses from this text") == "data_extraction"

    def test_analysis(self) -> None:
        assert _classify_task_type("analyze the performance metrics") == "analysis"

    def test_question_answering(self) -> None:
        assert _classify_task_type("what is the capital of France") == "question_answering"

    def test_general_fallback(self) -> None:
        assert _classify_task_type("asdfghjkl") == "general"

    def test_transformation(self) -> None:
        assert _classify_task_type("convert this CSV to JSON format") == "transformation"


# ---------------------------------------------------------------------------
# _estimate_complexity
# ---------------------------------------------------------------------------


class TestEstimateComplexity:
    def test_simple_prompt(self) -> None:
        score = _estimate_complexity("hello")
        assert 0.0 <= score <= 0.3

    def test_complex_prompt(self) -> None:
        prompt = (
            "Build an async distributed API with database migrations, "
            "authentication, and deploy the microservice pipeline"
        )
        score = _estimate_complexity(prompt)
        assert score >= 0.4

    def test_range(self) -> None:
        for prompt in ["x", "a b c", "build a complex distributed system"]:
            score = _estimate_complexity(prompt)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Store: prompt_history
# ---------------------------------------------------------------------------


class TestPromptHistory:
    @pytest.fixture()
    def store(self) -> LoopStore:
        return LoopStore(db_path=":memory:")

    def test_record_and_retrieve(self, store: LoopStore) -> None:
        store.record_prompt({
            "prompt_text": "write a function",
            "quality_score": 0.75,
            "grade": "B",
            "task_type": "code_generation",
        })
        history = store.get_prompt_history(limit=10)
        assert len(history) == 1
        assert history[0]["prompt_text"] == "write a function"
        assert history[0]["quality_score"] == 0.75

    def test_ordering(self, store: LoopStore) -> None:
        for i in range(5):
            store.record_prompt({
                "prompt_text": f"prompt_{i}",
                "quality_score": i * 0.2,
                "grade": "C",
            })
        history = store.get_prompt_history(limit=3)
        assert len(history) == 3
        # Most recent first
        assert history[0]["prompt_text"] == "prompt_4"

    def test_stats_empty(self, store: LoopStore) -> None:
        stats = store.get_prompt_stats(window=50)
        assert stats["total_prompts"] == 0
        assert stats["trend"] == "no_data"

    def test_stats_with_data(self, store: LoopStore) -> None:
        for score in [0.3, 0.5, 0.7, 0.8]:
            store.record_prompt({
                "prompt_text": "test",
                "quality_score": score,
                "grade": "C",
            })
        stats = store.get_prompt_stats(window=50)
        assert stats["total_prompts"] == 4
        assert stats["avg_quality"] == pytest.approx(0.575)

    def test_stats_grade_distribution(self, store: LoopStore) -> None:
        store.record_prompt({"prompt_text": "t", "quality_score": 0.9, "grade": "A"})
        store.record_prompt({"prompt_text": "t", "quality_score": 0.75, "grade": "B"})
        store.record_prompt({"prompt_text": "t", "quality_score": 0.75, "grade": "B"})
        stats = store.get_prompt_stats(window=50)
        assert stats["grade_distribution"]["A"] == 1
        assert stats["grade_distribution"]["B"] == 2


# ---------------------------------------------------------------------------
# Integration: _tool_intercept / _tool_prompt_stats / _tool_feedback
#   These require shared state initialization
# ---------------------------------------------------------------------------


class TestToolIntercept:
    """Integration tests using mock provider."""

    @pytest.fixture(autouse=True)
    def _setup_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import loopllm.mcp_server as mod

        # Reset shared state
        mod._store = None
        mod._priors = None
        mod._provider = None
        mod._status_path = None

        monkeypatch.setenv("LOOPLLM_DB", str(tmp_path / "test.db"))
        monkeypatch.setenv("LOOPLLM_PROVIDER", "mock")
        monkeypatch.setenv("LOOPLLM_MODEL", "test-model")

    def test_intercept_returns_valid_json(self) -> None:
        result = json.loads(_tool_intercept("fix it"))
        assert "route" in result
        assert "quality" in result
        assert "task_type" in result

    def test_intercept_routes_vague_to_elicit(self) -> None:
        result = json.loads(_tool_intercept("do something"))
        assert result["route"] in ("elicit", "elicit_then_refine")

    def test_intercept_routes_clear_to_refine(self) -> None:
        prompt = (
            "Write a Python function that takes a list of integers and returns "
            "the top 3 values as JSON. Must include type hints and handle edge "
            "cases like empty lists. The function should be called get_top_values."
        )
        result = json.loads(_tool_intercept(prompt))
        assert result["route"] == "refine"

    def test_intercept_logs_to_history(self) -> None:
        import loopllm.mcp_server as mod

        _tool_intercept("write a test")
        store = mod._get_store()
        history = store.get_prompt_history(limit=5)
        assert len(history) >= 1

    def test_prompt_stats_after_intercepts(self) -> None:
        _tool_intercept("fix it")
        _tool_intercept("write a detailed Python function")
        result = json.loads(_tool_prompt_stats(50))
        assert result["total_prompts"] == 2

    def test_feedback_records(self) -> None:
        result = json.loads(_tool_feedback(4, "code_generation", "good output"))
        assert result["status"] == "recorded"
        assert result["rating"] == 4

    def test_feedback_clamps_rating(self) -> None:
        result = json.loads(_tool_feedback(10))
        assert result["rating"] == 5

        result = json.loads(_tool_feedback(-1))
        assert result["rating"] == 1

    def test_intercept_writes_status_file(self, tmp_path: Path) -> None:
        import loopllm.mcp_server as mod

        _tool_intercept("hello world")
        status_path = mod._status_path
        assert status_path is not None
        assert status_path.exists()
        status = json.loads(status_path.read_text())
        assert status["tool"] == "intercept"
        assert "quality_score" in status["data"]
