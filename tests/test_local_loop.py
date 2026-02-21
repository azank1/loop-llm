"""Tests for LocalModelLoop, LoopIteration, and LocalLoopResult."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from loopllm.local_loop import LocalLoopResult, LocalModelLoop, LoopIteration


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_loop(**kwargs) -> LocalModelLoop:
    defaults = dict(
        base_url="http://localhost:11434",
        model="test-model",
        score_url="http://localhost:8765/score",
        quality_threshold=0.80,
        max_retries=3,
    )
    defaults.update(kwargs)
    return LocalModelLoop(**defaults)


def _mock_score(score: float, deficiencies: list[str] | None = None) -> dict:
    return {
        "output_score": score,
        "deficiencies": deficiencies or [],
    }


# ---------------------------------------------------------------------------
# LoopIteration dataclass
# ---------------------------------------------------------------------------

class TestLoopIteration:
    def test_fields_stored(self):
        it = LoopIteration(
            iteration=0,
            prompt="p",
            output="o",
            score=0.9,
            passed=True,
            deficiencies=[],
            latency_ms=42.0,
        )
        assert it.iteration == 0
        assert it.passed is True
        assert it.rewrite_used is False  # default

    def test_rewrite_used_flag(self):
        it = LoopIteration(0, "p", "o", 0.5, False, [], 10.0, rewrite_used=True)
        assert it.rewrite_used is True


# ---------------------------------------------------------------------------
# LocalLoopResult dataclass
# ---------------------------------------------------------------------------

class TestLocalLoopResult:
    def test_fields_stored(self):
        r = LocalLoopResult(output="hi", final_score=0.9, best_score=0.9,
                            total_iterations=1, converged=True)
        assert r.output == "hi"
        assert r.converged is True
        assert r.iterations == []


# ---------------------------------------------------------------------------
# LocalModelLoop construction
# ---------------------------------------------------------------------------

class TestLocalModelLoopInit:
    def test_defaults(self):
        loop = LocalModelLoop()
        assert loop.base_url == "http://localhost:11434"
        assert loop.model == "llama3.2"
        assert loop.quality_threshold == 0.80
        assert loop.max_retries == 3

    def test_trailing_slash_stripped(self):
        loop = LocalModelLoop(base_url="http://localhost:11434/")
        assert loop.base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# _rewrite_prompt
# ---------------------------------------------------------------------------

class TestRewritePrompt:
    def test_contains_score_header(self):
        loop = _make_loop(quality_threshold=0.80, max_retries=3)
        result = loop._rewrite_prompt(
            original_prompt="Write a function",
            previous_output="def f(): pass",
            score=0.55,
            deficiencies=["too short"],
            iteration=1,
        )
        assert "[LOOPLLM | score=0.55 | retry=1/3 | threshold=0.80]" in result

    def test_contains_original_prompt(self):
        loop = _make_loop()
        result = loop._rewrite_prompt("My task", "prev", 0.4, [], 1)
        assert "My task" in result

    def test_default_deficiency_when_empty(self):
        loop = _make_loop()
        result = loop._rewrite_prompt("task", "prev", 0.4, [], 1)
        assert "did not meet quality threshold" in result

    def test_deficiencies_listed(self):
        loop = _make_loop()
        result = loop._rewrite_prompt("task", "prev", 0.4, ["too vague", "missing examples"], 1)
        assert "too vague" in result
        assert "missing examples" in result

    def test_previous_output_truncated(self):
        loop = _make_loop()
        long_output = "x" * 1000
        result = loop._rewrite_prompt("task", long_output, 0.4, [], 1)
        # Only first 500 chars of previous output should appear
        assert long_output[:501] not in result
        assert long_output[:500] in result


# ---------------------------------------------------------------------------
# LocalModelLoop._call_model (mocked httpx)
# ---------------------------------------------------------------------------

class TestCallModel:
    def test_returns_message_content(self):
        loop = _make_loop()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "hello"}}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            output = loop._call_model("prompt")

        assert output == "hello"
        mock_post.assert_called_once()

    def test_falls_back_to_response_key(self):
        loop = _make_loop()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "fallback"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            output = loop._call_model("prompt")

        assert output == "fallback"

    def test_system_message_included(self):
        loop = _make_loop()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            loop._call_model("user prompt", system="be helpful")

        payload = mock_post.call_args[1]["json"]
        assert payload["messages"][0] == {"role": "system", "content": "be helpful"}
        assert payload["messages"][1] == {"role": "user", "content": "user prompt"}

    def test_raises_if_httpx_missing(self, monkeypatch):
        loop = _make_loop()
        import sys
        original = sys.modules.get("httpx")
        sys.modules["httpx"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, TypeError)):
                loop._call_model("prompt")
        finally:
            if original is not None:
                sys.modules["httpx"] = original
            else:
                del sys.modules["httpx"]


# ---------------------------------------------------------------------------
# LocalModelLoop._score (mocked httpx)
# ---------------------------------------------------------------------------

class TestScore:
    def test_returns_score_dict(self):
        loop = _make_loop()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"output_score": 0.88, "deficiencies": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            result = loop._score("prompt", "output")

        assert result["output_score"] == 0.88

    def test_fallback_on_http_error(self):
        loop = _make_loop(quality_threshold=0.80)
        with patch("httpx.post", side_effect=Exception("connection refused")):
            # output with 5 words should score 1.0 (min_words=5)
            result = loop._score("p", "one two three four five", min_words=5)

        assert "output_score" in result
        assert result["output_score"] == 1.0

    def test_fallback_short_output_scores_low(self):
        loop = _make_loop(quality_threshold=0.80)
        with patch("httpx.post", side_effect=Exception("unreachable")):
            result = loop._score("p", "one", min_words=10)

        assert result["output_score"] < 1.0


# ---------------------------------------------------------------------------
# LocalModelLoop.run — full integration with mocks
# ---------------------------------------------------------------------------

class TestRun:
    def test_converges_on_first_try(self):
        loop = _make_loop(quality_threshold=0.80, max_retries=3)

        with patch.object(loop, "_call_model", return_value="great output"):
            with patch.object(loop, "_score", return_value=_mock_score(0.90)):
                result = loop.run("write a function")

        assert result.converged is True
        assert result.total_iterations == 1
        assert result.output == "great output"
        assert result.final_score == 0.90

    def test_retries_on_low_score_then_succeeds(self):
        loop = _make_loop(quality_threshold=0.80, max_retries=3)
        model_outputs = ["poor output", "mediocre", "great output"]
        scores = [_mock_score(0.40), _mock_score(0.65), _mock_score(0.85)]

        with patch.object(loop, "_call_model", side_effect=model_outputs):
            with patch.object(loop, "_score", side_effect=scores):
                result = loop.run("write a function")

        assert result.converged is True
        assert result.total_iterations == 3
        assert result.output == "great output"
        assert result.best_score == 0.85

    def test_exhausts_retries_without_convergence(self):
        loop = _make_loop(quality_threshold=0.80, max_retries=3)

        with patch.object(loop, "_call_model", return_value="bad"):
            with patch.object(loop, "_score", return_value=_mock_score(0.40)):
                result = loop.run("impossible task")

        assert result.converged is False
        assert result.total_iterations == 3
        assert result.best_score == 0.40

    def test_best_score_tracked_across_iterations(self):
        loop = _make_loop(quality_threshold=0.95, max_retries=3)
        scores = [_mock_score(0.70), _mock_score(0.75), _mock_score(0.60)]

        with patch.object(loop, "_call_model", return_value="output"):
            with patch.object(loop, "_score", side_effect=scores):
                result = loop.run("task")

        assert result.best_score == 0.75
        assert result.converged is False

    def test_rewrite_used_flag_on_subsequent_iterations(self):
        loop = _make_loop(quality_threshold=0.80, max_retries=3)

        with patch.object(loop, "_call_model", return_value="ok"):
            with patch.object(loop, "_score", side_effect=[
                _mock_score(0.3), _mock_score(0.85),
            ]):
                result = loop.run("task")

        assert result.iterations[0].rewrite_used is False
        assert result.iterations[1].rewrite_used is True

    def test_system_message_forwarded(self):
        loop = _make_loop(quality_threshold=0.80, max_retries=1)

        with patch.object(loop, "_call_model", return_value="ok") as m:
            with patch.object(loop, "_score", return_value=_mock_score(0.9)):
                loop.run("task", system="be terse")

        m.assert_called_once_with("task", system="be terse")

    def test_iterations_list_populated(self):
        loop = _make_loop(quality_threshold=0.80, max_retries=2)

        with patch.object(loop, "_call_model", return_value="output"):
            with patch.object(loop, "_score", side_effect=[
                _mock_score(0.3, ["too short"]), _mock_score(0.85),
            ]):
                result = loop.run("task")

        assert len(result.iterations) == 2
        assert result.iterations[0].deficiencies == ["too short"]
        assert result.iterations[0].score == 0.3
        assert result.iterations[1].score == 0.85

    def test_empty_output_handled(self):
        loop = _make_loop(quality_threshold=0.80, max_retries=1)

        with patch.object(loop, "_call_model", return_value=""):
            with patch.object(loop, "_score", return_value=_mock_score(0.0)):
                result = loop.run("task")

        assert result.output == ""
        assert result.final_score == 0.0
        assert result.converged is False
