"""Tests for Conservative Dual-Verify step scoring."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from loopllm.evaluator_factory import build_evaluator
from loopllm.step_scorer import (
    conservative_dual_verify,
    keyword_criteria_score,
    legacy_self_report_score,
    score_channel_a,
)


def test_keyword_criteria_score_empty_criteria() -> None:
    score, deficiencies = keyword_criteria_score("anything", [])
    assert score == 0.9
    assert deficiencies == []


def test_score_channel_a_regex_fails_on_failed_tests() -> None:
    evaluator = build_evaluator("regex", required_patterns=["tests passed"])
    result = score_channel_a("pytest: 3 FAILED", "make tests pass", evaluator)
    assert result.score < 0.5
    assert not result.passed
    assert result.deficiencies


@pytest.mark.asyncio
async def test_cdv_blocks_self_grade_inflation() -> None:
    """Agent would self-report 0.99; regex requires 'tests passed'; artifact fails."""
    evaluator = build_evaluator("regex", required_patterns=["tests passed"])
    mock_ctx = MagicMock()
    mock_response = MagicMock()
    mock_response.content.text = json.dumps({
        "score": 0.99,
        "passed": True,
        "deficiencies": [],
        "feedback": "looks great",
    })
    mock_ctx.sample = AsyncMock(return_value=mock_response)

    result = await conservative_dual_verify(
        step_output="pytest: 3 FAILED, 0 passed",
        goal="make the failing test pass",
        quality_criteria=["tests pass"],
        evaluator=evaluator,
        ctx=mock_ctx,
    )

    assert result.source == "conservative_dual_verify"
    assert result.channel_a_score < 0.5
    assert result.channel_b_score == pytest.approx(0.99)
    assert result.final_score == result.channel_a_score
    assert result.final_score < 0.5


@pytest.mark.asyncio
async def test_cdv_channel_a_only_when_no_ctx() -> None:
    evaluator = build_evaluator("regex", required_patterns=["done"])
    result = await conservative_dual_verify(
        step_output="task done successfully",
        goal="finish task",
        quality_criteria=[],
        evaluator=evaluator,
        ctx=None,
    )
    assert result.source == "channel_a_only"
    assert result.channel_b_score is None
    assert result.final_score == result.channel_a_score


@pytest.mark.asyncio
async def test_cdv_min_fusion_when_both_channels() -> None:
    evaluator = build_evaluator("length", min_words=1)
    mock_ctx = MagicMock()
    mock_response = MagicMock()
    mock_response.content.text = '{"score": 0.4, "passed": false, "deficiencies": ["weak"], "feedback": "ok"}'
    mock_ctx.sample = AsyncMock(return_value=mock_response)

    result = await conservative_dual_verify(
        step_output="short",
        goal="write a long report",
        quality_criteria=["comprehensive"],
        evaluator=evaluator,
        ctx=mock_ctx,
    )
    assert result.final_score == min(result.channel_a_score, result.channel_b_score or 1.0)
    assert result.source == "conservative_dual_verify"


def test_legacy_self_report_score() -> None:
    result = legacy_self_report_score(0.85)
    assert result.source == "legacy_self_report"
    assert result.final_score == pytest.approx(0.85)


def test_dual_verify_score_to_dict() -> None:
    result = legacy_self_report_score(0.5)
    d = result.to_dict()
    assert d["score"] == 0.5
    assert d["score_source"] == "legacy_self_report"
    assert "channel_a_score" in d
