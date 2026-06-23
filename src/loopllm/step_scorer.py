"""Conservative Dual-Verify (CDV) scoring for agent-loop steps.

Channel A: deterministic evaluators (regex, JSON, completeness).
Channel B: separate critic via MCP sampling (verifier hat).
Final score: min(channel_a, channel_b) — the stricter channel wins.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from loopllm.engine import EvaluationResult, Evaluator
from loopllm.evaluator_factory import build_evaluator


@dataclass
class DualVerifyScore:
    """Result of Conservative Dual-Verify scoring for one agent-loop step."""

    final_score: float
    channel_a_score: float
    channel_b_score: float | None
    passed: bool
    deficiencies: list[str] = field(default_factory=list)
    channel_a_sub_scores: dict[str, float] = field(default_factory=dict)
    channel_b_feedback: str = ""
    source: str = "conservative_dual_verify"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for MCP verdict JSON."""
        out: dict[str, Any] = {
            "score": round(self.final_score, 4),
            "channel_a_score": round(self.channel_a_score, 4),
            "score_source": self.source,
            "passed": self.passed,
            "deficiencies": self.deficiencies,
            "channel_a_sub_scores": self.channel_a_sub_scores,
            "channel_b_feedback": self.channel_b_feedback,
        }
        if self.channel_b_score is not None:
            out["channel_b_score"] = round(self.channel_b_score, 4)
        return out


def keyword_criteria_score(output: str, quality_criteria: list[str]) -> tuple[float, list[str]]:
    """Fast keyword pre-check used by Channel B."""
    if not quality_criteria:
        return 0.9, []
    output_lower = output.lower()
    passed_criteria = [
        c for c in quality_criteria
        if any(word in output_lower for word in c.lower().split() if len(word) > 3)
    ]
    fast_score = len(passed_criteria) / len(quality_criteria)
    fast_deficiencies = [c for c in quality_criteria if c not in passed_criteria]
    return fast_score, fast_deficiencies


def score_channel_a(
    step_output: str,
    goal: str,
    evaluator: Evaluator,
) -> EvaluationResult:
    """Channel A: deterministic evaluator on the step artifact."""
    return evaluator.evaluate(
        step_output,
        context={"goal": goal},
    )


async def sample_text(ctx: Any, prompt: str, max_tokens: int = 2048) -> str:
    """Call ctx.sample() and return plain text content."""
    result = await ctx.sample(prompt, max_tokens=max_tokens)
    content = result.content
    return content.text if hasattr(content, "text") else str(content)


async def score_channel_b(
    step_output: str,
    goal: str,
    quality_criteria: list[str],
    ctx: Any,
) -> dict[str, Any]:
    """Channel B: critic sampling call in a separate verifier role."""
    fast_score, fast_deficiencies = keyword_criteria_score(step_output, quality_criteria)

    verify_result = await sample_text(
        ctx,
        f"You are an independent verifier — NOT the agent that produced this output.\n\n"
        f"GOAL:\n{goal}\n\n"
        f"STEP OUTPUT:\n{step_output[:3000]}\n\n"
        f"QUALITY CRITERIA: {quality_criteria}\n\n"
        f"Rate how close this output is to achieving the goal on a scale 0.0-1.0.\n"
        f"Reply ONLY with valid JSON:\n"
        f'{{"score":0.0,"passed":false,"deficiencies":["..."],"feedback":"..."}}',
        max_tokens=500,
    )
    parse_failed = False
    try:
        m = re.search(r"\{.*\}", verify_result, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        score = float(data.get("score", fast_score))
        score = max(0.0, min(1.0, score))
        passed = bool(data.get("passed", score >= 0.7))
        deficiencies = list(data.get("deficiencies", fast_deficiencies))
        feedback = str(data.get("feedback", ""))
    except Exception:  # noqa: BLE001
        # Conservative fallback: mark parse failure so the caller can substitute
        # Channel A score rather than the potentially-inflated fast_score (which
        # returns 0.9 when quality_criteria is empty).
        score = fast_score
        passed = fast_score >= 0.7
        deficiencies = fast_deficiencies
        feedback = verify_result[:300]
        parse_failed = True

    return {
        "score": score,
        "passed": passed,
        "deficiencies": deficiencies,
        "feedback": feedback,
        "keyword_match": fast_score,
        "parse_failed": parse_failed,
    }


async def conservative_dual_verify(
    step_output: str,
    goal: str,
    quality_criteria: list[str],
    evaluator: Evaluator,
    ctx: Any | None,
    *,
    quality_threshold: float = 0.7,
) -> DualVerifyScore:
    """Run Conservative Dual-Verify: min(Channel A, Channel B).

    ``quality_threshold`` is the session threshold from ``AgentLoopSession``.
    ``passed`` is set relative to this threshold so it is consistent with the
    guard-stack decision — it never contradicts the continue/stop verdict.
    """
    channel_a = score_channel_a(step_output, goal, evaluator)
    deficiencies = list(channel_a.deficiencies)

    if ctx is None:
        return DualVerifyScore(
            final_score=channel_a.score,
            channel_a_score=channel_a.score,
            channel_b_score=None,
            passed=channel_a.score >= quality_threshold and not deficiencies,
            deficiencies=deficiencies,
            channel_a_sub_scores=dict(channel_a.sub_scores),
            channel_b_feedback="",
            source="channel_a_only",
        )

    try:
        channel_b = await score_channel_b(step_output, goal, quality_criteria, ctx)
    except Exception:  # noqa: BLE001 — sampling unavailable; degrade gracefully
        return DualVerifyScore(
            final_score=channel_a.score,
            channel_a_score=channel_a.score,
            channel_b_score=None,
            passed=channel_a.score >= quality_threshold and not deficiencies,
            deficiencies=deficiencies,
            channel_a_sub_scores=dict(channel_a.sub_scores),
            channel_b_feedback="",
            source="channel_a_only",
        )
    # If the critic response failed to parse, its fast_score may be 0.9 (the
    # default when quality_criteria is empty), which would silently bypass the
    # conservative min(A, B) fusion.  Use channel_a.score as the fallback so the
    # fusion stays conservative even on malformed critic output.
    b_score = channel_a.score if channel_b.get("parse_failed") else channel_b["score"]
    final = min(channel_a.score, b_score)
    all_deficiencies = deficiencies + [
        d for d in channel_b["deficiencies"] if d not in deficiencies
    ]
    passed = final >= quality_threshold and not all_deficiencies

    return DualVerifyScore(
        final_score=final,
        channel_a_score=channel_a.score,
        channel_b_score=b_score,
        passed=passed,
        deficiencies=all_deficiencies,
        channel_a_sub_scores=dict(channel_a.sub_scores),
        channel_b_feedback=channel_b["feedback"],
        source="conservative_dual_verify",
    )


def legacy_self_report_score(
    score: float,
    *,
    quality_threshold: float = 0.7,
) -> DualVerifyScore:
    """Wrap a legacy agent self-reported score (not CDV)."""
    clamped = max(0.0, min(1.0, float(score)))
    return DualVerifyScore(
        final_score=clamped,
        channel_a_score=clamped,
        channel_b_score=None,
        passed=clamped >= quality_threshold,
        deficiencies=[],
        channel_a_sub_scores={},
        channel_b_feedback="",
        source="legacy_self_report",
    )


def build_step_evaluator(
    evaluator_type: str,
    quality_criteria: list[str],
    **kwargs: Any,
) -> Evaluator:
    """Build the Channel A evaluator for an agent-loop session."""
    return build_evaluator(
        evaluator_type,
        quality_criteria=quality_criteria,
        **kwargs,
    )
