"""Build evaluators from type strings — shared by MCP and CDV step scoring."""
from __future__ import annotations

from typing import Any

from loopllm.engine import CompositeEvaluator
from loopllm.evaluators import (
    CompletenessEvaluator,
    JSONSchemaEvaluator,
    LengthEvaluator,
    RegexEvaluator,
)


def build_evaluator(
    evaluator_type: str = "length",
    **kwargs: Any,
) -> (
    LengthEvaluator
    | RegexEvaluator
    | JSONSchemaEvaluator
    | CompositeEvaluator
    | CompletenessEvaluator
):
    """Build an evaluator from a type string and optional config.

    Args:
        evaluator_type: One of ``length``, ``json``, ``regex``, ``composite``,
            or ``completeness``.
        **kwargs: Evaluator-specific options (``required_patterns``,
            ``required_fields``, ``quality_criteria``, ``min_words``, etc.).

    Returns:
        A concrete evaluator instance.
    """
    if evaluator_type == "json":
        return JSONSchemaEvaluator(
            required_fields=kwargs.get("required_fields", []),
            field_types={},
        )
    if evaluator_type == "regex":
        return RegexEvaluator(
            required=kwargs.get("required_patterns", []),
            forbidden=kwargs.get("forbidden_patterns", []),
        )
    if evaluator_type == "completeness":
        criteria = kwargs.get("quality_criteria") or kwargs.get("required_aspects") or []
        return CompletenessEvaluator(required_aspects=list(criteria))
    if evaluator_type == "composite":
        evals: list[Any] = []
        criteria = kwargs.get("quality_criteria") or []
        if criteria:
            evals.append(CompletenessEvaluator(required_aspects=list(criteria)))
        if kwargs.get("required_fields"):
            evals.append(JSONSchemaEvaluator(required_fields=kwargs["required_fields"]))
        if kwargs.get("required_patterns"):
            evals.append(RegexEvaluator(required=kwargs["required_patterns"]))
        if not evals:
            evals.append(LengthEvaluator(
                min_words=kwargs.get("min_words", 5),
                max_words=kwargs.get("max_words", 10_000),
            ))
        else:
            evals.append(LengthEvaluator(
                min_words=kwargs.get("min_words", 1),
                max_words=kwargs.get("max_words", 10_000),
            ))
        return CompositeEvaluator(evaluators=evals)
    return LengthEvaluator(
        min_words=kwargs.get("min_words", 5),
        max_words=kwargs.get("max_words", 10_000),
    )
