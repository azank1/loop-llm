"""Built-in evaluators for scoring LLM outputs."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from loopllm.engine import EvaluationResult


class ThresholdEvaluator:
    """Evaluator that delegates scoring to a callable and applies a pass/fail threshold.

    Args:
        scorer: A callable ``(output, context) -> float`` returning a score in [0, 1].
        threshold: Minimum score to pass.
        name: Human-readable evaluator name.
    """

    def __init__(
        self,
        scorer: Callable[[str, dict[str, Any]], float],
        threshold: float = 0.7,
        name: str = "threshold",
    ) -> None:
        self.scorer = scorer
        self.threshold = threshold
        self.name = name

    def evaluate(self, output: str, context: dict[str, Any] | None = None) -> EvaluationResult:
        """Score *output* and return pass/fail based on threshold.

        Args:
            output: The text to evaluate.
            context: Optional context dict passed to the scorer.

        Returns:
            :class:`EvaluationResult` with clamped score.
        """
        ctx = context or {}
        raw = self.scorer(output, ctx)
        score = max(0.0, min(1.0, raw))
        passed = score >= self.threshold
        deficiencies: list[str] = []
        if not passed:
            deficiencies.append(
                f"{self.name}: score {score:.2f} below threshold {self.threshold:.2f}"
            )
        return EvaluationResult(
            score=score,
            passed=passed,
            deficiencies=deficiencies,
            sub_scores={self.name: score},
        )


class RegexEvaluator:
    """Evaluator that checks for required and forbidden regex patterns.

    Args:
        required: Patterns that must be present in the output.
        forbidden: Patterns that must NOT be present in the output.
    """

    def __init__(
        self,
        required: list[str] | None = None,
        forbidden: list[str] | None = None,
    ) -> None:
        self.required = [re.compile(p, re.IGNORECASE) for p in (required or [])]
        self.forbidden = [re.compile(p, re.IGNORECASE) for p in (forbidden or [])]
        self._required_raw = required or []
        self._forbidden_raw = forbidden or []

    def evaluate(self, output: str, context: dict[str, Any] | None = None) -> EvaluationResult:
        """Check *output* against required/forbidden patterns.

        Args:
            output: The text to evaluate.
            context: Unused; accepted for interface compatibility.

        Returns:
            :class:`EvaluationResult` with score = passing_checks / total_checks.
        """
        total_checks = len(self.required) + len(self.forbidden)
        if total_checks == 0:
            return EvaluationResult(score=1.0, passed=True)

        passing = 0
        deficiencies: list[str] = []
        all_required_present = True
        no_forbidden_found = True

        for pattern, raw in zip(self.required, self._required_raw):
            if pattern.search(output):
                passing += 1
            else:
                deficiencies.append(f"Required pattern not found: {raw}")
                all_required_present = False

        for pattern, raw in zip(self.forbidden, self._forbidden_raw):
            if not pattern.search(output):
                passing += 1
            else:
                deficiencies.append(f"Forbidden pattern found: {raw}")
                no_forbidden_found = False

        score = passing / total_checks
        passed = all_required_present and no_forbidden_found
        return EvaluationResult(
            score=score,
            passed=passed,
            deficiencies=deficiencies,
            sub_scores={"regex": score},
        )


class JSONSchemaEvaluator:
    """Evaluator that checks output is valid JSON matching a lightweight schema.

    Args:
        required_fields: Field names that must be present in the JSON object.
        field_types: Mapping of field names to expected Python types.
        must_be_object: If True, the top-level JSON value must be a dict.
    """

    def __init__(
        self,
        required_fields: list[str] | None = None,
        field_types: dict[str, type] | None = None,
        must_be_object: bool = True,
    ) -> None:
        self.required_fields = required_fields or []
        self.field_types = field_types or {}
        self.must_be_object = must_be_object

    def evaluate(self, output: str, context: dict[str, Any] | None = None) -> EvaluationResult:
        """Validate *output* as JSON against the configured schema.

        Weighted sub-scores:
            - ``json_valid``: 0.2
            - ``is_object``: 0.1
            - ``fields_present``: 0.4
            - ``type_correct``: 0.3

        Args:
            output: The text to evaluate as JSON.
            context: Unused; accepted for interface compatibility.

        Returns:
            :class:`EvaluationResult` with weighted composite score.
        """
        sub_scores: dict[str, float] = {}
        deficiencies: list[str] = []

        # Parse JSON
        try:
            data = json.loads(output)
            sub_scores["json_valid"] = 1.0
        except (json.JSONDecodeError, ValueError):
            return EvaluationResult(
                score=0.0,
                passed=False,
                deficiencies=["Invalid JSON"],
                sub_scores={"json_valid": 0.0, "is_object": 0.0, "fields_present": 0.0, "type_correct": 0.0},
            )

        # Check object type
        if self.must_be_object and not isinstance(data, dict):
            sub_scores["is_object"] = 0.0
            sub_scores["fields_present"] = 0.0
            sub_scores["type_correct"] = 0.0
            deficiencies.append("Top-level value is not a JSON object")
            score = 0.2 * sub_scores["json_valid"]
            return EvaluationResult(
                score=score,
                passed=False,
                deficiencies=deficiencies,
                sub_scores=sub_scores,
            )

        sub_scores["is_object"] = 1.0

        # Check required fields
        if self.required_fields:
            present = sum(1 for f in self.required_fields if f in data)
            sub_scores["fields_present"] = present / len(self.required_fields)
            for f in self.required_fields:
                if f not in data:
                    deficiencies.append(f"Missing required field: {f}")
        else:
            sub_scores["fields_present"] = 1.0

        # Check field types
        type_checks = [(f, t) for f, t in self.field_types.items() if f in data]
        if type_checks:
            correct = sum(1 for f, t in type_checks if isinstance(data[f], t))
            sub_scores["type_correct"] = correct / len(type_checks)
            for f, t in type_checks:
                if not isinstance(data[f], t):
                    deficiencies.append(
                        f"Field '{f}' has type {type(data[f]).__name__}, expected {t.__name__}"
                    )
        else:
            sub_scores["type_correct"] = 1.0

        # Weighted score
        score = (
            0.2 * sub_scores["json_valid"]
            + 0.1 * sub_scores["is_object"]
            + 0.4 * sub_scores["fields_present"]
            + 0.3 * sub_scores["type_correct"]
        )

        passed = score >= 0.7 and not deficiencies
        return EvaluationResult(
            score=score,
            passed=passed,
            deficiencies=deficiencies,
            sub_scores=sub_scores,
        )


class LengthEvaluator:
    """Evaluator that checks character and word count bounds.

    Args:
        min_chars: Minimum number of characters required.
        max_chars: Maximum number of characters allowed.
        min_words: Minimum number of words required.
        max_words: Maximum number of words allowed.
    """

    def __init__(
        self,
        min_chars: int = 0,
        max_chars: int = 100_000,
        min_words: int = 0,
        max_words: int = 100_000,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.max_words = max_words

    def evaluate(self, output: str, context: dict[str, Any] | None = None) -> EvaluationResult:
        """Check *output* length against configured bounds.

        Args:
            output: The text to evaluate.
            context: Unused; accepted for interface compatibility.

        Returns:
            :class:`EvaluationResult` with score 1.0 if all bounds met, 0.3 otherwise.
        """
        char_count = len(output)
        word_count = len(output.split())
        deficiencies: list[str] = []

        if char_count < self.min_chars:
            deficiencies.append(
                f"Too few characters: {char_count} < minimum {self.min_chars}"
            )
        if char_count > self.max_chars:
            deficiencies.append(
                f"Too many characters: {char_count} > maximum {self.max_chars}"
            )
        if word_count < self.min_words:
            deficiencies.append(
                f"Too few words: {word_count} < minimum {self.min_words}"
            )
        if word_count > self.max_words:
            deficiencies.append(
                f"Too many words: {word_count} > maximum {self.max_words}"
            )

        score = 1.0 if not deficiencies else 0.3
        passed = not deficiencies
        return EvaluationResult(
            score=score,
            passed=passed,
            deficiencies=deficiencies,
            sub_scores={"length": score},
        )
