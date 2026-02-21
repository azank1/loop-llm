"""Tests for the Bayesian intent elicitation layer."""
from __future__ import annotations

import json
from typing import Any


from loopllm.elicitation import (
    QUESTION_TYPES,
    ClarifyingQuestion,
    ElicitationSession,
    IntentRefiner,
    IntentSpec,
)
from loopllm.providers.mock import MockLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_question_response(questions: list[dict[str, Any]]) -> str:
    """Build a JSON response string for the analyze prompt."""
    return json.dumps(questions)


def _make_spec_response(
    task_type: str = "code_generation",
    refined_prompt: str = "Write a sorting function",
    complexity: float = 0.4,
) -> str:
    """Build a JSON response string for the refine prompt."""
    return json.dumps({
        "task_type": task_type,
        "refined_prompt": refined_prompt,
        "constraints": {"language": "python"},
        "quality_criteria": ["correctness", "readability"],
        "decomposition_hints": [],
        "estimated_complexity": complexity,
    })


def _mock_provider_for_session(
    questions: list[dict[str, Any]],
    task_type: str = "code_generation",
) -> MockLLMProvider:
    """Create a MockLLMProvider that responds correctly to the elicitation flow.

    The flow is: classify → analyze → refine.
    """
    return MockLLMProvider(responses=[
        task_type,                                    # classify_task
        _make_question_response(questions),           # analyze (first call)
        _make_spec_response(task_type=task_type),     # refine
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClarifyingQuestion:
    def test_defaults(self) -> None:
        q = ClarifyingQuestion(text="What scope?", question_type="scope")
        assert q.text == "What scope?"
        assert q.question_type == "scope"
        assert q.options is None
        assert q.information_gain == 0.0

    def test_with_options(self) -> None:
        q = ClarifyingQuestion(
            text="Preferred format?",
            question_type="format",
            options=["JSON", "YAML", "Plain text"],
        )
        assert len(q.options) == 3


class TestIntentSpec:
    def test_defaults(self) -> None:
        spec = IntentSpec()
        assert spec.task_type == "general"
        assert spec.estimated_complexity == 0.5
        assert spec.constraints == {}
        assert spec.quality_criteria == []

    def test_with_values(self) -> None:
        spec = IntentSpec(
            task_type="code_generation",
            original_prompt="Write sort",
            refined_prompt="Write a Python function that sorts a list using merge sort",
            constraints={"language": "python", "algorithm": "merge sort"},
            quality_criteria=["correctness", "O(n log n)"],
        )
        assert spec.task_type == "code_generation"
        assert "merge sort" in spec.refined_prompt


class TestIntentRefinerAnalyze:
    def test_generates_questions(self) -> None:
        questions_data = [
            {"question_type": "scope", "question": "What should the function do?", "options": None},
            {"question_type": "format", "question": "What format?", "options": ["JSON", "text"]},
        ]
        provider = MockLLMProvider(responses=[
            _make_question_response(questions_data),
        ])
        refiner = IntentRefiner(provider=provider)
        result = refiner.analyze("Write a function")

        assert len(result) == 2
        assert all(isinstance(q, ClarifyingQuestion) for q in result)
        assert result[0].question_type in QUESTION_TYPES

    def test_ranks_by_info_gain(self) -> None:
        questions_data = [
            {"question_type": "scope", "question": "Scope?"},
            {"question_type": "edge_cases", "question": "Edge cases?"},
            {"question_type": "format", "question": "Format?"},
        ]
        provider = MockLLMProvider(responses=[
            _make_question_response(questions_data),
        ])
        refiner = IntentRefiner(provider=provider)
        result = refiner.analyze("Write code")

        # All should have info_gain computed
        assert all(q.information_gain > 0 for q in result)
        # Should be sorted descending
        gains = [q.information_gain for q in result]
        assert gains == sorted(gains, reverse=True)

    def test_handles_invalid_json(self) -> None:
        provider = MockLLMProvider(responses=["not json at all"])
        refiner = IntentRefiner(provider=provider)
        result = refiner.analyze("Write code")
        assert result == []

    def test_handles_json_with_preamble(self) -> None:
        """Parser should extract JSON even with surrounding text."""
        questions_data = [
            {"question_type": "scope", "question": "What scope?"},
        ]
        wrapped = f"Here are some questions:\n{json.dumps(questions_data)}\nDone!"
        provider = MockLLMProvider(responses=[wrapped])
        refiner = IntentRefiner(provider=provider)
        result = refiner.analyze("Write code")
        assert len(result) == 1


class TestIntentRefinerRefine:
    def test_produces_spec(self) -> None:
        provider = MockLLMProvider(responses=[
            _make_spec_response(),
        ])
        refiner = IntentRefiner(provider=provider)
        spec = refiner.refine("Write a function", {"scope": "sorting"})

        assert isinstance(spec, IntentSpec)
        assert spec.task_type == "code_generation"
        assert spec.refined_prompt != ""
        assert len(spec.quality_criteria) > 0

    def test_handles_invalid_spec_json(self) -> None:
        provider = MockLLMProvider(responses=["Just a plain text response"])
        refiner = IntentRefiner(provider=provider)
        spec = refiner.refine("Write code", {"scope": "everything"})

        # Should fall back gracefully
        assert isinstance(spec, IntentSpec)
        assert spec.original_prompt == "Write code"

    def test_clamps_complexity(self) -> None:
        provider = MockLLMProvider(responses=[
            json.dumps({
                "task_type": "test",
                "refined_prompt": "test",
                "constraints": {},
                "quality_criteria": [],
                "decomposition_hints": [],
                "estimated_complexity": 5.0,  # Out of range
            })
        ])
        refiner = IntentRefiner(provider=provider)
        spec = refiner.refine("test", {})
        assert 0.0 <= spec.estimated_complexity <= 1.0


class TestIntentRefinerClassify:
    def test_classify_known_type(self) -> None:
        provider = MockLLMProvider(responses=["code_generation"])
        refiner = IntentRefiner(provider=provider)
        assert refiner.classify_task("Write a Python function") == "code_generation"

    def test_classify_unknown_defaults_to_general(self) -> None:
        provider = MockLLMProvider(responses=["something_weird"])
        refiner = IntentRefiner(provider=provider)
        assert refiner.classify_task("Do the thing") == "general"


class TestIntentRefinerSession:
    def test_run_session_with_answers(self) -> None:
        questions_data = [
            {"question_type": "scope", "question": "What scope?"},
            {"question_type": "format", "question": "What format?"},
        ]
        provider = _mock_provider_for_session(questions_data)
        refiner = IntentRefiner(provider=provider, max_questions=2, min_info_gain=0.0)

        session = refiner.run_session(
            "Write code",
            answer_func=lambda q: f"answer for {q.question_type}",
        )

        assert session.task_type == "code_generation"
        assert len(session.questions_asked) > 0
        assert len(session.answers) > 0
        assert session.refined_spec is not None

    def test_run_session_without_answers(self) -> None:
        """When no answer_func, questions are gathered but not answered."""
        questions_data = [
            {"question_type": "scope", "question": "What scope?"},
        ]
        provider = _mock_provider_for_session(questions_data)
        refiner = IntentRefiner(provider=provider, max_questions=2, min_info_gain=0.0)

        session = refiner.run_session("Write code")

        assert len(session.questions_asked) > 0
        assert len(session.answers) == 0
        # Should still produce a minimal spec
        assert session.refined_spec is not None
        assert session.refined_spec.refined_prompt == "Write code"

    def test_respects_max_questions(self) -> None:
        questions_data = [
            {"question_type": "scope", "question": "Q1?"},
            {"question_type": "format", "question": "Q2?"},
            {"question_type": "constraints", "question": "Q3?"},
            {"question_type": "examples", "question": "Q4?"},
            {"question_type": "edge_cases", "question": "Q5?"},
        ]
        # Need: classify, analyze(1st), analyze(2nd), refine
        provider = MockLLMProvider(responses=[
            "general",
            _make_question_response(questions_data),
            _make_question_response(questions_data[2:]),
            _make_spec_response(),
        ])
        refiner = IntentRefiner(provider=provider, max_questions=2, min_info_gain=0.0)

        session = refiner.run_session(
            "Do something",
            answer_func=lambda q: "yes",
        )
        assert len(session.questions_asked) <= 2


class TestInfoGainComputation:
    def test_info_gain_calculation(self) -> None:
        refiner = IntentRefiner(provider=MockLLMProvider(responses=[]))
        # scope has high default prior (alpha=3.0, beta=1.5)
        gain = refiner._compute_info_gain("scope")
        assert gain > 0

    def test_unknown_type_gets_prior(self) -> None:
        refiner = IntentRefiner(provider=MockLLMProvider(responses=[]))
        gain = refiner._compute_info_gain("unknown_type")
        assert gain > 0  # Should create a default prior


class TestObserveOutcome:
    def test_positive_outcome_updates_priors(self) -> None:
        provider = MockLLMProvider(responses=[])
        refiner = IntentRefiner(provider=provider)

        session = ElicitationSession(original_prompt="test")
        session.questions_asked = [
            ClarifyingQuestion(text="Q?", question_type="scope"),
            ClarifyingQuestion(text="Q?", question_type="format"),
        ]
        prior_scope_before = refiner._get_question_prior("scope").alpha

        refiner.observe_outcome(session, final_score=0.9)

        # Positive outcome should increase alpha
        assert refiner._get_question_prior("scope").alpha > prior_scope_before

    def test_negative_outcome_updates_priors(self) -> None:
        provider = MockLLMProvider(responses=[])
        refiner = IntentRefiner(provider=provider)

        session = ElicitationSession(original_prompt="test")
        session.questions_asked = [
            ClarifyingQuestion(text="Q?", question_type="scope"),
        ]
        prior_beta_before = refiner._get_question_prior("scope").beta

        refiner.observe_outcome(session, final_score=0.3)

        # Negative outcome should increase beta
        assert refiner._get_question_prior("scope").beta > prior_beta_before
