"""Bayesian intent elicitation layer.

Decomposes vague user prompts into structured specs through
information-gain-ranked clarifying questions.  Learns which
questions are most valuable per task type over time.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from loopllm.priors import AdaptivePriors, BetaPrior
from loopllm.provider import LLMProvider, LLMResponse

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Question taxonomy — each type carries its own prior
# ---------------------------------------------------------------------------

QUESTION_TYPES: list[str] = [
    "scope",        # What exactly should the output cover?
    "format",       # Desired output format / structure
    "constraints",  # Hard requirements, rules, boundaries
    "examples",     # Could you give an example of what you want?
    "edge_cases",   # How should corner cases be handled?
    "audience",     # Who is the target audience / consumer?
    "priority",     # What matters most if trade-offs are needed?
]

# Default priors for question effectiveness (positive-skew: asking is
# usually somewhat helpful, but we're uncertain).
_DEFAULT_QUESTION_PRIORS: dict[str, BetaPrior] = {
    "scope": BetaPrior(alpha=3.0, beta=1.5),
    "format": BetaPrior(alpha=2.5, beta=1.5),
    "constraints": BetaPrior(alpha=2.0, beta=1.5),
    "examples": BetaPrior(alpha=2.5, beta=2.0),
    "edge_cases": BetaPrior(alpha=1.5, beta=2.0),
    "audience": BetaPrior(alpha=2.0, beta=2.5),
    "priority": BetaPrior(alpha=2.0, beta=2.0),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClarifyingQuestion:
    """A question to ask the user, ranked by expected information gain.

    Attributes:
        text: The question text to present.
        question_type: Category from :data:`QUESTION_TYPES`.
        options: Optional multiple-choice options.
        information_gain: Expected information gain (higher = ask first).
        prior: The Beta prior tracking this question type's effectiveness.
    """

    text: str
    question_type: str
    options: list[str] | None = None
    information_gain: float = 0.0
    prior: BetaPrior = field(default_factory=BetaPrior)


@dataclass
class IntentSpec:
    """Structured specification produced by elicitation.

    This is the refined, unambiguous description of what the user wants.
    It feeds directly into the refinement loop and task orchestrator.

    Attributes:
        task_type: Classified task type (e.g. ``"code_generation"``).
        original_prompt: The user's original input.
        refined_prompt: The improved, expanded prompt.
        constraints: Key constraints extracted from answers.
        quality_criteria: How to judge the output.
        decomposition_hints: Suggested subtask breakdown.
        model_preference: Optional model preference.
        estimated_complexity: Estimated difficulty (0.0–1.0).
        context: Additional context from elicitation answers.
    """

    task_type: str = "general"
    original_prompt: str = ""
    refined_prompt: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    quality_criteria: list[str] = field(default_factory=list)
    decomposition_hints: list[str] = field(default_factory=list)
    model_preference: str | None = None
    estimated_complexity: float = 0.5
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ElicitationSession:
    """Tracks the state of a Bayesian elicitation conversation.

    Attributes:
        session_id: Unique identifier.
        original_prompt: The user's verbatim input.
        questions_asked: Questions that have been posed so far.
        answers: Mapping of question_type → user's answer.
        refined_spec: The final spec (populated after :meth:`IntentRefiner.refine`).
        task_type: Detected task type.
        model_id: Model in use.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    original_prompt: str = ""
    questions_asked: list[ClarifyingQuestion] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)
    refined_spec: IntentSpec | None = None
    task_type: str = "general"
    model_id: str = ""


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ANALYZE_PROMPT = """\
You are an intent-analysis assistant.  Given the user's prompt below,
identify what is AMBIGUOUS, MISSING, or ASSUMED.

For each gap you find, produce a JSON object with:
- "question_type": one of {question_types}
- "question": a concise clarifying question to ask the user
- "options": optional list of 2-4 suggested answers (or null)

Return a JSON array of these objects.  Return at most 5 questions.
Order them by importance (most important first).

User prompt:
\"\"\"
{prompt}
\"\"\"

Respond ONLY with the JSON array, no other text.
"""

_REFINE_PROMPT = """\
You are a prompt-engineering assistant.  Given the original user prompt
and their answers to clarifying questions, produce a structured
specification.

Original prompt:
\"\"\"
{prompt}
\"\"\"

Clarifying Q&A:
{qa_text}

Produce a JSON object with these fields:
- "task_type": a short category label (e.g. "code_generation", "summarization", "data_extraction")
- "refined_prompt": an improved, unambiguous version of the user's prompt that incorporates all their answers
- "constraints": an object of key-value constraints extracted from answers
- "quality_criteria": a list of 2-5 criteria for judging output quality
- "decomposition_hints": a list of subtask descriptions if the task should be broken down (empty list if atomic)
- "estimated_complexity": a float 0.0-1.0 (0=trivial, 1=very complex)

Respond ONLY with the JSON object, no other text.
"""

_CLASSIFY_PROMPT = """\
Classify the following user prompt into exactly one task type.
Choose from: code_generation, summarization, data_extraction, question_answering,
creative_writing, analysis, transformation, general.

User prompt: "{prompt}"

Respond with ONLY the task type label, nothing else.
"""


# ---------------------------------------------------------------------------
# IntentRefiner
# ---------------------------------------------------------------------------


class IntentRefiner:
    """Bayesian elicitation engine that turns vague prompts into structured specs.

    Uses an LLM to generate clarifying questions, ranks them by expected
    information gain (drawn from per-type Beta priors), and converges to
    a structured :class:`IntentSpec` once marginal gain drops below a
    learned threshold.

    Args:
        provider: LLM provider for meta-prompting.
        priors: Adaptive priors for learning question effectiveness.
        model: Default model to use for meta-prompts.
        min_info_gain: Stop asking when the best remaining question has
            information gain below this threshold.
        max_questions: Hard cap on questions per session.
    """

    def __init__(
        self,
        provider: LLMProvider,
        priors: AdaptivePriors | None = None,
        model: str = "gpt-4o-mini",
        min_info_gain: float = 0.15,
        max_questions: int = 5,
    ) -> None:
        self.provider = provider
        self.priors = priors or AdaptivePriors()
        self.model = model
        self.min_info_gain = min_info_gain
        self.max_questions = max_questions
        self._question_priors: dict[str, BetaPrior] = {
            qt: BetaPrior(alpha=p.alpha, beta=p.beta)
            for qt, p in _DEFAULT_QUESTION_PRIORS.items()
        }

    # -- question prior management -------------------------------------------

    def _get_question_prior(self, question_type: str) -> BetaPrior:
        """Get or create the effectiveness prior for a question type."""
        if question_type not in self._question_priors:
            self._question_priors[question_type] = BetaPrior(alpha=1.5, beta=1.5)
        return self._question_priors[question_type]

    def _compute_info_gain(self, question_type: str) -> float:
        """Compute expected information gain for a question type.

        Information gain = prior_mean * (1 - confidence).
        High impact + high uncertainty = high gain (explore).
        High impact + high confidence = moderate gain (exploit).
        Low impact = low gain regardless.
        """
        prior = self._get_question_prior(question_type)
        return prior.mean * (1.0 - prior.confidence)

    # -- core API ------------------------------------------------------------

    def classify_task(self, prompt: str) -> str:
        """Classify a prompt into a task type using the LLM.

        Args:
            prompt: The user's original prompt.

        Returns:
            A task type string.
        """
        classify_prompt = _CLASSIFY_PROMPT.format(prompt=prompt)
        response: LLMResponse = self.provider.complete(classify_prompt, self.model)
        task_type = response.content.strip().lower().replace('"', "").replace("'", "")
        # Validate against known types
        known_types = {
            "code_generation", "summarization", "data_extraction",
            "question_answering", "creative_writing", "analysis",
            "transformation", "general",
        }
        if task_type not in known_types:
            task_type = "general"
        return task_type

    def analyze(self, prompt: str) -> list[ClarifyingQuestion]:
        """Generate clarifying questions ranked by expected information gain.

        Sends the prompt to the LLM with a meta-prompt asking it to
        identify ambiguities, then ranks the resulting questions using
        per-type Bayesian priors.

        Args:
            prompt: The user's original prompt.

        Returns:
            List of :class:`ClarifyingQuestion` objects, sorted by
            information gain (highest first).
        """
        question_types_str = ", ".join(QUESTION_TYPES)
        analyze_prompt = _ANALYZE_PROMPT.format(
            prompt=prompt, question_types=question_types_str
        )

        response: LLMResponse = self.provider.complete(analyze_prompt, self.model)
        raw = response.content.strip()

        # Parse the JSON response
        questions = self._parse_questions(raw)

        # Score and sort by information gain
        for q in questions:
            q.prior = self._get_question_prior(q.question_type)
            q.information_gain = self._compute_info_gain(q.question_type)

        questions.sort(key=lambda q: q.information_gain, reverse=True)
        return questions

    def ask(self, session: ElicitationSession) -> ClarifyingQuestion | None:
        """Return the next best question to ask, or ``None`` if enough info gathered.

        Implements Bayesian stopping: stops when marginal info gain drops
        below :attr:`min_info_gain` or :attr:`max_questions` is reached.

        Args:
            session: The current elicitation session.

        Returns:
            The next question to ask, or ``None`` to stop.
        """
        # Check hard cap
        if len(session.questions_asked) >= self.max_questions:
            return None

        # Generate fresh questions if this is the first call
        if not session.questions_asked:
            candidates = self.analyze(session.original_prompt)
        else:
            # Re-analyze with context of previous answers
            qa_context = "\n".join(
                f"Q ({qt}): {q.text}\nA: {session.answers.get(qt, '(unanswered)')}"
                for q in session.questions_asked
                for qt in [q.question_type]
            )
            enriched = (
                f"{session.original_prompt}\n\n"
                f"Already answered:\n{qa_context}\n\n"
                f"What else is still unclear or ambiguous?"
            )
            candidates = self.analyze(enriched)

        # Filter out already-asked types
        asked_types = {q.question_type for q in session.questions_asked}
        candidates = [q for q in candidates if q.question_type not in asked_types]

        if not candidates:
            return None

        best = candidates[0]
        if best.information_gain < self.min_info_gain:
            return None

        return best

    def refine(
        self, prompt: str, answers: dict[str, str]
    ) -> IntentSpec:
        """Synthesize a structured :class:`IntentSpec` from prompt + answers.

        Uses the LLM to combine the original prompt with all gathered
        answers into a well-formed specification.

        Args:
            prompt: The user's original prompt.
            answers: Dict mapping question_type → user answer.

        Returns:
            A structured :class:`IntentSpec`.
        """
        qa_text = "\n".join(
            f"Q ({qt}): {ans}" for qt, ans in answers.items()
        )
        refine_prompt = _REFINE_PROMPT.format(prompt=prompt, qa_text=qa_text)

        response: LLMResponse = self.provider.complete(refine_prompt, self.model)
        raw = response.content.strip()

        spec = self._parse_spec(raw, prompt)
        return spec

    def run_session(
        self,
        prompt: str,
        answer_func: Any | None = None,
    ) -> ElicitationSession:
        """Run a complete elicitation session programmatically.

        If *answer_func* is ``None``, gathers questions without answers
        (useful for getting the question list).  Otherwise, calls
        ``answer_func(question)`` for each question to get the answer.

        Args:
            prompt: The user's original prompt.
            answer_func: Optional callable ``(ClarifyingQuestion) -> str``.

        Returns:
            The completed :class:`ElicitationSession`.
        """
        session = ElicitationSession(original_prompt=prompt)
        session.task_type = self.classify_task(prompt)

        while True:
            question = self.ask(session)
            if question is None:
                break

            session.questions_asked.append(question)

            if answer_func is not None:
                answer = answer_func(question)
                session.answers[question.question_type] = answer

        if session.answers:
            session.refined_spec = self.refine(prompt, session.answers)
        else:
            # No questions asked or no answers — create a minimal spec
            session.refined_spec = IntentSpec(
                task_type=session.task_type,
                original_prompt=prompt,
                refined_prompt=prompt,
            )

        return session

    def observe_outcome(
        self,
        session: ElicitationSession,
        final_score: float,
    ) -> None:
        """Update question priors based on the final outcome score.

        Questions asked in sessions with high final scores get positive
        updates; those in low-scoring sessions get negative updates.
        The threshold is 0.7 (above = positive).

        Args:
            session: The completed elicitation session.
            final_score: Final quality score of the task output.
        """
        success = final_score >= 0.7
        for q in session.questions_asked:
            prior = self._get_question_prior(q.question_type)
            prior.update(success)
            logger.debug(
                "question_prior_updated",
                question_type=q.question_type,
                success=success,
                new_mean=prior.mean,
            )

    # -- parsing helpers -----------------------------------------------------

    def _parse_questions(self, raw: str) -> list[ClarifyingQuestion]:
        """Parse LLM response into ClarifyingQuestion objects."""
        # Try to extract JSON array from the response
        raw = raw.strip()
        if not raw.startswith("["):
            # Try to find JSON array in the response
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                raw = raw[start : end + 1]
            else:
                logger.warning("failed_to_parse_questions", raw=raw[:200])
                return []

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", raw=raw[:200])
            return []

        questions: list[ClarifyingQuestion] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            qt = item.get("question_type", "general")
            if qt not in QUESTION_TYPES:
                qt = "scope"  # Default to scope for unknown types
            questions.append(
                ClarifyingQuestion(
                    text=item.get("question", ""),
                    question_type=qt,
                    options=item.get("options"),
                )
            )
        return questions

    def _parse_spec(self, raw: str, original_prompt: str) -> IntentSpec:
        """Parse LLM response into an IntentSpec."""
        raw = raw.strip()
        if not raw.startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start : end + 1]
            else:
                # Fallback: treat entire response as refined prompt
                return IntentSpec(
                    original_prompt=original_prompt,
                    refined_prompt=raw or original_prompt,
                )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return IntentSpec(
                original_prompt=original_prompt,
                refined_prompt=raw or original_prompt,
            )

        complexity = data.get("estimated_complexity", 0.5)
        if isinstance(complexity, str):
            try:
                complexity = float(complexity)
            except ValueError:
                complexity = 0.5
        complexity = max(0.0, min(1.0, complexity))

        return IntentSpec(
            task_type=data.get("task_type", "general"),
            original_prompt=original_prompt,
            refined_prompt=data.get("refined_prompt", original_prompt),
            constraints=data.get("constraints", {}),
            quality_criteria=data.get("quality_criteria", []),
            decomposition_hints=data.get("decomposition_hints", []),
            estimated_complexity=complexity,
        )
