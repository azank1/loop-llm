"""Elicitation demo: Bayesian intent refinement with MockLLMProvider.

Demonstrates the full elicitation flow:
1. User provides a vague prompt
2. IntentRefiner generates clarifying questions ranked by info gain
3. Questions are answered (programmatically in this demo)
4. A structured IntentSpec is produced
5. The spec feeds into the refinement loop
"""
from __future__ import annotations

import json

from loopllm.elicitation import ClarifyingQuestion, IntentRefiner
from loopllm.engine import LoopConfig, LoopedLLM
from loopllm.evaluators import LengthEvaluator
from loopllm.priors import AdaptivePriors, CallObservation
from loopllm.providers.mock import MockLLMProvider


def main() -> None:
    # --- Mock responses for the elicitation flow ---
    # The mock provider cycles through these responses in order:
    # 1. classify_task response
    # 2. analyze response (questions)
    # 3. refine response (spec)
    # 4-6. refinement loop responses

    questions_json = json.dumps([
        {
            "question_type": "scope",
            "question": "Should the function handle nested lists or only flat lists?",
            "options": ["Flat lists only", "Nested lists", "Both"],
        },
        {
            "question_type": "constraints",
            "question": "Are there any performance requirements (time/space complexity)?",
            "options": ["O(n log n) time", "O(n) space", "No constraints"],
        },
        {
            "question_type": "format",
            "question": "Should the function return a new list or sort in-place?",
            "options": ["New list", "In-place", "Support both"],
        },
    ])

    spec_json = json.dumps({
        "task_type": "code_generation",
        "refined_prompt": (
            "Write a Python function called `merge_sort` that takes a flat list of "
            "comparable elements and returns a new sorted list using the merge sort "
            "algorithm. The implementation should achieve O(n log n) time complexity "
            "and include a docstring with examples."
        ),
        "constraints": {
            "language": "python",
            "algorithm": "merge sort",
            "complexity": "O(n log n)",
            "mutation": "return new list",
        },
        "quality_criteria": [
            "Correct merge sort implementation",
            "O(n log n) time complexity",
            "Includes docstring with examples",
            "Handles empty and single-element lists",
        ],
        "decomposition_hints": [],
        "estimated_complexity": 0.3,
    })

    provider = MockLLMProvider(responses=[
        # Elicitation phase
        "code_generation",       # classify_task
        questions_json,          # analyze
        spec_json,               # refine
        # Refinement loop iterations
        "def merge_sort(lst): return sorted(lst)",
        (
            "def merge_sort(lst):\n"
            '    """Sort a list using merge sort.\n\n'
            "    Examples:\n"
            "        >>> merge_sort([3, 1, 2])\n"
            "        [1, 2, 3]\n"
            '    """\n'
            "    if len(lst) <= 1:\n"
            "        return lst\n"
            "    mid = len(lst) // 2\n"
            "    left = merge_sort(lst[:mid])\n"
            "    right = merge_sort(lst[mid:])\n"
            "    return merge(left, right)\n"
        ),
    ])

    # --- Step 1: Intent elicitation ---
    print("=" * 60)
    print("BAYESIAN INTENT ELICITATION DEMO")
    print("=" * 60)

    priors = AdaptivePriors()
    refiner = IntentRefiner(
        provider=provider,
        priors=priors,
        max_questions=3,
        min_info_gain=0.0,  # Don't stop early for the demo
    )

    # Simulated answers
    answers = {
        "scope": "Flat lists only",
        "constraints": "O(n log n) time",
        "format": "New list",
    }

    def answer_func(q: ClarifyingQuestion) -> str:
        answer = answers.get(q.question_type, "Not sure")
        print(f"\n  [{q.question_type.upper()}] {q.text}")
        if q.options:
            for i, opt in enumerate(q.options, 1):
                print(f"    {i}. {opt}")
        print(f"  → {answer}")
        return answer

    print(f'\nOriginal prompt: "Write a sorting function"')
    session = refiner.run_session("Write a sorting function", answer_func=answer_func)

    print(f"\n--- Elicitation Complete ---")
    print(f"Task type: {session.task_type}")
    print(f"Questions asked: {len(session.questions_asked)}")

    if session.refined_spec:
        print(f"Complexity: {session.refined_spec.estimated_complexity:.1f}")
        print(f"Refined prompt: {session.refined_spec.refined_prompt[:120]}...")
        print(f"Quality criteria:")
        for c in session.refined_spec.quality_criteria:
            print(f"  ✓ {c}")

    # --- Step 2: Refinement loop ---
    print(f"\n{'=' * 60}")
    print("REFINEMENT LOOP")
    print("=" * 60)

    evaluator = LengthEvaluator(min_chars=50, min_words=10, max_words=500)
    config = LoopConfig(max_iterations=3, quality_threshold=0.8)
    loop = LoopedLLM(provider=provider, config=config)

    prompt = session.refined_spec.refined_prompt if session.refined_spec else "Write a sorting function"
    result = loop.refine(prompt, evaluator)

    print(f"\nExit reason: {result.metrics.exit_reason.condition}")
    print(f"Iterations: {result.metrics.total_iterations}")
    print(f"Best score: {result.metrics.best_score:.3f}")
    print(f"Score trajectory: {[f'{s:.2f}' for s in result.metrics.score_trajectory]}")
    print(f"\nFinal output:\n{result.output}")

    # --- Step 3: Learn from outcome ---
    print(f"\n{'=' * 60}")
    print("LEARNING FROM OUTCOME")
    print("=" * 60)

    refiner.observe_outcome(session, final_score=result.metrics.best_score)
    print(f"Question priors updated based on final score: {result.metrics.best_score:.3f}")

    for qt in QUESTION_TYPES_USED:
        prior = refiner._get_question_prior(qt)
        print(f"  {qt}: mean={prior.mean:.3f}, confidence={prior.confidence:.3f}")


QUESTION_TYPES_USED = ["scope", "constraints", "format"]


if __name__ == "__main__":
    main()
