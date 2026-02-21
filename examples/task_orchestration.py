"""Task orchestration demo: multi-step workflow with MockLLMProvider.

Demonstrates:
1. IntentSpec creation from elicitation
2. Task decomposition via TaskOrchestrator
3. Dependency-ordered execution through LoopedLLM
4. Result assembly
"""
from __future__ import annotations

import json

from loopllm.elicitation import IntentSpec
from loopllm.priors import AdaptivePriors
from loopllm.providers.mock import MockLLMProvider
from loopllm.tasks import TaskOrchestrator, TaskState


def main() -> None:
    print("=" * 60)
    print("TASK ORCHESTRATION DEMO")
    print("=" * 60)

    # --- Create a complex spec that warrants decomposition ---
    spec = IntentSpec(
        task_type="code_generation",
        original_prompt="Build a calculator module",
        refined_prompt=(
            "Build a Python calculator module with basic operations "
            "(add, subtract, multiply, divide), input validation, "
            "and comprehensive unit tests."
        ),
        constraints={"language": "python", "style": "functional"},
        quality_criteria=[
            "All four operations implemented",
            "Division by zero handled",
            "Unit tests for each operation",
        ],
        decomposition_hints=[
            "Implement core operations",
            "Add input validation",
            "Write unit tests",
        ],
        estimated_complexity=0.6,
    )

    # --- Mock responses for the orchestration flow ---
    # 1. Decomposition response
    decomposition = json.dumps([
        {
            "title": "Implement operations",
            "description": "Create functions for add, subtract, multiply, and divide",
            "dependencies": [],
            "estimated_complexity": 0.3,
        },
        {
            "title": "Add validation",
            "description": "Add input validation and division by zero handling",
            "dependencies": ["Implement operations"],
            "estimated_complexity": 0.3,
        },
        {
            "title": "Write tests",
            "description": "Create unit tests for all operations and edge cases",
            "dependencies": ["Implement operations", "Add validation"],
            "estimated_complexity": 0.3,
        },
    ])

    # 2. Execution responses (one per subtask per iteration)
    ops_code = """\
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""

    validation_code = """\
def validate_input(value):
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected number, got {type(value).__name__}")
    return value

def safe_divide(a, b):
    validate_input(a)
    validate_input(b)
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""

    test_code = """\
import pytest
from calculator import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2

def test_multiply():
    assert multiply(4, 3) == 12

def test_divide():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(1, 0)
"""

    provider = MockLLMProvider(responses=[
        decomposition,     # plan() decomposition
        ops_code,          # execute task 1
        validation_code,   # execute task 2
        test_code,         # execute task 3
    ])

    # --- Run the orchestrator ---
    priors = AdaptivePriors()
    orchestrator = TaskOrchestrator(provider=provider, priors=priors)

    print(f"\nSpec: {spec.refined_prompt}")
    print(f"Complexity: {spec.estimated_complexity}")

    # Step 1: Plan
    plan = orchestrator.plan(spec)
    print(f"\n--- Task Plan ({len(plan.tasks)} subtasks) ---")
    for task in plan.execution_order():
        deps = [t.title for t in plan.tasks if t.id in task.dependencies]
        dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
        print(f"  [{task.state.value:12s}] {task.title}{dep_str}")

    # Step 2: Execute
    print(f"\n--- Executing ---")
    results = orchestrator.execute(plan)

    for task in plan.execution_order():
        status = "✓" if task.state == TaskState.COMPLETED else "✗"
        score = results[task.id].metrics.best_score if task.id in results else 0
        print(f"  {status} {task.title}: score={score:.3f}")

    # Step 3: Show results
    print(f"\n--- Results ---")
    for task in plan.execution_order():
        if task.id in results:
            print(f"\n  [{task.title}]")
            output = results[task.id].output
            for line in output.split("\n")[:5]:
                print(f"    {line}")
            if output.count("\n") > 5:
                print(f"    ... ({output.count(chr(10)) - 5} more lines)")

    # Step 4: Show priors
    print(f"\n--- Learned Priors ---")
    reports = priors.report_all()
    for r in reports:
        print(
            f"  {r['task_type']}/{r['model_id']}: "
            f"calls={r['total_calls']}, "
            f"converge={r['converge_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
