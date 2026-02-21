"""Tests for the task model and orchestrator."""
from __future__ import annotations

import json

import pytest

from loopllm.elicitation import IntentSpec
from loopllm.engine import RefinementResult
from loopllm.evaluators import CompletenessEvaluator, ConsistencyEvaluator
from loopllm.providers.mock import MockLLMProvider
from loopllm.store import LoopStore
from loopllm.tasks import Task, TaskOrchestrator, TaskPlan, TaskState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> LoopStore:
    return LoopStore(db_path=":memory:")


@pytest.fixture()
def simple_spec() -> IntentSpec:
    return IntentSpec(
        task_type="code_generation",
        original_prompt="Write a sort function",
        refined_prompt="Write a Python merge sort function with docstring",
        constraints={"language": "python"},
        quality_criteria=["correctness", "docstring"],
        estimated_complexity=0.2,  # Below decomposition threshold
    )


@pytest.fixture()
def complex_spec() -> IntentSpec:
    return IntentSpec(
        task_type="code_generation",
        original_prompt="Build a REST API",
        refined_prompt="Build a RESTful API with CRUD endpoints for a todo list",
        constraints={"framework": "flask"},
        quality_criteria=["endpoints", "error handling", "documentation"],
        decomposition_hints=["Define models", "Create endpoints", "Add error handling"],
        estimated_complexity=0.7,  # Above decomposition threshold
    )


# ---------------------------------------------------------------------------
# Task data model tests
# ---------------------------------------------------------------------------


class TestTask:
    def test_defaults(self) -> None:
        t = Task(title="Test task")
        assert t.title == "Test task"
        assert t.state == TaskState.PENDING
        assert t.dependencies == []
        assert t.id  # Should be auto-generated

    def test_state_transitions(self) -> None:
        t = Task(title="Test")
        assert t.state == TaskState.PENDING
        t.state = TaskState.IN_PROGRESS
        assert t.state == TaskState.IN_PROGRESS
        t.state = TaskState.COMPLETED
        assert t.state == TaskState.COMPLETED


class TestTaskState:
    def test_enum_values(self) -> None:
        assert TaskState.PENDING.value == "pending"
        assert TaskState.IN_PROGRESS.value == "in_progress"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.VERIFIED.value == "verified"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.BLOCKED.value == "blocked"

    def test_string_comparison(self) -> None:
        assert TaskState.PENDING == "pending"


class TestTaskPlan:
    def test_execution_order_no_deps(self) -> None:
        t1 = Task(title="A")
        t2 = Task(title="B")
        plan = TaskPlan(tasks=[t1, t2], dependency_graph={t1.id: [], t2.id: []})
        order = plan.execution_order()
        assert len(order) == 2

    def test_execution_order_with_deps(self) -> None:
        t1 = Task(title="First")
        t2 = Task(title="Second", dependencies=[t1.id])
        t3 = Task(title="Third", dependencies=[t2.id])

        plan = TaskPlan(
            tasks=[t3, t1, t2],  # Unordered
            dependency_graph={
                t1.id: [],
                t2.id: [t1.id],
                t3.id: [t2.id],
            },
        )

        order = plan.execution_order()
        ids = [t.id for t in order]
        assert ids.index(t1.id) < ids.index(t2.id)
        assert ids.index(t2.id) < ids.index(t3.id)

    def test_execution_order_diamond(self) -> None:
        """Diamond dependency: A → B, A → C, B → D, C → D."""
        a = Task(title="A")
        b = Task(title="B", dependencies=[a.id])
        c = Task(title="C", dependencies=[a.id])
        d = Task(title="D", dependencies=[b.id, c.id])

        plan = TaskPlan(
            tasks=[d, c, b, a],
            dependency_graph={
                a.id: [],
                b.id: [a.id],
                c.id: [a.id],
                d.id: [b.id, c.id],
            },
        )

        order = plan.execution_order()
        ids = [t.id for t in order]
        assert ids.index(a.id) < ids.index(b.id)
        assert ids.index(a.id) < ids.index(c.id)
        assert ids.index(b.id) < ids.index(d.id)
        assert ids.index(c.id) < ids.index(d.id)

    def test_cycle_raises(self) -> None:
        t1 = Task(title="A")
        t2 = Task(title="B")
        plan = TaskPlan(
            tasks=[t1, t2],
            dependency_graph={t1.id: [t2.id], t2.id: [t1.id]},
        )
        with pytest.raises(ValueError, match="cycle"):
            plan.execution_order()


# ---------------------------------------------------------------------------
# TaskOrchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestratorPlan:
    def test_simple_task_single_subtask(self, simple_spec: IntentSpec) -> None:
        """Simple tasks (complexity < 0.3) should not be decomposed."""
        provider = MockLLMProvider(responses=["result"])
        orchestrator = TaskOrchestrator(provider=provider)
        plan = orchestrator.plan(simple_spec)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].title == "Execute task"

    def test_complex_task_decomposed(self, complex_spec: IntentSpec) -> None:
        """Complex tasks should be decomposed via LLM."""
        subtasks_json = json.dumps([
            {"title": "Define models", "description": "Create data models", "dependencies": []},
            {"title": "Create endpoints", "description": "Build CRUD routes", "dependencies": ["Define models"]},
            {"title": "Add docs", "description": "Write API docs", "dependencies": ["Create endpoints"]},
        ])
        provider = MockLLMProvider(responses=[subtasks_json])
        orchestrator = TaskOrchestrator(provider=provider)
        plan = orchestrator.plan(complex_spec)

        assert len(plan.tasks) == 3
        titles = [t.title for t in plan.tasks]
        assert "Define models" in titles
        assert "Create endpoints" in titles

    def test_plan_with_store(self, complex_spec: IntentSpec, store: LoopStore) -> None:
        """Tasks should be persisted to the store."""
        subtasks_json = json.dumps([
            {"title": "Step 1", "description": "Do thing", "dependencies": []},
        ])
        provider = MockLLMProvider(responses=[subtasks_json])
        orchestrator = TaskOrchestrator(provider=provider, store=store)
        orchestrator.plan(complex_spec)

        # Should be in the store
        tasks = store.get_tasks()
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Step 1"

    def test_invalid_decomposition_fallback(self, complex_spec: IntentSpec) -> None:
        """Invalid LLM response should fall back to single task."""
        provider = MockLLMProvider(responses=["not valid json"])
        orchestrator = TaskOrchestrator(provider=provider)
        plan = orchestrator.plan(complex_spec)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].title == "Execute task"


class TestOrchestratorExecute:
    def test_execute_single_task(self, simple_spec: IntentSpec) -> None:
        provider = MockLLMProvider(responses=[
            "def merge_sort(lst): return sorted(lst)",  # Loop iteration 1
        ])
        orchestrator = TaskOrchestrator(provider=provider)
        plan = orchestrator.plan(simple_spec)
        results = orchestrator.execute(plan)

        assert len(results) == 1
        result = next(iter(results.values()))
        assert "merge_sort" in result.output

    def test_execute_preserves_order(self) -> None:
        t1 = Task(title="First", description="Write part 1")
        t2 = Task(title="Second", description="Write part 2", dependencies=[t1.id])

        plan = TaskPlan(
            tasks=[t2, t1],
            dependency_graph={t1.id: [], t2.id: [t1.id]},
        )

        provider = MockLLMProvider(responses=[
            "Part 1 result",   # t1 iteration 1
            "Part 2 result with Part 1 context",  # t2 iteration 1
        ])
        orchestrator = TaskOrchestrator(provider=provider)
        results = orchestrator.execute(plan)

        assert t1.state == TaskState.COMPLETED
        assert t2.state == TaskState.COMPLETED
        assert len(results) == 2

    def test_execute_with_store(self, store: LoopStore) -> None:
        """Task state should be updated in store during execution."""
        # Use a complex spec so plan() persists tasks to store
        complex_spec = IntentSpec(
            task_type="test",
            original_prompt="test",
            refined_prompt="Write some test code",
            estimated_complexity=0.7,
            decomposition_hints=["step one"],
        )
        subtasks_json = json.dumps([
            {"title": "Step 1", "description": "Do the work", "dependencies": []},
        ])
        provider = MockLLMProvider(responses=[
            subtasks_json,     # plan() decomposition
            "result output",   # execute task iteration
        ])
        orchestrator = TaskOrchestrator(provider=provider, store=store)
        plan = orchestrator.plan(complex_spec)
        orchestrator.execute(plan)

        # Task state should be updated in store
        task_id = plan.tasks[0].id
        stored = store.get_task(task_id)
        assert stored is not None
        assert stored["state"] == "completed"


class TestOrchestratorVerify:
    def test_verify_parses_response(self) -> None:
        verification = json.dumps({
            "overall_score": 0.85,
            "criteria_scores": {"correctness": 0.9, "docstring": 0.8},
            "issues": [],
        })
        provider = MockLLMProvider(responses=[verification])
        orchestrator = TaskOrchestrator(provider=provider)

        spec = IntentSpec(
            refined_prompt="test",
            quality_criteria=["correctness", "docstring"],
        )
        result = orchestrator.verify(spec, "some output")
        assert result.score == pytest.approx(0.85)
        assert result.passed is True

    def test_verify_with_issues(self) -> None:
        verification = json.dumps({
            "overall_score": 0.4,
            "criteria_scores": {"correctness": 0.3},
            "issues": ["Missing error handling"],
        })
        provider = MockLLMProvider(responses=[verification])
        orchestrator = TaskOrchestrator(provider=provider)

        spec = IntentSpec(refined_prompt="test", quality_criteria=["correctness"])
        result = orchestrator.verify(spec, "some output")
        assert result.score == pytest.approx(0.4)
        assert result.passed is False
        assert "Missing error handling" in result.deficiencies

    def test_verify_invalid_json(self) -> None:
        provider = MockLLMProvider(responses=["not json"])
        orchestrator = TaskOrchestrator(provider=provider)
        spec = IntentSpec(refined_prompt="test")
        result = orchestrator.verify(spec, "output")
        assert result.score == 0.5  # Fallback score


class TestOrchestratorRun:
    def test_full_pipeline_simple(self) -> None:
        """Full pipeline: classify → analyze → refine → execute."""
        provider = MockLLMProvider(responses=[
            # Elicitation: classify, analyze (no questions above threshold), refine
            "code_generation",
            "[]",  # No questions
            json.dumps({
                "task_type": "code_generation",
                "refined_prompt": "Write a sort function",
                "constraints": {},
                "quality_criteria": ["correctness"],
                "decomposition_hints": [],
                "estimated_complexity": 0.2,
            }),
            # Execution (single task, 1 iteration)
            "def sort(lst): return sorted(lst)",
        ])

        orchestrator = TaskOrchestrator(provider=provider)
        result = orchestrator.run("Write a sort function")

        assert isinstance(result, RefinementResult)
        assert len(result.output) > 0


# ---------------------------------------------------------------------------
# Task-aware evaluator tests
# ---------------------------------------------------------------------------


class TestConsistencyEvaluator:
    def test_high_overlap(self) -> None:
        dep = "The function uses merge sort algorithm for efficient sorting"
        output = "This implements the merge sort algorithm for sorting lists efficiently"
        evaluator = ConsistencyEvaluator(dependency_outputs=[dep])
        result = evaluator.evaluate(output)
        assert result.score > 0.3

    def test_no_overlap(self) -> None:
        dep = "Configure the database connection with PostgreSQL"
        output = "Hello world is a simple program that prints a greeting"
        evaluator = ConsistencyEvaluator(dependency_outputs=[dep])
        result = evaluator.evaluate(output)
        assert result.score < 0.5

    def test_empty_dependencies(self) -> None:
        evaluator = ConsistencyEvaluator(dependency_outputs=[])
        result = evaluator.evaluate("any output")
        assert result.score == 1.0
        assert result.passed is True

    def test_multiple_dependencies(self) -> None:
        deps = [
            "Module for sorting with merge sort",
            "Test cases for the sorting function",
        ]
        output = "The merge sort function passes all test cases for sorting"
        evaluator = ConsistencyEvaluator(dependency_outputs=deps)
        result = evaluator.evaluate(output)
        assert result.score > 0.0


class TestCompletenessEvaluator:
    def test_all_aspects_covered(self) -> None:
        aspects = ["error handling", "input validation"]
        output = "This function includes error handling and input validation checks"
        evaluator = CompletenessEvaluator(required_aspects=aspects)
        result = evaluator.evaluate(output)
        assert result.score == 1.0
        assert result.passed is True

    def test_missing_aspects(self) -> None:
        aspects = ["error handling", "input validation", "logging"]
        output = "This function handles errors gracefully"
        evaluator = CompletenessEvaluator(required_aspects=aspects)
        result = evaluator.evaluate(output)
        assert result.score < 1.0
        assert len(result.deficiencies) > 0

    def test_empty_aspects(self) -> None:
        evaluator = CompletenessEvaluator(required_aspects=[])
        result = evaluator.evaluate("anything")
        assert result.score == 1.0

    def test_partial_coverage(self) -> None:
        aspects = ["sorting algorithm", "documentation", "testing"]
        output = "The sorting algorithm is well documented with examples"
        evaluator = CompletenessEvaluator(required_aspects=aspects)
        result = evaluator.evaluate(output)
        # Should have partial score
        assert 0.0 < result.score < 1.0
