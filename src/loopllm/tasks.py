"""Task model and orchestrator for multi-step workflows.

Decomposes an :class:`IntentSpec` into a dependency-ordered graph
of subtasks, executes each through :class:`LoopedLLM`, and assembles
the final result.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from loopllm.elicitation import IntentRefiner, IntentSpec
from loopllm.engine import (
    EvaluationResult,
    LoopConfig,
    LoopedLLM,
    RefinementResult,
)
from loopllm.evaluators import LengthEvaluator
from loopllm.priors import AdaptivePriors, CallObservation
from loopllm.provider import LLMProvider
from loopllm.store import LoopStore

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Task state machine
# ---------------------------------------------------------------------------


class TaskState(str, Enum):
    """Lifecycle state of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Task data model
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A single unit of work in a task plan.

    Attributes:
        id: Unique identifier.
        parent_id: ID of the parent task (``None`` for root tasks).
        title: Short description.
        description: Full description / instructions.
        state: Current lifecycle state.
        dependencies: IDs of tasks that must complete before this one.
        intent_spec: Optional structured spec for this subtask.
        result: Refinement result once executed.
        metadata: Arbitrary extra data.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    title: str = ""
    description: str = ""
    state: TaskState = TaskState.PENDING
    dependencies: list[str] = field(default_factory=list)
    intent_spec: IntentSpec | None = None
    result: RefinementResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """An ordered collection of tasks with dependency information.

    Attributes:
        tasks: All tasks in the plan.
        dependency_graph: Mapping of task ID → list of dependency IDs.
        estimated_total_cost: Rough cost estimate (token-based).
        session_id: ID of the elicitation session that produced this plan.
    """

    tasks: list[Task] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    estimated_total_cost: float = 0.0
    session_id: str = ""

    def execution_order(self) -> list[Task]:
        """Return tasks in topological order respecting dependencies.

        Uses Kahn's algorithm.  Raises :class:`ValueError` if the
        dependency graph contains cycles.

        Returns:
            Tasks sorted so that each task appears after all its dependencies.
        """
        task_map = {t.id: t for t in self.tasks}
        in_degree: dict[str, int] = {t.id: 0 for t in self.tasks}
        adj: dict[str, list[str]] = {t.id: [] for t in self.tasks}

        for task in self.tasks:
            deps = self.dependency_graph.get(task.id, task.dependencies)
            for dep_id in deps:
                if dep_id in adj:
                    adj[dep_id].append(task.id)
                    in_degree[task.id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        ordered: list[Task] = []

        while queue:
            tid = queue.pop(0)
            ordered.append(task_map[tid])
            for next_id in adj.get(tid, []):
                in_degree[next_id] -= 1
                if in_degree[next_id] == 0:
                    queue.append(next_id)

        if len(ordered) != len(self.tasks):
            msg = "Dependency graph contains a cycle"
            raise ValueError(msg)

        return ordered


# ---------------------------------------------------------------------------
# Prompt templates for task decomposition
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are a task decomposition assistant.  Given the following structured
specification, break it into discrete subtasks that can be executed
independently (with explicit dependencies where needed).

Specification:
- Task type: {task_type}
- Prompt: {refined_prompt}
- Constraints: {constraints}
- Quality criteria: {quality_criteria}
- Decomposition hints: {decomposition_hints}
- Estimated complexity: {estimated_complexity}

Produce a JSON array of subtask objects.  Each subtask has:
- "title": short label (3-8 words)
- "description": detailed instructions for the subtask
- "dependencies": list of titles of subtasks this one depends on (empty for independent)
- "estimated_complexity": float 0.0-1.0

If the task is simple enough to do in one step, return a single-element array.
Return ONLY the JSON array.
"""

_ASSEMBLE_PROMPT = """\
You are a result assembler.  Given the following subtask results,
combine them into a single coherent output that addresses the original
prompt.

Original prompt: {original_prompt}

Subtask results:
{subtask_results}

Produce a single, cohesive output that integrates all subtask results.
Do not repeat the prompt or explain what you did — just produce the final output.
"""

_VERIFY_PROMPT = """\
You are a quality verifier.  Check whether the following output
addresses all the requirements from the original specification.

Specification:
- Prompt: {refined_prompt}
- Quality criteria: {quality_criteria}

Output to verify:
\"\"\"
{output}
\"\"\"

For each quality criterion, rate it 0.0-1.0.
Return a JSON object with:
- "overall_score": float 0.0-1.0
- "criteria_scores": object mapping criterion → score
- "issues": list of strings describing problems (empty if none)

Return ONLY the JSON object.
"""


# ---------------------------------------------------------------------------
# TaskOrchestrator
# ---------------------------------------------------------------------------


class TaskOrchestrator:
    """Decompose, execute, and verify multi-step LLM tasks.

    Integrates :class:`IntentRefiner` for elicitation, :class:`LoopedLLM`
    for per-subtask refinement, and :class:`AdaptivePriors` for learning
    optimal decomposition strategies.

    Args:
        provider: LLM provider for all calls.
        priors: Adaptive priors for learning.
        store: Optional persistent store.
        refiner: Optional intent refiner (created automatically if not given).
        model: Default model to use.
    """

    def __init__(
        self,
        provider: LLMProvider,
        priors: AdaptivePriors | None = None,
        store: LoopStore | None = None,
        refiner: IntentRefiner | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.provider = provider
        self.priors = priors or AdaptivePriors()
        self.store = store
        self.refiner = refiner or IntentRefiner(
            provider=provider, priors=self.priors, model=model
        )
        self.model = model

    # -- decomposition -------------------------------------------------------

    def plan(self, spec: IntentSpec) -> TaskPlan:
        """Decompose an :class:`IntentSpec` into a :class:`TaskPlan`.

        Uses the LLM to generate subtasks with dependency ordering.
        Simple tasks (complexity < 0.3) are kept as a single task.

        Args:
            spec: The structured specification.

        Returns:
            A :class:`TaskPlan` with ordered subtasks.
        """
        # Simple tasks don't need decomposition
        if spec.estimated_complexity < 0.3 and not spec.decomposition_hints:
            task = Task(
                title="Execute task",
                description=spec.refined_prompt,
                intent_spec=spec,
            )
            return TaskPlan(
                tasks=[task],
                dependency_graph={task.id: []},
            )

        # Use the LLM to decompose
        decompose_prompt = _DECOMPOSE_PROMPT.format(
            task_type=spec.task_type,
            refined_prompt=spec.refined_prompt,
            constraints=json.dumps(spec.constraints),
            quality_criteria=json.dumps(spec.quality_criteria),
            decomposition_hints=json.dumps(spec.decomposition_hints),
            estimated_complexity=spec.estimated_complexity,
        )

        response = self.provider.complete(decompose_prompt, self.model)
        raw = response.content.strip()

        tasks = self._parse_tasks(raw, spec)
        dep_graph = self._build_dependency_graph(tasks)

        plan = TaskPlan(
            tasks=tasks,
            dependency_graph=dep_graph,
        )

        # Persist tasks
        if self.store:
            for task in tasks:
                self.store.save_task({
                    "id": task.id,
                    "parent_id": task.parent_id,
                    "title": task.title,
                    "description": task.description,
                    "state": task.state.value,
                    "dependencies": task.dependencies,
                    "spec": {
                        "task_type": spec.task_type,
                        "refined_prompt": task.description,
                    } if task.intent_spec else None,
                })

        logger.info("task_plan_created", task_count=len(tasks))
        return plan

    # -- execution -----------------------------------------------------------

    def execute(
        self, plan: TaskPlan, model: str | None = None
    ) -> dict[str, RefinementResult]:
        """Execute all tasks in a plan in dependency order.

        Each task is refined using :class:`LoopedLLM` with adaptive
        exit conditions.  Prior task outputs are passed as context
        to dependent tasks.

        Args:
            plan: The task plan to execute.
            model: Model override (defaults to ``self.model``).

        Returns:
            Dict mapping task ID to :class:`RefinementResult`.
        """
        model = model or self.model
        results: dict[str, RefinementResult] = {}

        depth = self.priors.predict_optimal_depth(
            "orchestrated_subtask", model
        )
        config = LoopConfig(
            max_iterations=max(depth, 2),
            quality_threshold=0.75,
        )

        for task in plan.execution_order():
            task.state = TaskState.IN_PROGRESS
            if self.store:
                self.store.update_task_state(task.id, task.state.value)

            logger.info("executing_task", task_id=task.id, title=task.title)

            # Build context from dependency results
            dep_context = ""
            for dep_id in plan.dependency_graph.get(task.id, task.dependencies):
                if dep_id in results:
                    dep_context += f"\n--- Result from '{dep_id}' ---\n"
                    dep_context += results[dep_id].output + "\n"

            prompt = task.description
            if dep_context:
                prompt = (
                    f"{task.description}\n\n"
                    f"Context from previous steps:\n{dep_context}"
                )

            evaluator = LengthEvaluator(min_words=5, max_words=10_000)
            loop = LoopedLLM(provider=self.provider, config=config)

            try:
                result = loop.refine(prompt, evaluator, model=model)
                task.result = result
                task.state = TaskState.COMPLETED
                results[task.id] = result

                # Learn from this subtask execution
                obs = CallObservation(
                    task_type="orchestrated_subtask",
                    model_id=model,
                    scores=result.metrics.score_trajectory,
                    latencies_ms=[it.latency_ms for it in result.iterations],
                    converged=result.metrics.converged,
                    total_iterations=result.metrics.total_iterations,
                    max_iterations=config.max_iterations,
                    quality_threshold=config.quality_threshold,
                )
                self.priors.observe(obs)

            except Exception as exc:
                logger.error("task_failed", task_id=task.id, error=str(exc))
                task.state = TaskState.FAILED
                task.metadata["error"] = str(exc)

            if self.store:
                self.store.update_task_state(task.id, task.state.value)

        return results

    # -- verification --------------------------------------------------------

    def verify(
        self,
        spec: IntentSpec,
        output: str,
    ) -> EvaluationResult:
        """Verify a combined output against the original spec.

        Uses the LLM to check quality criteria, then parses
        the structured response into an :class:`EvaluationResult`.

        Args:
            spec: The original specification.
            output: The assembled output to verify.

        Returns:
            An :class:`EvaluationResult` with per-criterion scores.
        """
        verify_prompt = _VERIFY_PROMPT.format(
            refined_prompt=spec.refined_prompt,
            quality_criteria=json.dumps(spec.quality_criteria),
            output=output,
        )

        response = self.provider.complete(verify_prompt, self.model)
        return self._parse_verification(response.content)

    # -- full pipeline -------------------------------------------------------

    def run(
        self,
        prompt: str,
        model: str | None = None,
        answer_func: Any | None = None,
    ) -> RefinementResult:
        """Run the full pipeline: elicit → plan → execute → assemble.

        This is the main entry point for end-to-end task processing.

        Args:
            prompt: The user's original prompt.
            model: Model override.
            answer_func: Optional function for interactive elicitation.

        Returns:
            The final assembled :class:`RefinementResult`.
        """
        model = model or self.model

        # Step 1: Elicit intent
        logger.info("pipeline_elicit", prompt=prompt[:80])
        session = self.refiner.run_session(prompt, answer_func=answer_func)
        spec = session.refined_spec or IntentSpec(
            original_prompt=prompt, refined_prompt=prompt
        )

        # Step 2: Plan
        logger.info("pipeline_plan", task_type=spec.task_type)
        plan = self.plan(spec)
        plan.session_id = session.session_id

        # Step 3: Execute
        logger.info("pipeline_execute", task_count=len(plan.tasks))
        results = self.execute(plan, model=model)

        # Step 4: Assemble
        if len(results) == 1:
            # Single task — return directly
            final_result = next(iter(results.values()))
        else:
            # Multiple tasks — assemble results
            final_result = self._assemble(spec, plan, results, model)

        # Step 5: Learn from outcome
        self.refiner.observe_outcome(
            session, final_score=final_result.metrics.best_score
        )

        logger.info(
            "pipeline_complete",
            tasks=len(plan.tasks),
            best_score=final_result.metrics.best_score,
        )
        return final_result

    # -- assembly ------------------------------------------------------------

    def _assemble(
        self,
        spec: IntentSpec,
        plan: TaskPlan,
        results: dict[str, RefinementResult],
        model: str,
    ) -> RefinementResult:
        """Assemble subtask results into a single output."""
        subtask_text = ""
        for task in plan.execution_order():
            if task.id in results:
                subtask_text += (
                    f"\n--- {task.title} ---\n"
                    f"{results[task.id].output}\n"
                )

        assemble_prompt = _ASSEMBLE_PROMPT.format(
            original_prompt=spec.original_prompt,
            subtask_results=subtask_text,
        )

        evaluator = LengthEvaluator(min_words=10, max_words=10_000)
        config = LoopConfig(max_iterations=2, quality_threshold=0.8)
        loop = LoopedLLM(provider=self.provider, config=config)

        return loop.refine(assemble_prompt, evaluator, model=model)

    # -- parsing helpers -----------------------------------------------------

    def _parse_tasks(
        self, raw: str, spec: IntentSpec
    ) -> list[Task]:
        """Parse LLM decomposition response into Task objects."""
        raw = raw.strip()
        if not raw.startswith("["):
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                raw = raw[start : end + 1]
            else:
                # Can't parse — create a single task
                return [
                    Task(
                        title="Execute task",
                        description=spec.refined_prompt,
                        intent_spec=spec,
                    )
                ]

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return [
                Task(
                    title="Execute task",
                    description=spec.refined_prompt,
                    intent_spec=spec,
                )
            ]

        tasks: list[Task] = []
        title_to_id: dict[str, str] = {}

        for item in items:
            if not isinstance(item, dict):
                continue
            task = Task(
                title=item.get("title", "Subtask"),
                description=item.get("description", ""),
                intent_spec=spec,
                metadata={"estimated_complexity": item.get("estimated_complexity", 0.5)},
            )
            title_to_id[task.title] = task.id
            tasks.append(task)

        # Resolve title-based dependencies to IDs
        for item, task in zip(items, tasks):
            if not isinstance(item, dict):
                continue
            dep_titles = item.get("dependencies", [])
            for dt in dep_titles:
                if dt in title_to_id:
                    task.dependencies.append(title_to_id[dt])

        return tasks if tasks else [
            Task(title="Execute task", description=spec.refined_prompt, intent_spec=spec)
        ]

    def _build_dependency_graph(
        self, tasks: list[Task]
    ) -> dict[str, list[str]]:
        """Build a dependency graph from task objects."""
        return {task.id: list(task.dependencies) for task in tasks}

    def _parse_verification(self, raw: str) -> EvaluationResult:
        """Parse LLM verification response into an EvaluationResult."""
        raw = raw.strip()
        if not raw.startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start : end + 1]
            else:
                return EvaluationResult(
                    score=0.5, passed=False,
                    deficiencies=["Could not parse verification response"],
                )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return EvaluationResult(
                score=0.5, passed=False,
                deficiencies=["Invalid verification JSON"],
            )

        score = float(data.get("overall_score", 0.5))
        score = max(0.0, min(1.0, score))
        issues = data.get("issues", [])
        sub_scores = {
            str(k): float(v)
            for k, v in data.get("criteria_scores", {}).items()
        }

        return EvaluationResult(
            score=score,
            passed=score >= 0.7 and not issues,
            deficiencies=issues,
            sub_scores=sub_scores,
        )
