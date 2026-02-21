"""Confidence-driven plan registry for scored task management.

Each plan tracks a rolling confidence score derived from:
- Prompt quality scores (from loopllm_intercept)
- Output scores (from loopllm_verify_output / evaluators)

When rolling_confidence drops below the plan's threshold, the registry
signals that the current task should be refined or the plan should be
replanned before proceeding.

This is the backbone for Shrimp-style task management where every action
is gated by accumulated evidence of quality.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    REPLANNING = "replanning"


@dataclass
class TaskRecord:
    """A single task entry in a plan.

    Attributes:
        id: Unique short identifier.
        title: Short description.
        description: Full task description / prompt.
        status: Current lifecycle status.
        prompt_score: Quality score of the task prompt (0–1), if scored.
        output_score: Quality score of the task output (0–1), if scored.
        confidence: Combined confidence for this task (weighted avg of both scores).
        replan_count: How many times this task has been re-attempted.
        metadata: Arbitrary extra data (e.g. model used, latency).
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    prompt_score: float | None = None
    output_score: float | None = None
    confidence: float = 0.0
    replan_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_confidence(
        self,
        prompt_weight: float = 0.35,
        output_weight: float = 0.65,
    ) -> None:
        """Recalculate confidence from available scores.

        Prompt score has lower weight — it measures intent clarity.
        Output score has higher weight — it measures actual result quality.
        If only one score is available, that score IS the confidence.
        """
        p = self.prompt_score
        o = self.output_score
        if p is not None and o is not None:
            self.confidence = p * prompt_weight + o * output_weight
        elif p is not None:
            self.confidence = p
        elif o is not None:
            self.confidence = o
        else:
            self.confidence = 0.0
        self.updated_at = time.time()


@dataclass
class Plan:
    """A collection of ordered tasks with a rolling confidence score.

    Attributes:
        plan_id: Unique identifier.
        goal: The original high-level goal / prompt.
        tasks: Ordered list of task records.
        confidence_threshold: Minimum rolling confidence to continue without replanning.
        rolling_confidence: Weighted rolling average across all scored tasks.
        replan_count: Total number of replan events for this plan.
        created_at: Creation timestamp.
        metadata: Arbitrary extra data.
    """

    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""
    tasks: list[TaskRecord] = field(default_factory=list)
    confidence_threshold: float = 0.72
    rolling_confidence: float = 1.0   # starts optimistic
    replan_count: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- task accessors ------------------------------------------------------

    def get_task(self, task_id: str) -> TaskRecord | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def pending_tasks(self) -> list[TaskRecord]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def done_tasks(self) -> list[TaskRecord]:
        return [t for t in self.tasks
                if t.status in (TaskStatus.DONE,)]

    # -- confidence engine ---------------------------------------------------

    def recalculate_confidence(self, decay: float = 0.85) -> float:
        """Recalculate rolling confidence using exponential decay weighting.

        More recent task scores have higher weight.  Tasks with no scores
        yet are skipped (they don't penalise the plan until scored).

        Args:
            decay: Weight decay factor per task (0–1). Lower = more recency bias.

        Returns:
            Updated rolling_confidence value.
        """
        scored = [t for t in self.tasks if t.confidence > 0.0]
        if not scored:
            self.rolling_confidence = 1.0
            return self.rolling_confidence

        # Exponential weighting: most recent task has weight 1.0, prior tasks decay
        weights = [decay ** (len(scored) - 1 - i) for i in range(len(scored))]
        total_weight = sum(weights)
        self.rolling_confidence = sum(
            t.confidence * w for t, w in zip(scored, weights)
        ) / total_weight
        return self.rolling_confidence

    def needs_replan(self) -> bool:
        """Return True if rolling confidence is below the threshold."""
        return self.rolling_confidence < self.confidence_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "rolling_confidence": round(self.rolling_confidence, 4),
            "confidence_threshold": self.confidence_threshold,
            "needs_replan": self.needs_replan(),
            "replan_count": self.replan_count,
            "created_at": self.created_at,
            "task_count": len(self.tasks),
            "tasks": [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "status": t.status.value,
                    "prompt_score": t.prompt_score,
                    "output_score": t.output_score,
                    "confidence": round(t.confidence, 4),
                    "replan_count": t.replan_count,
                    "metadata": t.metadata,
                }
                for t in self.tasks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        """Reconstruct a Plan from a serialised dict (e.g. loaded from store)."""
        plan = cls(
            plan_id=data["plan_id"],
            goal=data.get("goal", ""),
            confidence_threshold=data.get("confidence_threshold", 0.72),
            rolling_confidence=data.get("rolling_confidence", 1.0),
            replan_count=data.get("replan_count", 0),
            created_at=data.get("created_at", time.time()),
        )
        for t in data.get("tasks", []):
            record = TaskRecord(
                id=t["id"],
                title=t.get("title", ""),
                description=t.get("description", ""),
                status=TaskStatus(t.get("status", "pending")),
                prompt_score=t.get("prompt_score"),
                output_score=t.get("output_score"),
                confidence=t.get("confidence", 0.0),
                replan_count=t.get("replan_count", 0),
                metadata=t.get("metadata", {}),
            )
            plan.tasks.append(record)
        return plan


class PlanRegistry:
    """In-memory registry of active plans.

    Thread-safe for concurrent MCP tool calls (uses a simple dict with
    no shared mutable state between plans).

    Usage::

        registry = PlanRegistry()
        plan_id = registry.create(goal="build a parser", tasks=[...])
        registry.score_prompt(plan_id, task_id, score=0.81)
        registry.score_output(plan_id, task_id, score=0.74)
        status = registry.get_status(plan_id)
        # status["needs_replan"] → True/False
    """

    def __init__(self) -> None:
        self._plans: dict[str, Plan] = {}

    # -- plan lifecycle ------------------------------------------------------

    def create(
        self,
        goal: str,
        tasks: list[dict[str, Any]],
        confidence_threshold: float = 0.72,
    ) -> Plan:
        """Create a new plan from a goal and list of task dicts.

        Each task dict should have at least: ``title``, ``description``.
        Optional: ``id``, ``metadata``.

        Args:
            goal: High-level goal text.
            tasks: List of task attribute dicts.
            confidence_threshold: Minimum rolling confidence before replan.

        Returns:
            The created :class:`Plan`.
        """
        plan = Plan(goal=goal, confidence_threshold=confidence_threshold)
        for t in tasks:
            plan.tasks.append(TaskRecord(
                id=t.get("id", uuid.uuid4().hex[:8]),
                title=t.get("title", ""),
                description=t.get("description", ""),
                metadata=t.get("metadata", {}),
            ))
        self._plans[plan.plan_id] = plan
        return plan

    def get(self, plan_id: str) -> Plan | None:
        return self._plans.get(plan_id)

    def delete(self, plan_id: str) -> bool:
        return bool(self._plans.pop(plan_id, None))

    def list_plans(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._plans.values()]

    def restore_from_store(self, store: Any) -> int:
        """Load all persisted plans from a LoopStore into this registry.

        Should be called once at server startup.  Skips plan IDs that are
        already in memory (idempotent).

        Args:
            store: A :class:`loopllm.store.LoopStore` instance.

        Returns:
            Number of plans loaded.
        """
        loaded = 0
        for plan_dict in store.load_all_plans():
            pid = plan_dict.get("plan_id")
            if pid and pid not in self._plans:
                self._plans[pid] = Plan.from_dict(plan_dict)
                loaded += 1
        return loaded

    # -- scoring API ---------------------------------------------------------

    def score_prompt(
        self,
        plan_id: str,
        task_id: str,
        score: float,
    ) -> dict[str, Any]:
        """Record a prompt quality score for a task.

        Args:
            plan_id: Plan identifier.
            task_id: Task identifier within the plan.
            score: Prompt quality score (0–1).

        Returns:
            Updated plan status dict.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            return {"error": f"Plan not found: {plan_id}"}
        task = plan.get_task(task_id)
        if task is None:
            return {"error": f"Task not found: {task_id} in plan {plan_id}"}

        task.prompt_score = max(0.0, min(1.0, score))
        task.update_confidence()
        plan.recalculate_confidence()
        return plan.to_dict()

    def score_output(
        self,
        plan_id: str,
        task_id: str,
        score: float,
        mark_done: bool = True,
    ) -> dict[str, Any]:
        """Record an output quality score for a task.

        Args:
            plan_id: Plan identifier.
            task_id: Task identifier within the plan.
            score: Output quality score (0–1).
            mark_done: If True and score >= threshold, mark task as DONE.

        Returns:
            Updated plan status dict.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            return {"error": f"Plan not found: {plan_id}"}
        task = plan.get_task(task_id)
        if task is None:
            return {"error": f"Task not found: {task_id} in plan {plan_id}"}

        task.output_score = max(0.0, min(1.0, score))
        task.update_confidence()
        plan.recalculate_confidence()

        if mark_done:
            if task.confidence >= plan.confidence_threshold:
                task.status = TaskStatus.DONE
            else:
                task.status = TaskStatus.FAILED

        # Trigger replan bookkeeping if needed
        if plan.needs_replan():
            plan.replan_count += 1
            task.replan_count += 1
            task.status = TaskStatus.REPLANNING

        return plan.to_dict()

    def mark_task(
        self,
        plan_id: str,
        task_id: str,
        status: str,
    ) -> dict[str, Any]:
        """Manually set a task's status."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return {"error": f"Plan not found: {plan_id}"}
        task = plan.get_task(task_id)
        if task is None:
            return {"error": f"Task not found: {task_id}"}
        try:
            task.status = TaskStatus(status)
        except ValueError:
            return {"error": f"Unknown status: {status}"}
        task.updated_at = time.time()
        return plan.to_dict()

    def get_status(self, plan_id: str) -> dict[str, Any]:
        plan = self._plans.get(plan_id)
        if plan is None:
            return {"error": f"Plan not found: {plan_id}"}
        return plan.to_dict()

    def next_task(self, plan_id: str) -> dict[str, Any] | None:
        """Return the next pending task, or None if the plan is complete/blocked."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return None
        pending = plan.pending_tasks()
        if not pending:
            return None
        t = pending[0]
        t.status = TaskStatus.IN_PROGRESS
        t.updated_at = time.time()
        return {
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "replan_count": t.replan_count,
            "needs_replan": plan.needs_replan(),
            "rolling_confidence": round(plan.rolling_confidence, 4),
        }


# ---------------------------------------------------------------------------
# Process-level singleton — shared across all MCP tool calls
# ---------------------------------------------------------------------------

_registry: PlanRegistry = PlanRegistry()


def get_registry() -> PlanRegistry:
    """Return the global PlanRegistry instance."""
    return _registry
