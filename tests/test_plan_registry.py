"""Tests for PlanRegistry, Plan, and TaskRecord."""
from __future__ import annotations

import pytest

from loopllm.plan_registry import (
    Plan,
    PlanRegistry,
    TaskRecord,
    TaskStatus,
    get_registry,
)


# ---------------------------------------------------------------------------
# TaskRecord
# ---------------------------------------------------------------------------

class TestTaskRecord:
    def test_default_id_generated(self):
        t = TaskRecord()
        assert t.id and len(t.id) == 8

    def test_update_confidence_both_scores(self):
        t = TaskRecord(prompt_score=0.8, output_score=0.6)
        t.update_confidence()
        # 0.8*0.35 + 0.6*0.65 = 0.28 + 0.39 = 0.67
        assert abs(t.confidence - 0.67) < 1e-9

    def test_update_confidence_prompt_only(self):
        t = TaskRecord(prompt_score=0.9)
        t.update_confidence()
        assert t.confidence == 0.9

    def test_update_confidence_output_only(self):
        t = TaskRecord(output_score=0.75)
        t.update_confidence()
        assert t.confidence == 0.75

    def test_update_confidence_no_scores(self):
        t = TaskRecord()
        t.update_confidence()
        assert t.confidence == 0.0

    def test_custom_weights(self):
        t = TaskRecord(prompt_score=1.0, output_score=0.0)
        t.update_confidence(prompt_weight=0.5, output_weight=0.5)
        assert abs(t.confidence - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class TestPlan:
    def _make_plan(self):
        p = Plan(goal="test", confidence_threshold=0.70)
        for i in range(3):
            p.tasks.append(TaskRecord(title=f"task-{i}"))
        return p

    def test_plan_starts_optimistic(self):
        p = Plan()
        assert p.rolling_confidence == 1.0

    def test_get_task(self):
        p = self._make_plan()
        tid = p.tasks[1].id
        assert p.get_task(tid) is p.tasks[1]

    def test_get_task_missing(self):
        p = self._make_plan()
        assert p.get_task("nonexistent") is None

    def test_pending_tasks_all_pending_initially(self):
        p = self._make_plan()
        assert len(p.pending_tasks()) == 3

    def test_done_tasks_empty_initially(self):
        p = self._make_plan()
        assert p.done_tasks() == []

    def test_needs_replan_false_when_above_threshold(self):
        p = Plan(confidence_threshold=0.70)
        p.rolling_confidence = 0.85
        assert p.needs_replan() is False

    def test_needs_replan_true_when_below_threshold(self):
        p = Plan(confidence_threshold=0.70)
        p.rolling_confidence = 0.50
        assert p.needs_replan() is True

    def test_recalculate_confidence_no_scored_tasks(self):
        p = self._make_plan()
        conf = p.recalculate_confidence()
        assert conf == 1.0

    def test_recalculate_confidence_single_task(self):
        p = Plan()
        t = TaskRecord(prompt_score=0.80, output_score=0.60)
        t.update_confidence()
        p.tasks.append(t)
        conf = p.recalculate_confidence()
        # single task → its confidence = 0.8*0.35 + 0.6*0.65 = 0.67
        assert abs(conf - 0.67) < 1e-9

    def test_recalculate_confidence_two_tasks_decay(self):
        p = Plan()
        for ps, os_ in [(0.9, 0.9), (0.3, 0.3)]:
            t = TaskRecord(prompt_score=ps, output_score=os_)
            t.update_confidence()
            p.tasks.append(t)
        conf = p.recalculate_confidence(decay=0.85)
        # scored: [t0 (conf≈0.9), t1 (conf≈0.3)]
        # weights: [0.85, 1.0], total=1.85
        # rolling = (0.9*0.85 + 0.3*1.0) / 1.85
        w0, w1 = 0.85, 1.0
        t0_conf = 0.9 * 0.35 + 0.9 * 0.65  # = 0.9
        t1_conf = 0.3 * 0.35 + 0.3 * 0.65  # = 0.3
        expected = (t0_conf * w0 + t1_conf * w1) / (w0 + w1)
        assert abs(conf - expected) < 1e-9

    def test_to_dict_structure(self):
        p = self._make_plan()
        d = p.to_dict()
        assert "plan_id" in d
        assert "goal" in d
        assert "rolling_confidence" in d
        assert "needs_replan" in d
        assert len(d["tasks"]) == 3


# ---------------------------------------------------------------------------
# PlanRegistry
# ---------------------------------------------------------------------------

class TestPlanRegistry:
    def _make_registry(self) -> PlanRegistry:
        return PlanRegistry()

    def _tasks(self):
        return [
            {"title": "parse JSON", "description": "Parse the input"},
            {"title": "write tests", "description": "Write pytest tests"},
        ]

    def test_create_returns_plan(self):
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks())
        assert plan.plan_id in reg._plans

    def test_get_returns_plan(self):
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks())
        assert reg.get(plan.plan_id) is plan

    def test_get_missing_returns_none(self):
        reg = self._make_registry()
        assert reg.get("nope") is None

    def test_delete_removes_plan(self):
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks())
        assert reg.delete(plan.plan_id) is True
        assert reg.get(plan.plan_id) is None

    def test_delete_missing_returns_false(self):
        reg = self._make_registry()
        assert reg.delete("ghost") is False

    def test_score_prompt_updates_task(self):
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks())
        tid = plan.tasks[0].id
        result = reg.score_prompt(plan.plan_id, tid, score=0.85)
        assert result["rolling_confidence"] == pytest.approx(0.85)
        assert plan.tasks[0].prompt_score == 0.85

    def test_score_output_updates_task(self):
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks())
        tid = plan.tasks[0].id
        reg.score_prompt(plan.plan_id, tid, score=0.85)
        result = reg.score_output(plan.plan_id, tid, score=0.55)
        # confidence = 0.85*0.35 + 0.55*0.65 = 0.2975 + 0.3575 = 0.655
        expected = 0.85 * 0.35 + 0.55 * 0.65
        assert result["rolling_confidence"] == pytest.approx(expected)

    def test_needs_replan_true_when_low(self):
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks(), confidence_threshold=0.72)
        tid = plan.tasks[0].id
        reg.score_prompt(plan.plan_id, tid, score=0.85)
        result = reg.score_output(plan.plan_id, tid, score=0.55)
        assert result["needs_replan"] is True

    def test_next_task_returns_first_pending(self):
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks())
        tid = plan.tasks[0].id
        reg.score_prompt(plan.plan_id, tid, score=0.9)

        nxt = reg.next_task(plan.plan_id)
        assert nxt is not None
        assert nxt["id"] == plan.tasks[0].id  # still pending (output not given yet)

    def test_mark_task_done_then_next_advances(self):
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks())
        tid0 = plan.tasks[0].id
        # mark first task done
        plan.tasks[0].status = TaskStatus.DONE
        nxt = reg.next_task(plan.plan_id)
        assert nxt is not None
        assert nxt["id"] == plan.tasks[1].id

    def test_all_done_returns_none(self):
        # next_task returns None when no tasks are pending (all done)
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks())
        for t in plan.tasks:
            t.status = TaskStatus.DONE
        nxt = reg.next_task(plan.plan_id)
        assert nxt is None

    def test_score_prompt_missing_plan_returns_error(self):
        # score_prompt returns an error dict for unknown plan IDs
        reg = self._make_registry()
        result = reg.score_prompt("ghost", "task", score=0.5)
        assert "error" in result

    def test_score_output_missing_task_returns_error(self):
        # score_output returns an error dict for unknown task IDs
        reg = self._make_registry()
        plan = reg.create("goal", self._tasks())
        result = reg.score_output(plan.plan_id, "ghost", score=0.5)
        assert "error" in result

    def test_list_plans(self):
        reg = self._make_registry()
        reg.create("goal A", self._tasks())
        reg.create("goal B", self._tasks())
        plans = reg.list_plans()
        assert len(plans) == 2


# ---------------------------------------------------------------------------
# get_registry singleton
# ---------------------------------------------------------------------------

class TestGetRegistry:
    def test_singleton_returns_same_instance(self):
        a = get_registry()
        b = get_registry()
        assert a is b

    def test_singleton_persists_plans(self):
        reg = get_registry()
        plan = reg.create("persistent goal", [{"title": "t", "description": "d"}])
        assert get_registry().get(plan.plan_id) is plan
        # cleanup so other tests are not affected
        reg.delete(plan.plan_id)
