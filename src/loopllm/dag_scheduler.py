"""DAG scheduler for IDE-compatible virtual sub-agents.

One IDE agent acts as the worker; loopllm schedules dependency-ordered nodes,
scores each submission via CDV, and merges verified outputs.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from loopllm.episodes import EpisodicStore, artifact_ref_hash, summarize_artifacts
from loopllm.step_scorer import build_step_evaluator, score_channel_a

logger = structlog.get_logger(__name__)

_DEFAULT_PASS_SCORE = 0.7


class NodeState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass
class DagNode:
    """One virtual sub-agent node in a DAG run."""

    id: str
    role: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    state: NodeState = NodeState.PENDING
    evaluator_type: str = "composite"
    evaluator_kwargs: dict[str, Any] = field(default_factory=dict)
    quality_criteria: list[str] = field(default_factory=list)
    verified_output: str = ""
    verified_score: float = 0.0
    deficiencies: list[str] = field(default_factory=list)


@dataclass
class DagRun:
    """In-memory DAG execution state."""

    run_id: str
    goal: str
    task_type: str
    model_id: str
    nodes: dict[str, DagNode] = field(default_factory=dict)
    merged_output: str = ""
    closed: bool = False

    def dependency_graph(self) -> dict[str, list[str]]:
        return {nid: list(node.dependencies) for nid, node in self.nodes.items()}

    def execution_order(self) -> list[DagNode]:
        """Topological order; raises ValueError on cycles."""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Unknown dependency {dep} for node {nid}")
                adj[dep].append(nid)
                in_degree[nid] += 1
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        ordered: list[DagNode] = []
        while queue:
            nid = queue.pop(0)
            ordered.append(self.nodes[nid])
            for child in adj[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(self.nodes):
            raise ValueError("Cycle detected in DAG dependencies")
        return ordered


class DagScheduler:
    """Compile, dispatch, verify, and merge DAG virtual sub-agents."""

    def __init__(self, episodic: EpisodicStore | None = None) -> None:
        self._runs: dict[str, DagRun] = {}
        self._episodic = episodic

    def compile(
        self,
        goal: str,
        nodes_spec: list[dict[str, Any]],
        *,
        task_type: str = "general",
        model_id: str = "unknown",
        recall_query: str | None = None,
    ) -> DagRun:
        """Build a DAG run from a list of node specifications."""
        run_id = uuid.uuid4().hex[:12]
        title_to_id: dict[str, str] = {}
        nodes: dict[str, DagNode] = {}

        for i, spec in enumerate(nodes_spec):
            nid = str(spec.get("id") or f"n{i + 1}")
            title = str(spec.get("title") or nid)
            title_to_id[title] = nid
            nodes[nid] = DagNode(
                id=nid,
                role=str(spec.get("role") or "implementer"),
                description=str(spec.get("description") or spec.get("title") or goal),
                dependencies=[str(d) for d in spec.get("dependencies", [])],
                evaluator_type=str(spec.get("evaluator_type") or "composite"),
                evaluator_kwargs=dict(spec.get("evaluator_kwargs") or {}),
                quality_criteria=list(spec.get("quality_criteria") or []),
            )
            if spec.get("required_patterns"):
                nodes[nid].evaluator_kwargs["required_patterns"] = list(
                    spec["required_patterns"]
                )

        # Resolve title-based dependencies to ids
        for node in nodes.values():
            resolved: list[str] = []
            for dep in node.dependencies:
                resolved.append(title_to_id.get(dep, dep))
            node.dependencies = resolved

        run = DagRun(
            run_id=run_id,
            goal=goal,
            task_type=task_type,
            model_id=model_id,
            nodes=nodes,
        )
        self._refresh_ready_states(run)
        self._runs[run_id] = run

        if self._episodic is not None:
            self._episodic.upsert_active_run(
                run_id,
                "dag",
                self.to_dict(run_id),
            )

        recall_context: list[dict[str, Any]] = []
        if self._episodic is not None and recall_query:
            recall_context = self._episodic.recall(
                recall_query or goal,
                task_type=task_type,
                k=3,
            )

        logger.info(
            "dag_compile",
            run_id=run_id,
            node_count=len(nodes),
            recall_hits=len(recall_context),
        )
        return run

    def ready(self, run_id: str) -> list[dict[str, Any]]:
        """Return frontier nodes with scoped worker prompts."""
        run = self._require(run_id)
        if run.closed:
            return []
        self._refresh_ready_states(run)
        frontier: list[dict[str, Any]] = []
        for node in run.nodes.values():
            if node.state != NodeState.READY:
                continue
            inputs = {
                dep_id: run.nodes[dep_id].verified_output
                for dep_id in node.dependencies
                if dep_id in run.nodes
            }
            frontier.append(
                {
                    "node_id": node.id,
                    "role": node.role,
                    "instructions": (
                        f"You are node {node.id} only ({node.role}). "
                        f"Do not replan the whole goal. Complete this node only."
                    ),
                    "description": node.description,
                    "inputs": inputs,
                    "evaluator": {
                        "evaluator_type": node.evaluator_type,
                        **node.evaluator_kwargs,
                        "quality_criteria": node.quality_criteria,
                    },
                }
            )
        return frontier

    def submit(
        self,
        run_id: str,
        node_id: str,
        step_output: str,
        *,
        pass_threshold: float = _DEFAULT_PASS_SCORE,
    ) -> dict[str, Any]:
        """Score a node artifact (Channel A) and mark verified or failed."""
        run = self._require(run_id)
        if run.closed:
            raise ValueError(f"DAG run already closed: {run_id}")
        if node_id not in run.nodes:
            raise KeyError(f"Unknown node: {node_id}")

        node = run.nodes[node_id]
        if node.state not in (NodeState.READY, NodeState.RUNNING):
            return {
                "run_id": run_id,
                "node_id": node_id,
                "accepted": False,
                "reason": f"Node not ready (state={node.state.value})",
            }

        node.state = NodeState.RUNNING
        evaluator = build_step_evaluator(
            node.evaluator_type,
            node.quality_criteria,
            **node.evaluator_kwargs,
        )
        node_goal = f"{run.goal} — node {node.id}: {node.description[:200]}"
        result = score_channel_a(step_output, node_goal, evaluator)
        passed = result.score >= pass_threshold and result.passed

        if passed:
            node.state = NodeState.VERIFIED
            node.verified_output = step_output
            node.verified_score = result.score
            node.deficiencies = list(result.deficiencies)
            self._refresh_ready_states(run)

            if self._episodic is not None:
                self._episodic.record_episode(
                    episode_type="plan_node",
                    goal=f"{run.goal} [{node.id}]",
                    task_type=run.task_type,
                    model_id=run.model_id,
                    summary=summarize_artifacts(
                        run.goal,
                        output=step_output,
                        score_final=result.score,
                    ),
                    artifact_ref=artifact_ref_hash(step_output),
                    score_final=result.score,
                    stop_reason="node_verified",
                )
        else:
            node.state = NodeState.FAILED
            node.deficiencies = list(result.deficiencies)

        self._persist_active(run)

        all_verified = all(
            n.state == NodeState.VERIFIED for n in run.nodes.values()
        )
        return {
            "run_id": run_id,
            "node_id": node_id,
            "accepted": passed,
            "score": round(result.score, 4),
            "deficiencies": list(result.deficiencies),
            "node_state": node.state.value,
            "dag_complete": all_verified,
            "ready_next": self.ready(run_id),
        }

    async def submit_async(
        self,
        run_id: str,
        node_id: str,
        step_output: str,
        ctx: Any | None = None,
        *,
        pass_threshold: float = _DEFAULT_PASS_SCORE,
    ) -> dict[str, Any]:
        """Submit with optional full CDV when *ctx* supports sampling."""
        run = self._require(run_id)
        if node_id not in run.nodes:
            raise KeyError(f"Unknown node: {node_id}")
        node = run.nodes[node_id]

        if ctx is not None and step_output:
            evaluator = build_step_evaluator(
                node.evaluator_type,
                node.quality_criteria,
                **node.evaluator_kwargs,
            )
            from loopllm.step_scorer import conservative_dual_verify

            dual = await conservative_dual_verify(
                step_output=step_output,
                goal=f"{run.goal} — {node.description[:200]}",
                quality_criteria=node.quality_criteria,
                evaluator=evaluator,
                ctx=ctx,
            )
            if dual.final_score < pass_threshold:
                node.state = NodeState.FAILED
                node.deficiencies = list(dual.deficiencies)
                self._persist_active(run)
                out = dual.to_dict()
                out.update({
                    "run_id": run_id,
                    "node_id": node_id,
                    "accepted": False,
                    "node_state": node.state.value,
                })
                return out

        return self.submit(
            run_id,
            node_id,
            step_output,
            pass_threshold=pass_threshold,
        )

    def merge(self, run_id: str) -> dict[str, Any]:
        """Stitch verified node outputs in topological order."""
        run = self._require(run_id)
        parts: list[str] = []
        for node in run.execution_order():
            if node.state != NodeState.VERIFIED:
                return {
                    "run_id": run_id,
                    "error": f"Node {node.id} not verified (state={node.state.value})",
                }
            parts.append(f"--- {node.id} ({node.role}) ---\n{node.verified_output}")

        run.merged_output = "\n\n".join(parts)
        run.closed = True

        if self._episodic is not None:
            self._episodic.record_episode(
                episode_type="dag",
                goal=run.goal,
                task_type=run.task_type,
                model_id=run.model_id,
                summary=summarize_artifacts(
                    run.goal,
                    output=run.merged_output[:300],
                    score_final=1.0,
                ),
                artifact_ref=artifact_ref_hash(run.merged_output),
                score_final=1.0,
                steps_used=len(run.nodes),
                stop_reason="dag_merged",
            )
            self._episodic.clear_active_run(run_id)

        return {
            "run_id": run_id,
            "merged_output": run.merged_output,
            "node_count": len(run.nodes),
        }

    def status(self, run_id: str) -> dict[str, Any]:
        """Full graph state for debugging."""
        return self.to_dict(run_id)

    def to_dict(self, run_id: str) -> dict[str, Any]:
        run = self._require(run_id)
        return {
            "run_id": run.run_id,
            "goal": run.goal,
            "task_type": run.task_type,
            "model_id": run.model_id,
            "closed": run.closed,
            "merged_output": run.merged_output,
            "nodes": {
                nid: {
                    "id": n.id,
                    "role": n.role,
                    "description": n.description,
                    "dependencies": n.dependencies,
                    "state": n.state.value,
                    "verified_score": round(n.verified_score, 4),
                    "verified_output": n.verified_output,
                    "deficiencies": n.deficiencies,
                    "evaluator_type": n.evaluator_type,
                    "evaluator_kwargs": n.evaluator_kwargs,
                    "quality_criteria": n.quality_criteria,
                }
                for nid, n in run.nodes.items()
            },
            "ready": [item["node_id"] for item in self.ready(run_id)],
        }

    def restore(self, state: dict[str, Any]) -> DagRun:
        """Restore a DAG run from serialized state (active_run recovery)."""
        run_id = state["run_id"]
        nodes: dict[str, DagNode] = {}
        for nid, raw in state.get("nodes", {}).items():
            nodes[nid] = DagNode(
                id=nid,
                role=raw.get("role", "implementer"),
                description=raw.get("description", ""),
                dependencies=list(raw.get("dependencies", [])),
                state=NodeState(raw.get("state", "pending")),
                evaluator_type=raw.get("evaluator_type", "composite"),
                evaluator_kwargs=dict(raw.get("evaluator_kwargs", {})),
                quality_criteria=list(raw.get("quality_criteria", [])),
                verified_output=raw.get("verified_output", ""),
                verified_score=float(raw.get("verified_score", 0.0)),
                deficiencies=list(raw.get("deficiencies", [])),
            )
        run = DagRun(
            run_id=run_id,
            goal=state.get("goal", ""),
            task_type=state.get("task_type", "general"),
            model_id=state.get("model_id", "unknown"),
            nodes=nodes,
            merged_output=state.get("merged_output", ""),
            closed=bool(state.get("closed", False)),
        )
        self._runs[run_id] = run
        return run

    def _refresh_ready_states(self, run: DagRun) -> None:
        for node in run.nodes.values():
            if node.state in (NodeState.VERIFIED, NodeState.FAILED):
                continue
            deps_ok = all(
                run.nodes[d].state == NodeState.VERIFIED
                for d in node.dependencies
            )
            if deps_ok and node.state == NodeState.PENDING:
                node.state = NodeState.READY

    def _persist_active(self, run: DagRun) -> None:
        if self._episodic is not None:
            self._episodic.upsert_active_run(run.run_id, "dag", self.to_dict(run.run_id))

    def _require(self, run_id: str) -> DagRun:
        if run_id not in self._runs:
            raise KeyError(f"Unknown DAG run: {run_id}")
        return self._runs[run_id]


def parse_nodes_json(raw: str) -> list[dict[str, Any]]:
    """Parse LLM JSON array of node specs."""
    text = raw.strip()
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        else:
            return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []
