"""Episodic memory: record and recall completed loop/plan outcomes.

Recall ranking is deterministic weighted keyword overlap (see
``store.search_episodes``). The public ``EpisodicStore.recall`` /
``store.search_episodes`` signature is stable, so the ranking backend can be
upgraded to SQLite FTS5 (a virtual table over goal+summary+tags, no new deps)
or an embedding index later without touching callers — deferred to v0.10.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from loopllm.store import LoopStore, _recall_terms

_MAX_SUMMARY = 500


def extract_tags(text: str, task_type: str) -> list[str]:
    """Derive searchable tags from goal text and task_type."""
    tags = [task_type] if task_type else []
    lower = text.lower()
    for keyword in (
        "pytest", "test", "bugfix", "refactor", "auth", "api",
        "json", "regex", "docker", "sql", "migrate",
    ):
        if keyword in lower and keyword not in tags:
            tags.append(keyword)
    return tags


def summarize_artifacts(
    goal: str,
    *,
    step_outputs: list[str] | None = None,
    output: str = "",
    stop_reason: str = "",
    score_final: float | None = None,
) -> str:
    """Build a compact episode summary without an LLM call."""
    parts: list[str] = [f"Goal: {goal[:120]}"]
    if score_final is not None:
        parts.append(f"score={score_final:.3f}")
    if stop_reason:
        parts.append(f"stop={stop_reason[:80]}")
    artifact = output or (step_outputs[-1] if step_outputs else "")
    if artifact:
        snippet = artifact.strip().replace("\n", " ")[:200]
        parts.append(f"artifact: {snippet}")
    summary = " | ".join(parts)
    return summary[:_MAX_SUMMARY]


def artifact_ref_hash(content: str) -> str:
    """Stable short hash pointer for an artifact body."""
    if not content:
        return ""
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"sha256:{digest}"


class EpisodicStore:
    """Record and recall episodes backed by :class:`LoopStore`."""

    def __init__(self, store: LoopStore, mirror_path: Path | None = None) -> None:
        self._store = store
        self._mirror_path = mirror_path

    def record_episode(
        self,
        *,
        episode_type: str,
        goal: str,
        task_type: str,
        model_id: str,
        summary: str,
        artifact_ref: str | None = None,
        tags: list[str] | None = None,
        score_final: float | None = None,
        steps_used: int | None = None,
        stop_reason: str | None = None,
    ) -> int:
        """Persist one episode row."""
        merged_tags = list(tags or [])
        for t in extract_tags(goal, task_type):
            if t not in merged_tags:
                merged_tags.append(t)
        return self._store.insert_episode(
            episode_type=episode_type,
            goal=goal,
            task_type=task_type,
            model_id=model_id,
            summary=summary[:_MAX_SUMMARY],
            artifact_ref=artifact_ref,
            tags=merged_tags,
            score_final=score_final,
            steps_used=steps_used,
            stop_reason=stop_reason,
        )

    def recall(
        self,
        query: str,
        *,
        task_type: str | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Keyword recall over past episodes."""
        return self._store.search_episodes(query, task_type=task_type, limit=k)

    def upsert_active_run(
        self,
        run_id: str,
        run_type: str,
        state: dict[str, Any],
    ) -> None:
        """Persist in-progress run; optionally mirror to JSON file."""
        self._store.upsert_active_run(run_id, run_type, state)
        if self._mirror_path is not None:
            self._mirror_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_id": run_id,
                "run_type": run_type,
                "state": state,
            }
            self._mirror_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear_active_run(self, run_id: str) -> None:
        """Remove active run after completion."""
        self._store.delete_active_run(run_id)
        if self._mirror_path is not None and self._mirror_path.exists():
            try:
                existing = json.loads(self._mirror_path.read_text(encoding="utf-8"))
                if existing.get("run_id") == run_id:
                    self._mirror_path.unlink()
            except (json.JSONDecodeError, OSError):
                pass

    def get_active_run(self, run_id: str) -> dict[str, Any] | None:
        return self._store.get_active_run(run_id)

    def list_active_runs(self, run_type: str | None = None) -> list[dict[str, Any]]:
        return self._store.list_active_runs(run_type=run_type)

    def run_status_snapshot(self) -> dict[str, Any]:
        """Return all active runs for IDE reload recovery."""
        runs = self.list_active_runs()
        return {"active_runs": runs, "count": len(runs)}


def tokenize_for_recall(text: str) -> list[str]:
    """Split text into recall terms (stopword-filtered, deduped)."""
    return _recall_terms(text)
