"""SQLite-backed persistence for priors, observations, and sessions."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import structlog

from loopllm.priors import (
    AdaptivePriors,
    BetaPrior,
    CallObservation,
    IterationProfile,
    NormalPrior,
    TaskModelPrior,
)

logger = structlog.get_logger(__name__)

SCHEMA_VERSION = 4

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS priors (
    key         TEXT PRIMARY KEY,
    task_type   TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    data        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type   TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    data        TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question_type   TEXT NOT NULL,
    task_type       TEXT NOT NULL,
    asked_count     INTEGER NOT NULL DEFAULT 0,
    positive_impact INTEGER NOT NULL DEFAULT 0,
    negative_impact INTEGER NOT NULL DEFAULT 0,
    avg_info_gain   REAL NOT NULL DEFAULT 0.0,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    original_prompt TEXT NOT NULL,
    task_type       TEXT,
    model_id        TEXT,
    questions_json  TEXT NOT NULL DEFAULT '[]',
    answers_json    TEXT NOT NULL DEFAULT '{}',
    spec_json       TEXT,
    final_score     REAL,
    created_at      TEXT NOT NULL,
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    parent_id       TEXT,
    session_id      TEXT,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    state           TEXT NOT NULL DEFAULT 'pending',
    dependencies    TEXT NOT NULL DEFAULT '[]',
    spec_json       TEXT,
    result_json     TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES tasks(id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_observations_task_model
    ON observations(task_type, model_id);
CREATE INDEX IF NOT EXISTS idx_questions_type
    ON questions(question_type, task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_session
    ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_state
    ON tasks(state);
"""


_SCHEMA_V2_SQL = """\
CREATE TABLE IF NOT EXISTS prompt_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    prompt_text     TEXT NOT NULL,
    quality_score   REAL NOT NULL,
    specificity     REAL NOT NULL DEFAULT 0.0,
    constraint_clarity REAL NOT NULL DEFAULT 0.0,
    context_completeness REAL NOT NULL DEFAULT 0.0,
    ambiguity       REAL NOT NULL DEFAULT 0.0,
    format_spec     REAL NOT NULL DEFAULT 0.0,
    task_type       TEXT NOT NULL DEFAULT 'general',
    complexity      REAL NOT NULL DEFAULT 0.0,
    route_chosen    TEXT NOT NULL DEFAULT 'refine',
    word_count      INTEGER NOT NULL DEFAULT 0,
    grade           TEXT NOT NULL DEFAULT 'C',
    session_context TEXT NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_prompt_history_ts
    ON prompt_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_prompt_history_grade
    ON prompt_history(grade);
"""

_SCHEMA_V3_SQL = """\
CREATE TABLE IF NOT EXISTS plans (
    plan_id         TEXT PRIMARY KEY,
    goal            TEXT NOT NULL,
    data            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""

_SCHEMA_V4_SQL = """\
CREATE TABLE IF NOT EXISTS learned_weights (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    weights     TEXT NOT NULL,
    n_updates   INTEGER NOT NULL DEFAULT 0,
    last_loss   REAL NOT NULL DEFAULT 0.0,
    updated_at  TEXT NOT NULL
);
"""


class LoopStore:
    """SQLite-backed store for loop-llm state.

    Thread-safe via a reentrant lock around all database operations.
    Uses WAL journal mode for concurrent read performance.

    Args:
        db_path: Path to the SQLite database file.  Use ``":memory:"``
            for an ephemeral in-memory store (useful for testing).
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # -- connection management -----------------------------------------------

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a thread-safe connection with WAL mode enabled."""
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(self.db_path)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.row_factory = sqlite3.Row
            yield self._conn

    def _ensure_schema(self) -> None:
        """Create tables or run migrations as needed."""
        with self._connection() as conn:
            # Check if schema_version table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                conn.executescript(_SCHEMA_SQL)
                conn.executescript(_SCHEMA_V2_SQL)
                conn.executescript(_SCHEMA_V3_SQL)
                conn.executescript(_SCHEMA_V4_SQL)
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
                conn.commit()
                logger.debug("store_schema_created", version=SCHEMA_VERSION)
            else:
                row = conn.execute("SELECT version FROM schema_version").fetchone()
                current = row["version"] if row else 0
                if current < SCHEMA_VERSION:
                    self._migrate(conn, current, SCHEMA_VERSION)

    def _migrate(self, conn: sqlite3.Connection, from_v: int, to_v: int) -> None:
        """Run schema migrations from *from_v* to *to_v*.

        Args:
            conn: Active database connection.
            from_v: Current schema version.
            to_v: Target schema version.
        """
        logger.info("store_migrating", from_version=from_v, to_version=to_v)
        if from_v < 2:
            conn.executescript(_SCHEMA_V2_SQL)
        if from_v < 3:
            conn.executescript(_SCHEMA_V3_SQL)
        if from_v < 4:
            conn.executescript(_SCHEMA_V4_SQL)
        conn.execute("UPDATE schema_version SET version = ?", (to_v,))
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- priors CRUD ---------------------------------------------------------

    def save_prior(self, key: str, prior: TaskModelPrior) -> None:
        """Upsert a serialised :class:`TaskModelPrior`.

        Args:
            key: Storage key (typically ``task_type::model_id``).
            prior: The prior to persist.
        """
        data = self._serialize_task_model_prior(prior)
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO priors (key, task_type, model_id, data, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       data = excluded.data,
                       updated_at = excluded.updated_at""",
                (key, prior.task_type, prior.model_id, json.dumps(data), now, now),
            )
            conn.commit()

    def load_prior(self, key: str) -> TaskModelPrior | None:
        """Load a :class:`TaskModelPrior` by key.

        Args:
            key: Storage key (typically ``task_type::model_id``).

        Returns:
            The deserialized prior, or ``None`` if not found.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT data FROM priors WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return self._deserialize_task_model_prior(json.loads(row["data"]))

    def load_all_priors(self) -> dict[str, TaskModelPrior]:
        """Load every stored prior.

        Returns:
            Dict mapping keys to :class:`TaskModelPrior` instances.
        """
        with self._connection() as conn:
            rows = conn.execute("SELECT key, data FROM priors").fetchall()
        result: dict[str, TaskModelPrior] = {}
        for row in rows:
            result[row["key"]] = self._deserialize_task_model_prior(
                json.loads(row["data"])
            )
        return result

    def delete_prior(self, key: str) -> bool:
        """Delete a prior by key.

        Args:
            key: Storage key to delete.

        Returns:
            True if a row was deleted.
        """
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM priors WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0

    # -- observations --------------------------------------------------------

    def record_observation(self, obs: CallObservation) -> int:
        """Append an observation to the log.

        Args:
            obs: The observation to record.

        Returns:
            The auto-generated row ID.
        """
        now = datetime.now(timezone.utc).isoformat()
        data = asdict(obs)
        with self._connection() as conn:
            cursor = conn.execute(
                """INSERT INTO observations (task_type, model_id, data, recorded_at)
                   VALUES (?, ?, ?, ?)""",
                (obs.task_type, obs.model_id, json.dumps(data), now),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_observations(
        self,
        task_type: str | None = None,
        model_id: str | None = None,
        limit: int = 100,
    ) -> list[CallObservation]:
        """Query observations with optional filters.

        Args:
            task_type: Filter by task type (optional).
            model_id: Filter by model ID (optional).
            limit: Maximum rows to return.

        Returns:
            List of :class:`CallObservation` instances, most recent first.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if task_type is not None:
            clauses.append("task_type = ?")
            params.append(task_type)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT data FROM observations {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        results: list[CallObservation] = []
        for row in rows:
            d = json.loads(row["data"])
            results.append(CallObservation(**d))
        return results

    def count_observations(
        self, task_type: str | None = None, model_id: str | None = None
    ) -> int:
        """Count observations with optional filters.

        Args:
            task_type: Filter by task type (optional).
            model_id: Filter by model ID (optional).

        Returns:
            Row count.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if task_type is not None:
            clauses.append("task_type = ?")
            params.append(task_type)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT COUNT(*) as cnt FROM observations {where}"

        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()
        return row["cnt"] if row else 0

    # -- question effectiveness tracking -------------------------------------

    def update_question_stats(
        self,
        question_type: str,
        task_type: str,
        *,
        positive: bool,
        info_gain: float = 0.0,
    ) -> None:
        """Update effectiveness statistics for a question type.

        Args:
            question_type: Category of the question (e.g. ``"scope"``).
            task_type: Task type context in which it was asked.
            positive: Whether the question had a positive impact on outcome.
            info_gain: Measured information gain from the answer.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            existing = conn.execute(
                "SELECT id, asked_count, positive_impact, negative_impact, avg_info_gain "
                "FROM questions WHERE question_type = ? AND task_type = ?",
                (question_type, task_type),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """INSERT INTO questions
                       (question_type, task_type, asked_count, positive_impact,
                        negative_impact, avg_info_gain, updated_at)
                       VALUES (?, ?, 1, ?, ?, ?, ?)""",
                    (
                        question_type,
                        task_type,
                        1 if positive else 0,
                        0 if positive else 1,
                        info_gain,
                        now,
                    ),
                )
            else:
                new_count = existing["asked_count"] + 1
                new_pos = existing["positive_impact"] + (1 if positive else 0)
                new_neg = existing["negative_impact"] + (0 if positive else 1)
                # Running average of info gain
                old_avg = existing["avg_info_gain"]
                new_avg = old_avg + (info_gain - old_avg) / new_count
                conn.execute(
                    """UPDATE questions SET
                           asked_count = ?, positive_impact = ?, negative_impact = ?,
                           avg_info_gain = ?, updated_at = ?
                       WHERE id = ?""",
                    (new_count, new_pos, new_neg, new_avg, now, existing["id"]),
                )
            conn.commit()

    def get_question_stats(
        self, task_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve question effectiveness statistics.

        Args:
            task_type: Optional filter by task type.

        Returns:
            List of dicts with question stats.
        """
        if task_type is not None:
            query = "SELECT * FROM questions WHERE task_type = ? ORDER BY avg_info_gain DESC"
            params: tuple[Any, ...] = (task_type,)
        else:
            query = "SELECT * FROM questions ORDER BY avg_info_gain DESC"
            params = ()

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "question_type": row["question_type"],
                "task_type": row["task_type"],
                "asked_count": row["asked_count"],
                "positive_impact": row["positive_impact"],
                "negative_impact": row["negative_impact"],
                "avg_info_gain": row["avg_info_gain"],
                "effectiveness": (
                    row["positive_impact"] / row["asked_count"]
                    if row["asked_count"] > 0
                    else 0.0
                ),
            }
            for row in rows
        ]

    # -- learned weights (online SGD) ----------------------------------------

    _DEFAULT_WEIGHTS = {
        "specificity": 0.25,
        "constraint_clarity": 0.20,
        "context_completeness": 0.20,
        "ambiguity": 0.20,
        "format_spec": 0.15,
    }

    def save_learned_weights(
        self,
        weights: dict[str, float],
        n_updates: int,
        last_loss: float,
    ) -> None:
        """Persist learned scoring weights.

        Args:
            weights: Dimension-name → weight mapping (must sum to ~1.0).
            n_updates: Cumulative number of SGD steps applied.
            last_loss: MSE from the most recent SGD step.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO learned_weights (id, weights, n_updates, last_loss, updated_at)
                   VALUES (1, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       weights    = excluded.weights,
                       n_updates  = excluded.n_updates,
                       last_loss  = excluded.last_loss,
                       updated_at = excluded.updated_at""",
                (json.dumps(weights), n_updates, last_loss, now),
            )
            conn.commit()

    def load_learned_weights(self) -> dict[str, float] | None:
        """Return the learned weight dict, or ``None`` if never saved.

        Returns:
            Dict mapping dimension names to weights, or None.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT weights FROM learned_weights WHERE id = 1"
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["weights"])

    def get_learned_weight_meta(self) -> dict[str, Any]:
        """Return metadata about the current learned weights.

        Returns:
            Dict with keys ``n_updates``, ``last_loss``, ``updated_at``.
            If no weights exist yet, returns zeroed defaults.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT n_updates, last_loss, updated_at FROM learned_weights WHERE id = 1"
            ).fetchone()
        if row is None:
            return {"n_updates": 0, "last_loss": 0.0, "updated_at": None}
        return dict(row)

    # -- session management --------------------------------------------------

    def create_session(
        self,
        session_id: str,
        original_prompt: str,
        task_type: str | None = None,
        model_id: str | None = None,
    ) -> None:
        """Create a new elicitation session.

        Args:
            session_id: Unique session identifier.
            original_prompt: The user's original prompt.
            task_type: Optional task type classification.
            model_id: Optional model being used.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO sessions
                   (session_id, original_prompt, task_type, model_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, original_prompt, task_type, model_id, now),
            )
            conn.commit()

    def update_session(
        self,
        session_id: str,
        *,
        questions: list[dict[str, Any]] | None = None,
        answers: dict[str, str] | None = None,
        spec: dict[str, Any] | None = None,
        final_score: float | None = None,
    ) -> None:
        """Update fields on an existing session.

        Args:
            session_id: Session to update.
            questions: Updated questions list.
            answers: Updated answers dict.
            spec: The refined IntentSpec as a dict.
            final_score: Final quality score achieved.
        """
        updates: list[str] = []
        params: list[Any] = []

        if questions is not None:
            updates.append("questions_json = ?")
            params.append(json.dumps(questions))
        if answers is not None:
            updates.append("answers_json = ?")
            params.append(json.dumps(answers))
        if spec is not None:
            updates.append("spec_json = ?")
            params.append(json.dumps(spec))
        if final_score is not None:
            updates.append("final_score = ?")
            params.append(final_score)
            updates.append("completed_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())

        if not updates:
            return

        params.append(session_id)
        sql = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?"

        with self._connection() as conn:
            conn.execute(sql, params)
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session data as a dict, or ``None``.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "original_prompt": row["original_prompt"],
            "task_type": row["task_type"],
            "model_id": row["model_id"],
            "questions": json.loads(row["questions_json"]),
            "answers": json.loads(row["answers_json"]),
            "spec": json.loads(row["spec_json"]) if row["spec_json"] else None,
            "final_score": row["final_score"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }

    # -- task management -----------------------------------------------------

    def save_task(self, task_data: dict[str, Any]) -> None:
        """Insert or update a task record.

        Args:
            task_data: Dict with keys matching the ``tasks`` table columns.
                Must include ``id`` and ``title``.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO tasks
                   (id, parent_id, session_id, title, description, state,
                    dependencies, spec_json, result_json, metadata_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       state = excluded.state,
                       result_json = excluded.result_json,
                       metadata_json = excluded.metadata_json,
                       updated_at = excluded.updated_at""",
                (
                    task_data["id"],
                    task_data.get("parent_id"),
                    task_data.get("session_id"),
                    task_data["title"],
                    task_data.get("description", ""),
                    task_data.get("state", "pending"),
                    json.dumps(task_data.get("dependencies", [])),
                    json.dumps(task_data.get("spec")) if task_data.get("spec") else None,
                    json.dumps(task_data.get("result")) if task_data.get("result") else None,
                    json.dumps(task_data.get("metadata", {})),
                    task_data.get("created_at", now),
                    now,
                ),
            )
            conn.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            Task data as a dict, or ``None``.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def get_tasks(
        self,
        session_id: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query tasks with optional filters.

        Args:
            session_id: Filter by session.
            state: Filter by state.
            limit: Maximum rows to return.

        Returns:
            List of task dicts, most recently updated first.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if state is not None:
            clauses.append("state = ?")
            params.append(state)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM tasks {where} ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_task(row) for row in rows]

    def update_task_state(self, task_id: str, state: str) -> None:
        """Update the state of a task.

        Args:
            task_id: Task identifier.
            state: New state value.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                "UPDATE tasks SET state = ?, updated_at = ? WHERE id = ?",
                (state, now, task_id),
            )
            conn.commit()

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a task dict."""
        return {
            "id": row["id"],
            "parent_id": row["parent_id"],
            "session_id": row["session_id"],
            "title": row["title"],
            "description": row["description"],
            "state": row["state"],
            "dependencies": json.loads(row["dependencies"]),
            "spec": json.loads(row["spec_json"]) if row["spec_json"] else None,
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # -- prompt history --------------------------------------------------------

    def record_prompt(self, entry: dict[str, Any]) -> int:
        """Append a prompt quality record to history.

        Args:
            entry: Dict with prompt analysis data including ``prompt_text``,
                ``quality_score``, dimension scores, ``task_type``, etc.

        Returns:
            The auto-generated row ID.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            cursor = conn.execute(
                """INSERT INTO prompt_history
                   (timestamp, prompt_text, quality_score, specificity,
                    constraint_clarity, context_completeness, ambiguity,
                    format_spec, task_type, complexity, route_chosen,
                    word_count, grade, session_context)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    entry.get("prompt_text", ""),
                    entry.get("quality_score", 0.0),
                    entry.get("specificity", 0.0),
                    entry.get("constraint_clarity", 0.0),
                    entry.get("context_completeness", 0.0),
                    entry.get("ambiguity", 0.0),
                    entry.get("format_spec", 0.0),
                    entry.get("task_type", "general"),
                    entry.get("complexity", 0.0),
                    entry.get("route_chosen", "refine"),
                    entry.get("word_count", 0),
                    entry.get("grade", "C"),
                    entry.get("session_context", "default"),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_prompt_history(
        self,
        limit: int = 100,
        session_context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve prompt history records.

        Args:
            limit: Maximum rows to return.
            session_context: Optional filter by session context.

        Returns:
            List of prompt history dicts, most recent first.
        """
        if session_context is not None:
            query = (
                "SELECT * FROM prompt_history WHERE session_context = ? "
                "ORDER BY id DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (session_context, limit)
        else:
            query = "SELECT * FROM prompt_history ORDER BY id DESC LIMIT ?"
            params = (limit,)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "prompt_text": row["prompt_text"],
                "quality_score": row["quality_score"],
                "specificity": row["specificity"],
                "constraint_clarity": row["constraint_clarity"],
                "context_completeness": row["context_completeness"],
                "ambiguity": row["ambiguity"],
                "format_spec": row["format_spec"],
                "task_type": row["task_type"],
                "complexity": row["complexity"],
                "route_chosen": row["route_chosen"],
                "word_count": row["word_count"],
                "grade": row["grade"],
                "session_context": row["session_context"],
            }
            for row in rows
        ]

    def get_prompt_stats(
        self,
        window: int = 50,
        session_context: str | None = None,
    ) -> dict[str, Any]:
        """Compute aggregate prompt quality statistics.

        Args:
            window: Number of recent prompts to consider.
            session_context: Optional filter by session context.

        Returns:
            Dict with stats: total, averages, trend, weak/strong areas.
        """
        history = self.get_prompt_history(limit=window, session_context=session_context)

        if not history:
            return {
                "total_prompts": 0,
                "avg_quality": 0.0,
                "trend": "no_data",
                "grade_distribution": {},
                "learning_curve": [],
                "weak_areas": [],
                "strong_areas": [],
            }

        # Reverse to chronological order
        history = list(reversed(history))

        scores = [h["quality_score"] for h in history]
        total = len(scores)

        # Compute avg for recent vs older
        avg_all = sum(scores) / total
        recent_10 = scores[-10:] if total >= 10 else scores
        avg_recent = sum(recent_10) / len(recent_10)

        if total >= 10:
            older_10 = scores[:10]
            avg_older = sum(older_10) / len(older_10)
            if avg_recent > avg_older + 0.05:
                trend = "improving"
            elif avg_recent < avg_older - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Grade distribution
        grades: dict[str, int] = {}
        for h in history:
            g = h["grade"]
            grades[g] = grades.get(g, 0) + 1

        # Dimension averages
        dims = {
            "specificity": [h["specificity"] for h in history],
            "constraint_clarity": [h["constraint_clarity"] for h in history],
            "context_completeness": [h["context_completeness"] for h in history],
            "format_spec": [h["format_spec"] for h in history],
        }
        dim_avgs = {k: sum(v) / len(v) for k, v in dims.items()}

        # Weak/strong
        sorted_dims = sorted(dim_avgs.items(), key=lambda x: x[1])
        weak = [d[0] for d in sorted_dims[:2] if d[1] < 0.6]
        strong = [d[0] for d in sorted_dims[-2:] if d[1] >= 0.6]

        # Learning curve (group by chunks of 5)
        curve: list[float] = []
        chunk_size = max(1, total // 10) if total >= 10 else 1
        for i in range(0, total, chunk_size):
            chunk = scores[i: i + chunk_size]
            curve.append(round(sum(chunk) / len(chunk), 3))

        return {
            "total_prompts": total,
            "avg_quality": round(avg_all, 3),
            "avg_quality_recent_10": round(avg_recent, 3),
            "trend": trend,
            "grade_distribution": grades,
            "learning_curve": curve,
            "dimension_averages": dim_avgs,
            "weak_areas": weak,
            "strong_areas": strong,
        }

    # -- serialization helpers -----------------------------------------------

    @staticmethod
    def _serialize_normal(p: NormalPrior) -> dict[str, Any]:
        return {
            "mean": p.mean,
            "variance": p.variance,
            "n_observations": p.n_observations,
            "_m2": p._m2,
            "decay": p.decay,
        }

    @staticmethod
    def _deserialize_normal(d: dict[str, Any]) -> NormalPrior:
        return NormalPrior(**d)

    @staticmethod
    def _serialize_beta(p: BetaPrior) -> dict[str, Any]:
        return {"alpha": p.alpha, "beta": p.beta}

    @staticmethod
    def _deserialize_beta(d: dict[str, Any]) -> BetaPrior:
        return BetaPrior(**d)

    def _serialize_task_model_prior(self, prior: TaskModelPrior) -> dict[str, Any]:
        """Serialise a :class:`TaskModelPrior` to a JSON-safe dict."""
        iterations_data: dict[str, Any] = {}
        for k, profile in prior.iterations.items():
            iterations_data[str(k)] = {
                "score": self._serialize_normal(profile.score),
                "score_delta": self._serialize_normal(profile.score_delta),
                "converge_prob": self._serialize_beta(profile.converge_prob),
                "latency_ms": self._serialize_normal(profile.latency_ms),
            }
        return {
            "task_type": prior.task_type,
            "model_id": prior.model_id,
            "created_at": prior.created_at,
            "updated_at": prior.updated_at,
            "total_calls": prior.total_calls,
            "iterations": iterations_data,
            "optimal_depth": self._serialize_normal(prior.optimal_depth),
            "overall_converge_rate": self._serialize_beta(prior.overall_converge_rate),
            "first_call_quality": self._serialize_normal(prior.first_call_quality),
        }

    def _deserialize_task_model_prior(self, data: dict[str, Any]) -> TaskModelPrior:
        """Deserialise a JSON dict back into a :class:`TaskModelPrior`."""
        iterations: dict[int, IterationProfile] = {}
        for k_str, idata in data.get("iterations", {}).items():
            iterations[int(k_str)] = IterationProfile(
                score=self._deserialize_normal(idata["score"]),
                score_delta=self._deserialize_normal(idata["score_delta"]),
                converge_prob=self._deserialize_beta(idata["converge_prob"]),
                latency_ms=self._deserialize_normal(idata["latency_ms"]),
            )
        return TaskModelPrior(
            task_type=data["task_type"],
            model_id=data["model_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            total_calls=data["total_calls"],
            iterations=iterations,
            optimal_depth=self._deserialize_normal(data["optimal_depth"]),
            overall_converge_rate=self._deserialize_beta(data["overall_converge_rate"]),
            first_call_quality=self._deserialize_normal(data["first_call_quality"]),
        )

    # -- plan persistence ----------------------------------------------------

    def save_plan(self, plan_dict: dict[str, Any]) -> None:
        """Upsert a plan (full JSON blob) into the plans table.

        Args:
            plan_dict: Output of ``Plan.to_dict()``, must include ``plan_id``.
        """
        now = datetime.now(timezone.utc).isoformat()
        plan_id = plan_dict["plan_id"]
        goal = plan_dict.get("goal", "")
        data = json.dumps(plan_dict)
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO plans (plan_id, goal, data, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(plan_id) DO UPDATE SET
                       data = excluded.data,
                       updated_at = excluded.updated_at""",
                (plan_id, goal, data, now, now),
            )
            conn.commit()

    def load_plan(self, plan_id: str) -> dict[str, Any] | None:
        """Load a plan dict by plan_id.

        Args:
            plan_id: The plan identifier.

        Returns:
            Plan dict or None if not found.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT data FROM plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["data"])

    def load_all_plans(self) -> list[dict[str, Any]]:
        """Load all persisted plans.

        Returns:
            List of plan dicts, most recently updated first.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT data FROM plans ORDER BY updated_at DESC"
            ).fetchall()
        return [json.loads(row["data"]) for row in rows]

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan from the store.

        Args:
            plan_id: The plan identifier.

        Returns:
            True if a row was deleted.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM plans WHERE plan_id = ?", (plan_id,)
            )
            conn.commit()
            return cursor.rowcount > 0


class SQLiteBackedPriors(AdaptivePriors):
    """Drop-in replacement for :class:`AdaptivePriors` backed by SQLite.

    Extends the base class but replaces the JSON file persistence with
    :class:`LoopStore`.  The in-memory prior dict is kept synchronised
    with the database.

    Args:
        store: The :class:`LoopStore` to use for persistence.
    """

    def __init__(self, store: LoopStore) -> None:
        # Bypass the parent __init__ to avoid JSON file loading
        self.store_path = None
        self._priors: dict[str, TaskModelPrior] = {}
        self._store = store
        self._load_from_store()

    def _load_from_store(self) -> None:
        """Load all priors from the SQLite store into memory."""
        self._priors = self._store.load_all_priors()

    def observe(self, observation: CallObservation) -> None:
        """Record an observation, updating both memory and database.

        Args:
            observation: The observation to incorporate.
        """
        # Use parent observe to update in-memory priors
        super().observe(observation)
        # Also log the raw observation
        self._store.record_observation(observation)
        # Persist the updated prior
        key = self._key(observation.task_type, observation.model_id)
        if key in self._priors:
            self._store.save_prior(key, self._priors[key])

    def _save(self) -> None:
        """Persist all priors to SQLite."""
        for key, prior in self._priors.items():
            self._store.save_prior(key, prior)

    def _load(self) -> None:
        """Load all priors from SQLite."""
        self._load_from_store()
