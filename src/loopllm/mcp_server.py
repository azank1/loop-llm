"""MCP server exposing loop-llm tools to IDE agents.

Provides iterative refinement, intent elicitation, task orchestration,
Bayesian meta-learning, and prompt quality analysis as MCP tools for
VS Code Copilot, Cursor, and other MCP-compatible clients.

Usage::

    loopllm mcp-server --provider ollama --model qwen2.5:0.5b
    # or
    python -m loopllm.mcp_server
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import structlog

from loopllm.agent_loop import AgentLoopController
from loopllm.elicitation import ElicitationSession, IntentRefiner, IntentSpec
from loopllm.engine import LoopConfig, LoopedLLM
from loopllm.evaluator_factory import build_evaluator
from loopllm.step_scorer import (
    conservative_dual_verify,
    legacy_self_report_score,
    build_step_evaluator,
)
from loopllm.priors import CallObservation
from loopllm.provider import LLMProvider
from loopllm.episodes import EpisodicStore, artifact_ref_hash, summarize_artifacts
from loopllm.plan_registry import get_registry
from loopllm.providers.agent import AgentPassthroughProvider
from loopllm.store import LoopStore, SQLiteBackedPriors
from loopllm.tasks import TaskOrchestrator

logger = structlog.get_logger(__name__)

try:
    from mcp.server.fastmcp import Context
except ImportError:
    Context = Any  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Shared state — initialised once per MCP server process
# ---------------------------------------------------------------------------

_store: LoopStore | None = None
_priors: SQLiteBackedPriors | None = None
_provider: LLMProvider | None = None
_default_model: str = "gpt-4o-mini"
_active_sessions: dict[str, dict[str, Any]] = {}
_agent_loop: AgentLoopController | None = None
_episodic: EpisodicStore | None = None
_status_path: Path | None = None
_history_path: Path | None = None
_episodes_feed_path: Path | None = None
# Last per-dimension scores computed by _score_prompt_quality — used by SGD.
_last_prompt_dims: dict[str, float] = {}


def _init_state() -> None:
    """Lazily initialise shared store, priors, and provider."""
    global _store, _priors, _provider, _default_model, _status_path, _history_path  # noqa: PLW0603

    if _store is not None:
        return

    db_path = Path(os.environ.get("LOOPLLM_DB", str(Path.home() / ".loopllm" / "store.db")))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _store = LoopStore(db_path=db_path)
    _priors = SQLiteBackedPriors(_store)
    _default_model = os.environ.get("LOOPLLM_MODEL", "gpt-4o-mini")
    _status_path = db_path.parent / "status.json"
    _history_path = db_path.parent / "prompt_history.json"
    _episodes_feed_path = db_path.parent / "episodes_feed.json"

    provider_name = os.environ.get("LOOPLLM_PROVIDER", "agent")
    _provider = _make_provider(provider_name)


def _write_episodes_feed(entry: dict[str, Any]) -> None:
    """Append one completed episode to episodes_feed.json (last 20 kept).

    The extension reads this file to display the Loop Monitor without
    needing a SQLite driver in the Node process.
    """
    _init_state()
    if _episodes_feed_path is None:
        return
    try:
        existing: list[dict[str, Any]] = []
        if _episodes_feed_path.exists():
            try:
                existing = json.loads(_episodes_feed_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(entry)
        # Keep the most recent 20 episodes
        trimmed = existing[-20:]
        _episodes_feed_path.write_text(json.dumps(trimmed, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass


def _make_provider(name: str) -> LLMProvider:
    """Create an LLM provider by name."""
    if name == "agent":
        return AgentPassthroughProvider()
    elif name == "mock":
        from loopllm.providers.mock import MockLLMProvider

        return MockLLMProvider(responses=[
            '{"result": "initial attempt"}',
            '{"result": "improved", "details": "comprehensive", "quality": "high"}',
        ])
    elif name == "ollama":
        from loopllm.providers.ollama import OllamaProvider

        base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaProvider(base_url=base_url)
    elif name == "openrouter":
        from loopllm.providers.openrouter import OpenRouterProvider

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY env var is required for openrouter provider")
        return OpenRouterProvider(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {name}")


def _get_provider(provider_override: str | None = None) -> LLMProvider:
    """Return the provider, applying optional per-call override."""
    _init_state()
    if provider_override and provider_override != os.environ.get("LOOPLLM_PROVIDER", "agent"):
        return _make_provider(provider_override)
    assert _provider is not None
    return _provider


def _is_agent_mode(provider_override: str | None = None) -> bool:
    """Return True when the active provider is the agent passthrough."""
    return isinstance(_get_provider(provider_override), AgentPassthroughProvider)


def _build_agent_refine_response(
    prompt: str,
    evaluator_type: str,
    min_words: int,
    max_words: int,
    required_fields: list[str],
    required_patterns: list[str],
    max_iterations: int,
    quality_threshold: float,
) -> str:
    """Build a structured agent-executable refinement response."""
    criteria: dict[str, Any] = {
        "evaluator_type": evaluator_type,
        "min_words": min_words,
        "max_words": max_words,
    }
    if required_fields:
        criteria["required_fields"] = required_fields
    if required_patterns:
        criteria["required_patterns"] = required_patterns

    instructions = (
        f"Generate a response to the prompt below. "
        f"Self-evaluate your response against the evaluator_criteria "
        f"and assign a quality score between 0.0 and 1.0. "
        f"If your score is below {quality_threshold}, revise and improve your response. "
        f"Repeat up to {max_iterations} times until your score is {quality_threshold} or higher. "
        f"Return only your best response."
    )
    return json.dumps({
        "mode": "agent_execute",
        "instructions": instructions,
        "prompt": prompt,
        "evaluator_criteria": criteria,
        "max_iterations": max_iterations,
        "quality_threshold": quality_threshold,
    }, indent=2)


def _get_model(model_override: str | None = None) -> str:
    """Return the model, applying optional per-call override."""
    _init_state()
    return model_override or _default_model


def _get_store() -> LoopStore:
    _init_state()
    assert _store is not None
    return _store


def _get_priors() -> SQLiteBackedPriors:
    _init_state()
    assert _priors is not None
    return _priors


def _get_agent_loop() -> AgentLoopController:
    """Return the process-wide agent-loop controller, bound to shared priors."""
    global _agent_loop  # noqa: PLW0603
    if _agent_loop is None:
        _agent_loop = AgentLoopController(_get_priors())
    return _agent_loop


def _get_episodic() -> EpisodicStore:
    global _episodic  # noqa: PLW0603
    if _episodic is None:
        db_path = Path(os.environ.get("LOOPLLM_DB", str(Path.home() / ".loopllm" / "store.db")))
        mirror = db_path.parent / "active_run.json"
        _episodic = EpisodicStore(_get_store(), mirror_path=mirror)
    return _episodic


def _build_evaluator(
    evaluator_type: str = "length",
    **kwargs: Any,
) -> Any:
    """Build an evaluator from a type string and optional config."""
    return build_evaluator(evaluator_type, **kwargs)


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a RefinementResult to a serialisable dict."""
    return {
        "output": result.output,
        "best_score": result.metrics.best_score,
        "final_score": result.metrics.final_score,
        "total_iterations": result.metrics.total_iterations,
        "converged": result.metrics.converged,
        "exit_reason": result.metrics.exit_reason.condition,
        "exit_message": result.metrics.exit_reason.message,
        "score_trajectory": result.metrics.score_trajectory,
    }


# ---------------------------------------------------------------------------
# Prompt quality scoring (heuristic-first)
# ---------------------------------------------------------------------------


def _score_prompt_quality(prompt: str) -> dict[str, Any]:
    """Score a prompt across multiple quality dimensions.

    Returns a dict with composite score, per-dimension scores, grade,
    issues, suggestions, and an ASCII gauge.
    """
    words = prompt.split()
    word_count = len(words)
    prompt_lower = prompt.lower()

    # --- Dimension: Specificity (0-1) ---
    specificity = 1.0
    vague_terms = ["something", "stuff", "thing", "things", "whatever", "somehow",
                   "do it", "make it", "fix it", "help me", "do this"]
    vague_hits = sum(1 for v in vague_terms if v in prompt_lower)
    specificity -= min(0.5, vague_hits * 0.15)
    if word_count < 5:
        specificity -= 0.3
    elif word_count < 10:
        specificity -= 0.15
    if word_count > 20:
        specificity += 0.1
    specificity = max(0.0, min(1.0, specificity))

    # --- Dimension: Constraint Clarity (0-1) ---
    constraint_clarity = 0.0
    constraint_words = ["must", "should", "require", "need", "format", "length",
                        "json", "csv", "return", "output", "include", "exclude",
                        "type", "schema", "limit", "exactly", "at least", "at most",
                        "no more than", "minimum", "maximum"]
    constraint_hits = sum(1 for c in constraint_words if c in prompt_lower)
    constraint_clarity = min(1.0, constraint_hits * 0.2)

    # --- Dimension: Context Completeness (0-1) ---
    context_completeness = 0.3  # base
    context_markers = ["because", "context", "background", "given that",
                       "for example", "e.g.", "such as", "like this",
                       "the goal is", "we need", "the purpose", "in order to"]
    ctx_hits = sum(1 for m in context_markers if m in prompt_lower)
    context_completeness += min(0.5, ctx_hits * 0.15)
    if word_count > 30:
        context_completeness += 0.1
    if word_count > 60:
        context_completeness += 0.1
    context_completeness = max(0.0, min(1.0, context_completeness))

    # --- Dimension: Ambiguity (0=no ambiguity, 1=highly ambiguous) ---
    ambiguity = 0.0
    ambiguous_pronouns = ["it", "this", "that", "they", "them", "those"]
    if word_count < 15:
        pronoun_hits = sum(1 for w in words if w.lower() in ambiguous_pronouns)
        ambiguity += min(0.4, pronoun_hits * 0.1)
    if not any(c in prompt for c in ["?", ".", "!", ":"]):
        ambiguity += 0.15
    if word_count < 5:
        ambiguity += 0.3
    if "?" in prompt and word_count < 8:
        ambiguity += 0.15
    ambiguity = max(0.0, min(1.0, ambiguity))

    # --- Dimension: Format Specification (0-1) ---
    format_spec = 0.0
    format_words = ["json", "csv", "xml", "html", "markdown", "yaml", "list",
                    "table", "code", "python", "javascript", "typescript",
                    "function", "class", "paragraph", "bullet points", "steps"]
    fmt_hits = sum(1 for f in format_words if f in prompt_lower)
    format_spec = min(1.0, fmt_hits * 0.25)

    # --- Composite Score ---
    _default_weights: dict[str, float] = {
        "specificity": 0.25,
        "constraint_clarity": 0.20,
        "context_completeness": 0.20,
        "ambiguity": 0.20,
        "format_spec": 0.15,
    }
    # Load adaptively learned weights if available.
    try:
        _init_state()
        _learned = _store.load_learned_weights() if _store else None
    except Exception:
        _learned = None
    weights = _learned if _learned is not None else _default_weights

    # Cache raw dimension values for SGD step at feedback time.
    global _last_prompt_dims  # noqa: PLW0603
    _last_prompt_dims = {
        "specificity": specificity,
        "constraint_clarity": constraint_clarity,
        "context_completeness": context_completeness,
        "ambiguity": ambiguity,
        "format_spec": format_spec,
    }

    composite = (
        weights["specificity"] * specificity
        + weights["constraint_clarity"] * constraint_clarity
        + weights["context_completeness"] * context_completeness
        + weights["ambiguity"] * (1.0 - ambiguity)
        + weights["format_spec"] * format_spec
    )
    composite = max(0.0, min(1.0, composite))

    # --- Grade ---
    if composite >= 0.85:
        grade = "A"
    elif composite >= 0.70:
        grade = "B"
    elif composite >= 0.55:
        grade = "C"
    elif composite >= 0.40:
        grade = "D"
    else:
        grade = "F"

    # --- Issues & Suggestions ---
    issues: list[str] = []
    suggestions: list[str] = []

    if specificity < 0.5:
        issues.append("Prompt is vague — lacks specific details")
        suggestions.append("Add concrete details about what you need")
    if constraint_clarity < 0.3:
        issues.append("No explicit constraints or requirements detected")
        suggestions.append("Specify output format, length, or quality requirements")
    if context_completeness < 0.4:
        issues.append("Insufficient context provided")
        suggestions.append("Add background, examples, or explain the goal")
    if ambiguity > 0.5:
        issues.append("High ambiguity — contains unclear references")
        suggestions.append("Replace pronouns (it, this, that) with specific nouns")
    if format_spec < 0.3:
        suggestions.append("Consider specifying the desired output format")

    # --- ASCII Gauge ---
    filled = int(composite * 10)
    gauge = "\u2588" * filled + "\u2591" * (10 - filled)
    pct = int(composite * 100)
    gauge_str = f"{gauge} {pct}% [{grade}]"

    return {
        "quality_score": round(composite, 3),
        "grade": grade,
        "gauge": gauge_str,
        "dimensions": {
            "specificity": round(specificity, 3),
            "constraint_clarity": round(constraint_clarity, 3),
            "context_completeness": round(context_completeness, 3),
            "ambiguity": round(ambiguity, 3),
            "format_spec": round(format_spec, 3),
        },
        "word_count": word_count,
        "issues": issues,
        "suggestions": suggestions,
    }


def _classify_task_type(prompt: str) -> str:
    """Fast heuristic task type classification (no LLM call)."""
    import re

    prompt_lower = prompt.lower()
    # Order matters: more specific patterns first, general last.
    # Multi-word phrases use plain substring match; single short words
    # use word-boundary regex to avoid false positives (e.g. "api" in "capital").
    patterns: list[tuple[str, list[str]]] = [
        ("creative_writing", ["write a story", "poem", "creative", "narrative",
                              "fiction", "blog post"]),
        ("summarization", ["summarize", "summary", "tldr", "condense",
                           "shorten"]),
        ("data_extraction", ["extract", "parse", "list all",
                             "pull out", "get all"]),
        ("transformation", ["convert", "transform", "translate", "reformat",
                            "restructure", "refactor"]),
        ("analysis", ["analyze", "analyse", "compare", "evaluate", "assess",
                      "review", "audit", "examine"]),
        ("question_answering", ["what is", "how does", "explain", "why ",
                                "describe", "define"]),
        ("code_generation", ["implement", "create a function", "build",
                             "code", "script", "program"]),
    ]

    # Short words needing word-boundary match to avoid substring false positives
    boundary_patterns: list[tuple[str, list[str]]] = [
        ("code_generation", [r"\bapi\b", r"\bclass\b"]),
    ]

    for task_type, keywords in patterns:
        if any(kw in prompt_lower for kw in keywords):
            return task_type

    for task_type, regexes in boundary_patterns:
        if any(re.search(rx, prompt_lower) for rx in regexes):
            return task_type

    # "write" alone → code_generation (after creative_writing checked above)
    if "write" in prompt_lower:
        return "code_generation"

    return "general"


def _estimate_complexity(prompt: str) -> float:
    """Estimate task complexity from 0.0 (trivial) to 1.0 (very complex)."""
    words = prompt.split()
    score = 0.0
    score += min(0.25, len(words) / 100)
    conjunctions = sum(1 for w in words if w.lower() in
                       ["and", "then", "also", "additionally", "plus"])
    score += min(0.2, conjunctions * 0.05)
    complex_kw = ["api", "database", "auth", "deploy", "test", "migrate",
                  "integrate", "concurrent", "async", "distributed",
                  "microservice", "pipeline", "architecture"]
    matches = sum(1 for kw in complex_kw if kw in prompt.lower())
    score += min(0.3, matches * 0.1)
    if prompt.count(",") > 3:
        score += 0.1
    if any(m in prompt.lower() for m in ["in addition", "as well as", "furthermore"]):
        score += 0.1
    return min(1.0, round(score, 3))


# ---------------------------------------------------------------------------
# Status file writer — enables near-real-time VS Code extension updates
# ---------------------------------------------------------------------------


def _write_status(tool_name: str, data: dict[str, Any]) -> None:
    """Write current status to ~/.loopllm/status.json for the VS Code extension."""
    if _status_path is None:
        return
    try:
        status = {
            "timestamp": time.time(),
            "tool": tool_name,
            "data": data,
        }
        _status_path.write_text(json.dumps(status, indent=2, default=str))
    except OSError:
        pass  # Never crash on status write failure


def _append_history(record: dict[str, Any]) -> None:
    """Append a prompt record to ~/.loopllm/prompt_history.json for the VS Code extension."""
    if _history_path is None:
        return
    try:
        history: list[dict[str, Any]] = []
        if _history_path.exists():
            try:
                history = json.loads(_history_path.read_text())
                if not isinstance(history, list):
                    history = []
            except (json.JSONDecodeError, OSError):
                history = []
        record["id"] = len(history) + 1
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        history.append(record)
        _history_path.write_text(json.dumps(history, indent=2, default=str))
    except OSError:
        pass  # Never crash on history write failure


# ---------------------------------------------------------------------------
# Tool implementations (sync — FastMCP wraps them in threads)
# ---------------------------------------------------------------------------


def _tool_intercept(prompt: str) -> str:
    """Analyse a prompt and recommend the best approach before acting."""
    store = _get_store()
    priors = _get_priors()
    model = _get_model()

    quality = _score_prompt_quality(prompt)
    task_type = _classify_task_type(prompt)
    complexity = _estimate_complexity(prompt)
    config = priors.suggest_config(task_type, model)

    q = quality["quality_score"]
    if q < 0.4:
        route = "elicit"
        reason = ("Prompt is too vague — clarifying questions will "
                  "significantly improve output quality")
        next_tool = "loopllm_elicitation_start"
    elif complexity > 0.6:
        route = "decompose"
        reason = (f"Complex task (complexity={complexity:.2f}) — "
                  "breaking into subtasks will produce better results")
        next_tool = "loopllm_plan_tasks"
    elif q < 0.6:
        route = "elicit_then_refine"
        reason = ("Prompt has gaps — quick elicitation then refinement "
                  "recommended")
        next_tool = "loopllm_elicitation_start"
    else:
        route = "refine"
        reason = "Prompt is clear enough — direct refinement loop"
        next_tool = "loopllm_refine"

    store.record_prompt({
        "prompt_text": prompt[:500],
        "quality_score": quality["quality_score"],
        "specificity": quality["dimensions"]["specificity"],
        "constraint_clarity": quality["dimensions"]["constraint_clarity"],
        "context_completeness": quality["dimensions"]["context_completeness"],
        "ambiguity": quality["dimensions"]["ambiguity"],
        "format_spec": quality["dimensions"]["format_spec"],
        "task_type": task_type,
        "complexity": complexity,
        "route_chosen": route,
        "word_count": quality["word_count"],
        "grade": quality["grade"],
    })

    # Also write to JSON history for the VS Code extension
    _append_history({
        "prompt_text": prompt[:500],
        "quality_score": quality["quality_score"],
        "specificity": quality["dimensions"]["specificity"],
        "constraint_clarity": quality["dimensions"]["constraint_clarity"],
        "context_completeness": quality["dimensions"]["context_completeness"],
        "ambiguity": quality["dimensions"]["ambiguity"],
        "format_spec": quality["dimensions"]["format_spec"],
        "task_type": task_type,
        "grade": quality["grade"],
    })

    result = {
        "route": route,
        "reason": reason,
        "next_tool": next_tool,
        "quality": quality,
        "task_type": task_type,
        "complexity": complexity,
        "prior_knowledge": config,
    }

    # Cheap recall hint: only when the prompt is clear enough to be worth it (M1).
    if q >= 0.6:
        try:
            hits = _get_episodic().recall(prompt, task_type=task_type, k=1)
        except Exception:  # noqa: BLE001
            hits = []
        if hits:
            result["recall_available"] = True
            result["recall_top_hit"] = hits[0].get("goal", "")
            result["recall_hint"] = (
                "Similar past work found — call loopllm_recall before planning."
            )
        else:
            result["recall_available"] = False

    _write_status("intercept", {
        "quality_score": quality["quality_score"],
        "grade": quality["grade"],
        "gauge": quality["gauge"],
        "task_type": task_type,
        "route": route,
    })

    return json.dumps(result, indent=2, default=str)


def _tool_prompt_stats(window: int = 50) -> str:
    """Get aggregate prompt quality statistics and learning curve."""
    store = _get_store()
    stats = store.get_prompt_stats(window=window)

    curve = stats.get("learning_curve", [])
    sparkline = ""
    if curve:
        spark_chars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
        mn = min(curve)
        mx = max(curve)
        rng = mx - mn if mx > mn else 1.0
        sparkline = "".join(
            spark_chars[int((v - mn) / rng * 8)] for v in curve
        )

    stats["sparkline"] = sparkline

    # normalise key name: expose both avg_score (canonical) and avg_quality (legacy)
    if "avg_quality" in stats and "avg_score" not in stats:
        stats["avg_score"] = stats["avg_quality"]
    elif "avg_score" in stats and "avg_quality" not in stats:
        stats["avg_quality"] = stats["avg_score"]

    _write_status("prompt_stats", {
        "total_prompts": stats.get("total_prompts", 0),
        "avg_score": stats.get("avg_score", 0),
        "trend": stats.get("trend", "no_data"),
    })

    return json.dumps(stats, indent=2, default=str)


def _update_scoring_weights(rating: int, dims: dict[str, float]) -> dict[str, Any]:
    """Run one online SGD step on the scoring dimension weights.

    Maps the 1-5 user rating to a [0, 1] target, computes the MSE gradient
    w.r.t. each weight, clips weights to [0.05, 0.50] and simplex-projects
    (sum-to-1), then persists to the store.

    Args:
        rating: User quality rating 1-5.
        dims: Raw per-dimension scores from ``_score_prompt_quality``.

    Returns:
        Dict with new weights, loss, and update count.
    """
    if not dims:
        return {"skipped": True, "reason": "no dimension cache"}

    store = _get_store()
    current_weights = store.load_learned_weights() or {
        "specificity": 0.25,
        "constraint_clarity": 0.20,
        "context_completeness": 0.20,
        "ambiguity": 0.20,
        "format_spec": 0.15,
    }

    target = (max(1, min(5, rating)) - 1) / 4.0  # map [1, 5] → [0.0, 1.0]

    # Predicted composite under current weights (ambiguity is inverted).
    y_hat = sum(
        current_weights[k] * (1.0 - dims[k] if k == "ambiguity" else dims[k])
        for k in current_weights
        if k in dims
    )
    loss = (y_hat - target) ** 2

    lr = 0.02  # learning rate
    new_weights = {}
    for k, w in current_weights.items():
        d = (1.0 - dims[k]) if k == "ambiguity" else dims.get(k, 0.0)
        grad = 2.0 * (y_hat - target) * d
        new_weights[k] = w - lr * grad

    # Project onto simplex: clip to [0.05, 0.50] then renormalise.
    for k in new_weights:
        new_weights[k] = max(0.05, min(0.50, new_weights[k]))
    total = sum(new_weights.values())
    for k in new_weights:
        new_weights[k] = round(new_weights[k] / total, 6)

    meta = store.get_learned_weight_meta()
    n_updates = meta["n_updates"] + 1
    store.save_learned_weights(new_weights, n_updates, loss)
    return {"weights": new_weights, "loss": round(loss, 6), "n_updates": n_updates}


def _tool_feedback(
    rating: int,
    task_type: str = "general",
    comment: str = "",
) -> str:
    """Record user quality feedback (1-5) to improve future predictions."""
    priors = _get_priors()
    model = _get_model()

    clamped = max(1, min(5, rating))
    normalized = clamped / 5.0

    obs = CallObservation(
        task_type=task_type,
        model_id=model,
        scores=[normalized],
        latencies_ms=[0.0],
        converged=normalized >= 0.8,
        total_iterations=1,
        max_iterations=1,
        quality_threshold=0.8,
    )
    priors.observe(obs)

    # --- Online SGD step: update scoring dimension weights. ---
    sgd_result = _update_scoring_weights(clamped, _last_prompt_dims)

    result: dict[str, Any] = {
        "recorded": True,
        "status": "ok",
        "rating": clamped,
        "normalized_score": normalized,
        "task_type": task_type,
        "model": model,
        "impact": ("Priors updated — future predictions for this task "
                   "type will be adjusted"),
        "weight_update": sgd_result,
    }
    if comment:
        result["comment"] = comment

    _write_status("feedback", {
        "rating": clamped,
        "task_type": task_type,
    })

    return json.dumps(result, indent=2)


def _tool_refine(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    max_iterations: int = 5,
    quality_threshold: float = 0.8,
    evaluator_type: str = "length",
    min_words: int = 5,
    max_words: int = 10000,
    required_fields: list[str] | None = None,
    required_patterns: list[str] | None = None,
) -> str:
    """Run the iterative refinement loop on a prompt."""
    prov = _get_provider(provider)
    mod = _get_model(model)
    priors = _get_priors()

    # Agent passthrough: delegate generation to the calling IDE agent.
    if isinstance(prov, AgentPassthroughProvider):
        return _build_agent_refine_response(
            prompt=prompt,
            evaluator_type=evaluator_type,
            min_words=min_words,
            max_words=max_words,
            required_fields=required_fields or [],
            required_patterns=required_patterns or [],
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
        )

    evaluator = _build_evaluator(
        evaluator_type,
        min_words=min_words,
        max_words=max_words,
        required_fields=required_fields or [],
        required_patterns=required_patterns or [],
    )

    config = LoopConfig(
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
    )
    loop = LoopedLLM(provider=prov, config=config)
    result = loop.refine(prompt, evaluator, model=mod)

    obs = CallObservation(
        task_type="mcp_refine",
        model_id=mod,
        scores=result.metrics.score_trajectory,
        latencies_ms=[it.latency_ms for it in result.iterations],
        converged=result.metrics.converged,
        total_iterations=result.metrics.total_iterations,
        max_iterations=config.max_iterations,
        quality_threshold=config.quality_threshold,
    )
    priors.observe(obs)

    result_dict = _result_to_dict(result)

    _write_status("refine", {
        "best_score": result.metrics.best_score,
        "iterations": result.metrics.total_iterations,
        "converged": result.metrics.converged,
    })

    return json.dumps(result_dict, indent=2)


def _tool_run_pipeline(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    max_iterations: int = 5,
    quality_threshold: float = 0.8,
    skip_elicitation: bool = False,
) -> str:
    """Run the full pipeline: elicit -> decompose -> execute -> verify."""
    prov = _get_provider(provider)
    mod = _get_model(model)
    store = _get_store()
    priors = _get_priors()

    # Agent passthrough: return a structured pipeline prompt for the calling agent.
    if isinstance(prov, AgentPassthroughProvider):
        quality = _score_prompt_quality(prompt)
        task_type = _classify_task_type(prompt)
        complexity = _estimate_complexity(prompt)
        instructions = (
            f"Execute the following pipeline for the user's prompt:\n"
            f"1. Elicit: identify any ambiguities and note clarifying questions.\n"
            f"2. Decompose: break into subtasks if complexity > 0.4 (detected: {complexity:.2f}).\n"
            f"3. Execute: generate a high-quality response for each subtask.\n"
            f"4. Verify: self-evaluate the combined output against quality_threshold={quality_threshold}.\n"
            f"Iterate on any failing subtasks (max {max_iterations} iterations total).\n"
            f"Return the final assembled result."
        )
        return json.dumps({
            "mode": "agent_execute",
            "instructions": instructions,
            "prompt": prompt,
            "task_type": task_type,
            "estimated_complexity": complexity,
            "prompt_quality": quality,
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
        }, indent=2)

    orchestrator = TaskOrchestrator(
        provider=prov,
        priors=priors,
        store=store,
        model=mod,
    )

    result = orchestrator.run(prompt, model=mod, answer_func=None)

    _write_status("pipeline", {
        "best_score": result.metrics.best_score,
        "iterations": result.metrics.total_iterations,
        "converged": result.metrics.converged,
    })

    return json.dumps(_result_to_dict(result), indent=2)


def _tool_classify_task(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Classify a prompt into a task type."""
    prov = _get_provider(provider)
    # Agent / fast path: deterministic heuristic; no LLM call needed.
    if isinstance(prov, AgentPassthroughProvider):
        return json.dumps({"task_type": _classify_task_type(prompt)})

    mod = _get_model(model)
    priors = _get_priors()
    refiner = IntentRefiner(provider=prov, priors=priors, model=mod)
    task_type = refiner.classify_task(prompt)
    return json.dumps({"task_type": task_type})


# Static question templates used in agent mode (no LLM required).
_STATIC_QUESTION_TEMPLATES: dict[str, dict[str, Any]] = {
    "scope": {
        "text": "What exactly should the output cover? What are the scope boundaries?",
        "options": None,
    },
    "format": {
        "text": "What output format is expected? (e.g., JSON, markdown, plain text, code)",
        "options": ["JSON", "Markdown", "Plain text", "Code", "Other"],
    },
    "constraints": {
        "text": "Are there any hard requirements, rules, or constraints to follow?",
        "options": None,
    },
    "examples": {
        "text": "Can you give an example of the expected input and/or output?",
        "options": None,
    },
    "audience": {
        "text": "Who will use this output? (e.g., developers, end-users, another system)",
        "options": ["Developers", "End-users", "Another system/API", "General audience"],
    },
    "priority": {
        "text": "If trade-offs are needed, what matters most?",
        "options": ["Speed", "Accuracy", "Brevity", "Completeness"],
    },
    "edge_cases": {
        "text": "How should edge cases, errors, or missing data be handled?",
        "options": None,
    },
}

# Fallback order when no historical stats are available.
_STATIC_QUESTION_ORDER = ["scope", "format", "constraints", "examples",
                           "audience", "priority", "edge_cases"]


def _next_static_question(
    asked_types: set[str],
    max_questions: int,
    n_asked: int,
) -> dict[str, Any] | None:
    """Return the next question selected via Thompson Sampling.

    Each question type maintains a Beta(alpha, beta) prior where alpha tracks
    historically positive-impact asks and beta tracks negative-impact asks.
    We draw one sample per candidate type and pick the argmax — exploration
    / exploitation without a fixed schedule.
    """
    import random  # stdlib — already available

    if n_asked >= max_questions:
        return None

    candidates = [qt for qt in _STATIC_QUESTION_TEMPLATES if qt not in asked_types]
    if not candidates:
        return None

    # Load per-type stats from store for Thompson Sampling.
    try:
        stats_list = _get_store().get_question_stats()
        stats: dict[str, dict[str, Any]] = {r["question_type"]: r for r in stats_list}
    except Exception:
        stats = {}

    scores: dict[str, float] = {}
    for qt in candidates:
        row = stats.get(qt, {})
        alpha = 1.0 + float(row.get("positive_impact", 0))
        beta  = 1.0 + float(row.get("negative_impact", 0))
        scores[qt] = random.betavariate(alpha, beta)

    best_qt = max(scores, key=lambda k: scores[k])
    tmpl = _STATIC_QUESTION_TEMPLATES[best_qt]
    return {
        "text": tmpl["text"],
        "question_type": best_qt,
        "options": tmpl["options"],
        "information_gain": round(scores[best_qt], 4),
    }


def _tool_analyze_prompt(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    max_questions: int = 5,
) -> str:
    """Analyze a prompt and generate ranked clarifying questions."""
    prov = _get_provider(provider)

    # Agent / fast path: derive questions from quality dimensions, no LLM.
    if isinstance(prov, AgentPassthroughProvider):
        quality = _score_prompt_quality(prompt)
        issues = quality.get("issues", [])
        # Map quality issues → most relevant question types
        issue_map = {
            "vague": "scope",
            "constraints": "constraints",
            "context": "examples",
            "ambiguity": "scope",
            "format": "format",
        }
        prioritized: list[str] = []
        for issue in issues:
            for keyword, qt in issue_map.items():
                if keyword in issue.lower() and qt not in prioritized:
                    prioritized.append(qt)
        # Fill remaining slots from default order
        for qt in _STATIC_QUESTION_ORDER:
            if qt not in prioritized:
                prioritized.append(qt)

        questions: list[dict[str, Any]] = []
        for qt in prioritized[:max_questions]:
            tmpl = _STATIC_QUESTION_TEMPLATES.get(qt, {})
            idx = len(questions)
            questions.append({
                "question": tmpl.get("text", qt),
                "question_type": qt,
                "options": tmpl.get("options"),
                "information_gain": round(0.5 - idx * 0.05, 4),
            })
        return json.dumps(questions, indent=2)

    mod = _get_model(model)
    priors = _get_priors()
    refiner = IntentRefiner(provider=prov, priors=priors, model=mod, max_questions=max_questions)
    cq_list = refiner.analyze(prompt)

    return json.dumps([
        {
            "question": q.text,
            "question_type": q.question_type,
            "options": q.options,
            "information_gain": round(q.information_gain, 4),
        }
        for q in cq_list
    ], indent=2)


def _tool_elicitation_start(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    max_questions: int = 3,
) -> str:
    """Start a new elicitation session."""
    prov = _get_provider(provider)
    mod = _get_model(model)
    store = _get_store()
    priors = _get_priors()

    session = ElicitationSession(original_prompt=prompt, model_id=mod)

    # Agent mode: deterministic task classification and static questions.
    if isinstance(prov, AgentPassthroughProvider):
        session.task_type = _classify_task_type(prompt)
        q_dict = _next_static_question(set(), max_questions, 0)
        _active_sessions[session.session_id] = {
            "session": session,
            "max_questions": max_questions,
            "agent_mode": True,
        }
        store.create_session(
            session_id=session.session_id,
            original_prompt=prompt,
            task_type=session.task_type,
            model_id=mod,
        )
        result: dict[str, Any] = {
            "session_id": session.session_id,
            "task_type": session.task_type,
        }
        if q_dict:
            from loopllm.elicitation import ClarifyingQuestion
            q = ClarifyingQuestion(
                text=q_dict["text"],
                question_type=q_dict["question_type"],
                options=q_dict["options"],
                information_gain=q_dict["information_gain"],
            )
            session.questions_asked.append(q)
            result["question"] = q_dict
            result["is_complete"] = False
        else:
            result["question"] = None
            result["is_complete"] = True
        return json.dumps(result, indent=2)

    refiner = IntentRefiner(provider=prov, priors=priors, model=mod, max_questions=max_questions)
    session.task_type = refiner.classify_task(prompt)
    question = refiner.ask(session)

    _active_sessions[session.session_id] = {
        "session": session,
        "refiner": refiner,
    }
    store.create_session(
        session_id=session.session_id,
        original_prompt=prompt,
        task_type=session.task_type,
        model_id=mod,
    )

    result2: dict[str, Any] = {
        "session_id": session.session_id,
        "task_type": session.task_type,
    }

    if question is not None:
        result2["question"] = {
            "text": question.text,
            "question_type": question.question_type,
            "options": question.options,
            "information_gain": round(question.information_gain, 4),
        }
        session.questions_asked.append(question)
        result2["is_complete"] = False
    else:
        result2["question"] = None
        result2["is_complete"] = True

    return json.dumps(result2, indent=2)


def _tool_elicitation_answer(session_id: str, answer: str) -> str:
    """Answer the current question and get the next one."""
    if session_id not in _active_sessions:
        return json.dumps({"error": f"Session not found: {session_id}"})

    state = _active_sessions[session_id]
    session: ElicitationSession = state["session"]
    store = _get_store()

    if session.questions_asked:
        last_q = session.questions_asked[-1]
        session.answers[last_q.question_type] = answer

    result: dict[str, Any] = {"session_id": session_id}

    # Agent mode: use static questions; no LLM needed.
    if state.get("agent_mode"):
        from loopllm.elicitation import ClarifyingQuestion
        asked_types = {q.question_type for q in session.questions_asked}
        max_q = state.get("max_questions", 3)
        q_dict = _next_static_question(asked_types, max_q, len(session.questions_asked))
        if q_dict:
            q = ClarifyingQuestion(
                text=q_dict["text"],
                question_type=q_dict["question_type"],
                options=q_dict["options"],
                information_gain=q_dict["information_gain"],
            )
            session.questions_asked.append(q)
            result["question"] = q_dict
            result["is_complete"] = False
        else:
            result["question"] = None
            result["is_complete"] = True
    else:
        refiner: IntentRefiner = state["refiner"]
        question = refiner.ask(session)
        if question is not None:
            result["question"] = {
                "text": question.text,
                "question_type": question.question_type,
                "options": question.options,
                "information_gain": round(question.information_gain, 4),
            }
            session.questions_asked.append(question)
            result["is_complete"] = False
        else:
            result["question"] = None
            result["is_complete"] = True

    store.update_session(
        session_id,
        answers=session.answers,
        questions=[
            {"text": q.text, "type": q.question_type}
            for q in session.questions_asked
        ],
    )

    return json.dumps(result, indent=2)


def _tool_elicitation_finish(session_id: str) -> str:
    """Finish an elicitation session and get the refined IntentSpec."""
    if session_id not in _active_sessions:
        return json.dumps({"error": f"Session not found: {session_id}"})

    state = _active_sessions[session_id]
    session: ElicitationSession = state["session"]
    store = _get_store()

    # Agent mode: synthesise spec deterministically from gathered answers.
    if state.get("agent_mode"):
        answers = session.answers
        constraints = {qt: ans for qt, ans in answers.items()
                       if qt in ("constraints", "format", "scope")}
        quality_criteria = []
        if "format" in answers:
            quality_criteria.append(f"Output must be in {answers['format']} format")
        if "constraints" in answers:
            quality_criteria.append(answers["constraints"])
        if "scope" in answers:
            quality_criteria.append(f"Scope: {answers['scope']}")
        context_parts = []
        if "examples" in answers:
            context_parts.append(f"Examples: {answers['examples']}")
        if "audience" in answers:
            context_parts.append(f"Audience: {answers['audience']}")
        context_str = ". ".join(context_parts)
        refined = session.original_prompt
        if context_str:
            refined = f"{session.original_prompt}. {context_str}."
        if "priority" in answers:
            refined += f" Prioritise: {answers['priority']}."
        complexity = _estimate_complexity(refined)
        spec = IntentSpec(
            task_type=session.task_type,
            original_prompt=session.original_prompt,
            refined_prompt=refined,
            constraints=constraints,
            quality_criteria=quality_criteria,
            estimated_complexity=complexity,
        )
    else:
        refiner: IntentRefiner = state["refiner"]
        if session.answers:
            spec = refiner.refine(session.original_prompt, session.answers)
        else:
            spec = IntentSpec(
                task_type=session.task_type,
                original_prompt=session.original_prompt,
                refined_prompt=session.original_prompt,
            )

    session.refined_spec = spec

    spec_dict = {
        "task_type": spec.task_type,
        "refined_prompt": spec.refined_prompt,
        "constraints": spec.constraints,
        "quality_criteria": spec.quality_criteria,
        "decomposition_hints": spec.decomposition_hints,
        "estimated_complexity": spec.estimated_complexity,
    }
    store.update_session(session_id, spec=spec_dict)

    del _active_sessions[session_id]

    return json.dumps({
        "session_id": session_id,
        "spec": spec_dict,
    }, indent=2)


def _tool_plan_tasks(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    estimated_complexity: float = 0.5,
) -> str:
    """Decompose a prompt into a task plan with dependencies."""
    prov = _get_provider(provider)

    # Agent passthrough: return a decomposition prompt for the calling agent.
    if isinstance(prov, AgentPassthroughProvider):
        task_type = _classify_task_type(prompt)
        instructions = (
            "Decompose the following task into an ordered list of subtasks. "
            "For each subtask provide: id (short unique string), title, description, "
            "and dependencies (list of ids that must complete first). "
            "Order them so they can be executed with dependencies satisfied. "
            "Return a JSON object with fields: task_count, tasks (array), execution_order (array of ids)."
        )
        return json.dumps({
            "mode": "agent_execute",
            "instructions": instructions,
            "prompt": prompt,
            "task_type": task_type,
            "estimated_complexity": estimated_complexity,
        }, indent=2)

    mod = _get_model(model)
    store = _get_store()
    priors = _get_priors()

    spec = IntentSpec(
        original_prompt=prompt,
        refined_prompt=prompt,
        estimated_complexity=estimated_complexity,
    )

    orchestrator = TaskOrchestrator(
        provider=prov, priors=priors, store=store, model=mod,
    )
    plan = orchestrator.plan(spec)

    return json.dumps({
        "task_count": len(plan.tasks),
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "state": t.state.value,
                "dependencies": t.dependencies,
            }
            for t in plan.tasks
        ],
        "execution_order": [t.id for t in plan.execution_order()],
    }, indent=2)


def _tool_verify_output(
    output: str,
    original_prompt: str,
    quality_criteria: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Verify an output against a prompt and quality criteria."""
    prov = _get_provider(provider)

    # Agent passthrough: score deterministically + ask agent for deeper check.
    if isinstance(prov, AgentPassthroughProvider):
        criteria = quality_criteria or []
        # Keyword-match criteria against output for a fast deterministic score
        output_lower = output.lower()
        passed_criteria = [c for c in criteria if any(
            word in output_lower for word in c.lower().split() if len(word) > 3
        )]
        score = (len(passed_criteria) / len(criteria)) if criteria else 0.9
        deficiencies = [c for c in criteria if c not in passed_criteria]
        agent_check = (
            f"Verify the following output against the original prompt and quality criteria.\n"
            f"Prompt: {original_prompt}\n"
            f"Criteria: {criteria}\n"
            f"Output: {output[:2000]}\n"
            f"List any deficiencies found and provide a quality score 0.0-1.0."
        )
        return json.dumps({
            "score": round(score, 3),
            "passed": score >= 0.7,
            "deficiencies": deficiencies,
            "sub_scores": {},
            "feedback": "Deterministic pre-check complete. Execute the instructions via the agent for deeper verification.",
            "mode": "agent_execute",
            "instructions": agent_check,
        }, indent=2)

    mod = _get_model(model)
    store = _get_store()
    priors = _get_priors()

    spec = IntentSpec(
        original_prompt=original_prompt,
        refined_prompt=original_prompt,
        quality_criteria=quality_criteria or [],
    )

    orchestrator = TaskOrchestrator(
        provider=prov, priors=priors, store=store, model=mod,
    )
    result = orchestrator.verify(spec, output)

    return json.dumps({
        "score": result.score,
        "passed": result.passed,
        "deficiencies": result.deficiencies,
        "sub_scores": result.sub_scores,
        "feedback": result.feedback,
    }, indent=2)


def _tool_report(
    task_type: str | None = None,
    model_id: str | None = None,
) -> str:
    """Show learned Bayesian priors and question effectiveness statistics."""
    priors = _get_priors()
    store = _get_store()

    if task_type and model_id:
        reports = [priors.report(task_type, model_id)]
    else:
        reports = priors.report_all()

    question_stats = store.get_question_stats()

    return json.dumps({
        "priors": reports,
        "question_effectiveness": question_stats,
    }, indent=2, default=str)


def _tool_suggest_config(
    task_type: str,
    model_id: str | None = None,
    cost_weight: float = 0.5,
) -> str:
    """Get a suggested LoopConfig based on learned beliefs."""
    priors = _get_priors()
    mod = model_id or _get_model()
    config = priors.suggest_config(task_type, mod, cost_weight)
    return json.dumps(config, indent=2, default=str)


# ---------------------------------------------------------------------------
# Adaptive agent-loop tools — Bayesian stop/continue control for agent loops
# ---------------------------------------------------------------------------


def _tool_loop_start(
    goal: str,
    task_type: str = "general",
    model_id: str | None = None,
    quality_threshold: float | None = None,
    cost_weight: float = 0.5,
    evaluator_type: str = "composite",
    quality_criteria: list[str] | None = None,
    required_patterns: list[str] | None = None,
    required_fields: list[str] | None = None,
    max_wall_ms: float = 300_000.0,
    max_tokens: int = 0,
) -> str:
    """Begin an adaptive agent-loop session with a CDV verifier recipe."""
    controller = _get_agent_loop()
    mod = model_id or _get_model()
    eval_kwargs: dict[str, Any] = {}
    if required_patterns:
        eval_kwargs["required_patterns"] = required_patterns
    if required_fields:
        eval_kwargs["required_fields"] = required_fields

    session = controller.start(
        goal=goal,
        task_type=task_type,
        model_id=mod,
        quality_threshold=quality_threshold,
        cost_weight=cost_weight,
        evaluator_type=evaluator_type,
        quality_criteria=quality_criteria,
        max_wall_ms=max_wall_ms,
        max_tokens=max_tokens,
        **eval_kwargs,
    )
    episodic = _get_episodic()
    episodic.upsert_active_run(session.session_id, "agent_loop", session.to_snapshot())

    # Surface similar past episodes so the agent doesn't start cold (M1 recall).
    similar = episodic.recall(goal, task_type=task_type, k=3)
    similar_brief = [
        {
            "goal": ep.get("goal", ""),
            "task_type": ep.get("task_type", ""),
            "summary": ep.get("summary", ""),
            "stop_reason": ep.get("stop_reason"),
            "score_final": ep.get("score_final"),
        }
        for ep in similar
    ]
    return json.dumps(
        {
            "session_id": session.session_id,
            "goal": session.goal,
            "task_type": session.task_type,
            "model_id": session.model_id,
            "suggested_budget": session.suggested_budget,
            "quality_threshold": round(session.quality_threshold, 3),
            "confidence": round(session.confidence, 3),
            "total_observations": session.total_observations,
            "evaluator_type": session.evaluator_type,
            "quality_criteria": session.quality_criteria,
            "similar_episodes": similar_brief,
            "memory_hint": (
                "Call loopllm_recall for more context on similar past runs."
                if similar_brief
                else "No similar episodes yet; this run will be recorded for next time."
            ),
            "guidance": (
                f"Run up to ~{session.suggested_budget} step(s). After each step, "
                f"call loopllm_loop_step with step_output=<artifact> (test log, diff, "
                f"summary). The server runs Conservative Dual-Verify (deterministic "
                f"checks + separate critic) — do NOT pass your own score. Stop when "
                f"the verdict says 'stop', then call loopllm_loop_end."
            ),
        },
        indent=2,
        default=str,
    )


async def _tool_loop_step(
    session_id: str,
    step_output: str = "",
    score: float | None = None,
    note: str = "",
    step_tokens: int = 0,
    ctx: Context[Any, Any, Any] | None = None,
) -> str:
    """Score a step artifact via CDV and return a continue/stop verdict."""
    controller = _get_agent_loop()
    try:
        session = controller.get_session(session_id)
    except KeyError as exc:
        return json.dumps({"error": str(exc)})

    cdv_meta: dict[str, Any] = {}
    if step_output:
        evaluator = build_step_evaluator(
            session.evaluator_type,
            session.quality_criteria,
            **session.evaluator_kwargs,
        )
        try:
            dual = await conservative_dual_verify(
                step_output=step_output,
                goal=session.goal,
                quality_criteria=session.quality_criteria,
                evaluator=evaluator,
                ctx=ctx,
                quality_threshold=session.quality_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({
                "error": f"CDV scoring failed: {exc}",
                "hint": "Pass step_output with the step artifact for verified scoring.",
            })
        final_score = dual.final_score
        cdv_meta = dual.to_dict()
    elif score is not None:
        dual = legacy_self_report_score(score, quality_threshold=session.quality_threshold)
        final_score = dual.final_score
        cdv_meta = dual.to_dict()
        cdv_meta["deprecation"] = (
            "Self-reported score is deprecated. Pass step_output for "
            "Conservative Dual-Verify scoring."
        )
    else:
        return json.dumps({
            "error": "Provide step_output (preferred) or score (legacy).",
            "hint": (
                "Submit the step artifact (test output, diff, summary) as "
                "step_output. The server scores it via Conservative Dual-Verify."
            ),
        })

    try:
        verdict = controller.step(
            session_id,
            final_score,
            note=note,
            step_output=step_output,
            step_tokens=step_tokens,
        )
    except KeyError as exc:
        return json.dumps({"error": str(exc)})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    verdict.update(cdv_meta)
    # Sampling transparency (M3): was Channel B (critic) actually consulted?
    full_cdv = bool(step_output) and cdv_meta.get("channel_b_score") is not None
    verdict["cdv_mode"] = "full" if full_cdv else "channel_a_only"
    verdict["cdv_mode_reason"] = (
        "Deterministic checks + independent critic (MCP sampling)."
        if full_cdv
        else "MCP sampling unavailable; scored by deterministic checks only."
    )

    # Checkpoint the verified step so the loop survives an MCP restart (M0).
    if verdict.get("decision") != "stop":
        try:
            episodic = _get_episodic()
            episodic.upsert_active_run(
                session_id, "agent_loop", controller.get_session(session_id).to_snapshot()
            )
        except Exception:  # noqa: BLE001 — checkpointing must never break a step
            logger.warning("step_checkpoint_failed", session_id=session_id, exc_info=True)

    return json.dumps(verdict, indent=2, default=str)


def _tool_loop_end(session_id: str, converged: bool | None = None) -> str:
    """Finalise an agent loop and learn from it for future budget predictions."""
    controller = _get_agent_loop()
    try:
        session = controller.get_session(session_id)
        summary = controller.end(session_id, converged)
    except KeyError as exc:
        return json.dumps({"error": str(exc)})

    episodic = _get_episodic()
    stop_reason = session.last_reason if session.last_decision == "stop" else "closed"
    episodic.record_episode(
        episode_type="agent_loop",
        goal=session.goal,
        task_type=session.task_type,
        model_id=session.model_id,
        summary=summarize_artifacts(
            session.goal,
            step_outputs=session.step_outputs,
            stop_reason=stop_reason,
            score_final=summary.get("final_score"),
        ),
        artifact_ref=artifact_ref_hash(
            session.step_outputs[-1] if session.step_outputs else ""
        ),
        score_final=summary.get("final_score"),
        steps_used=summary.get("steps_run"),
        stop_reason=stop_reason,
    )
    episodic.clear_active_run(session_id)
    _write_episodes_feed({
        "episode_type": "agent_loop",
        "goal": session.goal,
        "task_type": session.task_type,
        "model_id": session.model_id,
        "score_final": summary.get("final_score"),
        "steps_used": summary.get("steps_run"),
        "stop_reason": stop_reason,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return json.dumps(summary, indent=2, default=str)


def _tool_loop_status(session_id: str) -> str:
    """Inspect the current state of an active agent-loop session."""
    controller = _get_agent_loop()
    try:
        status = controller.status(session_id)
    except KeyError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(status, indent=2, default=str)


def _tool_recall(
    query: str,
    task_type: str | None = None,
    k: int = 5,
) -> str:
    """Recall similar past episodes by keyword search."""
    episodes = _get_episodic().recall(query, task_type=task_type, k=k)
    return json.dumps({"query": query, "count": len(episodes), "episodes": episodes}, indent=2)


def _tool_run_status() -> str:
    """Return active loop/plan snapshots for IDE reload recovery."""
    snapshot = _get_episodic().run_status_snapshot()
    controller = _get_agent_loop()
    loops: list[dict[str, Any]] = []
    for run in snapshot.get("active_runs", []):
        if run.get("run_type") == "agent_loop":
            sid = run.get("run_id", "")
            try:
                loops.append(controller.status(sid))
            except KeyError:
                loops.append(run)
    snapshot["active_agent_loops"] = loops
    return json.dumps(snapshot, indent=2, default=str)


def _tool_loop_resume(session_id: str | None = None) -> str:
    """Resume an agent loop after an IDE/MCP restart.

    With ``session_id`` omitted, resumes the single active loop or returns an
    ambiguity list when several are pending. Rehydrates the in-memory session so
    the next ``loopllm_loop_step`` succeeds.
    """
    controller = _get_agent_loop()
    episodic = _get_episodic()

    if session_id:
        # Already live in this process — nothing to restore.
        try:
            return json.dumps(
                {"resumed": True, "source": "memory", "status": controller.status(session_id)},
                indent=2,
                default=str,
            )
        except KeyError:
            pass
        run = episodic.get_active_run(session_id)
        if run is None or run.get("run_type") != "agent_loop":
            return json.dumps({"error": f"No active agent loop to resume: {session_id}"})
        controller.restore_from_snapshot(run["state"])
        return json.dumps(
            {"resumed": True, "source": "store", "status": controller.status(session_id)},
            indent=2,
            default=str,
        )

    runs = [
        r for r in episodic.list_active_runs(run_type="agent_loop")
        if not (r.get("state") or {}).get("closed")
    ]
    if not runs:
        return json.dumps({"resumed": False, "message": "No active agent loops to resume."})
    if len(runs) > 1:
        return json.dumps(
            {
                "resumed": False,
                "ambiguous": True,
                "message": "Multiple active loops; pass session_id.",
                "candidates": [
                    {
                        "session_id": r["run_id"],
                        "goal": (r.get("state") or {}).get("goal", ""),
                        "steps_used": len((r.get("state") or {}).get("scores", [])),
                    }
                    for r in runs
                ],
            },
            indent=2,
            default=str,
        )
    only = runs[0]
    sid = only["run_id"]
    try:
        controller.status(sid)  # already live?
        source = "memory"
    except KeyError:
        controller.restore_from_snapshot(only["state"])
        source = "store"
    return json.dumps(
        {"resumed": True, "source": source, "status": controller.status(sid)},
        indent=2,
        default=str,
    )


def _tool_list_tasks(
    state: str | None = None,
    limit: int = 20,
) -> str:
    """List tasks from the store."""
    store = _get_store()
    tasks = store.get_tasks(state=state, limit=limit)
    return json.dumps(tasks, indent=2, default=str)


def _tool_show_task(task_id: str) -> str:
    """Show detailed information about a specific task."""
    store = _get_store()
    task = store.get_task(task_id)
    if task is None:
        return json.dumps({"error": f"Task not found: {task_id}"})
    return json.dumps(task, indent=2, default=str)


# ---------------------------------------------------------------------------
# Plan Registry tools — confidence-driven task management
# ---------------------------------------------------------------------------


def _tool_plan_register(
    goal: str,
    tasks: list[dict[str, Any]],
    confidence_threshold: float = 0.72,
) -> str:
    """Create a new plan in the PlanRegistry.

    Each task in ``tasks`` should have at minimum a ``title`` and
    ``description``.  The registry assigns a ``plan_id`` and starts
    tracking rolling confidence as tasks are scored.

    Args:
        goal: High-level goal text for the plan.
        tasks: List of task dicts with ``title``, ``description`` (and
            optionally ``id``, ``metadata``).
        confidence_threshold: Rolling confidence must stay above this
            value or the plan flags ``needs_replan=True``.

    Returns:
        JSON plan dict including ``plan_id``, all tasks with initial
        scores, and ``rolling_confidence``.
    """
    registry = get_registry()
    plan = registry.create(
        goal=goal,
        tasks=tasks,
        confidence_threshold=confidence_threshold,
    )
    # Persist immediately so plans survive MCP server restarts
    store = _get_store()
    store.save_plan(plan.to_dict())
    _get_episodic().upsert_active_run(
        plan.plan_id,
        "plan",
        plan.to_dict(),
    )
    return json.dumps(plan.to_dict(), indent=2)


def _tool_plan_update(
    plan_id: str,
    task_id: str,
    prompt_score: float | None = None,
    output_score: float | None = None,
    mark_done: bool = True,
) -> str:
    """Update a task's prompt/output scores and recalculate plan confidence.

    Call this after:
    - Scoring the task prompt with ``loopllm_intercept``
      → pass the ``quality_score`` as ``prompt_score``
    - Generating and verifying the task output
      → pass the evaluation score as ``output_score``

    The registry recalculates ``rolling_confidence`` with exponential
    decay weighting (recent tasks count more) and sets
    ``needs_replan=True`` if confidence drops below the threshold.

    Args:
        plan_id: The plan to update.
        task_id: The task within the plan.
        prompt_score: Prompt quality score (0–1) from loopllm_intercept.
        output_score: Output quality score (0–1) from loopllm_verify_output.
        mark_done: If True, mark the task DONE when confidence >= threshold,
            or REPLANNING when below it.

    Returns:
        Updated plan dict with new ``rolling_confidence`` and
        ``needs_replan`` flag.
    """
    registry = get_registry()
    result: dict[str, Any] = {}

    if prompt_score is not None:
        result = registry.score_prompt(plan_id, task_id, prompt_score)
        if "error" in result:
            return json.dumps(result, indent=2)

    if output_score is not None:
        result = registry.score_output(plan_id, task_id, output_score, mark_done=mark_done)
        if "error" in result:
            return json.dumps(result, indent=2)

    if not result:
        result = registry.get_status(plan_id)

    # Persist updated plan state
    store = _get_store()
    plan = registry.get(plan_id)
    if plan:
        store.save_plan(plan.to_dict())
        _get_episodic().upsert_active_run(plan_id, "plan", plan.to_dict())
        if mark_done and output_score is not None:
            task = next((t for t in plan.tasks if t.id == task_id), None)
            if task is not None:
                _get_episodic().record_episode(
                    episode_type="plan_node",
                    goal=f"{plan.goal} [{task.title}]",
                    task_type="general",
                    model_id=_get_model(),
                    summary=summarize_artifacts(
                        plan.goal,
                        output=task.description[:200],
                        score_final=output_score,
                    ),
                    score_final=output_score,
                    stop_reason="plan_task_done",
                )

    return json.dumps(result, indent=2)


def _tool_plan_list() -> str:
    """List all active plans with their current status and confidence.

    Returns all plans tracked by the PlanRegistry — both in-memory and
    those restored from disk after a server restart.  Use this to get a
    Shrimp-style overview of all ongoing work.

    Returns:
        JSON with a ``plans`` list, each entry containing ``plan_id``,
        ``goal``, ``rolling_confidence``, ``needs_replan``, task counts
        by status, and the next pending task title.
    """
    registry = get_registry()
    store = _get_store()

    # Ensure any plans saved by previous server invocations are loaded
    registry.restore_from_store(store)

    plans = registry.list_plans()
    summary = []
    for p in plans:
        tasks = p.get("tasks", [])
        by_status: dict[str, int] = {}
        for t in tasks:
            s = t.get("status", "pending")
            by_status[s] = by_status.get(s, 0) + 1
        next_pending = next((t["title"] for t in tasks if t["status"] == "pending"), None)
        confidence = p.get("rolling_confidence", 1.0)
        filled = int(confidence * 10)
        gauge = f"{'█' * filled}{'░' * (10 - filled)} {int(confidence * 100)}%"
        summary.append({
            "plan_id": p["plan_id"],
            "goal": p["goal"],
            "gauge": gauge,
            "rolling_confidence": p["rolling_confidence"],
            "needs_replan": p["needs_replan"],
            "confidence_threshold": p.get("confidence_threshold", 0.72),
            "replan_count": p.get("replan_count", 0),
            "task_counts": by_status,
            "total_tasks": len(tasks),
            "next_task": next_pending,
        })
    return json.dumps({
        "total_plans": len(summary),
        "plans": summary,
    }, indent=2)


def _tool_plan_delete(plan_id: str) -> str:
    """Delete a plan from the registry and persistent store.

    Use this when a plan is complete or no longer needed.

    Args:
        plan_id: The plan to delete.

    Returns:
        JSON confirmation with ``deleted`` flag.
    """
    registry = get_registry()
    store = _get_store()
    in_memory = registry.delete(plan_id)
    on_disk = store.delete_plan(plan_id)
    _get_episodic().clear_active_run(plan_id)  # drop any recovery snapshot
    return json.dumps({
        "deleted": in_memory or on_disk,
        "plan_id": plan_id,
        "message": (
            f"Plan {plan_id} deleted." if (in_memory or on_disk)
            else f"Plan {plan_id} not found."
        ),
    }, indent=2)


def _tool_gauge(prompt: str) -> str:
    """Instantly score a prompt and return a visual quality gauge.

    Lighter than loopllm_intercept — no routing, no DB write, no elicitation.
    Use this for a quick visual quality check of any prompt or draft.

    Returns a gauge like:  ████████░░ 82% [A]
    plus the five dimension scores and a list of improvement suggestions.

    Args:
        prompt: The prompt text to score.

    Returns:
        JSON with ``gauge``, ``grade``, ``score``, ``dimensions``, and ``suggestions``.
    """
    quality = _score_prompt_quality(prompt)
    q = quality["quality_score"]
    dims = quality["dimensions"]

    # Sort dimensions worst→best so weakest areas appear first
    sorted_dims = sorted(dims.items(), key=lambda x: x[1])

    return json.dumps({
        "gauge": quality["gauge"],
        "grade": quality["grade"],
        "score": round(q, 3),
        "dimensions": {
            k: {
                "score": round(v, 3),
                "bar": "█" * int(v * 10) + "░" * (10 - int(v * 10)),
            }
            for k, v in sorted_dims
        },
        "suggestions": quality.get("suggestions", []),
        "issues": quality.get("issues", []),
        "word_count": quality.get("word_count", 0),
    }, indent=2)


def _tool_context_history(
    limit: int = 20,
    session_context: str | None = None,
    min_score: float | None = None,
) -> str:
    """Browse your prompt quality history with visual gauges.

    Returns your recent prompts with their scores, grades, and gauges so you
    can track how your prompting quality is evolving over time.

    Args:
        limit: Number of recent prompts to return (default 20, max 100).
        session_context: Filter to a specific session context tag.
        min_score: Only return prompts at or above this quality score (0-1).

    Returns:
        JSON with a ``history`` list (newest first) and aggregate ``summary``.
    """
    store = _get_store()
    limit = max(1, min(limit, 100))
    rows = store.get_prompt_history(limit=limit, session_context=session_context)

    if min_score is not None:
        rows = [r for r in rows if r["quality_score"] >= min_score]

    spark_chars = " ▁▂▃▄▅▆▇█"

    def _gauge(score: float, grade: str) -> str:
        filled = int(score * 10)
        return f"{'█' * filled}{'░' * (10 - filled)} {int(score * 100)}% [{grade}]"

    def _spark(score: float) -> str:
        idx = int(score * 8)
        return spark_chars[min(idx, 8)]

    formatted = []
    for r in rows:
        formatted.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "prompt_preview": r["prompt_text"][:80] + ("…" if len(r["prompt_text"]) > 80 else ""),
            "gauge": _gauge(r["quality_score"], r["grade"]),
            "grade": r["grade"],
            "score": round(r["quality_score"], 3),
            "task_type": r["task_type"],
            "route_chosen": r["route_chosen"],
            "session_context": r["session_context"],
            "dimensions": {
                "specificity": round(r["specificity"], 3),
                "constraint_clarity": round(r["constraint_clarity"], 3),
                "context_completeness": round(r["context_completeness"], 3),
                "ambiguity": round(r["ambiguity"], 3),
                "format_spec": round(r["format_spec"], 3),
            },
        })

    # Summary strip
    if rows:
        scores = [r["quality_score"] for r in rows]
        avg = sum(scores) / len(scores)
        sparkline = "".join(_spark(s) for s in reversed(scores))
        grade_dist: dict[str, int] = {}
        for r in rows:
            grade_dist[r["grade"]] = grade_dist.get(r["grade"], 0) + 1
        summary = {
            "total_shown": len(rows),
            "avg_score": round(avg, 3),
            "avg_gauge": _gauge(avg, next(
                g for g, lb in [("A", 0.85), ("B", 0.70), ("C", 0.55), ("D", 0.40), ("F", 0.0)]
                if avg >= lb
            )),
            "sparkline": sparkline,
            "grade_distribution": grade_dist,
        }
    else:
        summary = {"total_shown": 0, "avg_score": 0.0, "sparkline": ""}

    return json.dumps({
        "summary": summary,
        "history": formatted,
    }, indent=2)


def _tool_context_clear(session_context: str | None = None) -> str:
    """Clear stored prompt history.

    Wipes all (or session-scoped) prompt history records from the local DB.
    Use this to reset your quality baseline at the start of a new project
    or when switching contexts.

    Args:
        session_context: If provided, only clear records with this session tag.
                         If omitted, ALL prompt history is cleared.

    Returns:
        JSON with the count of records deleted and confirmation message.
    """
    store = _get_store()
    with store._connection() as conn:
        if session_context is not None:
            cursor = conn.execute(
                "DELETE FROM prompt_history WHERE session_context = ?",
                (session_context,),
            )
        else:
            cursor = conn.execute("DELETE FROM prompt_history")
        conn.commit()
        deleted = cursor.rowcount

    scope = f"session '{session_context}'" if session_context else "all sessions"
    return json.dumps({
        "deleted": deleted,
        "scope": scope,
        "message": f"Cleared {deleted} prompt history record(s) from {scope}.",
    }, indent=2)


def _tool_plan_next(plan_id: str) -> str:
    """Get and activate the next pending task in a plan.

    Returns the next PENDING task and marks it IN_PROGRESS, or signals
    that the plan is complete.  Also surfaces ``needs_replan`` and the
    current ``rolling_confidence`` so the agent can decide whether to
    pause and replan before proceeding.

    Args:
        plan_id: The plan to query.

    Returns:
        JSON with the next task details, ``needs_replan``, and
        ``rolling_confidence``; or ``{\"done\": true}`` if all tasks
        are finished.
    """
    registry = get_registry()
    task = registry.next_task(plan_id)
    if task is None:
        status = registry.get_status(plan_id)
        return json.dumps({
            "done": True,
            "plan_id": plan_id,
            "rolling_confidence": status.get("rolling_confidence", 0.0),
        }, indent=2)
    # Persist the in_progress status change
    store = _get_store()
    plan = registry.get(plan_id)
    if plan:
        store.save_plan(plan.to_dict())
    return json.dumps({**task, "done": False}, indent=2)


# ---------------------------------------------------------------------------
# Mid-execution MCP sampling helpers
# ---------------------------------------------------------------------------


async def _sample_text(ctx: Any, prompt: str, max_tokens: int = 2048) -> str:
    """Call ctx.sample() and return the plain text content."""
    result = await ctx.sample(prompt, max_tokens=max_tokens)
    content = result.content
    return content.text if hasattr(content, "text") else str(content)


async def _sampling_refine(
    ctx: Any,
    prompt: str,
    max_iterations: int,
    quality_threshold: float,
    evaluator_type: str,
    min_words: int,
    max_words: int,
    required_fields: list[str],
    required_patterns: list[str],
) -> str:
    """Iterative refinement loop executed entirely via MCP sampling calls."""
    evaluator = _build_evaluator(
        evaluator_type,
        min_words=min_words,
        max_words=max_words,
        required_fields=required_fields,
        required_patterns=required_patterns,
    )
    best_output = ""
    best_score = 0.0
    scores: list[float] = []
    current_prompt = prompt
    for i in range(max_iterations):
        output = await _sample_text(ctx, current_prompt, max_tokens=4096)
        ev = evaluator.evaluate(output)
        scores.append(ev.score)
        if ev.score > best_score:
            best_score = ev.score
            best_output = output
        if ev.passed or i == max_iterations - 1:
            break
        deficiency_str = "; ".join(ev.deficiencies) if ev.deficiencies else "low score"
        current_prompt = (
            f"{prompt}\n\n"
            f"[Iteration {i + 1} score: {ev.score:.2f}. Issues: {deficiency_str}. "
            f"Please improve your response to address these issues.]"
        )
    return json.dumps({
        "output": best_output,
        "best_score": round(best_score, 3),
        "converged": best_score >= quality_threshold,
        "iterations": len(scores),
        "score_trajectory": [round(s, 3) for s in scores],
        "via": "mcp_sampling",
    }, indent=2)


async def _sampling_run_pipeline(
    ctx: Any,
    prompt: str,
    max_iterations: int,
    quality_threshold: float,
    skip_elicitation: bool,
) -> str:
    """Full pipeline (elicit -> decompose -> execute -> verify) via MCP sampling."""
    quality = _score_prompt_quality(prompt)
    task_type = _classify_task_type(prompt)
    complexity = _estimate_complexity(prompt)
    num_samples = 0

    # Stage 1: elicit clarifying assumptions when prompt quality is weak
    refined_prompt = prompt
    if not skip_elicitation and quality["quality_score"] < 0.6:
        elicit_text = await _sample_text(
            ctx,
            f"The user asked: '{prompt}'\n"
            f"Identify the 1-2 most important ambiguities (prompt score: "
            f"{quality['quality_score']:.2f}). State your clarifying assumptions "
            f"clearly, then proceed based on those assumptions.",
            max_tokens=400,
        )
        num_samples += 1
        refined_prompt = f"{prompt}\n\n[Clarifying assumptions: {elicit_text}]"

    # Stage 2: decompose into subtasks when complexity warrants it
    subtask_list: list[dict[str, Any]] = []
    if complexity > 0.5:
        decomp_text = await _sample_text(
            ctx,
            f"Decompose this task into 2-5 ordered subtasks:\n{refined_prompt}\n\n"
            f"Task type: {task_type}\n"
            f"Reply ONLY with a JSON array: "
            f'[{{"id":"t1","title":"...","description":"..."}}]',
            max_tokens=800,
        )
        num_samples += 1
        try:
            m = re.search(r"\[.*\]", decomp_text, re.DOTALL)
            subtask_list = json.loads(m.group()) if m else []
        except Exception:  # noqa: BLE001
            subtask_list = []

    # Stage 3: execute each subtask (or the whole prompt if not decomposed)
    if subtask_list:
        parts: list[str] = []
        for t in subtask_list:
            part = await _sample_text(
                ctx,
                f"Task: {t.get('title', '')}\n{t.get('description', '')}\n\nContext: {refined_prompt}",
                max_tokens=2048,
            )
            num_samples += 1
            parts.append(part)
        output = "\n\n".join(parts)
    else:
        output = await _sample_text(ctx, refined_prompt, max_tokens=4096)
        num_samples += 1

    # Stage 4: ask the agent to self-rate the output
    verify_text = await _sample_text(
        ctx,
        f"Rate the quality of this response to the prompt '{refined_prompt[:200]}' "
        f"on a scale 0.0-1.0. Reply ONLY with a decimal number.",
        max_tokens=20,
    )
    num_samples += 1
    try:
        score_match = re.search(r"\d+\.?\d*", verify_text)
        best_score = float(score_match.group()) if score_match else 0.85
        if best_score > 1.0:
            best_score = best_score / 10.0
        best_score = min(1.0, best_score)
    except Exception:  # noqa: BLE001
        best_score = 0.85

    return json.dumps({
        "output": output,
        "best_score": round(best_score, 3),
        "converged": best_score >= quality_threshold,
        "iterations": num_samples,
        "score_trajectory": [round(best_score, 3)],
        "task_type": task_type,
        "subtasks": len(subtask_list) if subtask_list else 1,
        "via": "mcp_sampling",
    }, indent=2)


async def _sampling_plan_tasks(
    ctx: Any,
    prompt: str,
    estimated_complexity: float,
) -> str:
    """Decompose a prompt into a structured task plan via MCP sampling."""
    task_type = _classify_task_type(prompt)
    text = await _sample_text(
        ctx,
        f"Decompose this task into 2-6 ordered subtasks:\n{prompt}\n\n"
        f"Task type: {task_type}, Complexity: {estimated_complexity:.2f}\n\n"
        f"Reply ONLY with valid JSON:\n"
        f'{{"tasks":[{{"id":"t1","title":"...","description":"...","dependencies":[]}}],'
        f'"execution_order":["t1"]}}',
        max_tokens=1200,
    )
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        tasks = data.get("tasks", [])
        order = data.get("execution_order", [t.get("id", "") for t in tasks])
    except Exception:  # noqa: BLE001
        tasks = [{"id": "t1", "title": prompt[:60], "description": prompt, "dependencies": []}]
        order = ["t1"]
    return json.dumps({
        "task_count": len(tasks),
        "tasks": [
            {
                "id": t.get("id", f"t{i}"),
                "title": t.get("title", ""),
                "description": t.get("description", ""),
                "state": "pending",
                "dependencies": t.get("dependencies", []),
            }
            for i, t in enumerate(tasks, 1)
        ],
        "execution_order": order,
        "via": "mcp_sampling",
    }, indent=2)


async def _sampling_verify_output(
    ctx: Any,
    output: str,
    original_prompt: str,
    quality_criteria: list[str],
) -> str:
    """Verify output quality with a fast keyword pre-check then deep MCP sampling."""
    output_lower = output.lower()
    passed_criteria = [
        c for c in quality_criteria if any(
            word in output_lower for word in c.lower().split() if len(word) > 3
        )
    ]
    fast_score = (len(passed_criteria) / len(quality_criteria)) if quality_criteria else 0.9
    fast_deficiencies = [c for c in quality_criteria if c not in passed_criteria]

    verify_result = await _sample_text(
        ctx,
        f"Verify the following output against the original prompt and quality criteria.\n\n"
        f"ORIGINAL PROMPT:\n{original_prompt}\n\n"
        f"OUTPUT:\n{output[:3000]}\n\n"
        f"QUALITY CRITERIA: {quality_criteria}\n\n"
        f"Reply ONLY with valid JSON:\n"
        f'{{"score":0.0,"passed":false,"deficiencies":["..."],"feedback":"..."}}',
        max_tokens=500,
    )
    try:
        m = re.search(r"\{.*\}", verify_result, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        score = float(data.get("score", fast_score))
        passed = bool(data.get("passed", score >= 0.7))
        deficiencies = data.get("deficiencies", fast_deficiencies)
        feedback = data.get("feedback", "")
    except Exception:  # noqa: BLE001
        score = fast_score
        passed = fast_score >= 0.7
        deficiencies = fast_deficiencies
        feedback = verify_result[:300]

    return json.dumps({
        "score": round(score, 3),
        "passed": passed,
        "deficiencies": deficiencies,
        "sub_scores": {"keyword_match": round(fast_score, 3)},
        "feedback": feedback,
        "via": "mcp_sampling",
    }, indent=2)


# ---------------------------------------------------------------------------
# FastMCP registration
# ---------------------------------------------------------------------------


def create_mcp_server() -> Any:
    """Create and configure the FastMCP server with all tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "The mcp package is required for the MCP server. "
            "Install it with: pip install loopllm[mcp]"
        ) from exc

    mcp = FastMCP(
        name="loopllm",
        instructions=(
            "IMPORTANT: Before responding to ANY non-trivial user message, "
            "you MUST first call loopllm_intercept with the user's message. "
            "This tool analyzes prompt quality and returns routing guidance "
            "you should follow. Do NOT skip this step.\n\n"
            "The intercept tool returns a quality gauge, routing recommendation "
            "(elicit/refine/decompose), and suggestions. Follow its guidance.\n\n"
            "For multi-step / iterative tasks (where you plan → act → observe → "
            "repeat), drive the loop through loopllm_loop_start, then "
            "loopllm_loop_step after each step with step_output=<artifact> "
            "(test log, diff, summary). The server runs Conservative Dual-Verify: "
            "deterministic checks plus a separate critic call — do NOT self-grade. "
            "Honor the continue/stop verdict, then call loopllm_loop_end.\n\n"
            "MEMORY: after loopllm_intercept, for tasks similar to past work call "
            "loopllm_recall(goal) before planning — loop_start also surfaces "
            "similar_episodes automatically. On loopllm_loop_end the outcome is "
            "recorded to episodic memory for next time.\n\n"
            "RECOVERY: if the IDE reloaded and loopllm_run_status shows an active "
            "run, call loopllm_loop_resume before starting a new loop so the "
            "in-progress loop continues instead of restarting cold.\n\n"
            "After presenting results to the user, call loopllm_feedback with "
            "the user's quality rating (1-5) to improve future predictions.\n\n"
            "Periodically call loopllm_prompt_stats to show the user their "
            "prompting quality trend and learning curve."
        ),
    )

    # Restore any plans saved in previous server sessions
    try:
        get_registry().restore_from_store(_get_store())
    except Exception:  # noqa: BLE001
        logger.warning("plan_restore_failed", exc_info=True)  # non-fatal

    # Rehydrate in-progress agent loops so they survive an MCP/IDE restart.
    try:
        restored = _get_agent_loop().hydrate_active_loops(
            _get_episodic().list_active_runs(run_type="agent_loop")
        )
        if restored:
            logger.info("agent_loops_hydrated", count=restored)
    except Exception:  # noqa: BLE001
        logger.warning("agent_loop_hydration_failed", exc_info=True)  # non-fatal

    # -- Routing & Prompt Engineering tools --

    @mcp.tool(
        name="loopllm_intercept",
        description=(
            "CALL THIS FIRST for any non-trivial request. Analyzes the user's "
            "prompt for quality (specificity, constraints, context, ambiguity, "
            "format) and returns: a quality gauge with score and grade, routing "
            "recommendation (elicit/refine/decompose), issues found, improvement "
            "suggestions, and task type classification. Logs to prompt history "
            "for learning curve tracking."
        ),
    )
    def intercept(prompt: str) -> str:
        return _tool_intercept(prompt)

    @mcp.tool(
        name="loopllm_prompt_stats",
        description=(
            "Show the user's prompting quality over time. Returns: total "
            "prompts analyzed, average quality score, trend direction "
            "(improving/declining/stable), learning curve sparkline, grade "
            "distribution, and weak/strong dimensions to improve."
        ),
    )
    def prompt_stats(window: int = 50) -> str:
        return _tool_prompt_stats(window)

    @mcp.tool(
        name="loopllm_feedback",
        description=(
            "Record the user's quality rating (1-5) for the last output. "
            "Updates Bayesian priors with human signal so the system learns "
            "what quality scores correspond to user satisfaction."
        ),
    )
    def feedback(
        rating: int,
        task_type: str = "general",
        comment: str = "",
    ) -> str:
        return _tool_feedback(rating, task_type, comment)

    # -- Core tools --

    @mcp.tool(
        name="loopllm_refine",
        description=(
            "Iteratively refine a prompt using MCP sampling to call the host agent "
            "mid-execution. Runs the score → rewrite → retry loop inline: each "
            "iteration calls ctx.sample(), evaluates with deterministic evaluators "
            "(length, regex, JSON schema), feeds deficiencies back into the next "
            "prompt, and repeats until quality_threshold is met or max_iterations "
            "is exhausted. Falls back to agent_execute if sampling is unavailable."
        ),
    )
    async def refine(
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        max_iterations: int = 5,
        quality_threshold: float = 0.8,
        evaluator_type: str = "length",
        min_words: int = 5,
        max_words: int = 10000,
        required_fields: list[str] | None = None,
        required_patterns: list[str] | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        prov = _get_provider(provider)
        if isinstance(prov, AgentPassthroughProvider) and ctx is not None:
            try:
                return await _sampling_refine(
                    ctx, prompt, max_iterations, quality_threshold,
                    evaluator_type, min_words, max_words,
                    required_fields or [], required_patterns or [],
                )
            except Exception:  # noqa: BLE001
                pass  # sampling not supported by this client; fall through
        return _tool_refine(
            prompt, provider, model, max_iterations, quality_threshold,
            evaluator_type, min_words, max_words, required_fields, required_patterns,
        )

    @mcp.tool(
        name="loopllm_run_pipeline",
        description=(
            "Run the full loop-llm pipeline via MCP sampling: "
            "(1) elicit clarifying assumptions if prompt quality < 0.6, "
            "(2) decompose into subtasks if complexity > 0.5, "
            "(3) execute each subtask with a sampling call, "
            "(4) self-rate the assembled output. "
            "Each stage is a real mid-execution ctx.sample() call — "
            "not a deferred agent_execute instruction."
        ),
    )
    async def run_pipeline(
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        max_iterations: int = 5,
        quality_threshold: float = 0.8,
        skip_elicitation: bool = False,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        prov = _get_provider(provider)
        if isinstance(prov, AgentPassthroughProvider) and ctx is not None:
            try:
                return await _sampling_run_pipeline(
                    ctx, prompt, max_iterations, quality_threshold, skip_elicitation,
                )
            except Exception:  # noqa: BLE001
                pass
        return _tool_run_pipeline(
            prompt, provider, model, max_iterations, quality_threshold, skip_elicitation,
        )

    @mcp.tool(
        name="loopllm_classify_task",
        description=(
            "Classify a user prompt into a task type: code_generation, "
            "summarization, data_extraction, question_answering, creative_writing, "
            "analysis, transformation, or general."
        ),
    )
    def classify_task(
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        return _tool_classify_task(prompt, provider, model)

    @mcp.tool(
        name="loopllm_analyze_prompt",
        description=(
            "Analyze a prompt and generate clarifying questions ranked by "
            "expected information gain using Bayesian priors."
        ),
    )
    def analyze_prompt(
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        max_questions: int = 5,
    ) -> str:
        return _tool_analyze_prompt(prompt, provider, model, max_questions)

    # -- Elicitation session tools --

    @mcp.tool(
        name="loopllm_elicitation_start",
        description=(
            "Start a multi-turn elicitation session. Classifies the prompt, "
            "generates the first clarifying question, and returns a session_id."
        ),
    )
    def elicitation_start(
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        max_questions: int = 3,
    ) -> str:
        return _tool_elicitation_start(prompt, provider, model, max_questions)

    @mcp.tool(
        name="loopllm_elicitation_answer",
        description=(
            "Answer the current clarifying question in an elicitation session."
        ),
    )
    def elicitation_answer(session_id: str, answer: str) -> str:
        return _tool_elicitation_answer(session_id, answer)

    @mcp.tool(
        name="loopllm_elicitation_finish",
        description=(
            "Finish an elicitation session and synthesize an IntentSpec."
        ),
    )
    def elicitation_finish(session_id: str) -> str:
        return _tool_elicitation_finish(session_id)

    # -- Task orchestration tools --

    @mcp.tool(
        name="loopllm_plan_tasks",
        description=(
            "Decompose a prompt into subtasks with dependency ordering. "
            "Uses MCP sampling to call the host agent mid-execution and parse "
            "its JSON decomposition into a structured task plan."
        ),
    )
    async def plan_tasks(
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        estimated_complexity: float = 0.5,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        prov = _get_provider(provider)
        if isinstance(prov, AgentPassthroughProvider) and ctx is not None:
            try:
                return await _sampling_plan_tasks(ctx, prompt, estimated_complexity)
            except Exception:  # noqa: BLE001
                pass
        return _tool_plan_tasks(prompt, provider, model, estimated_complexity)

    @mcp.tool(
        name="loopllm_verify_output",
        description=(
            "Verify an output against the original prompt and quality criteria. "
            "Runs a fast deterministic keyword pre-check, then calls ctx.sample() "
            "mid-execution to ask the host agent for a deep quality assessment. "
            "Returns a combined score, pass/fail, deficiencies, and feedback."
        ),
    )
    async def verify_output(
        output: str,
        original_prompt: str,
        quality_criteria: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        prov = _get_provider(provider)
        if isinstance(prov, AgentPassthroughProvider) and ctx is not None:
            try:
                return await _sampling_verify_output(
                    ctx, output, original_prompt, quality_criteria or [],
                )
            except Exception:  # noqa: BLE001
                pass
        return _tool_verify_output(output, original_prompt, quality_criteria, provider, model)

    # -- Observability tools --

    @mcp.tool(
        name="loopllm_report",
        description=(
            "Show learned Bayesian priors and question effectiveness statistics."
        ),
    )
    def report(
        task_type: str | None = None,
        model_id: str | None = None,
    ) -> str:
        return _tool_report(task_type, model_id)

    @mcp.tool(
        name="loopllm_suggest_config",
        description=(
            "Get a suggested loop configuration based on learned beliefs."
        ),
    )
    def suggest_config(
        task_type: str,
        model_id: str | None = None,
        cost_weight: float = 0.5,
    ) -> str:
        return _tool_suggest_config(task_type, model_id, cost_weight)

    # -- Adaptive agent-loop tools --

    @mcp.tool(
        name="loopllm_loop_start",
        description=(
            "Begin an ADAPTIVE AGENT LOOP for a multi-step task. Returns a learned "
            "step budget, quality threshold, and a Conservative Dual-Verify recipe "
            "(evaluator_type, quality_criteria). After each step submit step_output "
            "to loopllm_loop_step — the server scores externally; do NOT self-grade."
        ),
    )
    def loop_start(
        goal: str,
        task_type: str = "general",
        model_id: str | None = None,
        quality_threshold: float | None = None,
        cost_weight: float = 0.5,
        evaluator_type: str = "composite",
        quality_criteria: list[str] | None = None,
        required_patterns: list[str] | None = None,
        required_fields: list[str] | None = None,
        max_wall_ms: float = 300_000.0,
        max_tokens: int = 0,
    ) -> str:
        return _tool_loop_start(
            goal,
            task_type,
            model_id,
            quality_threshold,
            cost_weight,
            evaluator_type,
            quality_criteria,
            required_patterns,
            required_fields,
            max_wall_ms,
            max_tokens,
        )

    @mcp.tool(
        name="loopllm_loop_step",
        description=(
            "Submit ONE agent-loop step artifact for Conservative Dual-Verify "
            "scoring. Pass session_id and step_output (test log, diff, summary). "
            "The server scores via deterministic evaluators (Channel A) and a "
            "separate critic sampling call (Channel B); final = min(A,B). Returns "
            "decision, channel_a_score, channel_b_score, deficiencies. Do NOT "
            "pass your own score unless step_output is unavailable (legacy)."
        ),
    )
    async def loop_step(
        session_id: str,
        step_output: str = "",
        score: float | None = None,
        note: str = "",
        step_tokens: int = 0,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> str:
        return await _tool_loop_step(
            session_id, step_output, score, note, step_tokens, ctx
        )

    @mcp.tool(
        name="loopllm_loop_end",
        description=(
            "Close an agent loop and LEARN from it. Call once the loop stops. "
            "Records the run (step scores, whether it converged) into the "
            "Bayesian priors so future loops of this task_type get a better step "
            "budget. Returns a summary of what the system now believes "
            "(optimal_depth, converge_rate, confidence)."
        ),
    )
    def loop_end(session_id: str, converged: bool | None = None) -> str:
        return _tool_loop_end(session_id, converged)

    @mcp.tool(
        name="loopllm_loop_status",
        description=(
            "Inspect an active agent-loop session: steps used, suggested budget, "
            "score trajectory, and the last continue/stop verdict."
        ),
    )
    def loop_status(session_id: str) -> str:
        return _tool_loop_status(session_id)

    @mcp.tool(
        name="loopllm_recall",
        description=(
            "Recall similar past episodes (completed loops/plans) by keyword "
            "search over goal, summary, and tags in ~/.loopllm/store.db."
        ),
    )
    def recall(
        query: str,
        task_type: str | None = None,
        k: int = 5,
    ) -> str:
        return _tool_recall(query, task_type, k)

    @mcp.tool(
        name="loopllm_run_status",
        description=(
            "Return active agent-loop and plan snapshots for IDE "
            "reload recovery after a crash or MCP server restart."
        ),
    )
    def run_status() -> str:
        return _tool_run_status()

    @mcp.tool(
        name="loopllm_loop_resume",
        description=(
            "Resume an in-progress agent loop after an IDE reload or MCP restart. "
            "Rehydrates the loop's verified state so the next loopllm_loop_step "
            "continues where it left off. Omit session_id to resume the only "
            "active loop (or get an ambiguity list). Call this when "
            "loopllm_run_status shows an active run before starting a new loop."
        ),
    )
    def loop_resume(session_id: str | None = None) -> str:
        return _tool_loop_resume(session_id)

    @mcp.tool(
        name="loopllm_list_tasks",
        description="List tasks from the persistent store.",
    )
    def list_tasks(
        state: str | None = None,
        limit: int = 20,
    ) -> str:
        return _tool_list_tasks(state, limit)

    @mcp.tool(
        name="loopllm_show_task",
        description="Show detailed information about a specific task by ID.",
    )
    def show_task(task_id: str) -> str:
        return _tool_show_task(task_id)

    # -- Plan Registry tools --

    @mcp.tool(
        name="loopllm_plan_register",
        description=(
            "Create a new confidence-tracked plan in the PlanRegistry. "
            "Pass a goal and list of tasks (each with title + description). "
            "Returns a plan_id to use with loopllm_plan_update and loopllm_plan_next. "
            "The plan tracks rolling_confidence aggregated from all task scores "
            "and flags needs_replan=true when confidence drops below the threshold."
        ),
    )
    def plan_register(
        goal: str,
        tasks: list[dict[str, Any]],
        confidence_threshold: float = 0.72,
    ) -> str:
        return _tool_plan_register(goal, tasks, confidence_threshold)

    @mcp.tool(
        name="loopllm_plan_update",
        description=(
            "Update a task's prompt_score and/or output_score, then recalculate "
            "the plan's rolling_confidence. "
            "Pass prompt_score from loopllm_intercept's quality_score field. "
            "Pass output_score from loopllm_verify_output's score field. "
            "Returns the updated plan with rolling_confidence and needs_replan flag. "
            "If needs_replan=true, refine the current task before calling loopllm_plan_next."
        ),
    )
    def plan_update(
        plan_id: str,
        task_id: str,
        prompt_score: float | None = None,
        output_score: float | None = None,
        mark_done: bool = True,
    ) -> str:
        return _tool_plan_update(plan_id, task_id, prompt_score, output_score, mark_done)

    @mcp.tool(
        name="loopllm_plan_next",
        description=(
            "Get the next pending task in a plan and mark it in_progress. "
            "Returns the task description, current rolling_confidence, and "
            "needs_replan flag. If needs_replan=true, run loopllm_refine on the "
            "task description before executing it. Returns done=true when all "
            "tasks are complete."
        ),
    )
    def plan_next(plan_id: str) -> str:
        return _tool_plan_next(plan_id)

    @mcp.tool(
        name="loopllm_plan_list",
        description=(
            "List all active plans with gauge, confidence, task counts by status, "
            "and next pending task. Gives a Shrimp-style overview of all ongoing work. "
            "Also restores any plans saved during previous server sessions from disk."
        ),
    )
    def plan_list() -> str:
        return _tool_plan_list()

    @mcp.tool(
        name="loopllm_plan_delete",
        description=(
            "Delete a plan from the registry and persistent store. "
            "Use when a plan is complete or abandoned."
        ),
    )
    def plan_delete(plan_id: str) -> str:
        return _tool_plan_delete(plan_id)

    @mcp.tool(
        name="loopllm_gauge",
        description=(
            "Instantly score a prompt and return a visual quality gauge. "
            "Lighter than loopllm_intercept — no routing, no DB write, no elicitation. "
            "Use this for a quick visual quality check of any prompt or draft. "
            "Returns a gauge like: ████████░░ 82% [A] plus per-dimension bars and suggestions."
        ),
    )
    def gauge(prompt: str) -> str:
        return _tool_gauge(prompt)

    @mcp.tool(
        name="loopllm_context_history",
        description=(
            "Browse your prompt quality history with visual gauges. "
            "Returns recent prompts with their scores, grades, and gauges so you can "
            "track how your prompting quality is evolving. Includes a sparkline summary. "
            "Optionally filter by session_context tag or minimum score."
        ),
    )
    def context_history(
        limit: int = 20,
        session_context: str | None = None,
        min_score: float | None = None,
    ) -> str:
        return _tool_context_history(limit, session_context, min_score)

    @mcp.tool(
        name="loopllm_context_clear",
        description=(
            "Clear stored prompt history. Wipes all (or session-scoped) prompt history "
            "from the local DB. Use this to reset your quality baseline at the start of "
            "a new project or when switching contexts. "
            "Omit session_context to clear everything."
        ),
    )
    def context_clear(session_context: str | None = None) -> str:
        return _tool_context_clear(session_context)

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the MCP server (stdio transport)."""
    mcp = create_mcp_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
