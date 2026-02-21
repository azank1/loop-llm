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
import time
from pathlib import Path
from typing import Any

from loopllm.elicitation import ElicitationSession, IntentRefiner, IntentSpec
from loopllm.engine import CompositeEvaluator, LoopConfig, LoopedLLM
from loopllm.evaluators import (
    JSONSchemaEvaluator,
    LengthEvaluator,
    RegexEvaluator,
)
from loopllm.priors import CallObservation
from loopllm.provider import LLMProvider
from loopllm.store import LoopStore, SQLiteBackedPriors
from loopllm.tasks import TaskOrchestrator

# ---------------------------------------------------------------------------
# Shared state — initialised once per MCP server process
# ---------------------------------------------------------------------------

_store: LoopStore | None = None
_priors: SQLiteBackedPriors | None = None
_provider: LLMProvider | None = None
_default_model: str = "gpt-4o-mini"
_active_sessions: dict[str, dict[str, Any]] = {}
_status_path: Path | None = None


def _init_state() -> None:
    """Lazily initialise shared store, priors, and provider."""
    global _store, _priors, _provider, _default_model, _status_path  # noqa: PLW0603

    if _store is not None:
        return

    db_path = Path(os.environ.get("LOOPLLM_DB", str(Path.home() / ".loopllm" / "store.db")))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _store = LoopStore(db_path=db_path)
    _priors = SQLiteBackedPriors(_store)
    _default_model = os.environ.get("LOOPLLM_MODEL", "gpt-4o-mini")
    _status_path = db_path.parent / "status.json"

    provider_name = os.environ.get("LOOPLLM_PROVIDER", "mock")
    _provider = _make_provider(provider_name)


def _make_provider(name: str) -> LLMProvider:
    """Create an LLM provider by name."""
    if name == "mock":
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
    if provider_override and provider_override != os.environ.get("LOOPLLM_PROVIDER", "mock"):
        return _make_provider(provider_override)
    assert _provider is not None
    return _provider


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


def _build_evaluator(
    evaluator_type: str = "length",
    **kwargs: Any,
) -> LengthEvaluator | RegexEvaluator | JSONSchemaEvaluator | CompositeEvaluator:
    """Build an evaluator from a type string and optional config."""
    if evaluator_type == "json":
        return JSONSchemaEvaluator(
            required_fields=kwargs.get("required_fields", []),
            field_types={},
        )
    elif evaluator_type == "regex":
        return RegexEvaluator(
            required=kwargs.get("required_patterns", []),
            forbidden=kwargs.get("forbidden_patterns", []),
        )
    elif evaluator_type == "composite":
        evals: list[Any] = []
        if kwargs.get("required_fields"):
            evals.append(JSONSchemaEvaluator(required_fields=kwargs["required_fields"]))
        if kwargs.get("required_patterns"):
            evals.append(RegexEvaluator(required=kwargs["required_patterns"]))
        evals.append(LengthEvaluator(
            min_words=kwargs.get("min_words", 5),
            max_words=kwargs.get("max_words", 10_000),
        ))
        return CompositeEvaluator(evaluators=evals)
    else:
        return LengthEvaluator(
            min_words=kwargs.get("min_words", 5),
            max_words=kwargs.get("max_words", 10_000),
        )


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
    weights = {
        "specificity": 0.25,
        "constraint_clarity": 0.20,
        "context_completeness": 0.20,
        "ambiguity": 0.20,
        "format_spec": 0.15,
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

    result = {
        "route": route,
        "reason": reason,
        "next_tool": next_tool,
        "quality": quality,
        "task_type": task_type,
        "complexity": complexity,
        "prior_knowledge": config,
    }

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

    _write_status("prompt_stats", {
        "total_prompts": stats.get("total_prompts", 0),
        "avg_quality": stats.get("avg_quality", 0),
        "trend": stats.get("trend", "no_data"),
    })

    return json.dumps(stats, indent=2, default=str)


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

    result: dict[str, Any] = {
        "status": "recorded",
        "rating": clamped,
        "normalized_score": normalized,
        "task_type": task_type,
        "model": model,
        "impact": ("Priors updated — future predictions for this task "
                   "type will be adjusted"),
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
    mod = _get_model(model)
    priors = _get_priors()

    refiner = IntentRefiner(provider=prov, priors=priors, model=mod)
    task_type = refiner.classify_task(prompt)
    return json.dumps({"task_type": task_type})


def _tool_analyze_prompt(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    max_questions: int = 5,
) -> str:
    """Analyze a prompt and generate ranked clarifying questions."""
    prov = _get_provider(provider)
    mod = _get_model(model)
    priors = _get_priors()

    refiner = IntentRefiner(provider=prov, priors=priors, model=mod, max_questions=max_questions)
    questions = refiner.analyze(prompt)

    return json.dumps([
        {
            "question": q.text,
            "question_type": q.question_type,
            "options": q.options,
            "information_gain": round(q.information_gain, 4),
        }
        for q in questions
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

    refiner = IntentRefiner(provider=prov, priors=priors, model=mod, max_questions=max_questions)
    session = ElicitationSession(original_prompt=prompt, model_id=mod)
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

    result: dict[str, Any] = {
        "session_id": session.session_id,
        "task_type": session.task_type,
    }

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

    return json.dumps(result, indent=2)


def _tool_elicitation_answer(session_id: str, answer: str) -> str:
    """Answer the current question and get the next one."""
    if session_id not in _active_sessions:
        return json.dumps({"error": f"Session not found: {session_id}"})

    state = _active_sessions[session_id]
    session: ElicitationSession = state["session"]
    refiner: IntentRefiner = state["refiner"]
    store = _get_store()

    if session.questions_asked:
        last_q = session.questions_asked[-1]
        session.answers[last_q.question_type] = answer

    question = refiner.ask(session)

    result: dict[str, Any] = {"session_id": session_id}

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
    refiner: IntentRefiner = state["refiner"]
    store = _get_store()

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
            "After presenting results to the user, call loopllm_feedback with "
            "the user's quality rating (1-5) to improve future predictions.\n\n"
            "Periodically call loopllm_prompt_stats to show the user their "
            "prompting quality trend and learning curve."
        ),
    )

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
            "Iteratively refine an LLM prompt with evaluation-driven feedback. "
            "Calls the LLM, evaluates the output with deterministic evaluators "
            "(length, regex, JSON schema), feeds deficiencies back as feedback, "
            "and repeats until quality threshold is met or iterations exhausted."
        ),
    )
    def refine(
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
        return _tool_refine(
            prompt, provider, model, max_iterations, quality_threshold,
            evaluator_type, min_words, max_words, required_fields, required_patterns,
        )

    @mcp.tool(
        name="loopllm_run_pipeline",
        description=(
            "Run the full loop-llm pipeline: classify task -> generate clarifying "
            "questions -> decompose into subtasks -> execute each through the "
            "refinement loop -> assemble final output."
        ),
    )
    def run_pipeline(
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        max_iterations: int = 5,
        quality_threshold: float = 0.8,
        skip_elicitation: bool = False,
    ) -> str:
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
            "Decompose a prompt into subtasks with dependency ordering."
        ),
    )
    def plan_tasks(
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        estimated_complexity: float = 0.5,
    ) -> str:
        return _tool_plan_tasks(prompt, provider, model, estimated_complexity)

    @mcp.tool(
        name="loopllm_verify_output",
        description=(
            "Verify an output against the original prompt and quality criteria."
        ),
    )
    def verify_output(
        output: str,
        original_prompt: str,
        quality_criteria: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
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
