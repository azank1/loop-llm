"""MCP server exposing loop-llm tools to IDE agents.

Provides iterative refinement, intent elicitation, task orchestration,
and Bayesian meta-learning as MCP tools for VS Code Copilot, Cursor,
and other MCP-compatible clients.

Usage::

    loopllm mcp-server --provider ollama --model qwen2.5:0.5b
    # or
    python -m loopllm.mcp_server
"""
from __future__ import annotations

import json
import os
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


def _init_state() -> None:
    """Lazily initialise shared store, priors, and provider."""
    global _store, _priors, _provider, _default_model  # noqa: PLW0603

    if _store is not None:
        return

    db_path = Path(os.environ.get("LOOPLLM_DB", str(Path.home() / ".loopllm" / "store.db")))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _store = LoopStore(db_path=db_path)
    _priors = SQLiteBackedPriors(_store)
    _default_model = os.environ.get("LOOPLLM_MODEL", "gpt-4o-mini")

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
# Tool implementations (sync — FastMCP wraps them in threads)
# ---------------------------------------------------------------------------


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

    # Record observation
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

    return json.dumps(_result_to_dict(result), indent=2)


def _tool_run_pipeline(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    max_iterations: int = 5,
    quality_threshold: float = 0.8,
    skip_elicitation: bool = False,
) -> str:
    """Run the full pipeline: elicit → decompose → execute → verify."""
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

    answer_func = None  # Non-interactive in MCP context

    result = orchestrator.run(prompt, model=mod, answer_func=answer_func)
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


# -- Elicitation session tools (stateful, multi-turn) --


def _tool_elicitation_start(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    max_questions: int = 3,
) -> str:
    """Start a new elicitation session. Returns session_id and first question."""
    prov = _get_provider(provider)
    mod = _get_model(model)
    store = _get_store()
    priors = _get_priors()

    refiner = IntentRefiner(provider=prov, priors=priors, model=mod, max_questions=max_questions)

    # Create session
    session = ElicitationSession(original_prompt=prompt, model_id=mod)
    session.task_type = refiner.classify_task(prompt)

    # Get first question
    question = refiner.ask(session)

    # Store in-memory state and persist
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


def _tool_elicitation_answer(
    session_id: str,
    answer: str,
) -> str:
    """Answer the current question and get the next one (or None if done)."""
    if session_id not in _active_sessions:
        return json.dumps({"error": f"Session not found: {session_id}"})

    state = _active_sessions[session_id]
    session: ElicitationSession = state["session"]
    refiner: IntentRefiner = state["refiner"]
    store = _get_store()

    # Record answer for the last asked question
    if session.questions_asked:
        last_q = session.questions_asked[-1]
        session.answers[last_q.question_type] = answer

    # Get next question
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

    # Persist answers
    store.update_session(
        session_id,
        answers=session.answers,
        questions=[
            {"text": q.text, "type": q.question_type}
            for q in session.questions_asked
        ],
    )

    return json.dumps(result, indent=2)


def _tool_elicitation_finish(
    session_id: str,
) -> str:
    """Finish an elicitation session and get the refined IntentSpec."""
    if session_id not in _active_sessions:
        return json.dumps({"error": f"Session not found: {session_id}"})

    state = _active_sessions[session_id]
    session: ElicitationSession = state["session"]
    refiner: IntentRefiner = state["refiner"]
    store = _get_store()

    # Refine into spec
    if session.answers:
        spec = refiner.refine(session.original_prompt, session.answers)
    else:
        spec = IntentSpec(
            task_type=session.task_type,
            original_prompt=session.original_prompt,
            refined_prompt=session.original_prompt,
        )

    session.refined_spec = spec

    # Persist
    spec_dict = {
        "task_type": spec.task_type,
        "refined_prompt": spec.refined_prompt,
        "constraints": spec.constraints,
        "quality_criteria": spec.quality_criteria,
        "decomposition_hints": spec.decomposition_hints,
        "estimated_complexity": spec.estimated_complexity,
    }
    store.update_session(session_id, spec=spec_dict)

    # Clean up in-memory state
    del _active_sessions[session_id]

    return json.dumps({
        "session_id": session_id,
        "spec": spec_dict,
    }, indent=2)


# -- Task orchestration tools --


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


# -- Observability tools --


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


def _tool_show_task(
    task_id: str,
) -> str:
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
    """Create and configure the FastMCP server with all tools registered.

    Returns:
        A configured ``FastMCP`` instance.

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
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
            "loop-llm: Iterative refinement engine with Bayesian adaptive exit. "
            "Use these tools to refine LLM outputs through evaluation-driven "
            "feedback loops, elicit intent through clarifying questions, "
            "decompose tasks, and learn optimal iteration depth over time."
        ),
    )

    # -- Core tools --

    @mcp.tool(
        name="loopllm_refine",
        description=(
            "Iteratively refine an LLM prompt with evaluation-driven feedback. "
            "Calls the LLM, evaluates the output with deterministic evaluators "
            "(length, regex, JSON schema), feeds deficiencies back as feedback, "
            "and repeats until quality threshold is met or iterations exhausted. "
            "Returns the best output across all iterations."
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
            "Run the full loop-llm pipeline: classify task → generate clarifying "
            "questions → decompose into subtasks → execute each through the "
            "refinement loop → assemble final output. Best for complex tasks "
            "that benefit from decomposition."
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
            "expected information gain. Questions are categorised (scope, format, "
            "constraints, examples, edge_cases, audience, priority) and scored "
            "using Bayesian priors that improve with use."
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
            "generates the first clarifying question, and returns a session_id "
            "for subsequent calls. Use loopllm_elicitation_answer to answer "
            "questions, then loopllm_elicitation_finish to get the refined spec."
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
            "Answer the current clarifying question in an elicitation session. "
            "Returns the next question (if any) or indicates the session is "
            "complete. Call loopllm_elicitation_finish when is_complete is true."
        ),
    )
    def elicitation_answer(
        session_id: str,
        answer: str,
    ) -> str:
        return _tool_elicitation_answer(session_id, answer)

    @mcp.tool(
        name="loopllm_elicitation_finish",
        description=(
            "Finish an elicitation session and synthesize a structured IntentSpec "
            "from the original prompt and all answers. The spec includes: "
            "refined_prompt, constraints, quality_criteria, decomposition_hints, "
            "and estimated_complexity."
        ),
    )
    def elicitation_finish(
        session_id: str,
    ) -> str:
        return _tool_elicitation_finish(session_id)

    # -- Task orchestration tools --

    @mcp.tool(
        name="loopllm_plan_tasks",
        description=(
            "Decompose a prompt into subtasks with dependency ordering. "
            "Returns a task plan with IDs, titles, descriptions, and "
            "execution order (topological sort). Simple tasks (complexity < 0.3) "
            "are kept as a single task."
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
            "Verify an output against the original prompt and quality criteria "
            "using LLM-based evaluation. Returns score, pass/fail, and "
            "specific deficiencies found."
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
            "Show learned Bayesian priors and question effectiveness statistics. "
            "Displays per-(task_type, model) beliefs: optimal iteration depth, "
            "convergence rate, first-call quality, per-iteration expected scores. "
            "Also shows which clarifying question types are most effective."
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
            "Get a suggested loop configuration based on learned beliefs for a "
            "given task type and model. Returns recommended max_iterations and "
            "quality_threshold optimised by historical observations."
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
        description=(
            "List tasks from the persistent store. Optionally filter by state "
            "(pending, in_progress, completed, verified, failed, blocked)."
        ),
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
    def show_task(
        task_id: str,
    ) -> str:
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
