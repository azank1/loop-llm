"""REST API server exposing loopllm scoring to local models.

Starts a lightweight HTTP server (FastAPI + uvicorn) that exposes:

  POST /score          — score a prompt+output pair, return quality metrics
  POST /rewrite        — score + return a rewritten prompt if below threshold
  GET  /intercept      — run loopllm_intercept on a prompt
  POST /plan/register  — create a new plan in the PlanRegistry
  POST /plan/update    — update task scores and get confidence status
  GET  /plan/{plan_id} — get full plan status
  GET  /health         — health check

This is the bridge that lets local models (Ollama, llama.cpp, LM Studio)
use loopllm as a scoring middleware without needing MCP tool-calling support.

Usage::

    loopllm serve --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import json
from typing import Any

from loopllm.mcp_server import (
    _init_state,
    _score_prompt_quality,
    _classify_task_type,
    _estimate_complexity,
    _build_evaluator,
    _tool_intercept,
)
from loopllm.evaluators import LengthEvaluator
from loopllm.plan_registry import get_registry


# ---------------------------------------------------------------------------
# Request / response models (Pydantic, only imported when FastAPI available)
# ---------------------------------------------------------------------------


def _get_app() -> Any:
    """Build and return the FastAPI application.

    Deferred import so the rest of the package doesn't require FastAPI.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError(
            "FastAPI and uvicorn are required for `loopllm serve`.\n"
            "Install with: pip install loopllm[serve]"
        ) from exc

    _init_state()
    app = FastAPI(
        title="loopllm scoring API",
        description=(
            "Quality scoring and prompt optimization middleware for local LLMs. "
            "POST your prompt+output to /score to get quality metrics and a "
            "rewritten prompt if needed."
        ),
        version="0.5.0",
    )

    # -- Pydantic models -------------------------------------------------------

    class ScoreRequest(BaseModel):
        prompt: str
        output: str
        evaluator_type: str = "length"
        min_words: int = 5
        max_words: int = 10_000
        required_fields: list[str] = []
        required_patterns: list[str] = []
        quality_threshold: float = 0.80

    class RewriteRequest(BaseModel):
        prompt: str
        output: str
        iteration: int = 0
        max_retries: int = 3
        evaluator_type: str = "length"
        min_words: int = 5
        max_words: int = 10_000
        quality_threshold: float = 0.80

    class InterceptRequest(BaseModel):
        prompt: str

    class PlanRegisterRequest(BaseModel):
        goal: str
        tasks: list[dict[str, Any]]
        confidence_threshold: float = 0.72

    class PlanUpdateRequest(BaseModel):
        plan_id: str
        task_id: str
        prompt_score: float | None = None
        output_score: float | None = None
        mark_done: bool = True

    class PlanNextRequest(BaseModel):
        plan_id: str

    # -- Endpoints -------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "loopllm"}

    @app.post("/score")
    def score(req: ScoreRequest) -> JSONResponse:
        """Score a prompt+output pair.

        Returns prompt_score, output_score, combined_score, passed,
        deficiencies, and grade.
        """
        # Prompt quality (heuristic)
        prompt_quality = _score_prompt_quality(req.prompt)
        prompt_score = prompt_quality["quality_score"]

        # Output quality (evaluator)
        evaluator = _build_evaluator(
            req.evaluator_type,
            min_words=req.min_words,
            max_words=req.max_words,
            required_fields=req.required_fields,
            required_patterns=req.required_patterns,
        )
        eval_result = evaluator.evaluate(req.output)
        output_score = eval_result.score

        # Combined (prompt has lower weight — see PlanRegistry)
        combined = prompt_score * 0.35 + output_score * 0.65
        passed = combined >= req.quality_threshold

        return JSONResponse({
            "prompt_score": round(prompt_score, 4),
            "output_score": round(output_score, 4),
            "combined_score": round(combined, 4),
            "passed": passed,
            "quality_threshold": req.quality_threshold,
            "deficiencies": eval_result.deficiencies,
            "prompt_grade": prompt_quality["grade"],
            "prompt_gauge": prompt_quality["gauge"],
            "prompt_issues": prompt_quality["issues"],
            "prompt_suggestions": prompt_quality["suggestions"],
        })

    @app.post("/rewrite")
    def rewrite(req: RewriteRequest) -> JSONResponse:
        """Score output and return a rewritten prompt if quality is below threshold.

        If passed=True the response also contains rewritten_prompt=null —
        meaning no retry is needed.
        """
        prompt_quality = _score_prompt_quality(req.prompt)
        prompt_score = prompt_quality["quality_score"]

        evaluator = _build_evaluator(
            req.evaluator_type,
            min_words=req.min_words,
            max_words=req.max_words,
        )
        eval_result = evaluator.evaluate(req.output)
        output_score = eval_result.score
        combined = prompt_score * 0.35 + output_score * 0.65
        passed = combined >= req.quality_threshold

        rewritten: str | None = None
        if not passed and req.iteration < req.max_retries:
            deficiency_str = (
                "\n".join(f"  - {d}" for d in eval_result.deficiencies)
                if eval_result.deficiencies
                else "  - Output did not meet quality threshold"
            )
            rewritten = (
                f"[LOOPLLM | score={combined:.2f} | "
                f"retry={req.iteration + 1}/{req.max_retries} | "
                f"threshold={req.quality_threshold:.2f}]\n"
                f"Your previous response scored {combined:.2f}/1.0.\n"
                f"Issues to fix:\n{deficiency_str}\n\n"
                f"Original task:\n{req.prompt}\n\n"
                f"Previous response (do not repeat):\n{req.output[:500]}\n\n"
                f"Please produce an improved response that addresses all issues."
            )

        return JSONResponse({
            "prompt_score": round(prompt_score, 4),
            "output_score": round(output_score, 4),
            "combined_score": round(combined, 4),
            "passed": passed,
            "quality_threshold": req.quality_threshold,
            "deficiencies": eval_result.deficiencies,
            "rewritten_prompt": rewritten,
            "should_retry": not passed and req.iteration < req.max_retries,
            "iteration": req.iteration,
        })

    @app.post("/intercept")
    def intercept(req: InterceptRequest) -> JSONResponse:
        """Run loopllm_intercept on a prompt (same as the MCP tool)."""
        result = _tool_intercept(req.prompt)
        return JSONResponse(json.loads(result))

    # -- Plan endpoints --------------------------------------------------------

    @app.post("/plan/register")
    def plan_register(req: PlanRegisterRequest) -> JSONResponse:
        """Create a new plan in the PlanRegistry."""
        registry = get_registry()
        plan = registry.create(
            goal=req.goal,
            tasks=req.tasks,
            confidence_threshold=req.confidence_threshold,
        )
        return JSONResponse(plan.to_dict())

    @app.post("/plan/update")
    def plan_update(req: PlanUpdateRequest) -> JSONResponse:
        """Score a task's prompt and/or output and get updated plan confidence."""
        registry = get_registry()
        result: dict[str, Any] = {}

        if req.prompt_score is not None:
            result = registry.score_prompt(req.plan_id, req.task_id, req.prompt_score)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])

        if req.output_score is not None:
            result = registry.score_output(
                req.plan_id, req.task_id, req.output_score, mark_done=req.mark_done
            )
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])

        if not result:
            result = registry.get_status(req.plan_id)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])

        return JSONResponse(result)

    @app.get("/plan/{plan_id}")
    def plan_status(plan_id: str) -> JSONResponse:
        """Get the current status and rolling confidence of a plan."""
        registry = get_registry()
        result = registry.get_status(plan_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return JSONResponse(result)

    @app.post("/plan/next")
    def plan_next(req: PlanNextRequest) -> JSONResponse:
        """Get and activate the next pending task in a plan."""
        registry = get_registry()
        task = registry.next_task(req.plan_id)
        if task is None:
            return JSONResponse({"done": True, "plan_id": req.plan_id})
        return JSONResponse({**task, "done": False})

    @app.get("/plan")
    def list_plans() -> JSONResponse:
        """List all active plans."""
        registry = get_registry()
        return JSONResponse({"plans": registry.list_plans()})

    return app


def run_server(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    """Start the loopllm scoring REST server.

    Args:
        host: Bind address.
        port: Port to listen on.
        reload: Enable auto-reload (development only).
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required for `loopllm serve`.\n"
            "Install with: pip install loopllm[serve]"
        ) from exc

    # Build the app once to surface import errors before uvicorn starts
    _get_app()

    uvicorn.run(
        "loopllm.serve:_get_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
