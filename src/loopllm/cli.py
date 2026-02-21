"""Command-line interface for loop-llm."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loopllm.elicitation import ClarifyingQuestion, IntentRefiner
from loopllm.engine import LoopConfig, LoopedLLM
from loopllm.evaluators import LengthEvaluator
from loopllm.priors import CallObservation
from loopllm.provider import LLMProvider
from loopllm.store import LoopStore, SQLiteBackedPriors
from loopllm.tasks import TaskOrchestrator


def _get_provider(name: str, **kwargs: Any) -> LLMProvider:
    """Instantiate an LLM provider by name.

    Args:
        name: Provider name (``"mock"``, ``"ollama"``, or ``"openrouter"``).
        **kwargs: Extra keyword arguments forwarded to the provider constructor.

    Returns:
        An :class:`LLMProvider` instance.

    Raises:
        SystemExit: If the provider name is unknown.
    """
    if name == "mock":
        from loopllm.providers.mock import MockLLMProvider

        responses = [
            '{"result": "initial attempt"}',
            '{"result": "improved version", "details": "added more info"}',
            '{"result": "refined output", "details": "comprehensive", "quality": "high"}',
        ]
        return MockLLMProvider(responses=responses)
    elif name == "ollama":
        from loopllm.providers.ollama import OllamaProvider

        return OllamaProvider(base_url=kwargs.get("base_url", "http://localhost:11434"))
    elif name == "openrouter":
        import os

        from loopllm.providers.openrouter import OpenRouterProvider

        api_key = kwargs.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("Error: OPENROUTER_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        return OpenRouterProvider(api_key=api_key)
    else:
        print(f"Unknown provider: {name}", file=sys.stderr)
        sys.exit(1)


def _get_store(db_path: str | None) -> LoopStore:
    """Create a LoopStore, defaulting to ~/.loopllm/store.db."""
    if db_path:
        path = Path(db_path)
    else:
        path = Path.home() / ".loopllm" / "store.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return LoopStore(db_path=path)


def _interactive_answer(question: ClarifyingQuestion) -> str:
    """Prompt the user for an answer to a clarifying question."""
    print(f"\n  [{question.question_type.upper()}] {question.text}")
    if question.options:
        for i, opt in enumerate(question.options, 1):
            print(f"    {i}. {opt}")
        print("    (enter number or free text)")

    answer = input("  > ").strip()

    # If they entered a number and we have options, map it
    if question.options and answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(question.options):
            answer = question.options[idx]

    return answer


def cmd_refine(args: argparse.Namespace) -> None:
    """Execute the ``refine`` subcommand: elicit intent then refine."""
    provider = _get_provider(args.provider)
    store = _get_store(args.db)
    priors = SQLiteBackedPriors(store)

    prompt = args.prompt
    model = args.model

    if args.no_questions:
        # Skip elicitation, go straight to refinement
        print(f"Refining: {prompt[:80]}...")
    else:
        # Run intent elicitation
        refiner = IntentRefiner(
            provider=provider,
            priors=priors,
            model=model,
            max_questions=args.max_questions,
        )

        print(f"Analyzing prompt: {prompt[:80]}...")
        session = refiner.run_session(prompt, answer_func=_interactive_answer)

        if session.refined_spec:
            print("\n--- Refined Spec ---")
            print(f"Task type: {session.refined_spec.task_type}")
            print(f"Complexity: {session.refined_spec.estimated_complexity:.1f}")
            print(f"Prompt: {session.refined_spec.refined_prompt[:200]}")
            if session.refined_spec.quality_criteria:
                print("Quality criteria:")
                for c in session.refined_spec.quality_criteria:
                    print(f"  - {c}")
            prompt = session.refined_spec.refined_prompt

    # Run refinement loop
    evaluator = LengthEvaluator(min_words=5, max_words=10_000)
    config = LoopConfig(
        max_iterations=args.max_iterations,
        quality_threshold=args.threshold,
    )
    loop = LoopedLLM(provider=provider, config=config)

    print(f"\nRunning refinement loop (max {config.max_iterations} iterations)...")
    result = loop.refine(prompt, evaluator, model=model)

    print("\n--- Result ---")
    print(f"Exit: {result.metrics.exit_reason.condition}")
    print(f"Iterations: {result.metrics.total_iterations}")
    print(f"Best score: {result.metrics.best_score:.3f}")
    print(f"Output:\n{result.output}")

    # Record observation
    obs = CallObservation(
        task_type="cli_refine",
        model_id=model,
        scores=result.metrics.score_trajectory,
        latencies_ms=[it.latency_ms for it in result.iterations],
        converged=result.metrics.converged,
        total_iterations=result.metrics.total_iterations,
        max_iterations=config.max_iterations,
        quality_threshold=config.quality_threshold,
    )
    priors.observe(obs)
    store.close()


def cmd_report(args: argparse.Namespace) -> None:
    """Execute the ``report`` subcommand: show learned priors."""
    store = _get_store(args.db)
    priors = SQLiteBackedPriors(store)
    reports = priors.report_all()

    if not reports:
        print("No observations recorded yet.")
    else:
        for r in reports:
            print(f"\n=== {r['task_type']} / {r['model_id']} ===")
            print(f"  Calls: {r['total_calls']}")
            print(f"  Optimal depth: {r['optimal_depth']}")
            print(f"  Converge rate: {r['converge_rate']:.1%}")
            print(f"  First-call quality: {r['first_call_quality']:.3f}")
            print(f"  Confidence: {r['confidence']:.3f}")
            if r.get("iterations"):
                print("  Iterations:")
                for k, v in r["iterations"].items():
                    print(
                        f"    {k}: score={v['expected_score']:.3f} "
                        f"delta={v['expected_delta']:.3f} "
                        f"converge={v['converge_prob']:.1%}"
                    )

    # Also show question stats
    stats = store.get_question_stats()
    if stats:
        print("\n=== Question Effectiveness ===")
        for s in stats:
            print(
                f"  {s['question_type']:15s}  "
                f"asked={s['asked_count']:3d}  "
                f"effectiveness={s['effectiveness']:.1%}  "
                f"info_gain={s['avg_info_gain']:.3f}"
            )

    store.close()


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` subcommand: full pipeline with task orchestration."""
    provider = _get_provider(args.provider)
    store = _get_store(args.db)
    priors = SQLiteBackedPriors(store)

    orchestrator = TaskOrchestrator(
        provider=provider,
        priors=priors,
        store=store,
        model=args.model,
    )

    answer_func = None if args.no_questions else _interactive_answer

    print(f"Running full pipeline: {args.prompt[:80]}...")
    result = orchestrator.run(
        args.prompt,
        model=args.model,
        answer_func=answer_func,
    )

    print("\n--- Result ---")
    print(f"Exit: {result.metrics.exit_reason.condition}")
    print(f"Iterations: {result.metrics.total_iterations}")
    print(f"Best score: {result.metrics.best_score:.3f}")
    print(f"Output:\n{result.output}")
    store.close()


def cmd_tasks_list(args: argparse.Namespace) -> None:
    """List tasks from the store."""
    store = _get_store(args.db)
    tasks = store.get_tasks(state=args.state, limit=args.limit)

    if not tasks:
        print("No tasks found.")
    else:
        for t in tasks:
            print(
                f"  [{t['state']:12s}] {t['id'][:8]}  {t['title']}"
            )
    store.close()


def cmd_tasks_show(args: argparse.Namespace) -> None:
    """Show details for a specific task."""
    store = _get_store(args.db)
    task = store.get_task(args.task_id)

    if task is None:
        print(f"Task not found: {args.task_id}")
    else:
        print(json.dumps(task, indent=2, default=str))
    store.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``loopllm`` CLI."""
    parser = argparse.ArgumentParser(
        prog="loopllm",
        description="Iterative refinement engine with Bayesian intent elicitation.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to SQLite database (default: ~/.loopllm/store.db)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- refine ---
    p_refine = subparsers.add_parser(
        "refine", help="Elicit intent and refine a prompt"
    )
    p_refine.add_argument("prompt", help="The prompt to refine")
    p_refine.add_argument(
        "--provider", default="mock",
        choices=["mock", "ollama", "openrouter"],
        help="LLM provider (default: mock)",
    )
    p_refine.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    p_refine.add_argument(
        "--no-questions", action="store_true",
        help="Skip intent elicitation",
    )
    p_refine.add_argument(
        "--max-questions", type=int, default=3,
        help="Maximum clarifying questions (default: 3)",
    )
    p_refine.add_argument(
        "--max-iterations", type=int, default=5,
        help="Maximum refinement iterations (default: 5)",
    )
    p_refine.add_argument(
        "--threshold", type=float, default=0.8,
        help="Quality threshold (default: 0.8)",
    )
    p_refine.set_defaults(func=cmd_refine)

    # --- run ---
    p_run = subparsers.add_parser(
        "run", help="Full pipeline: elicit → decompose → execute → verify"
    )
    p_run.add_argument("prompt", help="The prompt to process")
    p_run.add_argument(
        "--provider", default="mock",
        choices=["mock", "ollama", "openrouter"],
    )
    p_run.add_argument("--model", default="gpt-4o-mini")
    p_run.add_argument("--no-questions", action="store_true")
    p_run.add_argument("--max-questions", type=int, default=3)
    p_run.add_argument("--max-iterations", type=int, default=5)
    p_run.add_argument("--threshold", type=float, default=0.8)
    p_run.set_defaults(func=cmd_run)

    # --- report ---
    p_report = subparsers.add_parser(
        "report", help="Show learned priors and statistics"
    )
    p_report.set_defaults(func=cmd_report)

    # --- tasks ---
    p_tasks = subparsers.add_parser("tasks", help="Task management")
    tasks_sub = p_tasks.add_subparsers(dest="tasks_command")

    p_tlist = tasks_sub.add_parser("list", help="List tasks")
    p_tlist.add_argument("--state", default=None, help="Filter by state")
    p_tlist.add_argument("--limit", type=int, default=20)
    p_tlist.set_defaults(func=cmd_tasks_list)

    p_tshow = tasks_sub.add_parser("show", help="Show task details")
    p_tshow.add_argument("task_id", help="Task ID")
    p_tshow.set_defaults(func=cmd_tasks_show)

    # --- mcp-server ---
    p_mcp = subparsers.add_parser(
        "mcp-server", help="Start MCP server for IDE integration (VS Code, Cursor)"
    )
    p_mcp.add_argument(
        "--provider", default=None,
        choices=["agent", "mock", "ollama", "openrouter"],
        help="LLM provider (default: LOOPLLM_PROVIDER env or agent)",
    )
    p_mcp.add_argument(
        "--model", default=None,
        help="Default model (default: LOOPLLM_MODEL env or gpt-4o-mini)",
    )
    p_mcp.add_argument(
        "--db", default=None,
        help="Path to SQLite database (default: ~/.loopllm/store.db)",
    )
    p_mcp.set_defaults(func=cmd_mcp_server)

    # --- serve ---
    p_serve = subparsers.add_parser(
        "serve",
        help="Start REST scoring server for local models (Ollama, llama.cpp, etc.)",
    )
    p_serve.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    p_serve.add_argument(
        "--port", type=int, default=8765,
        help="Port to listen on (default: 8765)",
    )
    p_serve.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload (development only)",
    )
    p_serve.set_defaults(func=cmd_serve)

    return parser


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the loopllm scoring REST server."""
    try:
        from loopllm.serve import run_server
    except ImportError:
        print(
            "Error: FastAPI and uvicorn are required for `loopllm serve`.\n"
            "Install with: pip install loopllm[serve]",
            file=sys.stderr,
        )
        sys.exit(1)
    run_server(host=args.host, port=args.port, reload=args.reload)


def cmd_mcp_server(args: argparse.Namespace) -> None:
    """Start the MCP server for IDE integration."""
    import os

    # Pass CLI args as env vars so mcp_server.py picks them up
    if args.provider:
        os.environ["LOOPLLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LOOPLLM_MODEL"] = args.model
    if args.db:
        os.environ["LOOPLLM_DB"] = args.db

    try:
        from loopllm.mcp_server import main as mcp_main
    except ImportError:
        print(
            "Error: The mcp package is required for the MCP server.\n"
            "Install it with: pip install loopllm[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp_main()


def main() -> None:
    """Entry point for the ``loopllm`` CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
