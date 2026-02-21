# Contributing to loop-llm

Thanks for your interest in contributing! This document covers everything you need to get started.

## Getting Started

```bash
git clone https://github.com/azank1/loop-llm.git
cd loop-llm
pip install -e ".[dev]"
```

The `dev` extra installs everything needed for development: linting, type checking, and testing tools.

## Project Structure

```
src/loopllm/
├── engine.py           # Core refinement loop (LoopedLLM, LoopConfig)
├── evaluators.py       # Deterministic evaluators (Regex, JSON, Length, etc.)
├── priors.py           # Bayesian adaptive priors (Welford, Beta distributions)
├── adaptive_exit.py    # BayesianExitCondition
├── elicitation.py      # Intent elicitation sessions
├── tasks.py            # Task orchestration and planning
├── store.py            # SQLite persistence layer
├── provider.py         # LLMProvider protocol
├── mcp_server.py       # MCP server (13 tools for IDE integration)
├── cli.py              # CLI entry point
├── __main__.py         # python -m loopllm support
└── providers/
    ├── ollama.py       # Ollama provider
    └── openrouter.py   # OpenRouter provider
```

## Development Workflow

### Run Tests

```bash
pytest tests/ -v
```

### Type Checking

We use `mypy --strict` across the entire codebase:

```bash
mypy src/loopllm/
```

### Linting

We use `ruff` for linting and formatting:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

### All Checks

Before submitting a PR, make sure everything passes:

```bash
pytest tests/ -v && mypy src/loopllm/ && ruff check src/ tests/
```

## What to Work On

### Good First Issues

- Add new evaluators (see `evaluators.py` for the pattern)
- Add new LLM providers (see `providers/ollama.py` for the template)
- Improve CLI output formatting
- Documentation improvements

### Larger Contributions

- New exit conditions for the refinement loop
- Bayesian prior enhancements (different distribution families, decay strategies)
- Additional MCP tools
- Performance optimisation of the SQLite store

## Adding a New Provider

1. Create `src/loopllm/providers/your_provider.py`
2. Implement the `LLMProvider` protocol from `provider.py`:

```python
from loopllm.provider import LLMProvider

class YourProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        # Call your LLM API and return the response text
        ...
```

3. Add the provider to `_make_provider()` in `mcp_server.py` if you want MCP support
4. Add any new dependencies as an optional extra in `pyproject.toml`

## Adding a New Evaluator

1. Add your evaluator class to `src/loopllm/evaluators.py`
2. Implement the `Evaluator` protocol from `engine.py`:

```python
from loopllm.engine import EvaluationResult

class YourEvaluator:
    def evaluate(self, output: str, context: dict[str, Any] | None = None) -> EvaluationResult:
        score = ...  # 0.0 to 1.0
        return EvaluationResult(
            score=score,
            passed=score >= self.threshold,
            feedback=f"Your feedback here",
            details={"key": "value"},
        )
```

## Code Style

- Python 3.11+ features are welcome (`match`, `|` unions, etc.)
- All public functions need type annotations (`mypy --strict`)
- Use `structlog` for logging, not `print()` or `logging`
- Keep evaluators deterministic — no LLM-as-judge patterns

## Pull Requests

1. Fork the repo and create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all checks pass (tests, mypy, ruff)
4. Open a PR with a description of what changed and why

## Questions?

Open an issue on [GitHub](https://github.com/azank1/loop-llm/issues) or start a discussion.
