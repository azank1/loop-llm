# Contributing to PromptLoop (`loopllm`)

Thanks for your interest in improving PromptLoop. This guide covers local setup,
the branch/commit conventions, and the checks your change must pass.

## Local setup

```bash
git clone https://github.com/azank1/loop-llm
cd loop-llm
pip install -e ".[dev]"
python -m pytest tests/ -q          # 235 tests (231 pass, 4 integration skipped)
```

## Branch naming convention

Branches follow a compact `<initials>/<type>/<short>` scheme:

- `<initials>` — the author's initials (e.g. `az`)
- `<type>` — `ft` (feature), `fix`, `chore`, `docs`, `refactor`, `test`
- `<short>` — a terse, hyphenated descriptor

Examples: `az/ft/cdv-v07`, `az/fix/store-migration`, `az/docs/readme`.

Do **not** include tool, assistant, or AI names in branch names or commit
metadata.

## Authorship

All commits on this repository are authored by `azank1 <azanhyder49@gmail.com>`.
Use squash-merge PRs into `main`; delete feature branches after merge.

## Commits & pull requests

- Use clear, imperative commit subjects (e.g. `feat: adaptive agent loops`).
- Keep commits focused; add tests for any new tool or behavior under `tests/`.
- Open PRs against `main`. CI (Python 3.11–3.13) must be green.

## Checks (run before pushing)

```bash
ruff check src/ tests/
mypy --strict src/loopllm/
python -m pytest tests/ -q
```

## Where things live

| Area | File |
|---|---|
| Core refinement loop | `src/loopllm/engine.py` |
| Bayesian priors / learning | `src/loopllm/priors.py` |
| Adaptive agent loops | `src/loopllm/agent_loop.py` |
| Framework stop adapter (should_continue) | `src/loopllm/adapters.py` |
| Conservative Dual-Verify scoring | `src/loopllm/step_scorer.py` |
| Agent-loop guard stack | `src/loopllm/guards.py` |
| Evaluator factory | `src/loopllm/evaluator_factory.py` |
| Bayesian early stopping | `src/loopllm/adaptive_exit.py` |
| MCP tools (31) | `src/loopllm/mcp_server.py` |
| Episodic memory | `src/loopllm/episodes.py` |
| SQLite persistence (schema v5) | `src/loopllm/store.py` |
| CLI | `src/loopllm/cli.py` |
| Providers (agent/ollama/openrouter/mock) | `src/loopllm/providers/` |

By contributing, you agree your contributions are licensed under the project's
[MIT License](LICENSE).
