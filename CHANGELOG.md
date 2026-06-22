# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] — 2026-06-22

### Added
- **DAG virtual sub-agents** (`src/loopllm/dag_scheduler.py`): compile dependency-ordered
  nodes with scoped worker prompts; IDE agent executes one frontier at a time via
  `loopllm_dag_compile` / `dag_ready` / `dag_submit` / `dag_merge` (32 MCP tools).
- Per-node CDV via `submit_async` when MCP sampling is available.
- `complexity > 0.5` routes to DAG via `loopllm_intercept` and `loopllm_run_pipeline`.

## [0.8.0] — 2026-06-22

### Added
- **Episodic memory** (SQLite schema v5): `episodes` and `active_runs` tables in
  [`store.py`](src/loopllm/store.py).
- [`EpisodicStore`](src/loopllm/episodes.py): record completed loops/plans, keyword
  recall, crash-safe active run snapshots (`~/.loopllm/active_run.json` mirror).
- MCP tools: `loopllm_recall`, `loopllm_run_status`.
- Hooks: `loopllm_loop_end`, `loopllm_plan_register` / `plan_update` record episodes
  and persist active runs.
- Example: [`examples/dag_virtual_agents.py`](examples/dag_virtual_agents.py).
- Tests: `tests/test_episodes.py`, `tests/test_dag_scheduler.py`.

## [0.7.0] — 2026-06-22

### Added
- **Conservative Dual-Verify (CDV) for agent loops.** Agents submit `step_output`
  artifacts; the MCP server scores each step through two independent channels:
  - **Channel A:** deterministic evaluators (regex, JSON, completeness, composite).
  - **Channel B:** separate critic via MCP sampling (verifier hat).
  - **Final score:** `min(channel_a, channel_b)` — the stricter channel wins.
- New modules: `src/loopllm/step_scorer.py`, `src/loopllm/guards.py`,
  `src/loopllm/evaluator_factory.py`.
- Composable **guard stack** on verified scores: timeout, token budget, output-repeat,
  plateau, threshold, Bayesian ROI, max steps, budget exhausted.
- `loopllm_loop_start` accepts verifier recipe (`evaluator_type`, `quality_criteria`,
  `required_patterns`, `max_wall_ms`, `max_tokens`).
- `loopllm_loop_step` is now async and artifact-primary (`step_output`); legacy
  `score` self-report still works with deprecation warning.
- Verdict JSON exposes `channel_a_score`, `channel_b_score`, `score_source`,
  `deficiencies` for demos and debugging.
- Tests: `tests/test_step_scorer.py`, `tests/test_guards.py` (CDV inflation-block test).
- Package exports: `DualVerifyScore`, `conservative_dual_verify`, `GuardStack`.

### Changed
- `AgentLoopController` uses `GuardStack` instead of inline `_decide` logic.
- `AgentLoopSession` stores verifier config, step artifacts, token accumulators.
- `CallObservation.prompt_tokens` / `completion_tokens` populated on `loop_end`.
- README and demo doc reframed around Conservative Dual-Verify.

## [0.6.0] — 2026-06-21

### Added
- **Adaptive agent loops.** A new `AgentLoopController` (`src/loopllm/agent_loop.py`)
  brings the Bayesian early-exit machinery to an agent's own plan → act → observe
  loop. Lifecycle: `start` → `step` → `end`.
  - Suggests a learned step budget and quality threshold per `task_type` from
    `AdaptivePriors` — no training data required.
  - Returns a continue/stop verdict on each step (goal reached, plateau, low
    expected ROI, or budget exhausted) using the same logic as
    `BayesianExitCondition`.
  - Records every completed loop so future budgets sharpen over time.
- Four new MCP tools (now 28 total): `loopllm_loop_start`, `loopllm_loop_step`,
  `loopllm_loop_end`, `loopllm_loop_status`.
- `AgentLoopController` and `AgentLoopSession` exported from the package root.
- Runnable example `examples/agent_loop.py` and walkthrough
  `docs/demo/agent_loop_demo.md`.
- Reproducible benchmark `benchmarks/adaptive_vs_fixed.py` comparing adaptive
  stopping against fixed/threshold strategies (adaptive: ~41% fewer steps than a
  fixed 6-step budget at 99.7% goal-reach).
- Tests: `tests/test_agent_loop.py`, `tests/test_benchmark.py`.
- Repo hygiene: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, GitHub issue/PR
  templates, `[project.urls]` + classifiers in `pyproject.toml`, CI badge, and a
  committed `.cursor/mcp.json` for Cursor auto-detection.

### Changed
- README deduplicated and corrected (28 tools, 204 tests, schema v4); added
  "Adaptive agent loops" and "Benchmark" sections.

### Fixed
- Version drift: `__init__.__version__` now matches `pyproject.toml`.

## [0.5.0]

### Added
- Online SGD weight learning and Thompson Sampling for question ordering
  (SQLite schema v4).
- Prompt Lab sidebar panel in the VS Code extension.
- Expanded MCP surface to 24 tools, MCP Sampling, and persistent confidence-gated
  plans.

## [0.4.0]

### Added
- Prompt quality scoring system (5 deterministic dimensions, grade A–F) and the
  initial VS Code extension.
