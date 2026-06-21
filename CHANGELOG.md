# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Tests: `tests/test_agent_loop.py`.

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
