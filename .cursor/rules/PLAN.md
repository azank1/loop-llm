# Plan: PromptLoop — Architecture, Launch Polish & Milestones

## Context

v0.6.0 "adaptive agent loops" is **merged to `main`** (PR #1, PR #2). The goal is
to **harden the repo for a public launch on Hacker News and Twitter/X**, ASAP.
Launch-readiness is now **~7.5/10** (up from the original ~6.5/10 audit): M0 and
M1 are done; remaining gaps are the IDE visual asset, git tag, and optional PyPI.

Decisions locked with the user:
- **Polish + launch ASAP** (roadmap features come after launch).
- **Ship a quantitative benchmark** proving adaptive loops beat fixed
  `max_iterations` — HN respects honest numbers.
- **An IDE demo** (Cursor / VS Code) building on the screenshots already in the
  README.
- **Merge PR #1 into `main`** before launch — **done** (CI green, merged).
- Branch/authorship rule stays fixed: `az/<type>/<short>` branches, all commits
  authored solely by `azank1 <azanhyder49@gmail.com>`, no AI references anywhere
  (commits, PR bodies, code, launch copy).

## What we have today (system inventory)

A model-agnostic prompt-quality + loop-control system, no training data, ~7.7k LOC,
**204 tests** (200 passing + 4 integration skipped), CI on 3.11–3.13 (ruff +
mypy --strict + pytest). Layers:

- **Core loop** — `LoopedLLM.refine()` (`src/loopllm/engine.py`): call → evaluate →
  feedback → retry with threshold / convergence / custom exit conditions.
- **Bayesian learning** — `AdaptivePriors` (`src/loopllm/priors.py`): Beta-Binomial
  convergence priors, NormalPrior via Welford, Thompson Sampling for question order;
  `BayesianExitCondition` (`src/loopllm/adaptive_exit.py`).
- **Agent-loop control (new in 0.6.0)** — `AgentLoopController`
  (`src/loopllm/agent_loop.py`): generalizes the Bayesian early-exit to an agent's
  own plan→act→observe loop (`start`/`step`/`end`).
- **Persistence** — `LoopStore` + `SQLiteBackedPriors` (`src/loopllm/store.py`,
  schema v4).
- **Interfaces** — MCP server with 28 tools (`src/loopllm/mcp_server.py`), CLI
  (`src/loopllm/cli.py`), REST server (`src/loopllm/serve.py`), VS Code extension
  (`vscode-loopllm/`).
- **Providers** — agent passthrough, Ollama, OpenRouter, mock
  (`src/loopllm/providers/`).
- **Elicitation / tasks** — `IntentRefiner`, `TaskOrchestrator`, plan registry.

## Target architecture (how to describe it for the launch)

```
            ┌──────────────── Interfaces ────────────────┐
   MCP (28 tools) · CLI · REST · VS Code extension
            └───────────────────┬─────────────────────────┘
                                 │
   ┌───────────── Control plane ─────────────┐
   AgentLoopController   LoopedLLM   IntentRefiner   TaskOrchestrator
   (when to stop)        (refine)    (what to ask)   (decompose)
            └───────────────────┬─────────────────────────┘
                                 │ reuse
   ┌───────── Bayesian learning (no training data) ─────────┐
   AdaptivePriors: Beta-Binomial · Welford NormalPrior · Thompson Sampling
            └───────────────────┬─────────────────────────┘
                                 │ persist
        LoopStore / SQLiteBackedPriors  →  ~/.loopllm/store.db (schema v4)
                                 │ backends
        Providers: agent · ollama · openrouter · mock
```
One sentence for the post: *"A model-agnostic, zero-training Bayesian layer that
tells an agent loop when to stop — exposed as MCP tools your IDE picks up
automatically."*

## Milestones

### M0 — Launch-blocking polish  [`az/ft/launch-polish`] — **DONE**
- [x] Deduplicate `README.md` (removed duplicated second half).
- [x] Fix stale claims: 28 tools, **204 tests** (200 pass + 4 skipped), schema v4,
  `agent_loop.py` in key-files list.
- [x] PyPI badge (`loopllm`) + CI status badge.
- [x] `[project.urls]` in `pyproject.toml`.
- [x] VSIX dead link → build-from-source instructions.
- [x] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `.github/` issue + PR templates.

### M1 — Credibility proof: benchmark  [`az/ft/benchmark`] — **DONE**
- [x] `benchmarks/adaptive_vs_fixed.py` (seed=7, train/test split, 4 strategies).
- [x] Metrics table → `benchmarks/results/adaptive_vs_fixed.md`.
- [x] README Benchmark section synced (~41% fewer steps, 99.7% goal-reach).
- [x] `tests/test_benchmark.py` (determinism + adaptive ≥ fixed efficiency).

### M2 — IDE demo (Cursor / VS Code)  [`az/ft/demo`] — **IN PROGRESS**
- [x] `examples/agent_loop.py` runnable demo.
- [x] `docs/demo/agent_loop_demo.md` walkthrough + MCP flow.
- [x] Terminal cast → `img/agent_loop.svg` embedded in README.
- [ ] IDE screen-recording GIF (user-captured) for X thread — optional upgrade.

### M3 — Merge & release hygiene  — **PARTIAL**
- [x] PR #1 merged to `main`; CI green.
- [x] Version `0.6.0` in `pyproject.toml` + `CHANGELOG.md`.
- [x] Annotated tag `v0.6.0` on `main` (local — push with `git push origin main --tags`).
- [ ] PyPI publish (user token required — out of scope for agent).

### M4 — Launch assets & post  [`docs/launch/`] — **DRAFT-READY**
- [x] `docs/launch/show-hn.md`, `reddit.md`, `twitter-thread.md`, `CHECKLIST.md`.
- [x] Benchmark numbers wired into launch copy.
- [ ] Attach demo visual to twitter-thread before posting.
- [ ] User posts manually (nothing automatic).

### Post-launch roadmap (future, not now)
Framework adapters (LangChain/LlamaIndex/CrewAI), a hosted web playground, a VS
Code agent-loop dashboard (live budget/verdict), more providers, and expanding the
benchmark to real model runs.

## Critical files
- Done: `README.md`, `pyproject.toml`, `benchmarks/`, `tests/test_benchmark.py`,
  governance files, `docs/launch/*`.
- Remaining: `img/agent_loop.svg`, tag `v0.6.0`, refresh `CHECKLIST.md`.

## Verification
- `python -m pytest tests/ -q` → all green (incl. `test_benchmark.py`);
  `ruff check src/ tests/` and `mypy --strict src/loopllm/` clean.
- `python benchmarks/adaptive_vs_fixed.py` reproduces the README table from a fixed
  seed; numbers in README match its output exactly.
- README renders with **no duplicated sections**; every count (28 tools, **204**
  tests, schema v4) matches `grep`/`pytest --collect-only`; PyPI + CI badges resolve.
- `python -m build` still produces `loopllm-0.6.0`; `[project.urls]` shows on the
  built metadata.
- Fresh-clone smoke: `pip install -e ".[mcp]"`, open in VS Code/Cursor, confirm the
  server is auto-detected and `loopllm_loop_start/step/end` work end-to-end.
- All work on `az/<type>/<short>` branches, authored by `azank1`, no AI references.
