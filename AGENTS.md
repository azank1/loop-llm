# loop-llm — Agent Onboarding

This repo is **loop-llm**: an MCP server that acts as a prompt observer and quality loop for AI agents. When you work in this repo, you are both using loop-llm and developing it. This document tells you how to use the tools correctly from the start.

---

## First thing to do on every message

Call `loopllm_intercept` with the user's exact prompt text before doing anything else:

```
loopllm_intercept(prompt="<user's exact message>")
```

This scores the prompt across 5 quality dimensions, routes it to the right workflow, and logs it to the local SQLite store so the VS Code sidebar gauge updates. Skip this and the whole observation layer is blind.

---

## What this repo contains

```
src/loopllm/
  mcp_server.py     — all 24 MCP tools, SGD weight learning, Thompson Sampling
  store.py          — SQLite persistence (schema v4: priors, history, plans, learned_weights)
  priors.py         — Beta-Binomial and Normal priors, Welford's online algorithm
  adaptive_exit.py  — Bayesian early stopping for refinement loops
  evaluators.py     — deterministic output quality evaluators
  cli.py            — `loopllm score`, `loopllm serve`, `loopllm refine` commands

vscode-loopllm/
  src/promptLabProvider.ts   — live quality scratchpad sidebar panel
  src/dashboardProvider.ts   — prompt history chart
  src/statusWatcher.ts       — polls status.json for real-time gauge updates
```

Tests live in `tests/`. Run with `python -m pytest tests/ -q`.

---

## Primary tool for non-trivial tasks

```
loopllm_run_pipeline(prompt="<user request>")
```

Runs: score → elicit (if quality < 0.6) → decompose (if complex) → execute via MCP Sampling → verify → log. All in one call.

---

## Routing table (from `loopllm_intercept` response)

| `route` value | Action |
|---|---|
| `elicit` | Call `loopllm_elicitation_start` first |
| `decompose` | Call `loopllm_plan_tasks` first |
| `elicit_then_refine` | Ask one question if helpful, then proceed |
| `refine` | Proceed directly |

---

## All 24 tools

| Tool | Purpose |
|---|---|
| `loopllm_intercept` | **Call first on every message** |
| `loopllm_run_pipeline` | Full observe → elicit → execute → verify loop |
| `loopllm_gauge` | Instant score, no DB write |
| `loopllm_refine` | Score → sample → retry loop |
| `loopllm_verify_output` | Check output against quality criteria |
| `loopllm_plan_tasks` | Decompose goal into subtasks |
| `loopllm_elicitation_start/answer/finish` | Multi-turn clarification session |
| `loopllm_plan_register/next/update/list/delete` | Confidence-gated plan tracking |
| `loopllm_feedback` | Rate 1–5 → SGD weight update |
| `loopllm_prompt_stats` | Quality trend + learning curve |
| `loopllm_context_history/clear` | Browse or wipe prompt history |
| `loopllm_analyze_prompt` | Ranked clarifying questions via Thompson Sampling |
| `loopllm_classify_task` | Label task type |
| `loopllm_suggest_config` | Bayesian-optimal config for task type |
| `loopllm_list_tasks/show_task` | Task store queries |
| `loopllm_report` | Learned weights + prior effectiveness stats |

---

## How the system learns (so you know what you're developing)

- **Online SGD**: `loopllm_feedback(rating)` runs one gradient descent step on the 5 scoring dimension weights. After ~50 ratings weights converge to what this user actually values.
- **Thompson Sampling**: clarifying question types maintain Beta(α, β) priors; the one with the best random draw from its prior gets asked next — exploration/exploitation without a fixed schedule.
- **Beta-Binomial priors**: per-(task_type, model) convergence probability, updated with every observation, drives the adaptive early-exit from refinement loops.
- **Welford's online algorithm**: `NormalPrior` tracks running mean/variance of quality scores in O(1) memory with optional exponential decay.

---

## Development workflow

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q        # 186 tests
```

Branch `testing-production` is active development. `main` mirrors it.
