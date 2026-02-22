# loop-llm — Claude Code Onboarding

This repo is **loop-llm**: an MCP server that observes prompt quality, routes requests, and learns from feedback via online gradient descent and Thompson Sampling. You are both using it and developing it.

## Required on every message

Call `loopllm_intercept` with the user's exact prompt text before doing anything else. This scores quality, routes the request, and updates the VS Code gauge.

```python
loopllm_intercept(prompt="<user's exact message>")
```

Read the `route` field in the response:
- `elicit` → call `loopllm_elicitation_start` before answering
- `decompose` → call `loopllm_plan_tasks` before answering
- `elicit_then_refine` → optionally ask one question, then proceed
- `refine` → proceed directly

## Primary tool for non-trivial tasks

```python
loopllm_run_pipeline(prompt="<request>")
```

Runs the full loop: score → elicit if needed → decompose if complex → execute via MCP Sampling → verify → log.

## Repo structure

```
src/loopllm/
  mcp_server.py    — 24 MCP tools + SGD + Thompson Sampling
  store.py         — SQLite schema v4 (priors, history, plans, learned_weights)
  priors.py        — Beta-Binomial, NormalPrior (Welford's), AdaptivePriors
  adaptive_exit.py — Bayesian early stopping
  evaluators.py    — deterministic evaluators (length, regex, JSON schema)
  cli.py           — loopllm score / serve / refine commands

vscode-loopllm/   — VS Code extension (TypeScript)
tests/            — 186 tests, run with: python -m pytest tests/ -q
```

## Key implementation details

- `_score_prompt_quality()` in `mcp_server.py` loads learned weights from the store on every call; falls back to defaults `{specificity: 0.25, constraint_clarity: 0.20, context_completeness: 0.20, ambiguity: 0.20, format_spec: 0.15}`.
- `_tool_feedback()` calls `_update_scoring_weights()` which runs one SGD step: `w_i -= 0.02 * 2*(y_hat - target) * d_i`, then clips to [0.05, 0.50] and simplex-projects.
- `_next_static_question()` uses Thompson Sampling: draws `Beta(1 + positive_count, 1 + negative_count)` per question type, picks argmax.
- Schema v4 adds `learned_weights` table with `id=1` singleton row.
- MCP Sampling: `loopllm_refine`, `loopllm_run_pipeline`, `loopllm_plan_tasks`, `loopllm_verify_output` call `ctx.sample()` to invoke the host LLM inline; fall back to `agent_execute` passthrough.

## After responding

If the user rates the response:
```python
loopllm_feedback(rating=<1-5>, task_type="<type>")
```
