# Show HN draft

> Draft only — nothing is posted automatically. Review, tweak the links, and submit
> manually at https://news.ycombinator.com/submit

## Title options (pick one, ≤ 80 chars)

1. Show HN: Agents submit artifacts; two verifiers score them — Bayesian agent-loop stop
2. Show HN: Conservative Dual-Verify for agent loops, as an MCP server
3. Show HN: Stop your agent looping forever – statistically grounded loop control

## URL

https://github.com/azank1/loop-llm

## Text

Most agent loops stop on a fixed `max_iterations` or let the agent self-grade when
it's "done." The first wastes tokens; the second optimizes **reported** progress.
I wanted something principled: external verification plus learned depth.

PromptLoop v0.7 (`pip install loopllm`) is an MCP server with **Conservative
Dual-Verify (CDV)** for agent loops. After each step the agent submits a
`step_output` artifact (test log, diff, summary). The server scores it through
two independent channels and takes the stricter result:

- **Channel A:** deterministic evaluators (regex, JSON, completeness)
- **Channel B:** separate critic call via MCP sampling (verifier hat, not the worker)
- **Final score:** `min(channel_a, channel_b)` — either channel can veto inflation

Three tools drive the loop:

- `loopllm_loop_start(goal, task_type, required_patterns=[...])` → learned budget +
  verifier recipe
- `loopllm_loop_step(session_id, step_output="...")` → CDV score + continue/stop
  verdict with `channel_a_score`, `channel_b_score`, deficiencies
- `loopllm_loop_end(session_id)` → records verified trajectories; budgets sharpen

Stop reasons include: goal reached (verified score), plateau, low Bayesian ROI,
budget exhausted, timeout, token cap, repeated output.

The learning layer is deliberately boring — no PyTorch, no training data. Beta-Binomial
priors on convergence, Welford online variance on score deltas, Thompson Sampling for
elicitation. Everything persists to local SQLite. Works in Cursor, VS Code, or any
MCP client; same machinery powers prompt scoring (5 dimensions, SGD weights from your
1–5 ratings) and iterative refinement.

Benchmark (simulation, stated assumptions): adaptive uses ~41% fewer steps than a
fixed 6-step loop while reaching the bar on 99.7% of tasks (vs 94% fixed, 34% for a
small budget). Repro: `python benchmarks/adaptive_vs_fixed.py`.

Stack: Python 3.11+, FastMCP, SQLite. MIT. ~8k LOC, 219 tests, mypy --strict + ruff
in CI.

Repo: https://github.com/azank1/loop-llm
Demo: `docs/demo/agent_loop_demo.md` and `examples/agent_loop.py`

Caveats: Channel B is still an LLM critic (not ground truth); cold-start budgets
are task-type defaults until ~5 observations; CDV advises your loop via MCP rather
than running the agent harness for you. Feedback welcome — especially on verifier
recipes for coding tasks.
