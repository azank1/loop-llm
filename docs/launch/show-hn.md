# Show HN draft

> Draft only — nothing is posted automatically. Review, tweak the links, and submit
> manually at https://news.ycombinator.com/submit

## Title options (pick one, ≤ 80 chars)

1. Show HN: PromptLoop – adaptive agent loops that learn when to stop (no training)
2. Show HN: Bayesian stop button for agent loops, as an MCP server
3. Show HN: Stop your agent looping forever – statistically grounded loop control

## URL

https://github.com/azank1/loop-llm

## Text

Most agent loops stop on a fixed `max_iterations` or "let the LLM decide when it's
done." The first wastes tokens; the second quits early or never. I wanted
something in between that's actually principled.

PromptLoop (`pip install loopllm`) is an MCP server that, among other things, gives
your agent's own plan → act → observe loop a statistically-grounded stop button.
The flow is three tools:

- `loopllm_loop_start(goal, task_type)` → returns a learned step budget + quality
  threshold
- `loopllm_loop_step(session_id, score)` → after each step you report a 0–1
  progress score; it returns continue/stop with a reason
- `loopllm_loop_end(session_id)` → records the run so future budgets sharpen

The stop decision fires on: goal reached, plateaued progress, low expected ROI
(learned priors say the remaining gap is unlikely to close), or budget exhausted.

The "learning" is deliberately boring and dependency-light — no PyTorch, no
training data. It's closed-form Bayesian bookkeeping: Beta-Binomial priors on
convergence, Welford's algorithm for online mean/variance of score deltas, and
Thompson Sampling for clarifying questions. Predictions start from sensible priors
and sharpen per `(task_type, model)` as loops are observed. Everything persists to
a local SQLite file.

It works in Cursor, VS Code, or any MCP client, and also runs as a plain Python
library and CLI. The same machinery powers prompt scoring (5 deterministic
dimensions, weights tuned by online SGD from your 1–5 ratings) and an iterative
refinement loop.

Does it help? A reproducible benchmark (simulation, stated assumptions) vs fixed
budgets: adaptive uses ~41% fewer steps than a fixed 6-step loop while reaching the
quality bar on 99.7% of tasks (vs 94% fixed, 34% for a small budget). Repro:
`python benchmarks/adaptive_vs_fixed.py`.

Stack: Python 3.11+, FastMCP, SQLite. MIT licensed. ~7.7k LOC, 204 tests, mypy
--strict + ruff in CI.

Repo: https://github.com/azank1/loop-llm
Demo: examples/agent_loop.py (cold start → plateau stop → learned budget)

Honest caveats: the progress score is whatever signal you feed it (self-rating or
an evaluator), so garbage-in applies; cold-start budgets are just task-type
defaults until it has ~5 observations; and this advises *your* loop rather than
running the agent for you. Feedback very welcome — especially on the stopping rule
and what task types you'd want priors for.
