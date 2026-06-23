# PromptLoop — Agent Instructions

PromptLoop is an MCP server for prompt quality, refinement loops, and Conservative
Dual-Verify agent-loop control. See [README.md](../README.md) for architecture,
tools, and examples.

---

## Required on every message

Always call `loopllm_intercept` with the user's exact prompt before responding. Do not skip this even for simple requests.

```
loopllm_intercept(prompt="<user's exact message>")
```

Act on the returned `route`: `elicit` → clarify first | `decompose` → plan first | `refine` → proceed directly.

For non-trivial tasks use `loopllm_run_pipeline` as the entry point.

For multi-step iterative work, use `loopllm_loop_start` → `loopllm_loop_step(step_output=...)` → `loopllm_loop_end`. Submit step artifacts; do not self-grade. The verdict includes `cdv_mode` (`full` = independent critic consulted; `channel_a_only` = deterministic checks only).

---

## Memory & recovery (v0.8)

**Recall before planning.** For tasks similar to past work, call `loopllm_recall("<goal>")` before planning. `loopllm_loop_start` also returns `similar_episodes` automatically — read them. If `loopllm_intercept` returns `recall_available: true`, recall first. `loopllm_loop_end` records the outcome to episodic memory automatically.

**Resume after an IDE reload.** Call `loopllm_run_status`. If it shows an active run, call `loopllm_loop_resume` (optionally with `session_id`) **before** starting a new loop, so the in-progress loop continues instead of restarting cold.
