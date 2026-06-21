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

For multi-step iterative work, use `loopllm_loop_start` → `loopllm_loop_step(step_output=...)` → `loopllm_loop_end`. Submit step artifacts; do not self-grade.
