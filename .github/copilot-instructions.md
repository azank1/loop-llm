# PromptLoop — Agent Instructions

Full system prompt and tool reference: [`.agent/system.md`](../.agent/system.md)

Read that file first when starting a new session in this repo.


---

## Required on every message

Always call `loopllm_intercept` with the user's exact prompt before responding. Do not skip this even for simple requests.

```
loopllm_intercept(prompt="<user's exact message>")
```

Act on the returned `route`: `elicit` → clarify first | `decompose` → plan first | `refine` → proceed directly.

For non-trivial tasks use `loopllm_run_pipeline` as the entry point.
