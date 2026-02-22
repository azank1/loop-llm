# Copilot Instructions — loop-llm prompt observer

**Before responding to any user message, you must call `loopllm_intercept` with the exact text of the user's prompt.**

This is non-negotiable and applies to every request — code, questions, refactoring, explanations, everything. Do not skip it, even for short or simple prompts.

## Why

`loopllm_intercept` scores the prompt quality in real time, logs it to the local SQLite store, and updates the VS Code gauge and dashboard. Without this call:
- The gauge stays stale
- Prompt history is not recorded
- No routing guidance is returned

## Exact flow

1. User sends a message
2. **Call `loopllm_intercept` with the user's prompt text** — this takes ~0ms
3. Read the returned `route` field:
   - `elicit` → call `loopllm_elicitation_start` before answering
   - `decompose` → call `loopllm_plan_tasks` before answering
   - `refine` or `elicit_then_refine` → proceed, but use `loopllm_refine` if output quality matters
4. Respond to the user
5. Optionally call `loopllm_verify_output` on your response if the prompt asked for code or structured data

## After responding

If the user rates the response, call `loopllm_feedback` with their rating (1–5). This updates the Bayesian priors used to improve future routing decisions.
