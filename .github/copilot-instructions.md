# loop-llm — Agent Onboarding

Welcome to **loop-llm**, a prompt observer and quality loop that runs as your MCP server.

You have 24 tools available under the `loopllm_*` namespace. This file tells you how and when to use them.

---

## What loop-llm does

Every prompt a developer writes has measurable quality gaps — missing constraints, vague references, no output format, missing context. loop-llm scores prompts across 5 dimensions (specificity, constraint clarity, context completeness, ambiguity, format spec), routes them to the right tool (clarify → refine → verify), and learns from feedback via online gradient descent on scoring weights and Thompson Sampling for question ordering.

The system accumulates a prompt history in `~/.loopllm/store.db` and exports it to `prompt_history.json` for the VS Code sidebar gauge.

---

## Required: call `loopllm_intercept` on every user message

**Before responding to any user message, call `loopllm_intercept` with the exact text of the user's prompt.**

This is non-negotiable. It applies to every request — code, questions, explanations, refactoring, everything. Do not skip it.

```
loopllm_intercept(prompt="<user's exact message>")
```

Read the returned `route` field and act on it:

| route | what to do |
|---|---|
| `elicit` | call `loopllm_elicitation_start` before answering — prompt needs clarification |
| `decompose` | call `loopllm_plan_tasks` — complex task, break it down first |
| `elicit_then_refine` | optionally ask one clarifying question, then proceed |
| `refine` | prompt is clear — proceed directly, use `loopllm_refine` if quality matters |

---

## Primary workflow: `loopllm_run_pipeline`

For any non-trivial coding task, use the pipeline tool instead of answering directly:

```
loopllm_run_pipeline(prompt="<user request>")
```

This runs the full loop internally:
1. Score the prompt
2. Elicit clarifying questions if quality < 0.6
3. Decompose into subtasks if complex
4. Execute each subtask via MCP Sampling (calls the host LLM inline)
5. Verify the assembled output
6. Log everything to SQLite

---

## All 24 tools — quick reference

| Tool | When to use |
|---|---|
| `loopllm_intercept` | **ALWAYS FIRST** — score + route every prompt |
| `loopllm_run_pipeline` | Primary tool for any non-trivial task |
| `loopllm_gauge` | Instant score check, no DB write |
| `loopllm_refine` | Iterative improvement loop on a single prompt |
| `loopllm_verify_output` | Check your response against quality criteria |
| `loopllm_plan_tasks` | Decompose a goal into ordered subtasks |
| `loopllm_elicitation_start` | Begin a clarifying question session |
| `loopllm_elicitation_answer` | Record an answer to a clarifying question |
| `loopllm_elicitation_finish` | Synthesise answers into an intent spec |
| `loopllm_plan_register` | Create a confidence-gated plan in SQLite |
| `loopllm_plan_next` | Get next task; detects if replan is needed |
| `loopllm_plan_update` | Record task result + recalculate confidence |
| `loopllm_plan_list` | See all active plans with gauges |
| `loopllm_plan_delete` | Remove a completed or abandoned plan |
| `loopllm_feedback` | Rate a response 1–5 → triggers SGD weight update |
| `loopllm_prompt_stats` | Trend + learning curve across recent prompts |
| `loopllm_context_history` | Browse saved prompt history with sparklines |
| `loopllm_context_clear` | Wipe history (scoped or all) |
| `loopllm_analyze_prompt` | Generate clarifying questions ranked by information gain |
| `loopllm_classify_task` | Label a prompt's task type |
| `loopllm_suggest_config` | Get Bayesian-optimal loop config for a task type |
| `loopllm_list_tasks` | List tasks from the persistent store |
| `loopllm_show_task` | Detail a single task |
| `loopllm_report` | See learned weights, priors, question effectiveness |

---

## After responding

If the user rates the response, call:
```
loopllm_feedback(rating=<1-5>, task_type="<type>")
```
This runs one online SGD step on the scoring dimension weights so future scores better reflect what this user values.

---

## How the system learns

- **Scoring weights** start at `{specificity: 0.25, constraint_clarity: 0.20, context_completeness: 0.20, ambiguity: 0.20, format_spec: 0.15}` and shift via gradient descent each time `loopllm_feedback` is called
- **Question ordering** uses Thompson Sampling on Beta(α, β) priors — questions that historically improve output quality get prioritised
- **Adaptive exit** uses Beta-Binomial conjugate priors per (task_type, model) to decide when to stop the refinement loop early
- All state persists in `~/.loopllm/store.db` (SQLite, schema v4) and survives server restarts
