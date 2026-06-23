# PromptLoop v0.8.0 Live Audit

**Date:** 2026-06-24 | **Branch:** `main` (post-v0.8 merge) | **Tests baseline:** 238 → 239 after fixes

---

## 1. Bayesian System in Action

### 1a. Adaptive priors demo (`examples/agent_loop.py`)

```
=== Loop 26bafa5d6e9c (task_type=bugfix) ===
Suggested budget: 3 step(s) | threshold 0.80 | confidence 0.00 (from 0 past loops)
  step  1 | 0.45 |#########           | -> CONTINUE: Keep going: step 1/3, score=0.450 below 0.80
  step  2 | 0.85 |#################   | -> STOP: Goal reached: score=0.850 >= threshold 0.80 at step 2
  learned: optimal_depth=2.0 converge_rate=0.667 (obs=1)

=== Loop 902b543a2bf4 (task_type=bugfix) ===
Suggested budget: 3 step(s) | threshold 0.80 | confidence 0.09 (from 1 past loops)
  step  2 | 0.51 |##########          | -> CONTINUE
  step  3 | 0.51 |##########          | -> STOP: Progress plateaued (last deltas 0.0010, 0.0050 < 0.0100)
  learned: optimal_depth=2.5 converge_rate=0.5 (obs=2)

After 15 seeded historical loops:
  optimal_depth=2.06 converge_rate=0.895 confidence=0.63 first_call_quality=0.497

=== Loop 439d13891c7c (task_type=bugfix) ===
Suggested budget: 1 step(s) | threshold 0.88 | confidence 0.63 (from 17 past loops)
  step  1 | 0.55 | -> STOP: Step budget exhausted (1/1)
```

**What this shows:** The `AdaptivePriors` system works correctly. Cold start uses task-type defaults
(budget=3, threshold=0.8, confidence=0.0). After 17 observations the system tightens the budget
to 1 and raises the threshold to 0.88 — it has learned that bugfix loops converge early.
The `PlateauGuard` fires correctly when deltas drop below 0.01.

### 1b. CDV loop (scripted via `scripts/audit_cdv_loop.py`)

**Intercept (cold):** score=0.48 [D], route=`elicit_then_refine`, `recall_available=None`
(correctly gated behind quality >= 0.6).

**loop_start (cold):** `similar_episodes=[]`, `memory_hint="No similar episodes yet"`,
`suggested_budget=3`, `confidence=0.0`.

**loop_step step 1** (artifact missing `retry`):
```json
{ "decision": "continue", "score": 0.3333, "cdv_mode": "channel_a_only",
  "channel_a_score": 0.3333, "passed": false,
  "deficiencies": ["Aspect not addressed: retry", "Required pattern not found: retry"] }
```

**loop_step step 2** (artifact contains retry + raise + backoff):
```json
{ "decision": "stop", "score": 0.8889, "cdv_mode": "channel_a_only",
  "channel_a_score": 0.8889, "reason": "Goal reached: score=0.889 >= threshold 0.80 at step 2" }
```

**loop_end:** `final_score=0.8889`, `steps_run=2`, `converged=true`, `learned.optimal_depth=2.0`.

### 1c. Recovery contract

After `loop_start` + 1 step, clearing the in-memory controller and re-running
`hydrate_active_loops` restores the session from `active_runs`. `loopllm_run_status` shows the
full snapshot. `loopllm_loop_resume` returns `source="store"`, `resumed=true`. Subsequent
`loop_step` advances `steps_used` from 1 to 2 correctly.

### 1d. Episodic recall demo (`examples/episodic_recall_loop.py`)

```
recorded episode: 'fix flaky auth login tests' (steps=2)
recorded episode: 'migrate users table to v5 schema' (steps=2)
recorded episode: 'write API reference docs' (steps=2)

loopllm_recall('auth tests failing') -> 1 hit(s)
  - fix flaky auth login tests  (score=0.9, converged)
```

Keyword recall correctly ranks the auth episode first. On the next `loop_start` with a similar
goal, `similar_episodes` is injected automatically with `memory_hint`.

---

## 2. Bugs Found and Fixed

### Bug 1 — CDV `passed` flag used hardcoded 0.7, not session threshold (CRITICAL)
**File:** `src/loopllm/step_scorer.py`

`DualVerifyScore.passed` was set with `final >= 0.7` regardless of the session's
`quality_threshold`. An agent could see `passed=True` in the step verdict while the guard
stack fires `continue` (because `score < quality_threshold`, e.g. 0.8). Contradictory signals.

**Fix:** Added `quality_threshold: float = 0.7` keyword parameter to `conservative_dual_verify`
and `legacy_self_report_score`. Threaded `session.quality_threshold` from `_tool_loop_step`
into both calls. `passed` is now consistent with the guard-stack stop/continue decision.

---

### Bug 2 — Channel B JSON parse failure inflated score to 0.9 (HIGH)
**File:** `src/loopllm/step_scorer.py`

On critic response JSON parse failure, `score_channel_b` fell back to `fast_score`, which
returns **0.9** when `quality_criteria` is empty. This made `min(A, B)` fusion return the
inflated keyword score, silently bypassing the conservative design.

**Fix:** Added `parse_failed: bool` flag to the `score_channel_b` return dict. In
`conservative_dual_verify`, when `parse_failed=True`, substitute `channel_a.score` as the
Channel B fallback so the fusion stays conservative.

---

### Bug 3 (NEW, found in live testing) — `ctx.sample()` failure crashed entire loop step (HIGH)
**File:** `src/loopllm/step_scorer.py`

When a MCP client passes `ctx` but does not support sampling (e.g. the `CallMcpTool` client
in agent chat), `score_channel_b` raises `AttributeError: 'Context' object has no attribute
'sample'`. The outer `try/except` in `_tool_loop_step` caught this and returned
`{"error": "CDV scoring failed"}` — blocking the entire step rather than degrading gracefully.

**Fix:** Wrapped the `score_channel_b` call in `conservative_dual_verify` with a `try/except`
that falls back to `channel_a_only` mode on any sampling exception. Verified with a mock
`BadCtx` that raises `AttributeError`.

Note: this bug is only visible when calling `loopllm_loop_step` from agent chat (live MCP).
The scripted tests pass `ctx=None` explicitly, bypassing it. **The live MCP server requires
a restart to pick up this fix.**

---

### Bug 4 — `similar_episodes` stripped `task_type` from recall results (LOW)
**File:** `src/loopllm/mcp_server.py` `_tool_loop_start`

`similar_brief` only returned `goal, summary, stop_reason, score_final`. Agents could not
determine whether a recalled episode was the same task type without calling `loopllm_recall`
separately.

**Fix:** Added `"task_type": ep.get("task_type", "")` to `similar_brief`.

---

### Bug 5 — `_tool_loop_step` MCP CDV path had no integration test (MEDIUM)
**File:** `tests/test_mcp_episodic.py`

CDV scoring, guards, checkpointing, and `cdv_mode` field were each tested in isolation but the
full MCP handler wiring was not exercised end-to-end.

**Fix:** Added `test_tool_loop_step_cdv_path` covering the full path:
- `loop_start` → step 1 (fails criteria) → assert `cdv_mode="channel_a_only"`, `passed=False`,
  checkpoint in `active_runs`
- step 2 (passes criteria) → assert `decision=stop`, score ≥ threshold
- `loop_end` → `converged=True`, episode recorded
- next `loop_start` → `similar_episodes` injected with correct `task_type` (verifies Bug 4 fix)

---

### Bug 6 — Silent startup/checkpoint failures (MEDIUM)
**File:** `src/loopllm/mcp_server.py`

Three `except Exception: pass` sites:
1. Plan registry restore on startup (line ~2419)
2. Agent loop hydration on startup (line ~2429)
3. Step checkpoint write in `_tool_loop_step` (line ~1579)

Corrupt DB or snapshot caused the server to start in a broken state with no log output.

**Fix:** All three replaced with `logger.warning("...", exc_info=True)`. Non-fatal behavior
preserved; failures are now visible in the MCP output log.

---

## 3. IDE Usability Audit

**Test task:** "Add retry logic with exponential backoff to a download function; raise after 3 tries"
**IDE:** Cursor with loopllm v0.8.0 (31 tools, `~/.cursor/mcp.json`)

### Step-by-step observations

| Step | Tool | Result | Notes |
|------|------|--------|-------|
| 1 | `loopllm_intercept` | 54% [D], `elicit_then_refine` | Correctly detected missing constraints/context |
| 2 | `loopllm_elicitation_start` | Q: "Who will use this output?" (info_gain=0.93) | Thompson Sampling picked highest-gain question first |
| 3 | `loopllm_elicitation_answer` | Refined with audience context | Works correctly |
| 4 | `loopllm_intercept` (refined prompt) | 68% [C], `refine` | Score improved from D to C after elicitation |
| 5 | `loopllm_loop_start` | budget=3, confidence=0.0, similar_episodes=[] | Cold start; guidance text clear |
| 6 | `loopllm_loop_step` | ERROR: ctx.sample() failed | **Bug 3 (new)** — fixed in code, server restart required |
| 7 | `loopllm_loop_end` | Episode recorded | Works; `loopllm_recall` returns it immediately |

### Friction points

**F1 — MCP server restart required after code changes.**
The pipx editable install reads source files at import time. After patching `step_scorer.py`,
the running MCP process still has old code in memory. In Cursor: Settings → Tools & MCPs →
refresh the loopllm server (or fully restart Cursor). VS Code: reload window. This is expected
behavior but not obvious to users.

**F2 — Two parallel plan systems confuse agents.**
`loopllm_plan_tasks` (via `TaskOrchestrator`) and `loopllm_plan_register` / `plan_*` (via
`PlanRegistry`) are both registered. They have different lifecycles, storage formats, and
dependency models. The MCP instructions do not differentiate them. An agent asked to "plan and
execute" will pick arbitrarily. Recommendation: add a one-line description to each tool's
description clarifying the use case (orchestrated execution vs confidence-gated milestone plans).

**F3 — CDV `channel_a_only` mode is the norm, not the exception.**
Full CDV (Channel B critic via MCP sampling) only runs when the host MCP client declares
sampling capability. In practice, most IDE agent contexts don't expose this. Agents and users
should expect `channel_a_only` as the default mode. The `cdv_mode_reason` field communicates
this but is buried in a large JSON response.

**F4 — JSON responses are verbose; key fields not surfaced first.**
`loopllm_loop_step` returns 15+ fields. Agents (and the Loop Monitor) have to find `decision`
and `reason` buried in the middle. Recommendation: reorder serialization so `decision`,
`reason`, `score` appear first (or add a top-level `verdict` key).

**F5 — Loop Monitor extension requires server restart to show live data.**
The extension polls `active_runs/*.json` every 3s. This works correctly once the server is
running. After a Cursor restart (e.g. after code changes), the extension shows stale data
until the MCP server restarts and writes fresh active_run files.

**F6 — `recall_available` in `loopllm_intercept` gated at quality ≥ 0.6.**
Intentional and correct design — low-quality prompts are not worth recalling against. But
agents who call `loopllm_intercept` first (as instructed) on a D-grade prompt will never see
`recall_available=True`, and may not know to call `loopllm_recall` separately. Consider adding
a `recall_hint` even for lower-quality prompts when episodes exist for the task_type.

### What worked well

- The MCP server instructions (`IMPORTANT: call loopllm_intercept first...`) are effective:
  the Cursor agent called `loopllm_intercept` without being explicitly asked.
- Thompson Sampling question ordering is live and picks high-information-gain questions first
  (info_gain=0.93 for audience question).
- `loop_start` guidance text is clear and actionable.
- Recovery contract works end-to-end: kill → restart → `run_status` → `resume` → continue.
- `similar_episodes` injection is transparent (shows goal + summary + stop reason + score).

---

## 4. v0.9 Readiness Assessment

| Component | Current state | Gap for v0.9 |
|-----------|--------------|--------------|
| **DAG topology** | `TaskPlan.execution_order()` — Kahn sort, cycle detection — in `tasks.py:80-131` | Not MCP-exposed; `TaskOrchestrator.execute()` runs it sequentially, not as virtual sub-agents |
| **Per-node CDV** | `conservative_dual_verify` + `build_step_evaluator` in `step_scorer.py` | Wired only through `loopllm_loop_step` (single session). v0.9 needs per-node sessions. |
| **Per-node guard stack** | `GuardStack` + 8 guard types in `guards.py` | `GuardContext` is tied to one `AgentLoopSession`. v0.9 needs node-scoped sessions or a generalized context. |
| **Episode-seeded planning** | `EpisodicStore.recall` works | Not called from `TaskOrchestrator.plan()` or `loopllm_plan_tasks`. v0.9 `dag_compile` should inject similar past runs. |
| **`AdaptiveStopper` adapter** | `should_continue(state)` + `route(on_continue, on_stop)` in `adapters.py` | Ready for per-node use with LangGraph/while-loops. No changes needed. |
| **4 new dag MCP tools** | Not implemented | Need `dag_compile`, `dag_ready`, `dag_submit`, `dag_status`. |
| **Dependency graph** | `Task.dependencies: list[str]` + `TaskPlan.dependency_graph` | Title-string matching is fragile (LLM may duplicate titles). v0.9 needs ID-based deps. |
| **PlanRegistry** | Confidence-gated linear plans; MCP `loopllm_plan_*` | Linear only — no dep edges. Cannot be used as the dag scheduler directly. |

### Files v0.9 DAG tools would extend

| File | What v0.9 needs from it |
|------|------------------------|
| `src/loopllm/tasks.py` | `TaskPlan`, `Task`, `execution_order()` — extend with roles + MCP-callable DAG lifecycle |
| `src/loopllm/agent_loop.py` | `AgentLoopController` — one session per DAG node (call `start`/`step`/`end` per node) |
| `src/loopllm/step_scorer.py` | `conservative_dual_verify` — already node-agnostic, used per node |
| `src/loopllm/guards.py` | `GuardStack` + `default_guard_stack` — reuse per-node with node-specific budgets |
| `src/loopllm/episodes.py` | `EpisodicStore.recall` — call from `dag_compile` to seed node descriptions |
| `src/loopllm/store.py` | `active_runs` table — store per-node snapshots under `run_type="dag_node"` |
| `src/loopllm/mcp_server.py` | Add 4 new `@mcp.tool` registrations (dag_compile, dag_ready, dag_submit, dag_status) |

### Bugs from this audit that are blocking for v0.9 correctness

- **Bug 1 (threshold)** — Fixed. Per-node CDV would have had inconsistent `passed` flags if
  node thresholds differ from the default 0.7. Now uses `session.quality_threshold` correctly.
- **Bug 3 (ctx.sample)** — Fixed. Per-node steps called from agent chat would have crashed
  without graceful fallback. Now degrades to `channel_a_only`.

### Minimum v0.9 changes

1. Add `dag_compile(goal) -> DagRun` — calls `TaskOrchestrator.plan()`, converts to
   `{nodes: [{id, role, description, dependencies, evaluator_recipe}]}`, stores in `active_runs`
   as `run_type="dag"`.
2. Add `dag_ready(run_id) -> list[{node_id, scoped_instructions, input_artifacts}]` — Kahn
   frontier from `TaskPlan.execution_order()`.
3. Add `dag_submit(run_id, node_id, step_output) -> verdict` — creates a per-node
   `AgentLoopController` session, runs CDV, updates node state, unlocks dependents.
4. Add `dag_status(run_id) -> full graph state` — returns all node states + active_runs snapshot.

No new Python dependencies needed. Reuses existing guard stack, CDV, episodic store, and
active_runs schema.

---

## 5. Deliverables Summary

| Deliverable | Status |
|-------------|--------|
| `AUDIT.md` (this file) | Done |
| `scripts/audit_cdv_loop.py` | Done — runnable CDV + recovery + recall demo |
| Bug 1 fix (`step_scorer.py` threshold) | Done |
| Bug 2 fix (`step_scorer.py` Channel B parse fallback) | Done |
| Bug 3 fix (`step_scorer.py` ctx.sample graceful degradation) | Done |
| Bug 4 fix (`mcp_server.py` task_type in similar_brief) | Done |
| Bug 5 fix (`mcp_server.py` startup/checkpoint logging) | Done |
| Bug 6 new test (`test_mcp_episodic.py` CDV path integration) | Done |
| Test suite | 239 passed, 4 skipped (was 238) |
| ruff + mypy --strict | Clean |

**Server restart required:** The live MCP server in Cursor/VS Code/Antigravity must be
restarted (Settings → Tools & MCPs → refresh, or restart the IDE) to pick up Bug 3 fix.
