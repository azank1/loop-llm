# Demo: Adaptive Agent Loops — Conservative Dual-Verify

A walkthrough of `loopllm`'s adaptive agent loops with **Conservative Dual-Verify
(CDV)**: agents submit step artifacts; the server scores them through two
independent channels (deterministic evaluators + separate critic call); the
stricter score drives when to stop and what the system learns.

## Run the terminal demo

```bash
pip install -e ".[dev]"
python examples/agent_loop.py
```

This exercises `AgentLoopController` directly with pre-computed scores (library
API). For CDV via MCP, see below.

## What you'll see (terminal)

The demo drives three loops through `AgentLoopController` and prints each step's
continue/stop verdict.

### 1. Cold start — stops the moment the goal is reached

No history yet, so the controller falls back to the `bugfix` task-type default
(3 steps) and stops as soon as progress crosses the threshold:

```
=== Loop 23b8d17f7b57 (task_type=bugfix) ===
Suggested budget: 3 step(s) | threshold 0.80 | confidence 0.00 (from 0 past loops)
  step  1 | 0.45 |#########           | -> CONTINUE: step 1/3, score 0.450 below threshold 0.80
  step  2 | 0.85 |#################   | -> STOP: Goal reached: score=0.850 >= threshold 0.80 at step 2
  learned: optimal_depth=2.0 converge_rate=0.667 (obs=1)
```

### 2. Plateau guard — stops a stalled loop instead of burning budget

Three near-identical scores trigger the convergence band; the loop stops early
even though budget remains:

```
  step  3 | 0.51 |##########          | -> STOP: Progress plateaued (last deltas 0.0010, 0.0050 < 0.0100)
```

### 3. After learning — a confident, data-driven budget

Once the controller has seen 15+ `bugfix` loops that converge by step 2, its
beliefs reflect reality and it budgets accordingly:

```
Now the controller's beliefs are informed by real data:
  optimal_depth=2.06 converge_rate=0.895 confidence=0.63 first_call_quality=0.497
```

## Try CDV through MCP (Cursor / VS Code)

With the server running (`loopllm mcp-server`), use **Conservative Dual-Verify**
— submit `step_output`, not your own score:

```
loopllm_loop_start(
  goal="make the failing test pass",
  task_type="bugfix",
  required_patterns=["tests passed"],
)

loopllm_loop_step(
  session_id=<id>,
  step_output="pytest: 3 FAILED, 12 passed",
)
# → channel_a_score: 0.0 (regex miss), channel_b_score: 0.55, score: 0.0, decision: continue

loopllm_loop_step(
  session_id=<id>,
  step_output="pytest: 42 passed, 0 failed",
)
# → channel_a_score: 1.0, score_source: conservative_dual_verify, decision: stop

loopllm_loop_end(session_id=<id>)
```

### Cursor Agent chat script (screen recording)

Paste into Agent mode:

```
Run an adaptive agent loop with Conservative Dual-Verify:

1. loopllm_loop_start with goal="make the failing test pass" task_type="bugfix"
   required_patterns=["passed"]
2. loopllm_loop_step with step_output="pytest: 3 FAILED"
3. loopllm_loop_step with step_output="pytest: 42 passed, 0 failed"
4. loopllm_loop_end

After each loop_step, show the full JSON including channel_a_score,
channel_b_score, score_source, and deficiencies.
```

## Recording an asciinema cast (for the launch)

```bash
asciinema rec agent_loop.cast -c "python examples/agent_loop.py"
# then: asciinema upload agent_loop.cast
```
