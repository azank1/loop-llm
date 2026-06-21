# Demo: Adaptive Agent Loops

A 60-second walkthrough of `loopllm`'s adaptive agent loops — the feature that
gives an agent's own plan → act → observe loop a statistically-grounded stop button.

## Run it

```bash
pip install -e ".[dev]"
python examples/agent_loop.py
```

## What you'll see

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

## Try it through MCP (Claude Code / Cursor)

With the server running (`loopllm mcp-server`), drive any iterative task:

```
loopllm_loop_start  goal="make the failing test pass" task_type="bugfix"
loopllm_loop_step   session_id=<id> score=0.4
loopllm_loop_step   session_id=<id> score=0.9     # -> decision: "stop"
loopllm_loop_end    session_id=<id>               # -> learns optimal depth
```

## Recording an asciinema cast (for the launch)

```bash
asciinema rec agent_loop.cast -c "python examples/agent_loop.py"
# then: asciinema upload agent_loop.cast
```
