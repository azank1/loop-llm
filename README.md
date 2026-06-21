# PromptLoop

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&pause=1000&color=00CFFF&center=true&vCenter=true&width=700&lines=PromptLoop+%E2%80%94+Prompt+Quality+Loop;Bayesian+Adaptive+Exit+%2B+Thompson+Sampling;MCP+Server+%E2%80%94+28+Tools+for+VS+Code+%2B+Cursor;Online+Weight+Learning+via+SGD;Intent+Elicitation+%2B+Task+Decomposition;Zero+Training+%E2%80%94+Model+Agnostic)](https://github.com/azank1/loop-llm)

[![CI](https://github.com/azank1/loop-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/azank1/loop-llm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/badge/PyPI-loopllm-blue)](https://pypi.org/project/loopllm/)
[![VS Code Extension](https://img.shields.io/badge/VS%20Code-Extension-007ACC?logo=visual-studio-code)](https://github.com/azank1/loop-llm/tree/main/vscode-loopllm)

**A Bayesian prompt observer and quality loop for your AI agent.**

PromptLoop sits between you and your IDE's agent as an MCP server. Every prompt is scored, routed, refined if needed, and then verified — all inside a single tool call using MCP Sampling. The system **learns** over time: it runs online gradient descent on scoring dimension weights every time you rate a response, and selects clarifying questions via Thompson Sampling on Beta priors.

> The underlying CLI and tool API ship as `loopllm` (the original name). PromptLoop is the project brand.

---

## VS Code Extension

Install the companion extension for a live quality scratchpad and prompt history dashboard directly in the sidebar.

<table>
<tr>
<td width="50%" valign="top">

**Prompt Lab** — live quality scratchpad

![Prompt Lab](img/Screenshot_20260222_171552_Chrome.jpg)

Scores on every keystroke (350 ms debounce). Grade badge, 5 dimension bars, issues + suggestions tags, Copy and Send to Chat.

</td>
<td width="50%" valign="top">

**History** — learning curve + metrics

![History](img/Screenshot_20260222_171624_Chrome.jpg)

Learning curve sparkline, grade distribution, SGD learned weights per dimension. Updates after every `loopllm_feedback` call.

</td>
</tr>
</table>

Build and install the extension from source:

```bash
cd vscode-loopllm
npm install
npx @vscode/vsce package          # produces loopllm-prompt-gauge-0.1.0.vsix
code --install-extension loopllm-prompt-gauge-0.1.0.vsix
```

---

## Primary tool: `loopllm_run_pipeline`

This is the intended entry point for PromptLoop. A single call runs the full observe → elicit → refine → verify loop:

```
loopllm_run_pipeline("add retry logic to the download function")
```

What happens inside:
1. **Score** the prompt across 5 dimensions (< 1 ms, deterministic)
2. **Elicit** clarifying questions if quality < 0.6 — selected by Thompson Sampling on historical Beta priors
3. **Decompose** into subtasks if complexity > 0.5
4. **Execute** each subtask: `ctx.sample(prompt)` → evaluate → retry if below threshold
5. **Verify** the assembled output against quality criteria via a second `ctx.sample()` call
6. **Log** result + quality score to SQLite; update scoring weights via online SGD

Everything runs inline via MCP Sampling — no extra chat turns, no polling.

---

## Adaptive agent loops — Conservative Dual-Verify (CDV)

Most agent loops stop on a fixed `max_iterations` or let the agent self-grade when
it's "done." Both waste tokens or optimize **reported** progress. PromptLoop v0.7
introduces **Conservative Dual-Verify**: agents submit **step artifacts** (test
logs, diffs, summaries); the MCP server scores them through **two independent
channels** and feeds the **stricter** score into Bayesian stop/continue logic.

```python
channel_a = deterministic_evaluator.evaluate(step_output)   # regex, JSON, completeness
channel_b = critic_sample(step_output, goal, criteria)      # separate verifier call
final_score = min(channel_a, channel_b)                   # either channel can veto
```

Three MCP tools wrap any iterative task:

```
loopllm_loop_start(
  goal="refactor module and make tests pass",
  task_type="bugfix",
  required_patterns=["tests passed"],
)
  → { suggested_budget: 3, quality_threshold: 0.8, evaluator_type: "composite" }

# after each step, submit the artifact — do NOT self-grade:
loopllm_loop_step(session_id, step_output="pytest: 3 failed, 12 passed")
  → {
      decision: "continue",
      score: 0.0,
      channel_a_score: 0.0,
      channel_b_score: 0.55,
      score_source: "conservative_dual_verify",
      deficiencies: ["Required pattern not found: tests passed"],
    }

loopllm_loop_end(session_id)   → learns optimal depth from verified trajectories
```

`loopllm_loop_step` returns `stop` when any guard fires:

- **goal reached** — verified score ≥ threshold
- **plateau** — last three verified scores flatlined
- **low expected ROI** — Bayesian priors say further steps unlikely to help
- **budget / timeout / token cap / repeated output** — production stop stack

Every `loopllm_loop_end` records **verified** score trajectories so budgets sharpen
over time. See [`examples/agent_loop.py`](examples/agent_loop.py) for the controller
demo and [`docs/demo/agent_loop_demo.md`](docs/demo/agent_loop_demo.md) for CDV via MCP.

```python
from loopllm import AdaptivePriors, AgentLoopController

controller = AgentLoopController(AdaptivePriors())
session = controller.start("fix flaky test", task_type="bugfix")
verdict = controller.step(session.session_id, score=0.9)   # library API (pre-scored)
controller.end(session.session_id)
```

![Adaptive agent loop demo](img/agent_loop.svg)

What a run looks like (`python examples/agent_loop.py`):

```text
=== Loop (task_type=bugfix) ===
Suggested budget: 3 step(s) | threshold 0.80 | confidence 0.00 (from 0 past loops)
  step  1 | 0.45 |#########           | -> CONTINUE: step 1/3, score 0.450 below 0.80
  step  2 | 0.85 |#################   | -> STOP: Goal reached: 0.850 >= 0.80 at step 2

# after learning from 15+ converging bugfix loops:
Suggested budget: 1 step(s) | threshold 0.88 | confidence 0.63 (from 17 past loops)
  step  1 | ... | -> STOP: budget exhausted (1/1); escalate or accept
```

### Benchmark: adaptive vs fixed `max_iterations`

A reproducible simulation (`benchmarks/adaptive_vs_fixed.py`, seed=7, 300 test
tasks across 3 task types, threshold 0.80) comparing how a loop decides when to
stop:

| Strategy | Mean steps | Mean final score | % reaching 0.80 | Wasted steps | Efficiency (reach/step) |
|---|---|---|---|---|---|
| fixed (budget=2) | 2.00 | 0.698 | 34.3% | 0.00 | 17.2 |
| fixed (budget=6) | 6.00 | 0.939 | 94.0% | 2.50 | 15.7 |
| threshold (reactive) | 3.56 | 0.852 | 100.0% | 0.00 | 28.1 |
| **adaptive (loopllm)** | **3.56** | **0.852** | **99.7%** | **0.00** | **28.0** |

**Adaptive uses ~41% fewer steps than a fixed 6-step budget while reaching the bar
on 99.7% of tasks** (vs 94% for the fixed budget, 34% for a small one). It recovers
the efficiency of the reactive "stop at threshold" oracle — but predicts a budget
up front and stops on low expected ROI, so it doesn't depend on cheaply evaluating
every single step.

> Honest caveat: this is a simulation with stated assumptions (diminishing-returns
> curves + noise); it measures *decision efficiency given a quality signal*, not
> absolute model quality. Reproduce with `python benchmarks/adaptive_vs_fixed.py`.

---

## Quickstart

```bash
git clone https://github.com/azank1/loop-llm   # GitHub repo still named loop-llm
cd loop-llm
pip install -e ".[mcp]"
code .
```

`.vscode/mcp.json` is committed — VS Code picks up the server automatically.  
On first load verify with:

```
use loopllm_intercept with prompt: add retry logic to the download function
```

Then:

```
use loopllm_run_pipeline with prompt: add retry logic to the download function
```

---

## Install as a package

```bash
pip install loopllm[mcp]
```

Add `.vscode/mcp.json` to your project:

```json
{
  "servers": {
    "loopllm": {
      "type": "stdio",
      "command": "loopllm",
      "args": ["mcp-server", "--provider", "agent"]
    }
  }
}
```

Cursor users: `.cursor/mcp.json` is committed too (it uses `"mcpServers"` as the top-level key), so Cursor picks the server up automatically.

---

## How it works

```
You type a prompt
      ↓
loopllm_intercept          ← scores it across 5 dimensions (~0ms, deterministic)
      ↓
route decision
  < 0.4  → elicitation     ← Thompson Sampling picks the highest-gain question
  0.4–0.6 → elicit + refine
  ≥ 0.6  → refine directly
      ↓
loopllm_refine (if needed)
  → ctx.sample(prompt)     ← MCP Sampling: calls host LLM mid-execution
  → evaluate output        ← deterministic evaluators (length, regex, JSON schema)
  → if score < threshold: ctx.sample(improved prompt)   ← retry inline
  → return best result
      ↓
result logged to ~/.loopllm/store.db
      ↓
loopllm_feedback (optional rating 1–5)
  → one SGD step updates scoring dimension weights
```

**Scoring dimensions** (each 0–1, composited by learned weights into grade A–F):

| Dimension | What it catches |
|---|---|
| Specificity | Vague, generic requests |
| Constraint Clarity | Missing format, length, or rule requirements |
| Context Completeness | No background or goal stated |
| Ambiguity | Unclear references, pronouns without antecedents |
| Format Specification | No output format specified |

**MCP Sampling** — generation tools call `ctx.sample()` to invoke the host agent's LLM inline. The entire score→generate→evaluate→retry loop is a single tool call. Falls back to `agent_execute` passthrough if the client doesn't declare sampling capability.

---

## How the system learns

### Online Gradient Descent on scoring weights

Default weights are `{specificity: 0.25, constraint_clarity: 0.20, context_completeness: 0.20, ambiguity: 0.20, format_spec: 0.15}`. Each time you call `loopllm_feedback(rating)` one SGD step adjusts them:

$$\hat{y} = \sum_i w_i \cdot d_i \quad \text{(predicted quality, ambiguity inverted)}$$

$$\mathcal{L} = (\hat{y} - t)^2 \quad t = \frac{\text{rating} - 1}{4}$$

$$w_i \leftarrow w_i - 0.02 \cdot 2(\hat{y} - t) \cdot d_i$$

After each step weights are clipped to $[0.05, 0.50]$ and renormalised to sum to 1 (simplex projection). After ~50 ratings the weights converge to reflect what actually matters in your prompts. Persisted in `learned_weights` table (schema v4).

### Thompson Sampling for question ordering

Each question type (`scope`, `format`, `constraints`, etc.) maintains a $\text{Beta}(\alpha, \beta)$ prior where $\alpha$ = historical positive-impact count + 1 and $\beta$ = negative-impact count + 1. When the pipeline needs to ask a clarifying question it draws one sample per candidate type:

$$s_i \sim \text{Beta}(\alpha_i, \beta_i)$$

and selects $\arg\max_i s_i$. This is a multi-armed bandit: questions that historically improve output quality get asked more often (exploitation) while new/untested types still get surfaced occasionally (exploration).

### Beta-Binomial Bayesian priors

Every call observation updates a conjugate $\text{Beta}(\alpha, \beta)$ prior on per-(task\_type, model) convergence probability. The adaptive exit criterion for refinement loops is:

$$P(\text{converge on next iteration}) = \frac{\alpha}{\alpha + \beta}$$

The loop stops early when this probability weighted by expected improvement drop falls below a cost-threshold. Implemented in `adaptive_exit.py` using `BetaPrior.prob_above(threshold)` (normal approximation via `math.erf`).

### Welford's online algorithm for variance

`NormalPrior` tracks the running mean and variance of output quality scores with $O(1)$ memory using Welford's online algorithm with optional exponential decay ($\lambda = 0.95$) to down-weight stale observations:

$$\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}, \quad \sigma^2_n = \frac{(n-1)\sigma^2_{n-1} + (x_n - \mu_{n-1})(x_n - \mu_n)}{n}$$

### Rolling plan confidence

$$C_{\text{rolling}} = \frac{\sum_i c_i \cdot 0.85^{n-i}}{\sum_i 0.85^{n-i}}$$

where $c_i$ is the confidence of the $i$-th completed task. Exponential decay weights recent tasks more heavily. Plans gate on `confidence ≥ threshold` before advancing to the next task.

---

## Tools (28)

| Tool | What it does |
|---|---|
| `loopllm_run_pipeline` | **Primary.** Elicit → decompose → execute → verify in one call |
| `loopllm_intercept` | Score + route a prompt; logs to history |
| `loopllm_gauge` | Instant quality bars, no DB write |
| `loopllm_refine` | Score → sample → retry loop via MCP Sampling |
| `loopllm_plan_tasks` | Decompose a goal into ordered subtasks via MCP Sampling |
| `loopllm_verify_output` | Keyword pre-check + deep sample against quality criteria |
| `loopllm_elicitation_start/answer/finish` | Multi-turn clarifying question session |
| `loopllm_plan_register` | Create a confidence-gated plan saved to SQLite |
| `loopllm_plan_next` | Advance to next task; returns `needs_replan` if quality dropped |
| `loopllm_plan_update` | Record task scores; recalculates rolling confidence |
| `loopllm_plan_list` | Dashboard: all plans with gauges and task counts |
| `loopllm_plan_delete` | Remove a completed or abandoned plan |
| `loopllm_context_history` | Browse prompt history with sparklines |
| `loopllm_context_clear` | Wipe prompt history (scoped or all) |
| `loopllm_prompt_stats` | Prompting quality trend and learning curve |
| `loopllm_feedback` | Rate a response (1–5); triggers SGD weight update |
| `loopllm_suggest_config` | Bayesian-optimal loop config for a task type |
| `loopllm_loop_start` | Begin an adaptive agent loop; returns a learned step budget |
| `loopllm_loop_step` | Submit step artifact for Conservative Dual-Verify; returns continue/stop + channel scores |
| `loopllm_loop_end` | Close an agent loop and learn its optimal depth |
| `loopllm_loop_status` | Inspect an active agent-loop session |
| `loopllm_classify_task` | Label a prompt's task type |
| `loopllm_analyze_prompt` | Generate clarifying questions ranked by Thompson-sampled gain |
| `loopllm_list_tasks` | List tasks from the persistent store |
| `loopllm_show_task` | Detail view for a single task |
| `loopllm_report` | Learned weights, Bayesian priors, question effectiveness stats |

Plans and learned weights persist to `~/.loopllm/store.db` and survive server restarts (schema v4).

---

## Local models (no MCP)

```bash
pip install loopllm[serve]
loopllm serve --port 8765          # REST scoring middleware
```

```python
from loopllm.local_loop import LocalModelLoop

loop = LocalModelLoop(
    base_url="http://localhost:11434",
    model="llama3.2",
    score_url="http://localhost:8765/score",
    quality_threshold=0.80,
    max_retries=3,
)
result = loop.run("Write a Python function to parse JSON safely.")
print(result.output, result.best_score, result.converged)
```

---

## Contributing

```bash
git clone https://github.com/azank1/loop-llm   # GitHub repo still named loop-llm
cd loop-llm
pip install -e ".[dev]"
python -m pytest tests/ -q          # 204 tests (200 pass, 4 integration skipped), ~2s
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and branch naming
convention.

**Key files:**
- `src/loopllm/mcp_server.py` — all 28 MCP tools + SGD weight update + MCP Sampling helpers
- `src/loopllm/store.py` — SQLite persistence (schema v4: priors, history, plans, learned_weights)
- `src/loopllm/priors.py` — Beta-Binomial and Normal priors, Welford's algorithm
- `src/loopllm/adaptive_exit.py` — Bayesian early stopping
- `src/loopllm/agent_loop.py` — adaptive agent-loop controller (start/step/end)
- `src/loopllm/evaluators.py` — deterministic output evaluators

PRs welcome. Add tests for new tools in `tests/`.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOOPLLM_PROVIDER` | `agent` | `agent`, `ollama`, or `openrouter` |
| `LOOPLLM_MODEL` | `agent` | Model identifier (ignored in agent mode) |
| `LOOPLLM_DB` | `~/.loopllm/store.db` | SQLite store path |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |

---

## License

MIT
