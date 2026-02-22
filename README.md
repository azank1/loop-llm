# loop-llm

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&pause=1000&color=00CFFF&center=true&vCenter=true&width=700&lines=Iterative+Prompt+Refinement+Engine;Bayesian+Adaptive+Exit+%2B+Cost-Aware+Stopping;MCP+Server+%E2%80%94+24+Tools+for+VS+Code+%2B+Cursor;Prompt+Quality+Scoring+%2B+Online+Weight+Learning;Intent+Elicitation+%2B+Task+Decomposition;Zero+Training+%E2%80%94+Model+Agnostic)](https://github.com/azank1/loop-llm)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A prompt observer and quality loop for your AI agent.**

loop-llm sits between you and your IDE's agent as an MCP server. Every prompt is scored, routed, refined if needed, and then verified — all inside a single tool call using MCP Sampling. The system **learns** over time: it runs online gradient descent on scoring dimension weights every time you rate a response, and selects clarifying questions via Thompson Sampling on Beta priors.

---

## Primary tool: `loopllm_run_pipeline`

This is the intended entry point. A single call runs the full observe → elicit → refine → verify loop:

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

## Quickstart

```bash
git clone https://github.com/azank1/loop-llm
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

For Cursor use `.cursor/mcp.json` with `"mcpServers"` as the top-level key.

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

## Tools (24)

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
git clone https://github.com/azank1/loop-llm
cd loop-llm
pip install -e ".[dev]"
python -m pytest tests/ -q          # 186 tests, ~2s
```

**Branch strategy:**
- `main` — stable, released code
- `testing-production` — active development; PRs target this branch first

**Key files:**
- `src/loopllm/mcp_server.py` — all 24 MCP tools + SGD weight update + MCP Sampling helpers
- `src/loopllm/store.py` — SQLite persistence (schema v4: priors, history, plans, learned_weights)
- `src/loopllm/priors.py` — Beta-Binomial and Normal priors, Welford's algorithm
- `src/loopllm/adaptive_exit.py` — Bayesian early stopping
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


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A prompt observer and quality loop for your AI agent.**

loop-llm sits between you and your IDE's agent as an MCP server. Every prompt you write is intercepted, scored across five quality dimensions, and routed to the right tool — clarification, refinement, or task decomposition. It tracks your prompting history over time, lets you manage multi-task plans with confidence gating, and runs iterative improvement loops using the host agent's own LLM via MCP Sampling, with no separate model needed.

Think of it as a **prompt observer**: it watches what you ask, measures how well-formed it is, tells you where it's weak, and automatically improves it before the agent acts.

**Stack:** Python 3.11+, FastMCP (stdio), SQLite (Bayesian priors + prompt history + plans), deterministic heuristic scoring, MCP Sampling for mid-execution LLM calls.

---

## Quickstart

```bash
git clone https://github.com/azank1/loop-llm
cd loop-llm
pip install -e ".[mcp]"
code .
```

`.vscode/mcp.json` is committed — VS Code picks up the server automatically. If tools don't appear: **MCP: Restart Server → loopllm** in the Command Palette.

Verify in Copilot chat:
```
use loopllm_gauge to score this prompt: write me some code
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

For Cursor use `.cursor/mcp.json` with `"mcpServers"` as the top-level key.

---

## How it works

```
You type a prompt
      ↓
loopllm_intercept          ← scores it across 5 dimensions (~0ms, deterministic)
      ↓
route decision
  < 0.4  → elicitation     ← agent asks the 1-2 highest-gain clarifying questions
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
```

**Scoring dimensions** (each 0–1, composited into a grade A–F):

| Dimension | What it catches |
|---|---|
| Specificity | Vague, generic requests |
| Constraint Clarity | Missing format, length, or rule requirements |
| Context Completeness | No background or goal stated |
| Ambiguity | Unclear references, pronouns without antecedents |
| Format Specification | No output format specified |

**MCP Sampling** means generation tools (`loopllm_refine`, `loopllm_run_pipeline`, `loopllm_plan_tasks`, `loopllm_verify_output`) call `ctx.sample()` to invoke the host agent's LLM inline — the entire score→generate→evaluate→retry loop happens inside a single tool call, not across multiple chat turns. Falls back to `agent_execute` passthrough if the client doesn't declare sampling capability.

**Prompt history** is stored in SQLite with per-prompt scores, grades, task types, and session tags. Use `loopllm_context_history` to browse it, `loopllm_prompt_stats` for your trend sparkline, or `loopllm_context_clear` to reset.

---

## Tools (24)

| Tool | What it does |
|---|---|
| `loopllm_intercept` | Score + route a prompt — call this first on every request |
| `loopllm_gauge` | Instant quality bars, no DB write |
| `loopllm_refine` | Score → sample → retry loop via MCP sampling |
| `loopllm_run_pipeline` | Elicit → decompose → execute → verify in one tool call |
| `loopllm_plan_tasks` | Decompose a goal into ordered subtasks via MCP sampling |
| `loopllm_verify_output` | Keyword pre-check + deep sample against quality criteria |
| `loopllm_elicitation_start/answer/finish` | Multi-turn clarifying question session |
| `loopllm_plan_register` | Create a confidence-gated plan saved to SQLite |
| `loopllm_plan_next` | Advance to next task; returns `needs_replan` if quality dropped |
| `loopllm_plan_update` | Record task scores; recalculates rolling confidence |
| `loopllm_plan_list` | Dashboard: all plans with gauges and task counts |
| `loopllm_plan_delete` | Remove a completed or abandoned plan |
| `loopllm_context_history` | Browse prompt history with sparkline |
| `loopllm_context_clear` | Wipe prompt history (scoped or all) |
| `loopllm_prompt_stats` | Prompting quality trend and learning curve |
| `loopllm_feedback` | Rate a response (1–5) to improve future scoring |
| `loopllm_suggest_config` | Bayesian-optimal loop config for a task type |
| `loopllm_classify_task` | Label a prompt's task type |
| `loopllm_analyze_prompt` | Generate clarifying questions ranked by information gain |
| `loopllm_list_tasks` | List tasks from the persistent store |
| `loopllm_show_task` | Detail view for a single task |
| `loopllm_report` | Learned Bayesian priors and question effectiveness |

Plans persist to `~/.loopllm/store.db` and survive server restarts.

Rolling confidence:
```
confidence(task) = prompt_score × 0.35 + output_score × 0.65
rolling            = Σ(confidence_i × 0.85^(n-i)) / Σ(0.85^(n-i))
```

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
git clone https://github.com/azank1/loop-llm
cd loop-llm
pip install -e ".[dev]"
python -m pytest tests/ -q          # 186 tests, ~0.5s
```

**Branch strategy:**
- `main` — stable, released code
- `testing-production` — active development; PRs target this branch first

**Key files:**
- `src/loopllm/mcp_server.py` — all 24 MCP tools + MCP sampling helpers
- `src/loopllm/plan_registry.py` — confidence-gated plan tracking
- `src/loopllm/store.py` — SQLite persistence (schema v3)
- `src/loopllm/evaluators.py` — deterministic output evaluators
- `src/loopllm/serve.py` — REST server for local models
- `src/loopllm/local_loop.py` — Ollama wrapper with score→retry loop

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
