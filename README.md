# loop-llm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Prompt quality scoring and iterative refinement as an MCP server. Works with VS Code Copilot, Cursor, or any MCP-compatible client. No extra LLM required.

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

Every prompt is scored across five dimensions: **Specificity**, **Constraint Clarity**, **Context Completeness**, **Ambiguity**, and **Format Specification**. The composite score (0–1) routes automatically:

```
< 0.4   → elicitation  (agent asks clarifying questions)
0.4–0.6 → elicit then refine
≥ 0.6   → direct refinement
```

For generation tools (`loopllm_refine`, `loopllm_run_pipeline`, etc.), loopllm uses **MCP Sampling** — `ctx.sample()` calls the host agent's LLM mid-execution, runs evaluators on the result, and retries inline without ending the tool call. Falls back gracefully if the client has no sampling capability.

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
