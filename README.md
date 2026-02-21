# loop-llm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**v0.3.0** — A package from [Metaorcha](https://github.com/azank1), a platform currently under development researching planning and discovery. This release collects real-world usage data and validates core abstractions to inform future versions.

Iterative refinement engine for LLM applications with Bayesian adaptive exit. Loop any LLM call with pluggable evaluators, learned optimal depth, and cost-aware stopping. Works as an **MCP server** inside VS Code Copilot / Cursor, as a **CLI tool**, or as a **Python library**. Model-agnostic, zero training required.

---

## Install

```bash
pip install loopllm                  # core only
pip install loopllm[ollama]          # + Ollama (local models)
pip install loopllm[openrouter]      # + OpenRouter (cloud models)
pip install loopllm[mcp]             # + MCP server for IDE integration
pip install loopllm[all]             # everything
```

## Quick Start — MCP Server (recommended)

The fastest way to use loop-llm is through the MCP server inside your IDE.

### VS Code / GitHub Copilot

Add to `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "loopllm": {
      "type": "stdio",
      "command": "loopllm",
      "args": ["mcp-server", "--provider", "ollama"],
      "env": {
        "LOOPLLM_MODEL": "qwen2.5:0.5b",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "loopllm": {
      "command": "loopllm",
      "args": ["mcp-server", "--provider", "ollama"],
      "env": {
        "LOOPLLM_MODEL": "qwen2.5:0.5b",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

Once configured, the IDE agent (Copilot, Cursor) can call any of the 13 MCP tools directly in chat.

### MCP Tools

| Tool | Description |
|------|-------------|
| `loopllm_refine` | Iteratively refine a prompt with evaluation-driven feedback |
| `loopllm_run_pipeline` | Run the full refinement pipeline (classify → plan → refine → verify) |
| `loopllm_classify_task` | Classify a prompt by task type and complexity |
| `loopllm_analyze_prompt` | Analyze prompt quality and suggest improvements |
| `loopllm_elicitation_start` | Start an intent-elicitation session with clarifying questions |
| `loopllm_elicitation_answer` | Answer a clarifying question in an active session |
| `loopllm_elicitation_finish` | Finish elicitation and get the refined intent spec |
| `loopllm_plan_tasks` | Decompose a goal into a dependency-ordered task plan |
| `loopllm_verify_output` | Score output against evaluators and get a quality report |
| `loopllm_suggest_config` | Get Bayesian-optimised loop config for a task type |
| `loopllm_report` | Show refinement history and Bayesian prior statistics |
| `loopllm_list_tasks` | List recent tasks from the persistent store |
| `loopllm_show_task` | Show details of a specific task by ID |

## Quick Start — CLI

```bash
# Refine a prompt with Ollama
loopllm refine "Write a Python function to merge two sorted lists" \
  --provider ollama --model qwen2.5:0.5b --max-iter 5

# Start the MCP server manually
loopllm mcp-server --provider ollama --model qwen2.5:0.5b
```

## Quick Start — Python Library

```python
from loopllm import LoopedLLM, LoopConfig
from loopllm.providers.ollama import OllamaProvider
from loopllm.evaluators import JSONSchemaEvaluator

provider = OllamaProvider(model="qwen2.5:0.5b")
config = LoopConfig(max_iterations=5, quality_threshold=0.8)
loop = LoopedLLM(provider=provider, config=config)

evaluator = JSONSchemaEvaluator(
    required_fields=["name", "age", "email"],
    field_types={"age": int, "name": str},
)

result = loop.refine("Generate a JSON user profile.", evaluator)
print(f"Converged in {result.metrics.total_iterations} iterations")
print(f"Score: {result.metrics.best_score:.2f}")
print(f"Output: {result.output}")
```

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Prompt     │────▶│  LLM Call     │────▶│  Evaluator   │
└─────────────┘     └──────────────┘     └──────┬───────┘
       ▲                                         │
       │            ┌──────────────┐             │
       └────────────│  Feedback    │◀────────────┘
        (if needed) │  Builder     │    score < threshold?
                    └──────────────┘

Exit conditions (checked in order):
  1. quality_threshold  — score >= target
  2. convergence_delta  — improvement plateaued
  3. bayesian_exit      — learned priors say stop
  4. max_iterations     — hard cap reached
  5. timeout_ms         — wall-clock limit
```

## Bayesian Adaptive Exit

The `AdaptivePriors` system learns from past refinement runs to predict optimal loop depth:

- **Per-iteration profiles** — tracks expected score, delta, convergence probability, and latency at each iteration depth
- **Welford's online algorithm** — maintains exact running mean and variance with zero memory overhead
- **Beta priors** — models binary outcomes (converged/not) with proper uncertainty
- **Cost-aware stopping** — balances expected improvement against computational cost
- **SQLite persistence** — atomic save/load, WAL mode, thread-safe

```python
from loopllm import AdaptivePriors, CallObservation
from loopllm.adaptive_exit import BayesianExitCondition

priors = AdaptivePriors(store_path=Path("./priors.json"))

# After each run, record what happened
priors.observe(CallObservation(
    task_type="decompose", model_id="gpt-4o",
    scores=[0.3, 0.55, 0.78, 0.88], converged=True, ...
))

# Use learned beliefs to configure future runs
config = priors.suggest_config("decompose", "gpt-4o")
```

## Evaluators

All evaluators are deterministic — no LLM self-assessment.

| Evaluator | Description |
|-----------|-------------|
| `ThresholdEvaluator` | Wraps any `(output, context) -> float` scorer with a pass/fail threshold |
| `RegexEvaluator` | Checks for required / forbidden regex patterns |
| `JSONSchemaEvaluator` | Validates JSON structure, required fields, and field types |
| `LengthEvaluator` | Enforces character and word count bounds |
| `CompositeEvaluator` | Weighted combination of multiple evaluators |
| `CompletenessEvaluator` | Checks output completeness against expected criteria |
| `ConsistencyEvaluator` | Checks output consistency against reference content |

## Providers

| Provider | Install | Description |
|----------|---------|-------------|
| `OllamaProvider` | `pip install loopllm[ollama]` | Local models via [Ollama](https://ollama.com) |
| `OpenRouterProvider` | `pip install loopllm[openrouter]` | Cloud models via [OpenRouter](https://openrouter.ai) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOOPLLM_PROVIDER` | `ollama` | Provider backend (`ollama` or `openrouter`) |
| `LOOPLLM_MODEL` | `gpt-4o-mini` | Model identifier |
| `LOOPLLM_DB` | `~/.loopllm/store.db` | SQLite database path |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |

## Research Background

This project draws inspiration from iterative refinement approaches in LLM systems. The Ouro/LoopLM paper explores iterative refinement at the **model layer** using weight-tied transformer blocks. loop-llm operates at the **application layer**, wrapping any LLM API with evaluation-driven feedback loops and Bayesian meta-learning.

These are complementary approaches. Model-level looping optimises inference compute within the forward pass; application-level looping optimises prompt–response cycles with domain-specific evaluation.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and setup instructions.

## License

MIT — see [LICENSE](LICENSE) for details.
