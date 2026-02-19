# loop-llm

[![PyPI](https://img.shields.io/pypi/v/loopllm)](https://pypi.org/project/loopllm/)
[![CI](https://github.com/azank1/loop-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/azank1/loop-llm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**v1** — A package from [Metaorcha](https://github.com/azank1), a platform currently under development researching planning and discovery. This release collects real-world usage data and validates core abstractions to inform v2 and beyond.

Iterative refinement engine for LLM applications with Bayesian adaptive exit. Loop any LLM call with pluggable evaluators, learned optimal depth, and cost-aware stopping. Model-agnostic, zero training required.

## Install

```bash
pip install loopllm                  # core only
pip install loopllm[openrouter]      # + OpenRouter provider
pip install loopllm[ollama]          # + Ollama provider
pip install loopllm[all]             # all providers
```

## Quick Start

```python
from loopllm import LoopedLLM, LoopConfig
from loopllm.providers.mock import MockLLMProvider
from loopllm.evaluators import JSONSchemaEvaluator

provider = MockLLMProvider(responses=[
    "not json",
    '{"name": "Alice"}',
    '{"name": "Alice", "age": 30, "email": "alice@example.com"}',
])
config = LoopConfig(max_iterations=5, quality_threshold=0.8)
loop = LoopedLLM(provider=provider, config=config)
evaluator = JSONSchemaEvaluator(
    required_fields=["name", "age", "email"],
    field_types={"age": int, "name": str}
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

- **Per-iteration profiles**: tracks expected score, delta, convergence probability, and latency at each iteration depth
- **Welford's online algorithm**: maintains exact running mean and variance with zero memory overhead
- **Beta priors**: models binary outcomes (converged/not) with proper uncertainty
- **Cost-aware stopping**: balances expected improvement against computational cost
- **JSON persistence**: atomic save/load for production deployment

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

| Evaluator | Description |
|-----------|-------------|
| `ThresholdEvaluator` | Wraps any `(output, context) -> float` scorer with a pass/fail threshold |
| `RegexEvaluator` | Checks for required / forbidden regex patterns |
| `JSONSchemaEvaluator` | Validates JSON structure, required fields, and field types |
| `LengthEvaluator` | Enforces character and word count bounds |
| `CompositeEvaluator` | Weighted combination of multiple evaluators |

## Providers

| Provider | Description |
|----------|-------------|
| `OpenRouterProvider` | Calls OpenRouter API (`pip install loopllm[openrouter]`) |
| `OllamaProvider` | Calls local Ollama instance (`pip install loopllm[ollama]`) |
| `MockLLMProvider` | Cycles through pre-set responses — ideal for testing |

## Research Background

This project draws inspiration from iterative refinement approaches in LLM systems. The Ouro/LoopLM paper explores iterative refinement at the **model layer** using weight-tied transformer blocks. loop-llm operates at the **application layer**, wrapping any LLM API with evaluation-driven feedback loops and Bayesian meta-learning.

These are complementary, not competing approaches. Model-level looping optimises inference compute within the forward pass; application-level looping optimises prompt–response cycles with domain-specific evaluation.

## Contributing

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/azank1/loop-llm).

```bash
git clone https://github.com/azank1/loop-llm.git
cd loop-llm
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE) for details.
