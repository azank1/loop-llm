# Reddit launch drafts

> Drafts only — nothing is posted automatically. Review subreddit rules (most ban
> low-effort self-promotion; lead with substance, link last). Flair as "Project" /
> "Tooling" where required.

---

## r/LocalLLaMA

**Title:** I built a Bayesian "stop button" for agent loops — learns when to quit, no training data, runs as an MCP server

**Body:**

If you run local models in agent loops you've felt both failure modes: a fixed
`max_iterations` that burns tokens on a 7B that already plateaued, or "let the model
decide" that loops forever. I wanted a principled middle ground.

`loopllm` is an MCP server (also a plain Python lib + CLI) that gives your agent's
own loop an adaptive stop button:

```
loopllm_loop_start(goal, task_type)        -> learned step budget + threshold
loopllm_loop_step(session_id, score)       -> {decision: continue|stop, reason}
loopllm_loop_end(session_id)               -> records the run; budgets sharpen
```

It stops on goal reached / plateaued progress / low expected ROI / budget
exhausted. The learning is closed-form Bayesian (Beta-Binomial convergence priors +
Welford's online variance), keyed per `(task_type, model)` — so it learns that,
say, `qwen2.5:7b` needs ~4 steps for `decompose` but 1 for `validate`. No PyTorch,
no training set, persists to local SQLite.

First-class local support: point it at Ollama or any OpenAI-compatible endpoint
(OpenRouter, llama.cpp server). Model-agnostic by design.

Reproducible benchmark (simulation w/ stated assumptions): adaptive uses ~41%
fewer steps than a fixed 6-step budget while reaching the bar on 99.7% of tasks.
`python benchmarks/adaptive_vs_fixed.py`.

MIT, Python 3.11+, 204 tests. Repo + runnable demo:
https://github.com/azank1/loop-llm

Curious what task types you'd want priors for, and whether the stopping rule
matches your intuition for small models.

---

## r/MachineLearning

**Title:** [P] Adaptive stopping for LLM agent loops via Beta-Binomial convergence priors (no training data)

**Body:**

**What:** A lightweight controller that decides when an LLM agent loop should stop,
using closed-form Bayesian estimates instead of a fixed iteration cap or
LLM-as-judge.

**How it works:** Each loop reports a per-step progress score in [0, 1]. Per
`(task_type, model)` I maintain (a) a Beta-Binomial prior on per-iteration
convergence, and (b) NormalPriors over score deltas and latency updated online with
Welford's algorithm. The stop rule estimates whether the remaining quality gap is
likely to be bridged within the remaining budget (normal approximation to the Beta
CDF) and stops when the expected ROI is low — equivalent to the early-exit
condition used in the package's refinement loop. Optimal depth is the first
iteration where score ≥ threshold, averaged across observed runs. Clarifying
questions are ordered by Thompson Sampling over question-type Beta priors.

**Why not RL / a trained model:** zero training data, interpretable decisions
(every stop returns a reason like `E[delta]=0.03±0.02, threshold=0.80`), O(1) memory
per arm, and it degrades gracefully to task-type defaults during cold start
(< 5 observations).

**Limitations:** the progress signal quality bounds everything; the normal
approximation to the Beta is crude in the tails; priors are per
`(task_type, model)` so sparse arms stay at defaults.

Code (MIT), tests, and a runnable demo: https://github.com/azank1/loop-llm
Core files: `priors.py`, `agent_loop.py`, `adaptive_exit.py`. Feedback on the
stopping criterion and prior choices is very welcome.
