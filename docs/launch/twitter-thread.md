# X / Twitter thread draft

> Draft only — nothing is posted automatically. Attach the IDE screen-recording
> (or the screenshots in `img/`) to tweet 1; it drives the most engagement.

---

**1/**
Most agent loops stop on a fixed `max_iterations` or "let the model decide."
One burns tokens, the other loops forever.

I built a statistically-grounded stop button for agent loops — no training data.

`pip install loopllm` · MIT · works in Cursor & VS Code 🧵

[attach: demo GIF / screenshot]

---

**2/**
It's an MCP server. Three tools wrap any iterative task:

• loop_start(goal, task_type) → a *learned* step budget
• loop_step(score) → continue / stop, with a reason
• loop_end() → records the run so next time is smarter

Your IDE agent picks it up automatically.

---

**3/**
How it decides to stop:
• goal reached (score ≥ threshold)
• progress plateaued
• low expected ROI (priors say the gap won't close)
• budget exhausted

No PyTorch. Just Beta-Binomial priors + Welford's online variance, per
(task_type, model). Interpretable every step.

---

**4/**
Does it actually help? Reproducible benchmark vs fixed budgets:

adaptive uses ~41% fewer steps than a fixed 6-step loop
…while hitting the quality bar on 99.7% of tasks (vs 94%).

Honest: it's a simulation w/ stated assumptions. Repro:
`python benchmarks/adaptive_vs_fixed.py`

---

**5/**
Same Bayesian core also powers prompt scoring (5 dims, weights tuned by your
1–5 ratings via online SGD) and an iterative refinement loop. 28 MCP tools total.

Python 3.11+, FastMCP, SQLite. 204 tests, mypy --strict in CI.

---

**6/**
Repo + demo + benchmark:
https://github.com/azank1/loop-llm

Feedback very welcome — especially on the stopping rule and which task types
you'd want priors for. ⭐ if it's useful.

---

## Single-tweet (standalone) variant

Agent loops shouldn't run on a fixed max_iterations.

loopllm = a Bayesian "stop button" for agent loops, as an MCP server for
Cursor/VS Code. ~41% fewer steps than a fixed budget at 99.7% goal-reach, no
training data.

pip install loopllm · MIT
https://github.com/azank1/loop-llm
