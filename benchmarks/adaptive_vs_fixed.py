#!/usr/bin/env python3
"""Benchmark: adaptive Bayesian loop control vs fixed / threshold strategies.

This is a **reproducible simulation** (no live LLM calls) with explicitly stated
assumptions. Each task has a hidden diminishing-returns quality curve:

    score(t) = 1 - (1 - s0) * exp(-r * (t - 1)) + noise        # t = 1, 2, 3, ...

where ``s0`` (first-step quality) and ``r`` (improvement rate) are sampled per
task from task-type-specific ranges. Three task types ("easy", "medium", "hard")
converge at different depths, which is exactly the regime where a fixed
``max_iterations`` is wrong for *some* task and right for others.

We compare four ways to decide when to stop an agent loop:

* ``fixed_small`` — always run a small fixed budget
* ``fixed_large`` — always run a large fixed budget
* ``threshold``   — stop the moment score >= threshold (reactive; requires
  evaluating every step), else a hard cap
* ``adaptive``    — loopllm's ``AgentLoopController``, warmed on a train split,
  predicting a per-task-type budget and stopping on goal/plateau/low-ROI

Run it::

    python benchmarks/adaptive_vs_fixed.py

Honest caveats: results depend on the simulated curves and noise; this measures
*decision efficiency given a quality signal*, not absolute model quality. The
same harness can be pointed at real per-step scores (e.g. an Ollama loop) by
feeding observed scores instead of the synthetic curve.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import exp

from loopllm import AdaptivePriors, AgentLoopController, CallObservation

THRESHOLD = 0.8
CAP = 8
FIXED_SMALL = 2
FIXED_LARGE = 6
MODEL = "sim-model"

# (s0_low, s0_high, r_low, r_high) per task type.
TASK_TYPES: dict[str, tuple[float, float, float, float]] = {
    "easy": (0.60, 0.75, 0.8, 1.2),
    "medium": (0.40, 0.55, 0.4, 0.7),
    "hard": (0.25, 0.40, 0.2, 0.4),
}


@dataclass
class Task:
    task_type: str
    scores: list[float]  # length CAP, the hidden per-step quality curve


@dataclass
class Result:
    name: str
    mean_steps: float = 0.0
    mean_final: float = 0.0
    pct_threshold: float = 0.0
    mean_wasted: float = 0.0
    samples: int = 0
    _steps: list[int] = field(default_factory=list)
    _finals: list[float] = field(default_factory=list)
    _reached: list[bool] = field(default_factory=list)
    _wasted: list[int] = field(default_factory=list)

    def add(self, steps: int, final: float, reached: bool, wasted: int) -> None:
        self._steps.append(steps)
        self._finals.append(final)
        self._reached.append(reached)
        self._wasted.append(wasted)

    def finalize(self) -> Result:
        n = max(len(self._steps), 1)
        self.samples = len(self._steps)
        self.mean_steps = sum(self._steps) / n
        self.mean_final = sum(self._finals) / n
        self.pct_threshold = 100.0 * sum(self._reached) / n
        self.mean_wasted = sum(self._wasted) / n
        return self

    @property
    def efficiency(self) -> float:
        """Quality-reach per step spent — higher is better."""
        return self.pct_threshold / self.mean_steps if self.mean_steps else 0.0


def _first_cross(scores: list[float], upto: int) -> int | None:
    """1-based index of first step with score >= THRESHOLD within ``upto`` steps."""
    for i in range(upto):
        if scores[i] >= THRESHOLD:
            return i + 1
    return None


def generate_tasks(n_per_type: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    for task_type, (s0lo, s0hi, rlo, rhi) in TASK_TYPES.items():
        for _ in range(n_per_type):
            s0 = rng.uniform(s0lo, s0hi)
            r = rng.uniform(rlo, rhi)
            scores: list[float] = []
            for t in range(1, CAP + 1):
                base = 1.0 - (1.0 - s0) * exp(-r * (t - 1))
                val = base + rng.gauss(0.0, 0.02)
                scores.append(max(0.0, min(1.0, val)))
            tasks.append(Task(task_type=task_type, scores=scores))
    rng.shuffle(tasks)
    return tasks


def run_fixed(tasks: list[Task], budget: int, name: str) -> Result:
    res = Result(name)
    for task in tasks:
        steps = min(budget, CAP)
        window = task.scores[:steps]
        best = max(window)
        cross = _first_cross(task.scores, steps)
        reached = cross is not None
        wasted = max(0, steps - cross) if reached else 0
        res.add(steps, best, reached, wasted)
    return res.finalize()


def run_threshold(tasks: list[Task], name: str = "threshold") -> Result:
    res = Result(name)
    for task in tasks:
        cross = _first_cross(task.scores, CAP)
        steps = cross if cross is not None else CAP
        best = max(task.scores[:steps])
        res.add(steps, best, cross is not None, 0)
    return res.finalize()


def warm_priors(priors: AdaptivePriors, train: list[Task]) -> None:
    """Warm priors from historical full-loop logs (train split)."""
    for task in train:
        cross = _first_cross(task.scores, CAP)
        stop = cross if cross is not None else CAP
        scores = task.scores[:stop]
        priors.observe(
            CallObservation(
                task_type=task.task_type,
                model_id=MODEL,
                scores=scores,
                latencies_ms=[1000.0] * len(scores),
                converged=cross is not None,
                total_iterations=len(scores),
                max_iterations=CAP,
                quality_threshold=THRESHOLD,
            )
        )


def run_adaptive(tasks: list[Task], controller: AgentLoopController, name: str = "adaptive") -> Result:
    res = Result(name)
    for task in tasks:
        session = controller.start(
            goal="reach quality bar",
            task_type=task.task_type,
            model_id=MODEL,
            quality_threshold=THRESHOLD,
        )
        steps = 0
        for t in range(CAP):
            steps += 1
            verdict = controller.step(session.session_id, task.scores[t])
            if verdict["decision"] == "stop":
                break
        best = max(task.scores[:steps])
        cross = _first_cross(task.scores, steps)
        reached = cross is not None
        wasted = max(0, steps - cross) if reached else 0
        controller.end(session.session_id, converged=reached)
        res.add(steps, best, reached, wasted)
    return res.finalize()


def run_benchmark(seed: int = 7, n_per_type: int = 200) -> dict[str, Result]:
    tasks = generate_tasks(n_per_type, seed)
    split = len(tasks) // 2
    train, test = tasks[:split], tasks[split:]

    priors = AdaptivePriors()
    warm_priors(priors, train)
    controller = AgentLoopController(priors)

    return {
        "fixed_small": run_fixed(test, FIXED_SMALL, f"fixed (budget={FIXED_SMALL})"),
        "fixed_large": run_fixed(test, FIXED_LARGE, f"fixed (budget={FIXED_LARGE})"),
        "threshold": run_threshold(test),
        "adaptive": run_adaptive(test, controller),
    }


def render_table(results: dict[str, Result]) -> str:
    header = (
        "| Strategy | Mean steps | Mean final score | % reaching 0.80 "
        "| Wasted steps | Efficiency (reach/step) |\n"
        "|---|---|---|---|---|---|"
    )
    rows = []
    for r in results.values():
        rows.append(
            f"| {r.name} | {r.mean_steps:.2f} | {r.mean_final:.3f} "
            f"| {r.pct_threshold:.1f}% | {r.mean_wasted:.2f} | {r.efficiency:.1f} |"
        )
    return header + "\n" + "\n".join(rows)


def main() -> None:
    results = run_benchmark()
    table = render_table(results)
    print("Adaptive agent loops vs fixed / threshold strategies")
    print("(simulation, seed=7, 300 test tasks across 3 task types, threshold=0.80)\n")
    print(table)

    a, fl, fs = results["adaptive"], results["fixed_large"], results["fixed_small"]
    step_saving = 100.0 * (fl.mean_steps - a.mean_steps) / fl.mean_steps
    print(
        f"\nAdaptive uses {step_saving:.0f}% fewer steps than {fl.name} "
        f"while reaching the bar on {a.pct_threshold:.1f}% of tasks "
        f"(vs {fl.pct_threshold:.1f}% for {fl.name}, {fs.pct_threshold:.1f}% for {fs.name})."
    )

    out_dir = __import__("pathlib").Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "adaptive_vs_fixed.md").write_text(
        "# Benchmark results\n\n"
        "Reproduce with `python benchmarks/adaptive_vs_fixed.py` (seed=7).\n\n"
        + table
        + f"\n\nAdaptive uses {step_saving:.0f}% fewer steps than {fl.name} "
        f"at {a.pct_threshold:.1f}% goal-reach.\n"
    )


if __name__ == "__main__":
    main()
