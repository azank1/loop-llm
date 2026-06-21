"""Tests for the adaptive-vs-fixed benchmark harness (deterministic)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import adaptive_vs_fixed as bench  # noqa: E402


def test_benchmark_is_deterministic() -> None:
    """Same seed yields identical headline numbers."""
    r1 = bench.run_benchmark(seed=7, n_per_type=120)
    r2 = bench.run_benchmark(seed=7, n_per_type=120)
    assert r1["adaptive"].mean_steps == r2["adaptive"].mean_steps
    assert r1["adaptive"].pct_threshold == r2["adaptive"].pct_threshold


def test_adaptive_beats_fixed_budgets() -> None:
    """Adaptive spends fewer steps than the large fixed budget and reaches the
    bar far more often than the small fixed budget."""
    res = bench.run_benchmark(seed=7, n_per_type=200)
    adaptive = res["adaptive"]
    fixed_large = res["fixed_large"]
    fixed_small = res["fixed_small"]

    assert adaptive.mean_steps < fixed_large.mean_steps
    assert adaptive.pct_threshold > fixed_small.pct_threshold
    # Better quality-per-step than burning a fixed large budget.
    assert adaptive.efficiency >= fixed_large.efficiency
    # Reaches the bar on the vast majority of tasks.
    assert adaptive.pct_threshold >= 95.0


def test_table_renders() -> None:
    res = bench.run_benchmark(seed=7, n_per_type=60)
    table = bench.render_table(res)
    assert "Strategy" in table and "adaptive" in table
