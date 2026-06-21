# Benchmark results

Reproduce with `python benchmarks/adaptive_vs_fixed.py` (seed=7).

| Strategy | Mean steps | Mean final score | % reaching 0.80 | Wasted steps | Efficiency (reach/step) |
|---|---|---|---|---|---|
| fixed (budget=2) | 2.00 | 0.698 | 34.3% | 0.00 | 17.2 |
| fixed (budget=6) | 6.00 | 0.939 | 94.0% | 2.50 | 15.7 |
| threshold | 3.56 | 0.852 | 100.0% | 0.00 | 28.1 |
| adaptive | 3.56 | 0.852 | 99.7% | 0.00 | 28.0 |

Adaptive uses 41% fewer steps than fixed (budget=6) at 99.7% goal-reach.
