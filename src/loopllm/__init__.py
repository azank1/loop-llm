"""loop-llm: iterative refinement engine for LLM applications."""
from __future__ import annotations

from loopllm.adaptive_exit import BayesianExitCondition
from loopllm.engine import (
    CompositeEvaluator,
    EvaluationResult,
    ExitReason,
    IterationRecord,
    LoopConfig,
    LoopedLLM,
    LoopMetrics,
    RefinementResult,
)
from loopllm.priors import AdaptivePriors, CallObservation

__version__ = "0.1.0"

__all__ = [
    "LoopedLLM",
    "LoopConfig",
    "EvaluationResult",
    "ExitReason",
    "IterationRecord",
    "LoopMetrics",
    "RefinementResult",
    "CompositeEvaluator",
    "AdaptivePriors",
    "CallObservation",
    "BayesianExitCondition",
]
