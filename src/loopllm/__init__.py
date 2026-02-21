"""loop-llm: iterative refinement engine for LLM applications."""
from __future__ import annotations

from loopllm.adaptive_exit import BayesianExitCondition
from loopllm.elicitation import (
    ClarifyingQuestion,
    ElicitationSession,
    IntentRefiner,
    IntentSpec,
)
from loopllm.engine import (
    CompositeEvaluator,
    Evaluator,
    EvaluationResult,
    ExitConditionProtocol,
    ExitReason,
    IterationRecord,
    LoopConfig,
    LoopedLLM,
    LoopMetrics,
    RefinementResult,
)
from loopllm.priors import AdaptivePriors, CallObservation
from loopllm.store import LoopStore, SQLiteBackedPriors
from loopllm.tasks import Task, TaskOrchestrator, TaskPlan, TaskState

__version__ = "0.3.0"

__all__ = [
    # Engine
    "LoopedLLM",
    "LoopConfig",
    "EvaluationResult",
    "ExitReason",
    "IterationRecord",
    "LoopMetrics",
    "RefinementResult",
    "CompositeEvaluator",
    "Evaluator",
    "ExitConditionProtocol",
    # Priors
    "AdaptivePriors",
    "CallObservation",
    "BayesianExitCondition",
    # Elicitation
    "IntentRefiner",
    "IntentSpec",
    "ClarifyingQuestion",
    "ElicitationSession",
    # Store
    "LoopStore",
    "SQLiteBackedPriors",
    # Tasks
    "Task",
    "TaskPlan",
    "TaskState",
    "TaskOrchestrator",
]
