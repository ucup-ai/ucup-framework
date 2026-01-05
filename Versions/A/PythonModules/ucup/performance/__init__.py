"""
Performance Optimization Module for UCUP Framework

This module provides performance enhancements including Bayesian inference optimization,
asynchronous processing, memory management, and distributed computing capabilities.
"""

from .async_processing import (
    AsyncTaskPool,
    ParallelAgentExecutor,
    StreamingResultAggregator,
)
from .bayesian_optimizer import (
    BayesianInferenceCache,
    OptimizedBayesianNetwork,
    VariableElimination,
)
from .distributed import (
    ConsistentHashRing,
    DistributedCoordinatorProtocol,
    DistributedStateStore,
    WorkerNode,
)
from .memory_manager import AgentMemoryPool, LazyAgentLoader, StateCompressor

__all__ = [
    # Bayesian optimization
    "VariableElimination",
    "BayesianInferenceCache",
    "OptimizedBayesianNetwork",
    # Async processing
    "AsyncTaskPool",
    "ParallelAgentExecutor",
    "StreamingResultAggregator",
    # Memory management
    "AgentMemoryPool",
    "StateCompressor",
    "LazyAgentLoader",
    # Distributed computing
    "WorkerNode",
    "DistributedCoordinatorProtocol",
    "ConsistentHashRing",
    "DistributedStateStore",
]
