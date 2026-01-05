"""
Testing and Validation Package for UCUP Framework.

Provides comprehensive testing capabilities for framework components.
"""

from .agent_tester import AgentTester, TestResult
from .performance_validator import PerformanceValidator, PerformanceMetrics, BenchmarkResult
from .scenario_generator import ScenarioGenerator, ScenarioType, GeneratedScenario, ScenarioConfig
from .model_interfaces import ModelMetadata, ModelPrediction, UniversalInput, UniversalModelInterface

__all__ = [
    "AgentTester",
    "TestResult",
    "PerformanceValidator",
    "PerformanceMetrics",
    "BenchmarkResult",
    "ScenarioGenerator",
    "ScenarioType",
    "GeneratedScenario",
    "ScenarioConfig",
    "ModelMetadata",
    "ModelPrediction",
    "UniversalInput",
    "UniversalModelInterface"
]
