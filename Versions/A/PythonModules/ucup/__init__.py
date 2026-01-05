"""
UCUP Framework - macOS Unified Cognitive Uncertainty Processing.

A macOS-native framework inspired by UCUP, featuring:
- Core ML accelerated probabilistic reasoning
- Grand Central Dispatch coordination
- Vision and AVFoundation multimodal processing
- Security framework integration
- Native macOS performance optimizations

Copyright (c) 2025 UCUP Framework Contributors. All rights reserved.
Licensed under the Apache License, Version 2.0
"""

__version__ = "1.0.0"
__author__ = "UCUP Framework Contributors"

# Core imports
from .core.config import UCUPConfig
from .core.manager import UCUPManager
from .core.bridge import ObjectiveCBridge

# Probabilistic reasoning (Core ML accelerated)
from .probabilistic.engine import ProbabilisticEngine
from .probabilistic.agent import ProbabilisticAgent, ProbabilisticResult

# Coordination (GCD-based)
from .coordination.hierarchical import HierarchicalCoordinator
from .coordination.swarm import SwarmCoordinator
from .coordination.adaptive import AdaptiveCoordinator

# Multimodal processing (Vision + AVFoundation)
from .multimodal.fusion_engine import MultimodalFusionEngine

# Testing and validation
from .testing.agent_tester import AgentTester
from .testing.performance_validator import PerformanceValidator
from .testing.scenario_generator import ScenarioGenerator

# Observability
from .observability.monitor import SystemMonitor
from .observability.metrics import MetricsCollector
from .observability.visualizer import DecisionVisualizer

# Initialize the framework
def initialize_ucup(config_path: str = None) -> UCUPManager:
    """
    Initialize the UCUP framework with optional configuration.

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        UCUPManager: The main framework manager instance
    """
    config = UCUPConfig.load(config_path) if config_path else UCUPConfig.default()
    return UCUPManager(config)

# Quick setup for common use cases
def create_probabilistic_agent(model_path: str = None) -> ProbabilisticAgent:
    """Create a probabilistic agent with optional Core ML model."""
    manager = initialize_ucup()
    return ProbabilisticAgent(manager, model_path)

def create_multimodal_processor() -> MultimodalFusionEngine:
    """Create a multimodal processing engine."""
    manager = initialize_ucup()
    return MultimodalFusionEngine(manager)

def create_coordinator(strategy: str = "hierarchical") -> object:
    """Create a coordination system."""
    manager = initialize_ucup()
    strategies = {
        "hierarchical": lambda: HierarchicalCoordinator(manager),
        "swarm": lambda: SwarmCoordinator(manager),
        "adaptive": lambda: AdaptiveCoordinator(manager)
    }
    return strategies.get(strategy, strategies["hierarchical"])()

# Export main classes
__all__ = [
    # Core
    "UCUPConfig",
    "UCUPManager",
    "ObjectiveCBridge",
    "initialize_ucup",

    # Probabilistic
    "ProbabilisticEngine",
    "ProbabilisticAgent",
    "ProbabilisticResult",
    "create_probabilistic_agent",

    # Coordination
    "HierarchicalCoordinator",
    "SwarmCoordinator",
    "AdaptiveCoordinator",
    "create_coordinator",

    # Multimodal
    "MultimodalFusionEngine",
    "create_multimodal_processor",

    # Testing
    "AgentTester",
    "PerformanceValidator",
    "ScenarioGenerator",

    # Observability
    "SystemMonitor",
    "MetricsCollector",
    "DecisionVisualizer",
]
