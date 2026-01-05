"""
UCUP Manager - Main framework coordinator.
"""

import threading
import concurrent.futures
from typing import Dict, Any, Optional, Callable
from .config import UCUPConfig
from .bridge import ObjectiveCBridge


class UCUPManager:
    """Main manager for UCUP framework operations."""

    def __init__(self, config: UCUPConfig):
        self.config = config
        self.bridge = ObjectiveCBridge()
        self._thread_pool = None
        self._lock = threading.RLock()
        self._components = {}

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all framework components."""
        # Initialize thread pool for concurrent operations
        max_workers = self.config.get("gcd.max_concurrent_tasks", 4)
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Initialize Objective-C bridge
        if not self.bridge.initialize():
            raise RuntimeError("Failed to initialize Objective-C bridge")

        # Register core components
        self._components.update({
            "probabilistic_engine": None,  # Will be initialized on demand
            "coordinator": None,
            "multimodal_processor": None,
        })

    def get_probabilistic_engine(self):
        """Get or create the probabilistic engine."""
        if self._components["probabilistic_engine"] is None:
            from ..probabilistic.engine import ProbabilisticEngine
            self._components["probabilistic_engine"] = ProbabilisticEngine(self)
        return self._components["probabilistic_engine"]

    def get_coordinator(self, strategy: str = "hierarchical"):
        """Get or create a coordinator with specified strategy."""
        key = f"coordinator_{strategy}"
        if key not in self._components:
            if strategy == "hierarchical":
                from ..coordination.hierarchical import HierarchicalCoordinator
                self._components[key] = HierarchicalCoordinator(self)
            elif strategy == "swarm":
                from ..coordination.swarm import SwarmCoordinator
                self._components[key] = SwarmCoordinator(self)
            elif strategy == "adaptive":
                from ..coordination.adaptive import AdaptiveCoordinator
                self._components[key] = AdaptiveCoordinator(self)
            else:
                raise ValueError(f"Unknown coordination strategy: {strategy}")
        return self._components[key]

    def get_multimodal_processor(self):
        """Get or create the multimodal processor."""
        if self._components["multimodal_processor"] is None:
            from ..multimodal.fusion_engine import MultimodalFusionEngine
            self._components["multimodal_processor"] = MultimodalFusionEngine(self)
        return self._components["multimodal_processor"]

    def execute_async(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """
        Execute a function asynchronously using the thread pool.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future object for the async operation
        """
        return self._thread_pool.submit(func, *args, **kwargs)

    def execute_probabilistic_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a probabilistic reasoning task.

        Args:
            parameters: Task parameters

        Returns:
            Task results
        """
        return self.bridge.execute_probabilistic_task(parameters)

    def analyze_multimodal_data(self, data: list) -> Dict[str, Any]:
        """
        Analyze multimodal data.

        Args:
            data: List of multimodal data items

        Returns:
            Analysis results
        """
        return self.bridge.analyze_multimodal_data(data)

    def coordinate_tasks(self, tasks: list, concurrency: int = None) -> list:
        """
        Coordinate multiple tasks.

        Args:
            tasks: List of task definitions
            concurrency: Maximum concurrent tasks (optional)

        Returns:
            List of task results
        """
        if concurrency is None:
            concurrency = self.config.get("gcd.max_concurrent_tasks", 4)

        return self.bridge.coordinate_tasks(tasks, concurrency)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        return {
            "active_threads": threading.active_count(),
            "thread_pool_size": self._thread_pool._max_workers,
            "config_summary": {
                "core_ml_enabled": self.config.get("core_ml.enabled", True),
                "gcd_enabled": self.config.get("gcd.enable_threading", True),
                "vision_enabled": self.config.get("vision.enable_high_accuracy", True),
            }
        }

    def shutdown(self):
        """Shutdown the framework and cleanup resources."""
        with self._lock:
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)

            # Cleanup components
            for component in self._components.values():
                if hasattr(component, 'shutdown'):
                    component.shutdown()

            self._components.clear()

            # Cleanup bridge
            self.bridge.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
