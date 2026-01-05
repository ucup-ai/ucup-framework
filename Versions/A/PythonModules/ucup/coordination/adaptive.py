"""
Adaptive Coordinator for UCUP.

Implements adaptive coordination that learns and adjusts strategies based on performance.
"""

import time
import statistics
import threading
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from ..core.manager import UCUPManager
from .hierarchical import HierarchicalCoordinator
from .swarm import SwarmCoordinator


class PerformanceMetrics:
    """Tracks performance metrics for coordination strategies."""

    def __init__(self):
        self.execution_times: List[float] = []
        self.success_rates: List[float] = []
        self.resource_usage: List[float] = []
        self.task_completion_rates: List[float] = []

    def add_measurement(self, execution_time: float, success: bool,
                       resource_usage: float, tasks_completed: int, total_tasks: int):
        """Add a performance measurement."""
        self.execution_times.append(execution_time)
        self.success_rates.append(1.0 if success else 0.0)
        self.resource_usage.append(resource_usage)
        self.task_completion_rates.append(tasks_completed / total_tasks if total_tasks > 0 else 0)

    def get_average_performance(self) -> Dict[str, float]:
        """Get average performance metrics."""
        return {
            "avg_execution_time": statistics.mean(self.execution_times) if self.execution_times else 0,
            "avg_success_rate": statistics.mean(self.success_rates) if self.success_rates else 0,
            "avg_resource_usage": statistics.mean(self.resource_usage) if self.resource_usage else 0,
            "avg_completion_rate": statistics.mean(self.task_completion_rates) if self.task_completion_rates else 0,
            "measurements_count": len(self.execution_times)
        }


class AdaptiveCoordinator:
    """Adaptive coordinator that learns optimal strategies."""

    def __init__(self, manager: UCUPManager):
        self.manager = manager
        self.config = manager.config
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._running = False
        self._lock = threading.RLock()

        # Strategy performance tracking
        self.strategy_metrics = {
            "hierarchical": PerformanceMetrics(),
            "swarm": PerformanceMetrics()
        }

        # Current best strategy
        self.current_strategy = "hierarchical"
        self.learning_rate = 0.1
        self.min_measurements = 5  # Minimum measurements before adaptation

        # Initialize coordinators
        self.coordinators = {
            "hierarchical": HierarchicalCoordinator(manager),
            "swarm": SwarmCoordinator(manager, num_agents=8)
        }

    def coordinate_tasks(self, tasks: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Coordinate tasks using adaptive strategy selection.

        Args:
            tasks: List of task definitions
            **kwargs: Additional coordination parameters

        Returns:
            Coordination results with strategy information
        """
        with self._lock:
            self._running = True
            start_time = time.time()

            try:
                # Analyze task characteristics to select strategy
                selected_strategy = self._select_optimal_strategy(tasks)

                # Execute with selected strategy
                coordinator = self.coordinators[selected_strategy]
                result = coordinator.coordinate_tasks(tasks, **kwargs)

                # Track performance
                execution_time = time.time() - start_time
                self._record_performance(selected_strategy, result, execution_time)

                # Adapt strategy if enough data
                if self._should_adapt():
                    self._adapt_strategy()

                # Add strategy information to result
                result.update({
                    "selected_strategy": selected_strategy,
                    "strategy_adaptation": True,
                    "performance_tracked": True
                })

                return result

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
            finally:
                self._running = False

    def _select_optimal_strategy(self, tasks: List[Dict[str, Any]]) -> str:
        """Select the optimal coordination strategy based on task characteristics and performance history."""

        # Analyze task characteristics
        task_analysis = self._analyze_tasks(tasks)

        # If we don't have enough measurements, use analysis-based selection
        if not self._has_sufficient_data():
            return self._select_strategy_by_analysis(task_analysis)

        # Use performance history for selection
        return self._select_strategy_by_performance(task_analysis)

    def _analyze_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze task characteristics."""
        total_tasks = len(tasks)

        # Count task types
        task_types = {}
        dependencies = 0
        avg_complexity = 0

        for task in tasks:
            task_type = task.get("type", "generic")
            task_types[task_type] = task_types.get(task_type, 0) + 1

            if task.get("dependencies"):
                dependencies += len(task["dependencies"])

            # Estimate complexity
            complexity = len(task.get("parameters", {})) + len(task.get("data", []))
            avg_complexity += complexity

        avg_complexity /= total_tasks if total_tasks > 0 else 1

        return {
            "total_tasks": total_tasks,
            "task_types": task_types,
            "total_dependencies": dependencies,
            "avg_complexity": avg_complexity,
            "has_dependencies": dependencies > 0,
            "task_diversity": len(task_types)
        }

    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient performance data for adaptation."""
        total_measurements = sum(len(metrics.execution_times)
                                for metrics in self.strategy_metrics.values())
        return total_measurements >= self.min_measurements * len(self.strategy_metrics)

    def _select_strategy_by_analysis(self, task_analysis: Dict[str, Any]) -> str:
        """Select strategy based on task analysis when limited performance data."""

        # Hierarchical works well for dependent tasks
        if task_analysis["has_dependencies"] and task_analysis["total_dependencies"] > task_analysis["total_tasks"]:
            return "hierarchical"

        # Swarm works well for diverse, independent tasks
        if task_analysis["task_diversity"] > 2 and not task_analysis["has_dependencies"]:
            return "swarm"

        # Complex tasks benefit from hierarchical structure
        if task_analysis["avg_complexity"] > 5:
            return "hierarchical"

        # Default to hierarchical for stability
        return "hierarchical"

    def _select_strategy_by_performance(self, task_analysis: Dict[str, Any]) -> str:
        """Select strategy based on performance history."""

        # Get current performance metrics
        hierarchical_perf = self.strategy_metrics["hierarchical"].get_average_performance()
        swarm_perf = self.strategy_metrics["swarm"].get_average_performance()

        # Calculate composite scores (weighted combination of metrics)
        hierarchical_score = self._calculate_strategy_score(hierarchical_perf, task_analysis)
        swarm_score = self._calculate_strategy_score(swarm_perf, task_analysis)

        return "hierarchical" if hierarchical_score >= swarm_score else "swarm"

    def _calculate_strategy_score(self, performance: Dict[str, float],
                                task_analysis: Dict[str, Any]) -> float:
        """Calculate a composite score for strategy performance."""
        # Weights for different performance aspects
        weights = {
            "success_rate": 0.4,
            "completion_rate": 0.3,
            "execution_time": -0.2,  # Negative because lower is better
            "resource_usage": -0.1   # Negative because lower is better
        }

        score = 0.0

        # Normalize execution time (lower is better, so invert)
        if performance["avg_execution_time"] > 0:
            normalized_time = 1.0 / (1.0 + performance["avg_execution_time"])
            score += weights["execution_time"] * normalized_time
        else:
            score += weights["execution_time"] * 1.0

        # Normalize resource usage (lower is better, so invert)
        if performance["avg_resource_usage"] > 0:
            normalized_resource = 1.0 / (1.0 + performance["avg_resource_usage"])
            score += weights["resource_usage"] * normalized_resource
        else:
            score += weights["resource_usage"] * 1.0

        # Direct metrics (higher is better)
        score += weights["success_rate"] * performance["avg_success_rate"]
        score += weights["completion_rate"] * performance["avg_completion_rate"]

        return score

    def _record_performance(self, strategy: str, result: Dict[str, Any], execution_time: float):
        """Record performance metrics for a strategy."""
        if strategy not in self.strategy_metrics:
            return

        success = result.get("success", False)
        total_tasks = result.get("total_tasks", 0)

        # Estimate resource usage (simplified)
        resource_usage = execution_time * 0.1  # Rough estimate

        # Count completed tasks
        results = result.get("results", [])
        completed_tasks = sum(1 for r in results if r.get("status") == "completed")

        self.strategy_metrics[strategy].add_measurement(
            execution_time, success, resource_usage, completed_tasks, total_tasks
        )

    def _should_adapt(self) -> bool:
        """Determine if strategy adaptation should occur."""
        # Adapt every N measurements or when performance varies significantly
        total_measurements = sum(len(metrics.execution_times)
                                for metrics in self.strategy_metrics.values())

        return total_measurements % 10 == 0  # Adapt every 10 measurements

    def _adapt_strategy(self):
        """Adapt coordination parameters based on performance."""
        # Analyze recent performance trends
        recent_hierarchical = self._get_recent_performance("hierarchical", 5)
        recent_swarm = self._get_recent_performance("swarm", 5)

        if recent_hierarchical and recent_swarm:
            # Adjust learning parameters based on performance stability
            hierarchical_variance = statistics.variance(recent_hierarchical) if len(recent_hierarchical) > 1 else 0
            swarm_variance = statistics.variance(recent_swarm) if len(recent_swarm) > 1 else 0

            # If swarm is more stable, increase its preference
            if swarm_variance < hierarchical_variance:
                self.learning_rate *= 1.05  # Slight increase in adaptation rate
            else:
                self.learning_rate *= 0.95  # Slight decrease

            # Keep learning rate in reasonable bounds
            self.learning_rate = max(0.01, min(0.5, self.learning_rate))

    def _get_recent_performance(self, strategy: str, n_recent: int) -> List[float]:
        """Get recent performance measurements."""
        metrics = self.strategy_metrics[strategy]
        return metrics.success_rates[-n_recent:] if len(metrics.success_rates) >= n_recent else []

    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status and metrics."""
        strategy_performance = {}
        for strategy, metrics in self.strategy_metrics.items():
            strategy_performance[strategy] = metrics.get_average_performance()

        return {
            "current_strategy": self.current_strategy,
            "learning_rate": self.learning_rate,
            "strategy_performance": strategy_performance,
            "total_measurements": sum(len(m.execution_times) for m in self.strategy_metrics.values()),
            "adaptation_ready": self._has_sufficient_data()
        }

    def force_strategy(self, strategy: str):
        """Force the use of a specific strategy (for testing/debugging)."""
        if strategy in self.coordinators:
            self.current_strategy = strategy

    def reset_learning(self):
        """Reset learning state and performance metrics."""
        for metrics in self.strategy_metrics.values():
            metrics.execution_times.clear()
            metrics.success_rates.clear()
            metrics.resource_usage.clear()
            metrics.task_completion_rates.clear()

        self.learning_rate = 0.1
        self.current_strategy = "hierarchical"

    def shutdown(self):
        """Shutdown the adaptive coordinator."""
        with self._lock:
            self._running = False
            for coordinator in self.coordinators.values():
                if hasattr(coordinator, 'shutdown'):
                    coordinator.shutdown()

            if self._executor:
                self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        status = self.get_adaptation_status()
        return f"AdaptiveCoordinator(strategy={status['current_strategy']}, learning_rate={status['learning_rate']:.3f})"
