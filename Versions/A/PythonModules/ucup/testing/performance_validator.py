"""
Performance Validation Module for UCUP Framework.

Provides performance benchmarking and validation capabilities for framework components.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from statistics import mean, median, stdev
from ..core.manager import UCUPManager


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    execution_time: float
    cpu_usage: float
    memory_usage: float
    throughput: Optional[float] = None
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    error_rate: float = 0.0
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "operation_name": self.operation_name,
            "execution_time": self.execution_time,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "throughput": self.throughput,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "error_rate": self.error_rate,
            "timestamp": self.timestamp
        }


class BenchmarkResult:
    """Result of a benchmark test."""

    def __init__(self, benchmark_name: str, metrics: List[PerformanceMetrics],
                 summary: Dict[str, Any]):
        self.benchmark_name = benchmark_name
        self.metrics = metrics
        self.summary = summary
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": self.summary,
            "timestamp": self.timestamp
        }


class PerformanceValidator:
    """Performance validation and benchmarking framework."""

    def __init__(self, manager: UCUPManager):
        self.manager = manager
        self.config = manager.config
        self._benchmark_results = []
        self._metrics_lock = threading.Lock()

    def benchmark_operation(self, operation: Callable, operation_name: str,
                          iterations: int = 100, warmup_iterations: int = 10,
                          concurrent: bool = False, concurrency_level: int = 4) -> BenchmarkResult:
        """
        Benchmark a specific operation.

        Args:
            operation: The operation to benchmark (callable)
            operation_name: Name of the operation
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            concurrent: Whether to run concurrently
            concurrency_level: Number of concurrent threads

        Returns:
            BenchmarkResult with detailed metrics
        """
        print(f"Starting benchmark: {operation_name}")

        # Warmup phase
        self._run_warmup(operation, warmup_iterations)

        # Benchmark phase
        if concurrent:
            return self._run_concurrent_benchmark(operation, operation_name,
                                                 iterations, concurrency_level)
        else:
            return self._run_sequential_benchmark(operation, operation_name, iterations)

    def validate_performance_thresholds(self, operation: Callable,
                                      operation_name: str,
                                      thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate that operation meets performance thresholds.

        Args:
            operation: The operation to validate
            operation_name: Name of the operation
            thresholds: Dictionary of threshold requirements

        Returns:
            Validation results
        """
        # Run benchmark
        result = self.benchmark_operation(operation, operation_name,
                                        iterations=50, warmup_iterations=5)

        # Check thresholds
        validation_results = {}
        summary = result.summary

        for metric_name, threshold in thresholds.items():
            if metric_name in summary:
                actual_value = summary[metric_name]
                passed = self._check_threshold(metric_name, actual_value, threshold)
                validation_results[metric_name] = {
                    "required": threshold,
                    "actual": actual_value,
                    "passed": passed
                }

        overall_passed = all(r["passed"] for r in validation_results.values())

        return {
            "operation": operation_name,
            "overall_passed": overall_passed,
            "validations": validation_results,
            "benchmark_summary": summary
        }

    def compare_implementations(self, implementations: Dict[str, Callable],
                              test_data: Any, iterations: int = 50) -> Dict[str, Any]:
        """
        Compare performance of different implementations.

        Args:
            implementations: Dict of implementation name -> callable
            test_data: Data to pass to each implementation
            iterations: Number of iterations per implementation

        Returns:
            Comparison results
        """
        results = {}

        for name, implementation in implementations.items():
            def wrapped_op():
                return implementation(test_data)

            result = self.benchmark_operation(wrapped_op, name, iterations=iterations)
            results[name] = result.to_dict()

        # Generate comparison summary
        comparison = self._generate_comparison_summary(results)

        return {
            "implementations": list(implementations.keys()),
            "results": results,
            "comparison": comparison
        }

    def _run_warmup(self, operation: Callable, iterations: int):
        """Run warmup iterations."""
        for _ in range(iterations):
            try:
                operation()
            except Exception:
                pass  # Ignore warmup errors

    def _run_sequential_benchmark(self, operation: Callable, operation_name: str,
                                iterations: int) -> BenchmarkResult:
        """Run sequential benchmark."""
        execution_times = []
        cpu_usages = []
        memory_usages = []
        errors = 0

        process = psutil.Process()

        for i in range(iterations):
            start_time = time.time()
            start_cpu = process.cpu_percent()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            try:
                operation()
            except Exception:
                errors += 1

            end_time = time.time()
            end_cpu = process.cpu_percent()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB

            execution_time = end_time - start_time
            cpu_usage = max(0, end_cpu - start_cpu)  # Avoid negative values
            memory_usage = end_memory

            execution_times.append(execution_time)
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)

        # Calculate statistics
        avg_execution_time = mean(execution_times)
        avg_cpu = mean(cpu_usages)
        avg_memory = mean(memory_usages)

        # Calculate percentiles
        sorted_times = sorted(execution_times)
        p50 = median(sorted_times)
        p95 = sorted_times[int(0.95 * len(sorted_times))]
        p99 = sorted_times[int(0.99 * len(sorted_times))]

        throughput = iterations / sum(execution_times) if execution_times else 0
        error_rate = errors / iterations

        # Create metrics
        metrics = [
            PerformanceMetrics(
                operation_name=f"{operation_name}_iter_{i}",
                execution_time=execution_times[i],
                cpu_usage=cpu_usages[i],
                memory_usage=memory_usages[i]
            ) for i in range(iterations)
        ]

        summary = {
            "iterations": iterations,
            "average_execution_time": avg_execution_time,
            "average_cpu_usage": avg_cpu,
            "average_memory_usage": avg_memory,
            "throughput": throughput,
            "latency_p50": p50,
            "latency_p95": p95,
            "latency_p99": p99,
            "error_rate": error_rate,
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "execution_time_stdev": stdev(execution_times) if len(execution_times) > 1 else 0
        }

        result = BenchmarkResult(operation_name, metrics, summary)

        with self._metrics_lock:
            self._benchmark_results.append(result)

        return result

    def _run_concurrent_benchmark(self, operation: Callable, operation_name: str,
                                iterations: int, concurrency_level: int) -> BenchmarkResult:
        """Run concurrent benchmark."""
        results = []
        errors = []

        def worker(worker_id: int):
            worker_results = self._run_sequential_benchmark(
                operation, f"{operation_name}_worker_{worker_id}",
                iterations // concurrency_level
            )
            results.append(worker_results)

        # Start worker threads
        threads = []
        for i in range(concurrency_level):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Aggregate results
        all_metrics = []
        total_iterations = 0
        total_time = 0
        total_errors = 0

        for result in results:
            all_metrics.extend(result.metrics)
            total_iterations += result.summary["iterations"]
            total_time += result.summary["average_execution_time"] * result.summary["iterations"]
            total_errors += result.summary["error_rate"] * result.summary["iterations"]

        # Calculate aggregate statistics
        avg_execution_time = total_time / total_iterations if total_iterations > 0 else 0
        error_rate = total_errors / total_iterations if total_iterations > 0 else 0

        all_execution_times = [m.execution_time for m in all_metrics]
        if all_execution_times:
            sorted_times = sorted(all_execution_times)
            p50 = median(sorted_times)
            p95 = sorted_times[int(0.95 * len(sorted_times))]
            p99 = sorted_times[int(0.99 * len(sorted_times))]
            throughput = len(all_execution_times) / sum(all_execution_times)
        else:
            p50 = p95 = p99 = throughput = 0

        summary = {
            "iterations": total_iterations,
            "concurrency_level": concurrency_level,
            "average_execution_time": avg_execution_time,
            "throughput": throughput,
            "latency_p50": p50,
            "latency_p95": p95,
            "latency_p99": p99,
            "error_rate": error_rate
        }

        result = BenchmarkResult(operation_name, all_metrics, summary)

        with self._metrics_lock:
            self._benchmark_results.append(result)

        return result

    def _check_threshold(self, metric_name: str, actual: float, threshold: float) -> bool:
        """Check if metric meets threshold requirement."""
        # For most metrics, lower is better (time, cpu, memory)
        # For throughput, higher is better
        if metric_name in ["throughput"]:
            return actual >= threshold
        else:
            return actual <= threshold

    def _generate_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison summary between implementations."""
        if not results:
            return {}

        # Find best and worst performers
        metrics_to_compare = ["average_execution_time", "throughput", "average_cpu_usage"]

        comparison = {}
        for metric in metrics_to_compare:
            values = {}
            for name, result in results.items():
                summary = result.get("summary", {})
                if metric in summary:
                    values[name] = summary[metric]

            if values:
                if metric in ["throughput"]:
                    best = max(values.items(), key=lambda x: x[1])
                    worst = min(values.items(), key=lambda x: x[1])
                else:
                    best = min(values.items(), key=lambda x: x[1])
                    worst = max(values.items(), key=lambda x: x[1])

                comparison[metric] = {
                    "best": {"implementation": best[0], "value": best[1]},
                    "worst": {"implementation": worst[0], "value": worst[1]},
                    "values": values
                }

        return comparison

    def get_benchmark_history(self) -> List[Dict[str, Any]]:
        """Get history of all benchmark results."""
        with self._metrics_lock:
            return [r.to_dict() for r in self._benchmark_results]

    def clear_history(self):
        """Clear benchmark history."""
        with self._metrics_lock:
            self._benchmark_results.clear()

    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        import json

        history = self.get_benchmark_history()
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
