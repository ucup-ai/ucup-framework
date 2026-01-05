"""
Metrics Collector Module for UCUP Framework.

Provides comprehensive metrics collection, aggregation, and analysis capabilities.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from statistics import mean, median, stdev, quantiles
from ..core.manager import UCUPManager


@dataclass
class MetricValue:
    """Individual metric measurement."""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class AggregatedMetric:
    """Aggregated metric statistics."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    stddev: Optional[float]
    p50: float
    p95: float
    p99: float
    start_time: float
    end_time: float
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "stddev": self.stddev,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tags": self.tags
        }


class MetricsCollector:
    """Comprehensive metrics collection and analysis system."""

    def __init__(self, manager: UCUPManager):
        self.manager = manager
        self.config = manager.config
        self._collection_active = False
        self._collection_thread = None
        self._raw_metrics = defaultdict(lambda: deque(maxlen=10000))  # Per metric storage
        self._aggregated_metrics = {}
        self._collection_intervals = {}  # Custom collection intervals
        self._metric_callbacks = defaultdict(list)  # Callbacks for metric updates
        self._lock = threading.Lock()

        # Default collection intervals (seconds)
        self._default_intervals = {
            "system.cpu": 5.0,
            "system.memory": 5.0,
            "system.disk": 30.0,
            "system.network": 10.0,
            "application.response_time": 1.0,
            "application.throughput": 1.0,
            "application.error_rate": 1.0,
        }

    def start_collection(self):
        """Start metrics collection."""
        if self._collection_active:
            return

        self._collection_active = True
        self._collection_thread = threading.Thread(target=self._collection_loop)
        self._collection_thread.daemon = True
        self._collection_thread.start()

    def stop_collection(self):
        """Stop metrics collection."""
        self._collection_active = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)

    def record_metric(self, name: str, value: Union[int, float],
                     tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a metric value."""
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata or {}
        )

        with self._lock:
            self._raw_metrics[name].append(metric)

        # Trigger callbacks
        self._trigger_callbacks(name, metric)

    def record_counter(self, name: str, increment: Union[int, float] = 1,
                      tags: Dict[str, str] = None):
        """Record a counter metric (incrementing value)."""
        # For counters, we store the increment value
        self.record_metric(name, increment, tags, {"type": "counter"})

    def record_gauge(self, name: str, value: Union[int, float],
                    tags: Dict[str, str] = None):
        """Record a gauge metric (point-in-time value)."""
        self.record_metric(name, value, tags, {"type": "gauge"})

    def record_histogram(self, name: str, value: Union[int, float],
                        tags: Dict[str, str] = None):
        """Record a histogram metric (distribution of values)."""
        self.record_metric(name, value, tags, {"type": "histogram"})

    def record_timer(self, name: str, duration: float,
                    tags: Dict[str, str] = None):
        """Record a timer metric (duration)."""
        self.record_metric(name, duration, tags, {"type": "timer"})

    def get_raw_metrics(self, name: str, limit: int = 1000) -> List[MetricValue]:
        """Get raw metric values."""
        with self._lock:
            return list(self._raw_metrics[name])[-limit:]

    def get_aggregated_metrics(self, name: str, window_seconds: float = 300.0) -> Optional[AggregatedMetric]:
        """Get aggregated metrics for a time window."""
        now = time.time()
        start_time = now - window_seconds

        with self._lock:
            metrics = [m for m in self._raw_metrics[name]
                      if m.timestamp >= start_time]

        if not metrics:
            return None

        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]

        # Calculate statistics
        count = len(values)
        total = sum(values)
        min_val = min(values)
        max_val = max(values)
        mean_val = mean(values)
        median_val = median(values)
        stddev_val = stdev(values) if count > 1 else None

        # Calculate percentiles
        if count >= 4:
            percentiles = quantiles(values, n=100)
            p50 = percentiles[49]  # 50th percentile (0-indexed)
            p95 = percentiles[94]  # 95th percentile
            p99 = percentiles[98]  # 99th percentile
        else:
            p50 = p95 = p99 = median_val

        # Use common tags from metrics (if any)
        common_tags = {}
        if metrics:
            # Find tags that appear in all metrics
            all_tags = [set(m.tags.keys()) for m in metrics]
            common_tag_keys = set.intersection(*all_tags) if all_tags else set()

            for tag_key in common_tag_keys:
                # Check if all metrics have the same value for this tag
                tag_values = [m.tags[tag_key] for m in metrics]
                if len(set(tag_values)) == 1:
                    common_tags[tag_key] = tag_values[0]

        return AggregatedMetric(
            name=name,
            count=count,
            sum=total,
            min=min_val,
            max=max_val,
            mean=mean_val,
            median=median_val,
            stddev=stddev_val,
            p50=p50,
            p95=p95,
            p99=p99,
            start_time=min(timestamps),
            end_time=max(timestamps),
            tags=common_tags
        )

    def get_all_metric_names(self) -> List[str]:
        """Get list of all collected metric names."""
        with self._lock:
            return list(self._raw_metrics.keys())

    def get_metrics_summary(self, window_seconds: float = 300.0) -> Dict[str, Any]:
        """Get summary of all metrics."""
        metric_names = self.get_all_metric_names()
        summary = {
            "total_metrics": len(metric_names),
            "metrics": {},
            "collection_window": window_seconds
        }

        for name in metric_names:
            aggregated = self.get_aggregated_metrics(name, window_seconds)
            if aggregated:
                summary["metrics"][name] = aggregated.to_dict()

        return summary

    def add_metric_callback(self, metric_name: str, callback: Callable[[MetricValue], None]):
        """Add callback for metric updates."""
        self._metric_callbacks[metric_name].append(callback)

    def set_collection_interval(self, metric_pattern: str, interval: float):
        """Set custom collection interval for metrics matching pattern."""
        self._collection_intervals[metric_pattern] = interval

    def export_metrics(self, filepath: str, format: str = "json", window_seconds: float = 3600.0):
        """Export metrics to file."""
        summary = self.get_metrics_summary(window_seconds)

        if format == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        elif format == "csv":
            self._export_to_csv(filepath, summary)

    def _collection_loop(self):
        """Main metrics collection loop."""
        last_collection = defaultdict(float)

        while self._collection_active:
            try:
                current_time = time.time()

                # Collect system metrics
                self._collect_system_metrics()

                # Collect application metrics
                self._collect_application_metrics()

                # Check custom collection intervals
                for pattern, interval in self._collection_intervals.items():
                    if current_time - last_collection[pattern] >= interval:
                        self._collect_custom_metrics(pattern)
                        last_collection[pattern] = current_time

            except Exception as e:
                # Record collection errors as metrics
                self.record_metric("collection.errors", 1,
                                 tags={"error": str(e)},
                                 metadata={"type": "collection_error"})

            time.sleep(1.0)  # Base collection interval

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.record_gauge("system.cpu.percent", cpu_percent)

            cpu_times = psutil.cpu_times_percent()
            self.record_gauge("system.cpu.user", cpu_times.user)
            self.record_gauge("system.cpu.system", cpu_times.system)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge("system.memory.percent", memory.percent)
            self.record_gauge("system.memory.used_mb", memory.used / 1024 / 1024)
            self.record_gauge("system.memory.available_mb", memory.available / 1024 / 1024)

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_gauge("system.disk.percent", disk.percent)
            self.record_gauge("system.disk.used_gb", disk.used / 1024 / 1024 / 1024)
            self.record_gauge("system.disk.free_gb", disk.free / 1024 / 1024 / 1024)

            # Network metrics
            network = psutil.net_io_counters()
            self.record_counter("system.network.bytes_sent", network.bytes_sent)
            self.record_counter("system.network.bytes_recv", network.bytes_recv)

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.record_metric("system.collection.errors", 1,
                             tags={"component": "system"},
                             metadata={"error": str(e)})

    def _collect_application_metrics(self):
        """Collect application-level metrics."""
        try:
            # Framework-specific metrics
            if hasattr(self.manager, 'get_stats'):
                stats = self.manager.get_stats()
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.record_gauge(f"application.{key}", value)

            # Probabilistic engine metrics
            if hasattr(self.manager, 'get_probabilistic_engine'):
                engine = self.manager.get_probabilistic_engine()
                if hasattr(engine, 'get_stats'):
                    engine_stats = engine.get_stats()
                    for key, value in engine_stats.items():
                        if isinstance(value, (int, float)):
                            self.record_gauge(f"application.probabilistic.{key}", value)

        except Exception as e:
            self.record_metric("application.collection.errors", 1,
                             tags={"component": "application"},
                             metadata={"error": str(e)})

    def _collect_custom_metrics(self, pattern: str):
        """Collect custom metrics for a pattern."""
        # This would be extended to collect metrics matching specific patterns
        # For now, it's a placeholder
        pass

    def _trigger_callbacks(self, metric_name: str, metric: MetricValue):
        """Trigger callbacks for a metric update."""
        callbacks = self._metric_callbacks[metric_name]
        for callback in callbacks:
            try:
                callback(metric)
            except Exception:
                pass  # Don't let callback errors break collection

    def _export_to_csv(self, filepath: str, summary: Dict[str, Any]):
        """Export metrics summary to CSV."""
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(["Metric", "Count", "Sum", "Min", "Max", "Mean", "Median", "P50", "P95", "P99"])

            # Write data
            for name, data in summary.get("metrics", {}).items():
                row = [
                    name,
                    data.get("count", 0),
                    data.get("sum", 0),
                    data.get("min", 0),
                    data.get("max", 0),
                    data.get("mean", 0),
                    data.get("median", 0),
                    data.get("p50", 0),
                    data.get("p95", 0),
                    data.get("p99", 0)
                ]
                writer.writerow(row)

    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear stored metrics."""
        with self._lock:
            if metric_name:
                self._raw_metrics[metric_name].clear()
            else:
                self._raw_metrics.clear()

    def get_metric_stats(self, name: str) -> Dict[str, Any]:
        """Get basic statistics for a metric."""
        with self._lock:
            metrics = list(self._raw_metrics[name])

        if not metrics:
            return {"name": name, "count": 0}

        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]

        return {
            "name": name,
            "count": len(values),
            "latest_value": values[-1] if values else None,
            "latest_timestamp": timestamps[-1] if timestamps else None,
            "time_range": max(timestamps) - min(timestamps) if timestamps else 0,
            "value_range": max(values) - min(values) if values else 0
        }

    def __repr__(self) -> str:
        metric_count = len(self._raw_metrics)
        total_values = sum(len(deque) for deque in self._raw_metrics.values())
        return f"MetricsCollector(metrics={metric_count}, total_values={total_values})"
