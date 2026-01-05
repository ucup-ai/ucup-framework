"""
Example Monitoring Plugin for UCUP Framework

This plugin demonstrates how to create custom monitoring and observability tools.
"""

import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, List

from ..plugins import MonitorPlugin, PluginMetadata


class MetricsMonitorPlugin(MonitorPlugin):
    """
    Advanced metrics monitoring plugin with Prometheus integration.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="metrics_monitor",
            version="1.0.0",
            description="Advanced metrics collection and monitoring",
            author="DevOps Team",
            dependencies=["prometheus_client"],
        )

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the monitoring plugin."""
        self.config = config or {}
        self.metrics_store = defaultdict(list)
        self.alerts = []
        self.collection_interval = self.config.get("interval", 60)
        self.retention_period = self.config.get("retention_hours", 24) * 3600

        # Initialize Prometheus metrics if available
        try:
            from prometheus_client import Counter, Gauge, Histogram

            self.request_count = Counter(
                "ucup_requests_total", "Total requests", ["agent_id", "method"]
            )
            self.response_time = Histogram(
                "ucup_response_time_seconds", "Response time", ["agent_id"]
            )
            self.error_count = Counter(
                "ucup_errors_total", "Total errors", ["agent_id", "error_type"]
            )
            self.confidence_gauge = Gauge(
                "ucup_confidence", "Current confidence level", ["agent_id"]
            )
            self.prometheus_enabled = True
        except ImportError:
            self.prometheus_enabled = False

        return True

    def shutdown(self) -> bool:
        """Clean up monitoring resources."""
        self.metrics_store.clear()
        self.alerts.clear()
        return True

    def get_monitor_name(self) -> str:
        """Return monitor name."""
        return "metrics_monitor"

    def start_monitoring(self, agent_id: str, config: Dict[str, Any]) -> bool:
        """Start monitoring an agent."""
        # Initialize agent-specific metrics
        if agent_id not in self.metrics_store:
            self.metrics_store[agent_id] = {
                "start_time": time.time(),
                "requests": 0,
                "errors": 0,
                "response_times": [],
                "confidence_scores": [],
                "last_activity": time.time(),
            }
        return True

    def stop_monitoring(self, agent_id: str) -> bool:
        """Stop monitoring an agent."""
        if agent_id in self.metrics_store:
            # Generate final metrics report
            self._generate_final_report(agent_id)
            del self.metrics_store[agent_id]
        return True

    def collect_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Collect current metrics for an agent."""
        if agent_id not in self.metrics_store:
            return {}

        metrics = self.metrics_store[agent_id]
        current_time = time.time()

        # Calculate derived metrics
        uptime = current_time - metrics["start_time"]
        avg_response_time = (
            sum(metrics["response_times"]) / len(metrics["response_times"])
            if metrics["response_times"]
            else 0
        )
        avg_confidence = (
            sum(metrics["confidence_scores"]) / len(metrics["confidence_scores"])
            if metrics["confidence_scores"]
            else 0
        )

        # Calculate rates (per minute)
        request_rate = metrics["requests"] / max(uptime / 60, 1)
        error_rate = metrics["errors"] / max(metrics["requests"], 1)

        result = {
            "agent_id": agent_id,
            "uptime_seconds": uptime,
            "total_requests": metrics["requests"],
            "total_errors": metrics["errors"],
            "request_rate_per_minute": request_rate,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "avg_confidence": avg_confidence,
            "last_activity": metrics["last_activity"],
        }

        # Update Prometheus metrics if available
        if self.prometheus_enabled:
            self.confidence_gauge.labels(agent_id=agent_id).set(avg_confidence)

        return result

    def record_request(
        self,
        agent_id: str,
        method: str = "execute",
        response_time: float = None,
        confidence: float = None,
        error: bool = False,
    ):
        """Record a request for monitoring."""
        if agent_id not in self.metrics_store:
            return

        metrics = self.metrics_store[agent_id]
        metrics["requests"] += 1
        metrics["last_activity"] = time.time()

        if error:
            metrics["errors"] += 1

        if response_time is not None:
            metrics["response_times"].append(response_time)
            # Keep only recent response times
            if len(metrics["response_times"]) > 1000:
                metrics["response_times"] = metrics["response_times"][-1000:]

        if confidence is not None:
            metrics["confidence_scores"].append(confidence)
            # Keep only recent confidence scores
            if len(metrics["confidence_scores"]) > 1000:
                metrics["confidence_scores"] = metrics["confidence_scores"][-1000:]

        # Update Prometheus metrics
        if self.prometheus_enabled:
            self.request_count.labels(agent_id=agent_id, method=method).inc()
            if response_time is not None:
                self.response_time.labels(agent_id=agent_id).observe(response_time)
            if error:
                self.error_count.labels(
                    agent_id=agent_id, error_type="execution_error"
                ).inc()

    def check_alerts(self, agent_id: str) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        metrics = self.collect_metrics(agent_id)

        if not metrics:
            return alerts

        # High error rate alert
        if metrics["error_rate"] > 0.1:  # > 10% error rate
            alerts.append(
                {
                    "type": "error_rate",
                    "severity": "warning",
                    "message": f"High error rate: {metrics['error_rate']:.1%}",
                    "agent_id": agent_id,
                    "value": metrics["error_rate"],
                }
            )

        # Low confidence alert
        if metrics["avg_confidence"] < 0.5:
            alerts.append(
                {
                    "type": "low_confidence",
                    "severity": "info",
                    "message": f"Low average confidence: {metrics['avg_confidence']:.2f}",
                    "agent_id": agent_id,
                    "value": metrics["avg_confidence"],
                }
            )

        # No activity alert
        inactivity_threshold = self.config.get("inactivity_alert_seconds", 300)
        if time.time() - metrics["last_activity"] > inactivity_threshold:
            alerts.append(
                {
                    "type": "inactivity",
                    "severity": "warning",
                    "message": f"No activity for {inactivity_threshold} seconds",
                    "agent_id": agent_id,
                    "value": time.time() - metrics["last_activity"],
                }
            )

        return alerts

    def get_performance_report(
        self, agent_id: str, time_range: str = "1h"
    ) -> Dict[str, Any]:
        """Generate performance report for an agent."""
        # This would analyze metrics over time
        # For now, return current metrics
        return self.collect_metrics(agent_id)

    def _generate_final_report(self, agent_id: str):
        """Generate final report when monitoring stops."""
        metrics = self.collect_metrics(agent_id)
        # In practice, this would save to database or log
        print(f"Final metrics report for {agent_id}: {metrics}")


class TracingMonitorPlugin(MonitorPlugin):
    """
    Advanced tracing and debugging monitor plugin.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="tracing_monitor",
            version="1.0.0",
            description="Advanced tracing and debugging monitoring",
            author="Debugging Team",
        )

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize tracing monitor."""
        self.config = config or {}
        self.trace_buffer = defaultdict(list)
        self.max_traces_per_agent = self.config.get("max_traces", 100)
        self.trace_retention = self.config.get("retention_hours", 1) * 3600
        return True

    def shutdown(self) -> bool:
        """Clean up tracing resources."""
        self.trace_buffer.clear()
        return True

    def get_monitor_name(self) -> str:
        """Return monitor name."""
        return "tracing_monitor"

    def start_monitoring(self, agent_id: str, config: Dict[str, Any]) -> bool:
        """Start tracing for an agent."""
        return True

    def stop_monitoring(self, agent_id: str) -> bool:
        """Stop tracing for an agent."""
        if agent_id in self.trace_buffer:
            del self.trace_buffer[agent_id]
        return True

    def collect_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Collect tracing metrics."""
        traces = self.trace_buffer.get(agent_id, [])
        if not traces:
            return {"trace_count": 0}

        # Analyze traces
        total_traces = len(traces)
        avg_execution_time = (
            sum(t.get("execution_time", 0) for t in traces) / total_traces
        )
        error_traces = sum(1 for t in traces if t.get("error"))
        success_rate = (total_traces - error_traces) / total_traces

        return {
            "trace_count": total_traces,
            "avg_execution_time": avg_execution_time,
            "error_rate": error_traces / total_traces,
            "success_rate": success_rate,
        }

    def record_trace(self, agent_id: str, trace_data: Dict[str, Any]):
        """Record a trace for an agent."""
        if agent_id not in self.trace_buffer:
            self.trace_buffer[agent_id] = []

        traces = self.trace_buffer[agent_id]
        traces.append({**trace_data, "timestamp": time.time()})

        # Maintain buffer size
        if len(traces) > self.max_traces_per_agent:
            traces.pop(0)

        # Clean old traces
        current_time = time.time()
        self.trace_buffer[agent_id] = [
            t for t in traces if current_time - t["timestamp"] < self.trace_retention
        ]

    def get_recent_traces(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent traces for debugging."""
        traces = self.trace_buffer.get(agent_id, [])
        return traces[-limit:] if traces else []

    def analyze_performance_patterns(self, agent_id: str) -> Dict[str, Any]:
        """Analyze performance patterns from traces."""
        traces = self.trace_buffer.get(agent_id, [])
        if not traces:
            return {"analysis": "No traces available"}

        # Simple pattern analysis
        execution_times = [t.get("execution_time", 0) for t in traces]
        confidence_scores = [t.get("confidence", 0) for t in traces]

        return {
            "total_traces": len(traces),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "avg_confidence": sum(confidence_scores) / len(confidence_scores),
            "performance_trend": "stable",  # Would analyze trends
            "error_patterns": [],  # Would identify error patterns
        }
