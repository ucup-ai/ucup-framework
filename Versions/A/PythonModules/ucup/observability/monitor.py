"""
System Monitor Module for UCUP Framework.

Provides real-time system monitoring and health checking capabilities.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
from ..core.manager import UCUPManager


@dataclass
class SystemMetrics:
    """Container for system metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: tuple
    process_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "load_average": self.load_average,
            "process_count": self.process_count
        }


@dataclass
class HealthStatus:
    """System health status."""
    overall_status: str  # "healthy", "warning", "critical"
    component_statuses: Dict[str, str]
    issues: List[str]
    recommendations: List[str]
    timestamp: float

    def __post_init__(self):
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary."""
        return {
            "overall_status": self.overall_status,
            "component_statuses": self.component_statuses,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp
        }


class Alert:
    """System alert."""

    def __init__(self, alert_type: str, severity: str, message: str,
                 component: str, threshold: Any = None, actual_value: Any = None):
        self.alert_type = alert_type
        self.severity = severity  # "info", "warning", "error", "critical"
        self.message = message
        self.component = component
        self.threshold = threshold
        self.actual_value = actual_value
        self.timestamp = time.time()
        self.acknowledged = False
        self.resolved = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "component": self.component,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }


class SystemMonitor:
    """Real-time system monitoring and health checking."""

    def __init__(self, manager: UCUPManager):
        self.manager = manager
        self.config = manager.config
        self._monitoring_active = False
        self._monitor_thread = None
        self._metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self._alerts = deque(maxlen=500)  # Keep last 500 alerts
        self._health_checks = {}
        self._alert_callbacks = []
        self._lock = threading.Lock()

        # Default thresholds
        self._thresholds = {
            "cpu_percent": {"warning": 70.0, "critical": 90.0},
            "memory_percent": {"warning": 75.0, "critical": 90.0},
            "disk_usage_percent": {"warning": 80.0, "critical": 95.0},
        }

        # Initialize system info
        self._system_info = self._get_system_info()

    def start_monitoring(self, interval: float = 5.0):
        """Start continuous system monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self._collect_metrics()

    def get_metrics_history(self, limit: int = 100) -> List[SystemMetrics]:
        """Get recent metrics history."""
        with self._lock:
            return list(self._metrics_history)[-limit:]

    def get_health_status(self) -> HealthStatus:
        """Get current system health status."""
        metrics = self.get_current_metrics()
        return self._assess_health(metrics)

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        with self._lock:
            return [alert for alert in self._alerts if not alert.resolved]

    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            return list(self._alerts)[-limit:]

    def acknowledge_alert(self, alert_index: int):
        """Acknowledge an alert."""
        with self._lock:
            if 0 <= alert_index < len(self._alerts):
                self._alerts[alert_index].acknowledged = True

    def resolve_alert(self, alert_index: int):
        """Mark an alert as resolved."""
        with self._lock:
            if 0 <= alert_index < len(self._alerts):
                self._alerts[alert_index].resolved = True

    def set_threshold(self, metric: str, warning: float, critical: float):
        """Set monitoring thresholds for a metric."""
        self._thresholds[metric] = {"warning": warning, "critical": critical}

    def add_health_check(self, name: str, check_function: Callable[[], Dict[str, Any]]):
        """Add a custom health check."""
        self._health_checks[name] = check_function

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self._alert_callbacks.append(callback)

    def get_system_info(self) -> Dict[str, Any]:
        """Get static system information."""
        return self._system_info.copy()

    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = self._collect_metrics()

                with self._lock:
                    self._metrics_history.append(metrics)

                # Check for alerts
                self._check_thresholds(metrics)

                # Run health checks
                self._run_health_checks()

            except Exception as e:
                self._create_alert("error", "critical", f"Monitoring error: {str(e)}", "monitor")

            time.sleep(interval)

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        load_avg = psutil.getloadavg()

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            load_average=load_avg,
            process_count=len(psutil.pids())
        )

    def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against thresholds and create alerts."""
        metrics_dict = metrics.to_dict()

        for metric_name, thresholds in self._thresholds.items():
            if metric_name in metrics_dict:
                value = metrics_dict[metric_name]

                if value >= thresholds["critical"]:
                    self._create_alert("threshold", "critical",
                                     f"Critical {metric_name}: {value:.1f} >= {thresholds['critical']}",
                                     "system", thresholds["critical"], value)
                elif value >= thresholds["warning"]:
                    self._create_alert("threshold", "warning",
                                     f"Warning {metric_name}: {value:.1f} >= {thresholds['warning']}",
                                     "system", thresholds["warning"], value)

    def _run_health_checks(self):
        """Run custom health checks."""
        for name, check_func in self._health_checks.items():
            try:
                result = check_func()
                if not result.get("healthy", True):
                    severity = result.get("severity", "warning")
                    message = result.get("message", f"Health check {name} failed")
                    self._create_alert("health_check", severity, message, name)
            except Exception as e:
                self._create_alert("health_check", "error",
                                 f"Health check {name} error: {str(e)}", name)

    def _assess_health(self, metrics: SystemMetrics) -> HealthStatus:
        """Assess overall system health."""
        component_statuses = {}
        issues = []
        recommendations = []

        # CPU assessment
        if metrics.cpu_percent > 90:
            component_statuses["cpu"] = "critical"
            issues.append("CPU usage extremely high")
            recommendations.append("Consider scaling or optimizing CPU-intensive operations")
        elif metrics.cpu_percent > 70:
            component_statuses["cpu"] = "warning"
            issues.append("CPU usage high")
            recommendations.append("Monitor CPU usage trends")
        else:
            component_statuses["cpu"] = "healthy"

        # Memory assessment
        if metrics.memory_percent > 90:
            component_statuses["memory"] = "critical"
            issues.append("Memory usage extremely high")
            recommendations.append("Check for memory leaks or increase memory allocation")
        elif metrics.memory_percent > 75:
            component_statuses["memory"] = "warning"
            issues.append("Memory usage high")
            recommendations.append("Monitor memory usage and consider optimization")
        else:
            component_statuses["memory"] = "healthy"

        # Disk assessment
        if metrics.disk_usage_percent > 95:
            component_statuses["disk"] = "critical"
            issues.append("Disk usage extremely high")
            recommendations.append("Free up disk space or add storage")
        elif metrics.disk_usage_percent > 80:
            component_statuses["disk"] = "warning"
            issues.append("Disk usage high")
            recommendations.append("Monitor disk usage and plan for expansion")
        else:
            component_statuses["disk"] = "healthy"

        # Overall status
        if "critical" in component_statuses.values():
            overall_status = "critical"
        elif "warning" in component_statuses.values():
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return HealthStatus(
            overall_status=overall_status,
            component_statuses=component_statuses,
            issues=issues,
            recommendations=recommendations
        )

    def _create_alert(self, alert_type: str, severity: str, message: str,
                     component: str, threshold: Any = None, actual_value: Any = None):
        """Create and dispatch an alert."""
        alert = Alert(alert_type, severity, message, component, threshold, actual_value)

        with self._lock:
            self._alerts.append(alert)

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception:
                pass  # Don't let callback errors break monitoring

    def _get_system_info(self) -> Dict[str, Any]:
        """Get static system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "disk_total_gb": psutil.disk_usage('/').total / 1024 / 1024 / 1024,
            "platform": psutil.platform(),
            "python_version": psutil.python_version(),
            "boot_time": psutil.boot_time()
        }

    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics history to file."""
        import json

        metrics_data = [m.to_dict() for m in self.get_metrics_history()]

        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        elif format == "csv":
            import csv
            if metrics_data:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
                    writer.writeheader()
                    writer.writerows(metrics_data)

    def __repr__(self) -> str:
        health = self.get_health_status()
        return f"SystemMonitor(status={health.overall_status}, alerts={len(self.get_active_alerts())})"
