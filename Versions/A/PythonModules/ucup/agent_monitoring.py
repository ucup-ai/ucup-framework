"""
Agent Validation and Monitoring Framework for UCUP

This module provides comprehensive agent validation, health monitoring,
performance tracking, and alerting capabilities for UCUP framework agents.

Features:
- Agent health and performance monitoring
- Validation of agent behavior and outputs
- Automated alerting and reporting
- Integration with UCUP metrics and reliability systems
- CLI commands for monitoring and validation
- Real-time dashboards and analytics
"""

import asyncio
import json
import logging
import statistics
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AgentHealth(Enum):
    """Agent health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AgentMetrics:
    """Metrics collected for an agent."""

    agent_id: str
    timestamp: datetime
    request_count: int = 0
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0  # requests per second
    memory_usage: float = 0.0  # MB
    cpu_usage: float = 0.0  # percentage
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationIssue:
    """A validation issue found during agent monitoring."""

    agent_id: str
    issue_type: str
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class AgentHealthReport:
    """Comprehensive health report for an agent."""

    agent_id: str
    health_status: AgentHealth
    overall_score: float  # 0-100
    metrics: AgentMetrics
    issues: List[ValidationIssue]
    recommendations: List[str]
    last_updated: datetime
    uptime_seconds: float


class AgentValidator(ABC):
    """Abstract base class for agent validators."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def validate(
        self, agent_id: str, agent_data: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate an agent and return any issues found."""
        pass

    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Return the validation rules this validator implements."""
        pass


class PerformanceValidator(AgentValidator):
    """Validates agent performance metrics."""

    def __init__(self):
        super().__init__("performance")
        self.thresholds = {
            "max_response_time": 5.0,  # seconds
            "max_error_rate": 0.05,  # 5%
            "min_throughput": 1.0,  # requests/second
            "max_memory_usage": 1024,  # MB
            "max_cpu_usage": 80.0,  # percentage
        }

    async def validate(
        self, agent_id: str, agent_data: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate performance metrics."""
        issues = []
        metrics = agent_data.get("metrics", {})

        # Check response time
        response_time = metrics.get("response_time_avg", 0)
        if response_time > self.thresholds["max_response_time"]:
            issues.append(
                ValidationIssue(
                    agent_id=agent_id,
                    issue_type="performance_response_time",
                    severity=ValidationSeverity.WARNING
                    if response_time < 10
                    else ValidationSeverity.ERROR,
                    message=f"Response time too high: {response_time:.2f}s",
                    details={
                        "current": response_time,
                        "threshold": self.thresholds["max_response_time"],
                    },
                    timestamp=datetime.now(),
                )
            )

        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.thresholds["max_error_rate"]:
            issues.append(
                ValidationIssue(
                    agent_id=agent_id,
                    issue_type="performance_error_rate",
                    severity=ValidationSeverity.ERROR,
                    message=f"Error rate too high: {error_rate:.1%}",
                    details={
                        "current": error_rate,
                        "threshold": self.thresholds["max_error_rate"],
                    },
                    timestamp=datetime.now(),
                )
            )

        # Check throughput
        throughput = metrics.get("throughput", 0)
        if throughput < self.thresholds["min_throughput"]:
            issues.append(
                ValidationIssue(
                    agent_id=agent_id,
                    issue_type="performance_throughput",
                    severity=ValidationSeverity.WARNING,
                    message=f"Throughput too low: {throughput:.2f} req/s",
                    details={
                        "current": throughput,
                        "threshold": self.thresholds["min_throughput"],
                    },
                    timestamp=datetime.now(),
                )
            )

        # Check memory usage
        memory_usage = metrics.get("memory_usage", 0)
        if memory_usage > self.thresholds["max_memory_usage"]:
            issues.append(
                ValidationIssue(
                    agent_id=agent_id,
                    issue_type="performance_memory",
                    severity=ValidationSeverity.WARNING,
                    message=f"Memory usage high: {memory_usage:.0f}MB",
                    details={
                        "current": memory_usage,
                        "threshold": self.thresholds["max_memory_usage"],
                    },
                    timestamp=datetime.now(),
                )
            )

        # Check CPU usage
        cpu_usage = metrics.get("cpu_usage", 0)
        if cpu_usage > self.thresholds["max_cpu_usage"]:
            issues.append(
                ValidationIssue(
                    agent_id=agent_id,
                    issue_type="performance_cpu",
                    severity=ValidationSeverity.ERROR,
                    message=f"CPU usage too high: {cpu_usage:.1f}%",
                    details={
                        "current": cpu_usage,
                        "threshold": self.thresholds["max_cpu_usage"],
                    },
                    timestamp=datetime.now(),
                )
            )

        return issues

    def get_validation_rules(self) -> Dict[str, Any]:
        """Return performance validation rules."""
        return {
            "response_time_max": self.thresholds["max_response_time"],
            "error_rate_max": self.thresholds["max_error_rate"],
            "throughput_min": self.thresholds["min_throughput"],
            "memory_max_mb": self.thresholds["max_memory_usage"],
            "cpu_max_percent": self.thresholds["max_cpu_usage"],
        }


class BehaviorValidator(AgentValidator):
    """Validates agent behavior and outputs."""

    def __init__(self):
        super().__init__("behavior")

    async def validate(
        self, agent_id: str, agent_data: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate agent behavior."""
        issues = []
        behavior = agent_data.get("behavior", {})

        # Check for consistent responses
        responses = behavior.get("recent_responses", [])
        if len(responses) > 10:
            # Check for repetitive responses
            unique_responses = len(set(str(r) for r in responses))
            if unique_responses / len(responses) < 0.5:
                issues.append(
                    ValidationIssue(
                        agent_id=agent_id,
                        issue_type="behavior_repetitive",
                        severity=ValidationSeverity.WARNING,
                        message="Agent showing repetitive response patterns",
                        details={"unique_ratio": unique_responses / len(responses)},
                        timestamp=datetime.now(),
                    )
                )

        # Check for appropriate confidence levels
        confidence_scores = behavior.get("confidence_scores", [])
        if confidence_scores:
            avg_confidence = statistics.mean(confidence_scores)
            if avg_confidence < 0.3:
                issues.append(
                    ValidationIssue(
                        agent_id=agent_id,
                        issue_type="behavior_low_confidence",
                        severity=ValidationSeverity.WARNING,
                        message=f"Agent showing low confidence: {avg_confidence:.2f}",
                        details={"average_confidence": avg_confidence},
                        timestamp=datetime.now(),
                    )
                )

        # Check for appropriate uncertainty quantification
        uncertainty_scores = behavior.get("uncertainty_scores", [])
        if uncertainty_scores:
            high_uncertainty = sum(1 for u in uncertainty_scores if u > 0.8)
            if high_uncertainty / len(uncertainty_scores) > 0.5:
                issues.append(
                    ValidationIssue(
                        agent_id=agent_id,
                        issue_type="behavior_high_uncertainty",
                        severity=ValidationSeverity.WARNING,
                        message="Agent frequently showing high uncertainty",
                        details={
                            "high_uncertainty_ratio": high_uncertainty
                            / len(uncertainty_scores)
                        },
                        timestamp=datetime.now(),
                    )
                )

        return issues

    def get_validation_rules(self) -> Dict[str, Any]:
        """Return behavior validation rules."""
        return {
            "min_unique_response_ratio": 0.5,
            "min_avg_confidence": 0.3,
            "max_high_uncertainty_ratio": 0.5,
        }


class AgentMonitor:
    """Main agent monitoring and validation system."""

    def __init__(self, metrics_dir: Optional[Path] = None):
        self.metrics_dir = metrics_dir or Path.home() / ".ucup" / "monitoring"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.validators: List[AgentValidator] = [
            PerformanceValidator(),
            BehaviorValidator(),
        ]

        self.agent_metrics: Dict[str, List[AgentMetrics]] = {}
        self.agent_issues: Dict[str, List[ValidationIssue]] = {}
        self.alert_handlers: List[Callable] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None

        self.logger = logging.getLogger(__name__)

    def register_validator(self, validator: AgentValidator):
        """Register a new validator."""
        self.validators.append(validator)

    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)

    async def record_agent_metrics(self, agent_id: str, metrics_data: Dict[str, Any]):
        """Record metrics for an agent."""
        metrics = AgentMetrics(
            agent_id=agent_id, timestamp=datetime.now(), **metrics_data
        )

        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = []

        self.agent_metrics[agent_id].append(metrics)

        # Keep only last 1000 metrics per agent
        if len(self.agent_metrics[agent_id]) > 1000:
            self.agent_metrics[agent_id] = self.agent_metrics[agent_id][-1000:]

        # Trigger validation
        await self.validate_agent(agent_id, {"metrics": metrics_data})

    async def validate_agent(
        self, agent_id: str, agent_data: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate an agent and return issues."""
        all_issues = []

        for validator in self.validators:
            try:
                issues = await validator.validate(agent_id, agent_data)
                all_issues.extend(issues)
            except Exception as e:
                self.logger.error(
                    f"Validator {validator.name} failed for agent {agent_id}: {e}"
                )

        # Store issues
        if agent_id not in self.agent_issues:
            self.agent_issues[agent_id] = []

        self.agent_issues[agent_id].extend(all_issues)

        # Keep only last 100 issues per agent
        if len(self.agent_issues[agent_id]) > 100:
            self.agent_issues[agent_id] = self.agent_issues[agent_id][-100:]

        # Trigger alerts for critical issues
        critical_issues = [
            i for i in all_issues if i.severity == ValidationSeverity.CRITICAL
        ]
        if critical_issues:
            await self._trigger_alerts(agent_id, critical_issues)

        return all_issues

    async def _trigger_alerts(self, agent_id: str, issues: List[ValidationIssue]):
        """Trigger alert handlers for critical issues."""
        for handler in self.alert_handlers:
            try:
                await handler(agent_id, issues)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")

    def get_agent_health_report(self, agent_id: str) -> Optional[AgentHealthReport]:
        """Generate a comprehensive health report for an agent."""
        if agent_id not in self.agent_metrics or not self.agent_metrics[agent_id]:
            return None

        metrics = self.agent_metrics[agent_id][-1]  # Latest metrics
        issues = self.agent_issues.get(agent_id, [])

        # Calculate health score
        health_score = self._calculate_health_score(metrics, issues)

        # Determine health status
        if health_score >= 90:
            health_status = AgentHealth.HEALTHY
        elif health_score >= 70:
            health_status = AgentHealth.DEGRADED
        elif health_score >= 50:
            health_status = AgentHealth.UNHEALTHY
        else:
            health_status = AgentHealth.CRITICAL

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, issues)

        # Calculate uptime (simplified - based on monitoring duration)
        uptime = 0.0
        if len(self.agent_metrics[agent_id]) > 1:
            first_metric = self.agent_metrics[agent_id][0]
            last_metric = self.agent_metrics[agent_id][-1]
            uptime = (last_metric.timestamp - first_metric.timestamp).total_seconds()

        return AgentHealthReport(
            agent_id=agent_id,
            health_status=health_status,
            overall_score=health_score,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations,
            last_updated=datetime.now(),
            uptime_seconds=uptime,
        )

    def _calculate_health_score(
        self, metrics: AgentMetrics, issues: List[ValidationIssue]
    ) -> float:
        """Calculate a health score from 0-100."""
        score = 100.0

        # Performance penalties
        if metrics.response_time_avg > 2.0:
            score -= min(30, (metrics.response_time_avg - 2.0) * 10)
        if metrics.error_rate > 0.01:
            score -= min(40, metrics.error_rate * 1000)
        if metrics.memory_usage > 512:
            score -= min(20, (metrics.memory_usage - 512) / 10)
        if metrics.cpu_usage > 50:
            score -= min(20, metrics.cpu_usage - 50)

        # Issue penalties
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                score -= 20
            elif issue.severity == ValidationSeverity.ERROR:
                score -= 10
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 5

        return max(0.0, min(100.0, score))

    def _generate_recommendations(
        self, metrics: AgentMetrics, issues: List[ValidationIssue]
    ) -> List[str]:
        """Generate recommendations based on metrics and issues."""
        recommendations = []

        if metrics.response_time_avg > 2.0:
            recommendations.append(
                "Consider optimizing agent response time - current average is high"
            )

        if metrics.error_rate > 0.01:
            recommendations.append(
                "Investigate and reduce error rate - currently above 1%"
            )

        if metrics.memory_usage > 512:
            recommendations.append(
                "Monitor memory usage - consider memory optimization"
            )

        if metrics.cpu_usage > 50:
            recommendations.append("High CPU usage detected - review agent efficiency")

        for issue in issues:
            if "repetitive" in issue.issue_type:
                recommendations.append(
                    "Diversify agent responses to avoid repetitive patterns"
                )
            elif "low_confidence" in issue.issue_type:
                recommendations.append(
                    "Improve agent confidence through better training or calibration"
                )
            elif "high_uncertainty" in issue.issue_type:
                recommendations.append(
                    "Address high uncertainty - consider additional training data"
                )

        if not recommendations:
            recommendations.append("Agent health is good - continue monitoring")

        return recommendations

    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )

    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Periodic health checks for all monitored agents
                for agent_id in list(self.agent_metrics.keys()):
                    health_report = self.get_agent_health_report(agent_id)
                    if health_report and health_report.health_status in [
                        AgentHealth.UNHEALTHY,
                        AgentHealth.CRITICAL,
                    ]:
                        await self._trigger_alerts(agent_id, health_report.issues)

                # Save metrics periodically
                self._save_metrics()

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

            await asyncio.sleep(interval_seconds)

    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            metrics_file = self.metrics_dir / "agent_metrics.json"
            data = {
                "agent_metrics": {
                    agent_id: [m.__dict__ for m in metrics]
                    for agent_id, metrics in self.agent_metrics.items()
                },
                "agent_issues": {
                    agent_id: [i.__dict__ for i in issues]
                    for agent_id, issues in self.agent_issues.items()
                },
                "last_saved": datetime.now().isoformat(),
            }

            with open(metrics_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def load_metrics(self):
        """Load metrics from disk."""
        try:
            metrics_file = self.metrics_dir / "agent_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    data = json.load(f)

                # Restore metrics
                for agent_id, metrics_list in data.get("agent_metrics", {}).items():
                    self.agent_metrics[agent_id] = [
                        AgentMetrics(**m) for m in metrics_list
                    ]

                # Restore issues
                for agent_id, issues_list in data.get("agent_issues", {}).items():
                    self.agent_issues[agent_id] = [
                        ValidationIssue(**i) for i in issues_list
                    ]

        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")

    def get_all_agent_reports(self) -> Dict[str, AgentHealthReport]:
        """Get health reports for all monitored agents."""
        reports = {}
        for agent_id in self.agent_metrics.keys():
            report = self.get_agent_health_report(agent_id)
            if report:
                reports[agent_id] = report
        return reports


# Global monitor instance
_global_monitor: Optional[AgentMonitor] = None


def get_global_monitor() -> AgentMonitor:
    """Get or create the global agent monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = AgentMonitor()
        _global_monitor.load_metrics()  # Load persisted metrics
    return _global_monitor


async def record_agent_metrics(agent_id: str, metrics_data: Dict[str, Any]):
    """Convenience function to record agent metrics."""
    monitor = get_global_monitor()
    await monitor.record_agent_metrics(agent_id, metrics_data)


async def validate_agent(
    agent_id: str, agent_data: Dict[str, Any]
) -> List[ValidationIssue]:
    """Convenience function to validate an agent."""
    monitor = get_global_monitor()
    return await monitor.validate_agent(agent_id, agent_data)


def get_agent_health_report(agent_id: str) -> Optional[AgentHealthReport]:
    """Convenience function to get agent health report."""
    monitor = get_global_monitor()
    return monitor.get_agent_health_report(agent_id)


def get_all_agent_reports() -> Dict[str, AgentHealthReport]:
    """Convenience function to get all agent reports."""
    monitor = get_global_monitor()
    return monitor.get_all_agent_reports()


async def start_agent_monitoring(interval_seconds: int = 60):
    """Start global agent monitoring."""
    monitor = get_global_monitor()
    await monitor.start_monitoring(interval_seconds)


async def stop_agent_monitoring():
    """Stop global agent monitoring."""
    monitor = get_global_monitor()
    await monitor.stop_monitoring()
