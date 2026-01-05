"""
Comprehensive error handling system for UCUP Framework.

This module provides a hierarchical error system with proper error classification,
recovery strategies, and logging integration for all UCUP components.
"""

import asyncio
import functools
import logging
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors that can occur in UCUP."""

    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    RUNTIME = "runtime"
    NETWORK = "network"
    RESOURCE = "resource"
    PLUGIN = "plugin"
    COORDINATION = "coordination"
    PROBABILISTIC = "probabilistic"
    DEPLOYMENT = "deployment"
    SECURITY = "security"


@dataclass
class ErrorContext:
    """Additional context information for errors."""

    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None


@dataclass
class RecoveryStrategy:
    """A strategy for recovering from an error."""

    name: str
    description: str
    action: Callable[[Any], Any]
    priority: int = 0  # Higher priority = preferred strategy
    requires_user_input: bool = False
    risk_level: str = "low"


@dataclass
class UCUPError:
    """
    Base error class for all UCUP-related errors.

    Provides structured error information with context, severity, and recovery options.
    """

    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    error_code: str
    context: ErrorContext
    timestamp: datetime = field(default_factory=datetime.now)
    original_exception: Optional[Exception] = None
    recovery_strategies: List[RecoveryStrategy] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.original_exception:
            self.metadata["original_exception_type"] = type(
                self.original_exception
            ).__name__
            self.metadata["original_exception_message"] = str(self.original_exception)

        # Add stack trace if available
        if not self.context.stack_trace:
            self.context.stack_trace = traceback.format_exc()

    def __str__(self) -> str:
        return f"[{self.category.value.upper()}] {self.error_code}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "error_code": self.error_code,
            "code": self.error_code,
            "context": {
                "component": self.context.component,
                "operation": self.context.operation,
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "agent_id": self.context.agent_id,
                "task_id": self.context.task_id,
                "metadata": self.context.metadata,
            },
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "recovery_strategies": [
                {
                    "name": strategy.name,
                    "description": strategy.description,
                    "priority": strategy.priority,
                    "requires_user_input": strategy.requires_user_input,
                    "risk_level": strategy.risk_level,
                }
                for strategy in self.recovery_strategies
            ],
        }

    def get_recovery_suggestions(self) -> List[RecoveryStrategy]:
        """Get prioritized recovery strategies."""
        return sorted(self.recovery_strategies, key=lambda x: x.priority, reverse=True)

    def can_auto_recover(self) -> bool:
        """Check if this error can be automatically recovered."""
        return any(
            not strategy.requires_user_input for strategy in self.recovery_strategies
        )


class ValidationError(UCUPError):
    """Error raised during data validation."""

    def __init__(
        self,
        message: str,
        field: str,
        value: Any = None,
        expected_type: Optional[type] = None,
        suggested_fix: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        **kwargs,
    ):
        if not context:
            context = ErrorContext(component="validator", operation="validate")

        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            error_code="VALIDATION_ERROR",
            context=context,
            **kwargs,
        )

        self.field = field
        self.value = value
        self.expected_type = expected_type
        self.suggested_fix = suggested_fix

        # Add validation-specific metadata
        self.metadata.update(
            {
                "field": field,
                "value_preview": str(value)[:100] if value is not None else None,
                "expected_type": expected_type.__name__ if expected_type else None,
                "suggested_fix": suggested_fix,
            }
        )

        # Add recovery strategies for validation errors
        if suggested_fix:
            self.recovery_strategies.append(
                RecoveryStrategy(
                    name="apply_suggested_fix",
                    description=f"Apply suggested fix: {suggested_fix}",
                    action=self._apply_fix,
                    priority=10,
                )
            )

        self.recovery_strategies.extend(
            [
                RecoveryStrategy(
                    name="skip_validation",
                    description="Skip validation and continue with warning",
                    action=self._skip_validation,
                    priority=2,
                    risk_level="medium",
                ),
                RecoveryStrategy(
                    name="use_default_value",
                    description="Use default value for the field",
                    action=self._use_default_value,
                    priority=1,
                ),
            ]
        )

    def _apply_fix(self, context: Dict[str, Any]) -> Any:
        """Apply the suggested fix if possible."""
        # Implementation would depend on the specific fix
        return context

    def _skip_validation(self, context: Dict[str, Any]) -> Any:
        """Skip validation and add warning."""
        logging.warning(f"Skipped validation for field '{self.field}': {self.message}")
        return context

    def _use_default_value(self, context: Dict[str, Any]) -> Any:
        """Use a default value for the invalid field."""
        # Implementation would provide field-specific defaults
        return context


class ConfigurationError(UCUPError):
    """Error in configuration loading or validation."""

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_section: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        **kwargs,
    ):
        if not context:
            context = ErrorContext(component="config", operation="load")

        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            error_code="CONFIG_ERROR",
            context=context,
            **kwargs,
        )

        self.config_file = config_file
        self.config_section = config_section

        self.metadata.update(
            {"config_file": config_file, "config_section": config_section}
        )

        self.recovery_strategies.extend(
            [
                RecoveryStrategy(
                    name="reload_config",
                    description="Reload the configuration file",
                    action=self._reload_config,
                    priority=8,
                ),
                RecoveryStrategy(
                    name="use_backup_config",
                    description="Fall back to backup configuration",
                    action=self._use_backup_config,
                    priority=7,
                ),
                RecoveryStrategy(
                    name="interactive_config",
                    description="Prompt user to provide configuration values",
                    action=self._interactive_config,
                    priority=5,
                    requires_user_input=True,
                ),
                RecoveryStrategy(
                    name="use_defaults",
                    description="Use default configuration values",
                    action=self._use_defaults,
                    priority=3,
                ),
            ]
        )

    def _reload_config(self, context: Dict[str, Any]) -> Any:
        """Reload configuration from file."""
        from .config import ConfigLoader

        loader = ConfigLoader()
        return loader.load_config(self.config_file)

    def _use_backup_config(self, context: Dict[str, Any]) -> Any:
        """Use backup configuration."""
        # Implementation would find and load backup config
        return context

    def _interactive_config(self, context: Dict[str, Any]) -> Any:
        """Prompt user for configuration values."""
        # Implementation would prompt user
        return context

    def _use_defaults(self, context: Dict[str, Any]) -> Any:
        """Use default configuration values."""
        # Implementation would provide default config
        return context


class RuntimeError(UCUPError):
    """Error during runtime execution."""

    def __init__(
        self,
        message: str,
        operation: str,
        component: str,
        can_retry: bool = True,
        context: Optional[ErrorContext] = None,
        **kwargs,
    ):
        if not context:
            context = ErrorContext(component=component, operation=operation)

        super().__init__(
            message=message,
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.HIGH,
            error_code="RUNTIME_ERROR",
            context=context,
            **kwargs,
        )

        self.can_retry = can_retry
        self.operation = operation
        self.component = component

        self.metadata.update(
            {"can_retry": can_retry, "operation": operation, "component": component}
        )

        self.recovery_strategies.extend(
            [
                RecoveryStrategy(
                    name="retry_operation",
                    description="Retry the failed operation",
                    action=self._retry_operation,
                    priority=9 if can_retry else 0,
                ),
                RecoveryStrategy(
                    name="switch_to_fallback",
                    description="Switch to fallback implementation",
                    action=self._switch_fallback,
                    priority=7,
                ),
                RecoveryStrategy(
                    name="reduce_load",
                    description="Reduce system load and retry",
                    action=self._reduce_load,
                    priority=6,
                ),
                RecoveryStrategy(
                    name="graceful_degradation",
                    description="Continue with reduced functionality",
                    action=self._graceful_degradation,
                    priority=4,
                ),
            ]
        )

    def _retry_operation(self, context: Dict[str, Any]) -> Any:
        """Retry the failed operation."""
        # Implementation would retry the operation
        return context

    def _switch_fallback(self, context: Dict[str, Any]) -> Any:
        """Switch to fallback implementation."""
        # Implementation would switch to backup system
        return context

    def _reduce_load(self, context: Dict[str, Any]) -> Any:
        """Reduce system load."""
        # Implementation would throttle or reduce concurrency
        return context

    def _graceful_degradation(self, context: Dict[str, Any]) -> Any:
        """Enable graceful degradation."""
        # Implementation would enable fallback modes
        return context


class NetworkError(UCUPError):
    """Error related to network operations."""

    def __init__(
        self,
        message: str,
        operation: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        context: Optional[ErrorContext] = None,
        **kwargs,
    ):
        if not context:
            context = ErrorContext(component="network", operation=operation)

        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            error_code="NETWORK_ERROR",
            context=context,
            **kwargs,
        )

        self.endpoint = endpoint
        self.status_code = status_code

        self.metadata.update({"endpoint": endpoint, "status_code": status_code})

        self.recovery_strategies.extend(
            [
                RecoveryStrategy(
                    name="retry_with_backoff",
                    description="Retry with exponential backoff",
                    action=self._retry_with_backoff,
                    priority=8,
                ),
                RecoveryStrategy(
                    name="switch_endpoint",
                    description="Switch to backup endpoint",
                    action=self._switch_endpoint,
                    priority=7,
                ),
                RecoveryStrategy(
                    name="offline_mode",
                    description="Switch to offline/cached mode",
                    action=self._offline_mode,
                    priority=5,
                ),
                RecoveryStrategy(
                    name="user_notification",
                    description="Notify user of network issues",
                    action=self._notify_user,
                    priority=3,
                    requires_user_input=True,
                ),
            ]
        )

    def _retry_with_backoff(self, context: Dict[str, Any]) -> Any:
        """Retry operation with backoff."""
        # Implementation would retry with backoff
        return context

    def _switch_endpoint(self, context: Dict[str, Any]) -> Any:
        """Switch to backup endpoint."""
        # Implementation would switch network endpoints
        return context

    def _offline_mode(self, context: Dict[str, Any]) -> Any:
        """Enable offline mode."""
        # Implementation would enable cached/offline functionality
        return context

    def _notify_user(self, context: Dict[str, Any]) -> Any:
        """Notify user of network issues."""
        # Implementation would show user notification
        return context


class PluginError(UCUPError):
    """Error related to plugin operations."""

    def __init__(
        self,
        message: str,
        plugin_name: str,
        plugin_version: Optional[str] = None,
        operation: str = "load",
        context: Optional[ErrorContext] = None,
        **kwargs,
    ):
        if not context:
            context = ErrorContext(component="plugin_system", operation=operation)

        super().__init__(
            message=message,
            category=ErrorCategory.PLUGIN,
            severity=ErrorSeverity.HIGH,
            error_code="PLUGIN_ERROR",
            context=context,
            **kwargs,
        )

        self.plugin_name = plugin_name
        self.plugin_version = plugin_version

        self.metadata.update(
            {
                "plugin_name": plugin_name,
                "plugin_version": plugin_version,
                "operation": operation,
            }
        )

        self.recovery_strategies.extend(
            [
                RecoveryStrategy(
                    name="reload_plugin",
                    description="Reload the plugin",
                    action=self._reload_plugin,
                    priority=8,
                ),
                RecoveryStrategy(
                    name="skip_plugin",
                    description="Skip loading this plugin and continue",
                    action=self._skip_plugin,
                    priority=7,
                ),
                RecoveryStrategy(
                    name="update_plugin",
                    description="Update plugin to latest version",
                    action=self._update_plugin,
                    priority=6,
                ),
                RecoveryStrategy(
                    name="disable_plugin",
                    description="Disable the plugin",
                    action=self._disable_plugin,
                    priority=4,
                ),
            ]
        )

    def _reload_plugin(self, context: Dict[str, Any]) -> Any:
        """Reload the plugin."""
        # Implementation would reload plugin
        return context

    def _skip_plugin(self, context: Dict[str, Any]) -> Any:
        """Skip loading the problematic plugin."""
        # Implementation would skip plugin loading
        return context

    def _update_plugin(self, context: Dict[str, Any]) -> Any:
        """Update plugin to latest version."""
        # Implementation would update plugin
        return context

    def _disable_plugin(self, context: Dict[str, Any]) -> Any:
        """Disable the plugin."""
        # Implementation would disable plugin
        return context


class CoordinationError(UCUPError):
    """Error in multi-agent coordination."""

    def __init__(
        self,
        message: str,
        coordination_type: str,
        affected_agents: List[str],
        context: Optional[ErrorContext] = None,
        **kwargs,
    ):
        if not context:
            context = ErrorContext(component="coordination", operation="coordinate")

        super().__init__(
            message=message,
            category=ErrorCategory.COORDINATION,
            severity=ErrorSeverity.HIGH,
            error_code="COORDINATION_ERROR",
            context=context,
            **kwargs,
        )

        self.coordination_type = coordination_type
        self.affected_agents = affected_agents

        self.metadata.update(
            {
                "coordination_type": coordination_type,
                "affected_agents": affected_agents,
                "agent_count": len(affected_agents),
            }
        )

        self.recovery_strategies.extend(
            [
                RecoveryStrategy(
                    name="rebalance_agents",
                    description="Rebalance tasks across remaining agents",
                    action=self._rebalance_agents,
                    priority=8,
                ),
                RecoveryStrategy(
                    name="switch_coordination",
                    description="Switch to different coordination strategy",
                    action=self._switch_coordination,
                    priority=7,
                ),
                RecoveryStrategy(
                    name="isolate_failed_agents",
                    description="Isolate and skip failed agents",
                    action=self._isolate_agents,
                    priority=6,
                ),
                RecoveryStrategy(
                    name="simplify_coordination",
                    description="Simplify coordination to basic mode",
                    action=self._simplify_coordination,
                    priority=5,
                ),
            ]
        )

    def _rebalance_agents(self, context: Dict[str, Any]) -> Any:
        """Rebalance tasks across agents."""
        # Implementation would redistribute tasks
        return context

    def _switch_coordination(self, context: Dict[str, Any]) -> Any:
        """Switch coordination strategy."""
        # Implementation would switch to alternative coordination
        return context

    def _isolate_agents(self, context: Dict[str, Any]) -> Any:
        """Isolate failed agents."""
        # Implementation would exclude failed agents
        return context

    def _simplify_coordination(self, context: Dict[str, Any]) -> Any:
        """Simplify coordination mode."""
        # Implementation would use basic coordination
        return context


class ProbabilisticError(UCUPError):
    """Error in probabilistic reasoning components."""

    def __init__(
        self,
        message: str,
        reasoning_component: str,
        confidence_level: Optional[float] = None,
        context: Optional[ErrorContext] = None,
        **kwargs,
    ):
        if not context:
            context = ErrorContext(component="probabilistic", operation="reason")

        super().__init__(
            message=message,
            category=ErrorCategory.PROBABILISTIC,
            severity=ErrorSeverity.MEDIUM,
            error_code="PROBABILISTIC_ERROR",
            context=context,
            **kwargs,
        )

        self.reasoning_component = reasoning_component
        self.confidence_level = confidence_level

        self.metadata.update(
            {
                "reasoning_component": reasoning_component,
                "confidence_level": confidence_level,
            }
        )

        self.recovery_strategies.extend(
            [
                RecoveryStrategy(
                    name="reduce_complexity",
                    description="Reduce reasoning complexity",
                    action=self._reduce_complexity,
                    priority=8,
                ),
                RecoveryStrategy(
                    name="switch_reasoning_strategy",
                    description="Switch to alternative reasoning strategy",
                    action=self._switch_reasoning,
                    priority=7,
                ),
                RecoveryStrategy(
                    name="increase_confidence_threshold",
                    description="Increase confidence threshold for decisions",
                    action=self._increase_threshold,
                    priority=6,
                ),
                RecoveryStrategy(
                    name="fallback_to_deterministic",
                    description="Fall back to deterministic reasoning",
                    action=self._fallback_deterministic,
                    priority=5,
                ),
            ]
        )

    def _reduce_complexity(self, context: Dict[str, Any]) -> Any:
        """Reduce reasoning complexity."""
        # Implementation would simplify reasoning
        return context

    def _switch_reasoning(self, context: Dict[str, Any]) -> Any:
        """Switch reasoning strategy."""
        # Implementation would switch to alternative strategy
        return context

    def _increase_threshold(self, context: Dict[str, Any]) -> Any:
        """Increase confidence threshold."""
        # Implementation would adjust thresholds
        return context

    def _fallback_deterministic(self, context: Dict[str, Any]) -> Any:
        """Fall back to deterministic reasoning."""
        # Implementation would use deterministic methods
        return context


class ErrorHandler:
    """
    Central error handling and recovery system.

    Provides error classification, logging, recovery strategies, and monitoring.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[UCUPError] = []
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}
        self.error_thresholds: Dict[ErrorCategory, int] = {}
        self.setup_default_handlers()

    def setup_default_handlers(self):
        """Set up default error recovery handlers."""
        self.recovery_handlers = {
            ErrorCategory.VALIDATION: self._handle_validation_error,
            ErrorCategory.CONFIGURATION: self._handle_configuration_error,
            ErrorCategory.RUNTIME: self._handle_runtime_error,
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.PLUGIN: self._handle_plugin_error,
            ErrorCategory.COORDINATION: self._handle_coordination_error,
            ErrorCategory.PROBABILISTIC: self._handle_probabilistic_error,
        }

        # Set error thresholds for alarming
        self.error_thresholds = {
            ErrorCategory.CRITICAL: 5,
            ErrorCategory.HIGH: 10,
            ErrorCategory.MEDIUM: 20,
            ErrorCategory.LOW: 50,
        }

    def handle_error(
        self, error: UCUPError, context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Handle an error with appropriate logging, recovery, and monitoring.
        """
        # Log the error
        self.log_error(error)

        # Add to error history
        self.error_history.append(error)

        # Check for error patterns and thresholds
        self._check_error_patterns(error)
        self._check_error_thresholds()

        # Attempt recovery
        recovery_result = self._attempt_recovery(error, context)

        # Monitor and alert if necessary
        self._monitor_error(error, recovery_result)

        return recovery_result

    def log_error(self, error: UCUPError):
        """Log error with appropriate level based on severity."""
        log_message = f"{error.error_code}: {error.message}"
        log_context = {
            "error_code": error.error_code,
            "category": error.category.value,
            "severity": error.severity.value,
            "component": error.context.component,
            "operation": error.context.operation,
            "session_id": error.context.session_id,
            "agent_id": error.context.agent_id,
        }

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra=log_context)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra=log_context)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra=log_context)
        else:
            self.logger.info(log_message, extra=log_context)

    def _attempt_recovery(
        self, error: UCUPError, context: Optional[Dict[str, Any]]
    ) -> Any:
        """Attempt to recover from the error."""
        handler = self.recovery_handlers.get(error.category)
        if handler:
            try:
                context = context or {}
                return handler(error, context)
            except Exception as e:
                self.logger.error(f"Error recovery failed: {e}")
                return None

        # Default recovery - re-raise the error
        raise error

    def _handle_validation_error(
        self, error: ValidationError, context: Dict[str, Any]
    ) -> Any:
        """Handle validation errors."""
        # Try suggested fix first
        if error.suggested_fix and error.can_auto_recover():
            strategies = error.get_recovery_suggestions()
            for strategy in strategies:
                if not strategy.requires_user_input:
                    try:
                        return strategy.action(context)
                    except Exception:
                        continue

        # Log validation issue
        self.logger.warning(
            f"Validation error in field '{error.field}': {error.message}"
        )
        return context

    def _handle_configuration_error(
        self, error: ConfigurationError, context: Dict[str, Any]
    ) -> Any:
        """Handle configuration errors."""
        # Try to reload config
        strategies = error.get_recovery_suggestions()
        for strategy in strategies:
            if strategy.name == "reload_config":
                try:
                    return strategy.action(context)
                except Exception:
                    continue

        return context

    def _handle_runtime_error(
        self, error: RuntimeError, context: Dict[str, Any]
    ) -> Any:
        """Handle runtime errors."""
        # Try retry if possible
        if error.can_retry:
            strategies = error.get_recovery_suggestions()
            for strategy in strategies:
                if strategy.name == "retry_operation":
                    try:
                        return strategy.action(context)
                    except Exception:
                        continue

        # Try fallback strategies
        for strategy in error.recovery_strategies:
            if strategy.name == "switch_to_fallback":
                try:
                    return strategy.action(context)
                except Exception:
                    continue

        return context

    def _handle_network_error(
        self, error: NetworkError, context: Dict[str, Any]
    ) -> Any:
        """Handle network errors."""
        # Try retry strategies
        for strategy in error.recovery_strategies:
            if strategy.name in ["retry_with_backoff", "switch_endpoint"]:
                try:
                    return strategy.action(context)
                except Exception:
                    continue

        return context

    def _handle_plugin_error(self, error: PluginError, context: Dict[str, Any]) -> Any:
        """Handle plugin errors."""
        # Try to skip problematic plugins
        for strategy in error.recovery_strategies:
            if strategy.name == "skip_plugin":
                try:
                    return strategy.action(context)
                except Exception:
                    continue

        return context

    def _handle_coordination_error(
        self, error: CoordinationError, context: Dict[str, Any]
    ) -> Any:
        """Handle coordination errors."""
        # Try to rebalance or switch strategy
        for strategy in error.recovery_strategies:
            if strategy.name in ["rebalance_agents", "switch_coordination"]:
                try:
                    return strategy.action(context)
                except Exception:
                    continue

        return context

    def _handle_probabilistic_error(
        self, error: ProbabilisticError, context: Dict[str, Any]
    ) -> Any:
        """Handle probabilistic reasoning errors."""
        # Try fallback strategies
        for strategy in error.recovery_strategies:
            if strategy.name == "fallback_to_deterministic":
                try:
                    return strategy.action(context)
                except Exception:
                    continue

        return context

    def _check_error_patterns(self, error: UCUPError):
        """Check for error patterns that might indicate systemic issues."""
        recent_errors = [
            e for e in self.error_history[-10:] if e.category == error.category
        ]

        if len(recent_errors) >= 3:
            self.logger.warning(
                f"Pattern detected: {len(recent_errors)} {error.category.value} errors in recent history"
            )

    def _check_error_thresholds(self):
        """Check if error thresholds have been exceeded."""
        for category in ErrorCategory:
            recent_errors = [
                e
                for e in self.error_history[-100:]
                if e.category == category
                and e.severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL)
            ]

            threshold = self.error_thresholds.get(category, 10)
            if len(recent_errors) > threshold:
                self.logger.error(
                    f"Error threshold exceeded for {category.value}: "
                    f"{len(recent_errors)} errors in recent history"
                )

    def _monitor_error(self, error: UCUPError, recovery_result: Any):
        """Monitor error for alerting and analytics."""
        # In a production system, this would send metrics to monitoring systems
        pass

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors for monitoring."""
        summary = {
            "total_errors": len(self.error_history),
            "errors_by_category": {},
            "errors_by_severity": {},
            "recent_errors": len([e for e in self.error_history[-100:]]),
        }

        for category in ErrorCategory:
            summary["errors_by_category"][category.value] = len(
                [e for e in self.error_history if e.category == category]
            )

        for severity in ErrorSeverity:
            summary["errors_by_severity"][severity.value] = len(
                [e for e in self.error_history if e.severity == severity]
            )

        return summary


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(
    error: Union[UCUPError, Exception], context: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Handle an error using the global error handler.

    This function can handle both UCUPError instances and raw exceptions,
    converting the latter to appropriate UCUPError types.
    """
    handler = get_error_handler()

    if isinstance(error, UCUPError):
        return handler.handle_error(error, context)
    else:
        # Convert to UCUPError
        ucup_error = UCUPError(
            message=str(error),
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.MEDIUM,
            error_code="GENERIC_ERROR",
            context=ErrorContext(component="unknown", operation="unknown"),
            original_exception=error,
        )
        return handler.handle_error(ucup_error, context)


def error_handler(catch_exceptions: bool = True):
    """
    Decorator for functions that provides comprehensive error handling.

    Usage:
        @error_handler()
        async def my_function():
            # Function implementation
    """

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": len(args),
                    "kwargs": list(kwargs.keys()),
                }
                return handle_error(e, context)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args": len(args),
                    "kwargs": list(kwargs.keys()),
                }
                return handle_error(e, context)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
