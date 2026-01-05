"""
Base classes and interfaces for UCUP plugin system.

Defines the core plugin architecture including plugin types, status,
metadata, and base interfaces for different plugin categories.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type


@dataclass
class PluginConfig:
    """Configuration for a plugin."""

    settings: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.settings[key] = value


class PluginError(Exception):
    """Base exception for plugin-related errors."""

    pass


class PluginType(Enum):
    """Types of plugins supported by UCUP."""

    BENCHMARK = "benchmark"
    COORDINATOR = "coordinator"
    OBSERVABILITY = "observability"
    TESTING = "testing"
    RELIABILITY = "reliability"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


class PluginStatus(Enum):
    """Status of a plugin."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    ucup_version_min: str = "2.0.0"
    ucup_version_max: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    homepage: Optional[str] = None
    documentation: Optional[str] = None
    license: str = "MIT"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "ucup_version_min": self.ucup_version_min,
            "ucup_version_max": self.ucup_version_max,
            "tags": list(self.tags),
            "homepage": self.homepage,
            "documentation": self.documentation,
            "license": self.license,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class PluginInterface(ABC):
    """Base interface that all plugins must implement."""

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with configuration.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of the plugin."""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this plugin provides."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the plugin.

        Returns:
            Dictionary with health status information
        """
        pass


class PluginBase(PluginInterface):
    """
    Base implementation of plugin interface with common functionality.

    All UCUP plugins should inherit from this class or implement PluginInterface.
    """

    def __init__(self, metadata: PluginMetadata):
        """Initialize plugin with metadata."""
        self.metadata = metadata
        self.status = PluginStatus.UNLOADED
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"ucup.plugin.{metadata.name}")
        self._initialized = False
        self._error: Optional[Exception] = None

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return self.metadata

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            self.status = PluginStatus.LOADING
            self.config = config

            # Perform custom initialization
            await self._initialize_impl(config)

            self.status = PluginStatus.LOADED
            self._initialized = True
            self.logger.info(f"Plugin {self.metadata.name} initialized successfully")
            return True

        except Exception as e:
            self.status = PluginStatus.ERROR
            self._error = e
            self.logger.error(f"Plugin {self.metadata.name} initialization failed: {e}")
            return False

    async def _initialize_impl(self, config: Dict[str, Any]) -> None:
        """
        Override this method to provide custom initialization logic.

        Args:
            config: Configuration dictionary
        """
        pass

    async def shutdown(self) -> None:
        """Clean shutdown of the plugin."""
        try:
            await self._shutdown_impl()
            self.status = PluginStatus.UNLOADED
            self._initialized = False
            self.logger.info(f"Plugin {self.metadata.name} shut down successfully")
        except Exception as e:
            self.logger.error(f"Plugin {self.metadata.name} shutdown error: {e}")
            raise

    async def _shutdown_impl(self) -> None:
        """Override this method to provide custom shutdown logic."""
        pass

    def get_capabilities(self) -> List[str]:
        """
        Return list of capabilities this plugin provides.
        Override in subclasses to specify actual capabilities.
        """
        return []

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the plugin."""
        return {
            "status": self.status.value,
            "initialized": self._initialized,
            "error": str(self._error) if self._error else None,
            "metadata": self.metadata.to_dict(),
        }

    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    def get_status(self) -> PluginStatus:
        """Get current plugin status."""
        return self.status

    def get_error(self) -> Optional[Exception]:
        """Get last error if any."""
        return self._error


# Alias for backward compatibility
Plugin = PluginBase


class AgentPlugin(PluginBase):
    """Base class for agent plugins."""

    def __init__(self, metadata: PluginMetadata):
        """Initialize agent plugin."""
        if metadata.plugin_type != PluginType.CUSTOM:  # Agent plugins use CUSTOM type
            metadata.plugin_type = PluginType.CUSTOM
        super().__init__(metadata)

    @abstractmethod
    def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create and return an agent instance."""
        pass

    @abstractmethod
    def get_supported_capabilities(self) -> Set[str]:
        """Return supported agent capabilities."""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        pass

    def get_capabilities(self) -> List[str]:
        """Return plugin capabilities."""
        return ["agent_creation", "capability_validation", "config_validation"]


class StrategyPlugin(PluginBase):
    """Base class for strategy plugins."""

    def __init__(self, metadata: PluginMetadata):
        """Initialize strategy plugin."""
        if (
            metadata.plugin_type != PluginType.CUSTOM
        ):  # Strategy plugins use CUSTOM type
            metadata.plugin_type = PluginType.CUSTOM
        super().__init__(metadata)

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name."""
        pass

    @abstractmethod
    def execute_strategy(self, context: Dict[str, Any]) -> Any:
        """Execute strategy with context."""
        pass

    def get_capabilities(self) -> List[str]:
        """Return plugin capabilities."""
        return ["strategy_execution", "context_processing"]


class MonitorPlugin(PluginBase):
    """Base class for monitoring plugins."""

    def __init__(self, metadata: PluginMetadata):
        """Initialize monitor plugin."""
        if metadata.plugin_type != PluginType.OBSERVABILITY:
            metadata.plugin_type = PluginType.OBSERVABILITY
        super().__init__(metadata)

    @abstractmethod
    def get_monitor_name(self) -> str:
        """Return monitor name."""
        pass

    def get_capabilities(self) -> List[str]:
        """Return plugin capabilities."""
        return ["monitoring", "metrics_collection", "health_checking"]


class BenchmarkPlugin(PluginBase):
    """Base class for benchmark integration plugins."""

    def __init__(self, metadata: PluginMetadata):
        """Initialize benchmark plugin."""
        if metadata.plugin_type != PluginType.BENCHMARK:
            metadata.plugin_type = PluginType.BENCHMARK
        super().__init__(metadata)

    @abstractmethod
    async def run_benchmark(
        self, agent: Any, benchmark_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run benchmark evaluation on the agent.

        Args:
            agent: Agent to benchmark
            benchmark_config: Benchmark-specific configuration

        Returns:
            Dictionary with benchmark results
        """
        pass

    @abstractmethod
    async def get_benchmark_info(self) -> Dict[str, Any]:
        """
        Get information about the benchmark.

        Returns:
            Dictionary with benchmark information (metrics, categories, etc.)
        """
        pass

    def get_capabilities(self) -> List[str]:
        """Return benchmark plugin capabilities."""
        return ["benchmark_execution", "metrics_collection", "result_analysis"]


class CoordinatorPlugin(PluginBase):
    """Base class for coordinator strategy plugins."""

    def __init__(self, metadata: PluginMetadata):
        """Initialize coordinator plugin."""
        if metadata.plugin_type != PluginType.COORDINATOR:
            metadata.plugin_type = PluginType.COORDINATOR
        super().__init__(metadata)

    @abstractmethod
    async def coordinate(
        self, agents: List[Any], task: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents to complete a task.

        Args:
            agents: List of agents to coordinate
            task: Task specification
            context: Execution context

        Returns:
            Dictionary with coordination results
        """
        pass

    @abstractmethod
    def get_coordination_strategy(self) -> str:
        """Return the coordination strategy name."""
        pass

    def get_capabilities(self) -> List[str]:
        """Return coordinator plugin capabilities."""
        return ["multi_agent_coordination", "task_distribution", "result_aggregation"]


class ObservabilityPlugin(PluginBase):
    """Base class for observability plugins."""

    def __init__(self, metadata: PluginMetadata):
        """Initialize observability plugin."""
        if metadata.plugin_type != PluginType.OBSERVABILITY:
            metadata.plugin_type = PluginType.OBSERVABILITY
        super().__init__(metadata)

    @abstractmethod
    async def trace_decision(
        self, agent_id: str, decision: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """
        Trace an agent decision.

        Args:
            agent_id: Agent identifier
            decision: Decision information
            context: Decision context
        """
        pass

    @abstractmethod
    async def export_traces(
        self,
        agent_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Export traces based on filters.

        Args:
            agent_id: Optional agent ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of trace records
        """
        pass

    def get_capabilities(self) -> List[str]:
        """Return observability plugin capabilities."""
        return ["decision_tracing", "trace_export", "monitoring"]


class TestingPlugin(PluginBase):
    """Base class for testing plugins."""

    def __init__(self, metadata: PluginMetadata):
        """Initialize testing plugin."""
        if metadata.plugin_type != PluginType.TESTING:
            metadata.plugin_type = PluginType.TESTING
        super().__init__(metadata)

    @abstractmethod
    async def generate_tests(
        self, agent: Any, test_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate test cases for the agent.

        Args:
            agent: Agent to test
            test_config: Test generation configuration

        Returns:
            List of generated test cases
        """
        pass

    @abstractmethod
    async def run_tests(
        self, agent: Any, test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run test cases on the agent.

        Args:
            agent: Agent to test
            test_cases: List of test cases

        Returns:
            Dictionary with test results
        """
        pass

    def get_capabilities(self) -> List[str]:
        """Return testing plugin capabilities."""
        return ["test_generation", "test_execution", "result_analysis"]


# Plugin decorator for easy registration
def plugin(
    name: str,
    version: str,
    author: str,
    description: str,
    plugin_type: PluginType,
    **kwargs,
) -> Callable:
    """
    Decorator to mark a class as a UCUP plugin and auto-register it.

    Usage:
        @plugin(
            name="MyBenchmark",
            version="1.0.0",
            author="Author Name",
            description="My custom benchmark",
            plugin_type=PluginType.BENCHMARK
        )
        class MyBenchmarkPlugin(BenchmarkPlugin):
            ...
    """

    def decorator(cls: Type[PluginBase]) -> Type[PluginBase]:
        # Create metadata
        metadata = PluginMetadata(
            name=name,
            version=version,
            author=author,
            description=description,
            plugin_type=plugin_type,
            **kwargs,
        )

        # Store metadata as class attribute
        cls._ucup_plugin_metadata = metadata

        # Mark class as plugin
        cls._is_ucup_plugin = True

        return cls

    return decorator
