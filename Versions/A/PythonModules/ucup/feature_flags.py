#!/usr/bin/env python3
"""
Feature Flags for UCUP Framework

Provides a centralized system for managing experimental and beta features
across the UCUP ecosystem, allowing safe rollout and testing of new components.

Copyright (c) 2025 UCUP Framework Contributors
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class FeatureFlag(Enum):
    """Available feature flags in UCUP."""

    # Core experimental features
    ADVANCED_PROBABILISTIC = "advanced_probabilistic"
    MULTIMODAL_FUSION = "multimodal_fusion"
    SELF_HEALING_COORDINATOR = "self_healing_coordinator"
    PREDICTIVE_FAILURE_DETECTION = "predictive_failure_detection"
    CAUSAL_ANALYSIS_ENGINE = "causal_analysis_engine"

    # Plugin system features
    PLUGIN_ARCHITECTURE = "plugin_architecture"
    DYNAMIC_PLUGIN_LOADING = "dynamic_plugin_loading"
    PLUGIN_SANDBOXING = "plugin_sandboxing"

    # TOON optimization features
    TOON_OPTIMIZATION = "toon_optimization"
    ADAPTIVE_SCHEMA_GENERATION = "adaptive_schema_generation"
    COST_AWARE_OPTIMIZATION = "cost_aware_optimization"

    # Debugging and observability
    TIME_TRAVEL_DEBUGGING = "time_travel_debugging"
    VISUAL_DEBUGGER_UI = "visual_debugger_ui"
    PRODUCTION_TRACING = "production_tracing"

    # Enterprise features
    ENTERPRISE_MONITORING = "enterprise_monitoring"
    AUDIT_LOGGING = "audit_logging"
    COMPLIANCE_REPORTING = "compliance_reporting"

    # UI and IDE features
    ADVANCED_DASHBOARDS = "advanced_dashboards"
    COLLABORATION_FEATURES = "collaboration_features"
    PERFORMANCE_PROFILING = "performance_profiling"

    # AI-assisted development
    AI_CODE_GENERATION = "ai_code_generation"
    SMART_REFACTORING = "smart_refactoring"
    INTELLIGENT_SUGGESTIONS = "intelligent_suggestions"


class FeatureFlagState(Enum):
    """Feature flag states."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    BETA = "beta"  # Enabled but may have issues
    EXPERIMENTAL = "experimental"  # May be unstable


@dataclass
class FeatureFlagConfig:
    """Configuration for a single feature flag."""

    name: str
    state: FeatureFlagState = FeatureFlagState.DISABLED
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    requires_restart: bool = False
    beta_warning: str = ""
    experimental_warning: str = "This feature is experimental and may be unstable"


@dataclass
class FeatureFlagsConfig:
    """Complete feature flags configuration."""

    version: str = "1.0"
    flags: Dict[str, FeatureFlagConfig] = field(default_factory=dict)
    global_enabled: bool = True
    environment_overrides: Dict[str, str] = field(default_factory=dict)


class FeatureFlagManager:
    """
    Centralized manager for UCUP feature flags.

    Provides methods to check feature flag status, manage dependencies,
    and handle environment-based overrides.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._validate_dependencies()

    def _get_default_config_path(self) -> Path:
        """Get default feature flags config path."""
        return Path.home() / ".ucup" / "feature_flags.json"

    def _load_config(self) -> FeatureFlagsConfig:
        """Load feature flags configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                return self._data_to_config(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Invalid feature flags config, using defaults: {e}")
        return self._get_default_config()

    def _data_to_config(self, data: Dict[str, Any]) -> FeatureFlagsConfig:
        """Convert JSON data to FeatureFlagsConfig."""
        flags = {}
        for name, flag_data in data.get("flags", {}).items():
            flags[name] = FeatureFlagConfig(
                name=name,
                state=FeatureFlagState(flag_data.get("state", "disabled")),
                description=flag_data.get("description", ""),
                dependencies=flag_data.get("dependencies", []),
                requires_restart=flag_data.get("requires_restart", False),
                beta_warning=flag_data.get("beta_warning", ""),
                experimental_warning=flag_data.get("experimental_warning", ""),
            )

        return FeatureFlagsConfig(
            version=data.get("version", "1.0"),
            flags=flags,
            global_enabled=data.get("global_enabled", True),
            environment_overrides=data.get("environment_overrides", {}),
        )

    def _get_default_config(self) -> FeatureFlagsConfig:
        """Get default feature flags configuration."""
        return FeatureFlagsConfig(
            flags={
                # Core features - enabled by default
                FeatureFlag.TOON_OPTIMIZATION.value: FeatureFlagConfig(
                    name=FeatureFlag.TOON_OPTIMIZATION.value,
                    state=FeatureFlagState.ENABLED,
                    description="TOON token optimization for cost savings",
                ),
                # Experimental features - disabled by default
                FeatureFlag.ADVANCED_PROBABILISTIC.value: FeatureFlagConfig(
                    name=FeatureFlag.ADVANCED_PROBABILISTIC.value,
                    state=FeatureFlagState.EXPERIMENTAL,
                    description="Advanced probabilistic reasoning strategies",
                    experimental_warning="Advanced probabilistic features may impact performance",
                ),
                FeatureFlag.MULTIMODAL_FUSION.value: FeatureFlagConfig(
                    name=FeatureFlag.MULTIMODAL_FUSION.value,
                    state=FeatureFlagState.EXPERIMENTAL,
                    description="Real-time multimodal data fusion",
                    experimental_warning="Multimodal fusion is experimental and may have accuracy issues",
                ),
                FeatureFlag.SELF_HEALING_COORDINATOR.value: FeatureFlagConfig(
                    name=FeatureFlag.SELF_HEALING_COORDINATOR.value,
                    state=FeatureFlagState.BETA,
                    description="Self-healing agent coordination",
                    beta_warning="Self-healing may occasionally trigger false recoveries",
                ),
                FeatureFlag.PREDICTIVE_FAILURE_DETECTION.value: FeatureFlagConfig(
                    name=FeatureFlag.PREDICTIVE_FAILURE_DETECTION.value,
                    state=FeatureFlagState.BETA,
                    description="ML-based failure prediction",
                    beta_warning="Prediction accuracy may vary based on usage patterns",
                ),
                FeatureFlag.CAUSAL_ANALYSIS_ENGINE.value: FeatureFlagConfig(
                    name=FeatureFlag.CAUSAL_ANALYSIS_ENGINE.value,
                    state=FeatureFlagState.EXPERIMENTAL,
                    description="Graph-based root cause analysis",
                    experimental_warning="Causal analysis is experimental and may be computationally intensive",
                ),
                # Plugin features
                FeatureFlag.PLUGIN_ARCHITECTURE.value: FeatureFlagConfig(
                    name=FeatureFlag.PLUGIN_ARCHITECTURE.value,
                    state=FeatureFlagState.ENABLED,
                    description="Plugin system for extensibility",
                ),
                FeatureFlag.DYNAMIC_PLUGIN_LOADING.value: FeatureFlagConfig(
                    name=FeatureFlag.DYNAMIC_PLUGIN_LOADING.value,
                    state=FeatureFlagState.BETA,
                    description="Runtime plugin loading",
                    dependencies=[FeatureFlag.PLUGIN_ARCHITECTURE.value],
                    beta_warning="Dynamic loading may affect startup performance",
                ),
                # Debugging features
                FeatureFlag.TIME_TRAVEL_DEBUGGING.value: FeatureFlagConfig(
                    name=FeatureFlag.TIME_TRAVEL_DEBUGGING.value,
                    state=FeatureFlagState.EXPERIMENTAL,
                    description="Time-travel debugging capabilities",
                    experimental_warning="Time travel debugging is experimental and may not work with all agent types",
                ),
                FeatureFlag.VISUAL_DEBUGGER_UI.value: FeatureFlagConfig(
                    name=FeatureFlag.VISUAL_DEBUGGER_UI.value,
                    state=FeatureFlagState.BETA,
                    description="Visual debugger interface",
                    dependencies=[FeatureFlag.TIME_TRAVEL_DEBUGGING.value],
                    beta_warning="Visual debugger may have UI inconsistencies",
                ),
                # Enterprise features
                FeatureFlag.ENTERPRISE_MONITORING.value: FeatureFlagConfig(
                    name=FeatureFlag.ENTERPRISE_MONITORING.value,
                    state=FeatureFlagState.ENABLED,
                    description="Enterprise-grade monitoring and alerting",
                ),
                FeatureFlag.AUDIT_LOGGING.value: FeatureFlagConfig(
                    name=FeatureFlag.AUDIT_LOGGING.value,
                    state=FeatureFlagState.ENABLED,
                    description="Comprehensive audit logging",
                    dependencies=[FeatureFlag.ENTERPRISE_MONITORING.value],
                ),
                # Development features
                FeatureFlag.AI_CODE_GENERATION.value: FeatureFlagConfig(
                    name=FeatureFlag.AI_CODE_GENERATION.value,
                    state=FeatureFlagState.EXPERIMENTAL,
                    description="AI-assisted code generation",
                    experimental_warning="AI code generation is experimental and may produce incorrect code",
                ),
                FeatureFlag.PERFORMANCE_PROFILING.value: FeatureFlagConfig(
                    name=FeatureFlag.PERFORMANCE_PROFILING.value,
                    state=FeatureFlagState.ENABLED,
                    description="Advanced performance profiling",
                ),
            }
        )

    def _validate_dependencies(self):
        """Validate feature flag dependencies."""
        for flag_name, flag_config in self.config.flags.items():
            for dep in flag_config.dependencies:
                if dep not in self.config.flags:
                    print(
                        f"Warning: Feature flag '{flag_name}' depends on unknown flag '{dep}'"
                    )
                elif not self.is_enabled(dep):
                    print(
                        f"Warning: Feature flag '{flag_name}' depends on disabled flag '{dep}'"
                    )

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled."""
        if not self.config.global_enabled:
            return False

        # Check environment override first
        env_override = self.config.environment_overrides.get(flag_name)
        if env_override:
            env_value = os.environ.get(env_override)
            if env_value:
                return env_value.lower() in ("true", "1", "yes", "enabled")

        # Check feature flag config
        if flag_name not in self.config.flags:
            return False

        flag_config = self.config.flags[flag_name]
        state = flag_config.state

        # Check dependencies
        for dep in flag_config.dependencies:
            if not self.is_enabled(dep):
                return False

        return state in (
            FeatureFlagState.ENABLED,
            FeatureFlagState.BETA,
            FeatureFlagState.EXPERIMENTAL,
        )

    def get_flag_state(self, flag_name: str) -> Optional[FeatureFlagState]:
        """Get the state of a feature flag."""
        if flag_name not in self.config.flags:
            return None
        return self.config.flags[flag_name].state

    def get_flag_config(self, flag_name: str) -> Optional[FeatureFlagConfig]:
        """Get the configuration for a feature flag."""
        return self.config.flags.get(flag_name)

    def enable_flag(self, flag_name: str) -> bool:
        """Enable a feature flag."""
        if flag_name not in self.config.flags:
            print(f"Warning: Unknown feature flag '{flag_name}'")
            return False

        flag_config = self.config.flags[flag_name]

        # Check if enabling would violate dependencies
        for dep in flag_config.dependencies:
            if not self.is_enabled(dep):
                print(f"Cannot enable '{flag_name}': dependency '{dep}' is not enabled")
                return False

        if flag_config.requires_restart:
            print(f"Note: Enabling '{flag_name}' requires a restart to take effect")

        flag_config.state = FeatureFlagState.ENABLED
        self._save_config()
        return True

    def disable_flag(self, flag_name: str) -> bool:
        """Disable a feature flag."""
        if flag_name not in self.config.flags:
            print(f"Warning: Unknown feature flag '{flag_name}'")
            return False

        flag_config = self.config.flags[flag_name]

        # Check if disabling would break dependents
        dependents = self._get_dependents(flag_name)
        if dependents:
            print(
                f"Warning: Disabling '{flag_name}' may affect: {', '.join(dependents)}"
            )

        if flag_config.requires_restart:
            print(f"Note: Disabling '{flag_name}' requires a restart to take effect")

        flag_config.state = FeatureFlagState.DISABLED
        self._save_config()
        return True

    def _get_dependents(self, flag_name: str) -> List[str]:
        """Get list of feature flags that depend on the given flag."""
        dependents = []
        for name, config in self.config.flags.items():
            if flag_name in config.dependencies:
                dependents.append(name)
        return dependents

    def list_flags(self) -> Dict[str, Dict[str, Any]]:
        """List all feature flags with their status."""
        result = {}
        for name, config in self.config.flags.items():
            result[name] = {
                "state": config.state.value,
                "enabled": self.is_enabled(name),
                "description": config.description,
                "dependencies": config.dependencies,
                "requires_restart": config.requires_restart,
                "warning": self._get_warning_text(config),
            }
        return result

    def _get_warning_text(self, config: FeatureFlagConfig) -> str:
        """Get appropriate warning text for a flag."""
        if config.state == FeatureFlagState.BETA and config.beta_warning:
            return f"BETA: {config.beta_warning}"
        elif (
            config.state == FeatureFlagState.EXPERIMENTAL
            and config.experimental_warning
        ):
            return f"EXPERIMENTAL: {config.experimental_warning}"
        return ""

    def _save_config(self):
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self._config_to_data(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save feature flags config: {e}")

    def _config_to_data(self) -> Dict[str, Any]:
        """Convert FeatureFlagsConfig to JSON-serializable data."""
        return {
            "version": self.config.version,
            "global_enabled": self.config.global_enabled,
            "flags": {
                name: {
                    "state": config.state.value,
                    "description": config.description,
                    "dependencies": config.dependencies,
                    "requires_restart": config.requires_restart,
                    "beta_warning": config.beta_warning,
                    "experimental_warning": config.experimental_warning,
                }
                for name, config in self.config.flags.items()
            },
            "environment_overrides": self.config.environment_overrides,
        }


# Global feature flag manager instance
_feature_manager = None


def get_feature_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance."""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureFlagManager()
    return _feature_manager


def is_feature_enabled(flag_name: str) -> bool:
    """Check if a feature flag is enabled (convenience function)."""
    return get_feature_manager().is_enabled(flag_name)


def require_feature(flag_name: str, message: Optional[str] = None):
    """Require a feature flag to be enabled, raise error if not."""
    if not is_feature_enabled(flag_name):
        error_msg = message or f"Feature '{flag_name}' is not enabled"
        raise ValueError(error_msg)
