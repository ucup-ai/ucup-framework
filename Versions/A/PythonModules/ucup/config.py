"""
Configuration DSL for UCUP Framework.

This module provides a domain-specific language for configuring complex
agent networks, coordination patterns, and monitoring pipelines using YAML.

Copyright (c) 2025 UCUP Framework Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    # Fallback if pydantic not available
    class BaseModel:
        pass

    ValidationError = Exception

from .errors import ConfigurationError, ErrorContext, get_error_handler
from .feature_flags import FeatureFlagManager, get_feature_manager

# Import validation and error handling
from .validation import ValidationReport, get_ucup_validator

# from .plugins import get_plugin_manager  # Plugin system now implemented


@dataclass
class ConfigReference:
    """Represents a reference to another configuration element."""

    ref_type: str  # "agents", "coordination", etc.
    name: str
    path: str = ""  # For nested references like "agents.customer_service"

    def resolve(self, config: Dict[str, Any]) -> Any:
        """Resolve the reference within a configuration."""
        parts = self.path.split(".") if self.path else [self.name]
        current = config

        for part in parts:
            if part in current:
                current = current[part]
            else:
                raise ValueError(f"Reference {self.path} not found in configuration")

        return current


class ConfigContext:
    """Context for configuration resolution and validation."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.variables: Dict[str, Any] = {}
        self.references: Dict[str, ConfigReference] = {}
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.includes: List[Path] = []

    def set_variable(self, name: str, value: Any):
        """Set a configuration variable."""
        self.variables[name] = value

    def get_variable(self, name: str) -> Any:
        """Get a configuration variable."""
        if name.startswith("$"):
            # Environment variable
            env_name = name[1:]
            return os.environ.get(env_name)
        return self.variables.get(name)

    def add_template(self, name: str, template: Dict[str, Any]):
        """Add a configuration template."""
        self.templates[name] = template

    def resolve_template(
        self, template_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolve a template with optional overrides."""
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")

        template = self.templates[template_name].copy()
        if overrides:
            self._deep_merge(template, overrides)

        return template

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Deep merge override into base dictionary."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


class ConfigLoader:
    """Loads and processes UCUP configuration files."""

    def __init__(self, context: Optional[ConfigContext] = None):
        self.context = context or ConfigContext()
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file or directory."""
        config_path = Path(config_path)

        if config_path.is_file():
            return self._load_single_config(config_path)
        elif config_path.is_dir():
            return self._load_config_directory(config_path)
        else:
            raise FileNotFoundError(f"Configuration path not found: {config_path}")

    def _load_single_config(self, config_path: Path) -> Dict[str, Any]:
        """Load a single configuration file."""
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Process includes first
        if "includes" in raw_config:
            self._process_includes(raw_config["includes"], config_path.parent)

        # Process templates
        if "templates" in raw_config:
            for name, template in raw_config["templates"].items():
                self.context.add_template(name, template)

        # Process variables
        if "variables" in raw_config:
            for name, value in raw_config["variables"].items():
                resolved_value = self._resolve_variables(value)
                self.context.set_variable(name, resolved_value)

        # Process the main configuration
        config = self._process_config_section(raw_config)
        return config

    def _load_config_directory(self, config_dir: Path) -> Dict[str, Any]:
        """Load configuration from a directory of files."""
        config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

        if not config_files:
            raise ValueError(f"No configuration files found in {config_dir}")

        # Load main config first
        main_config = None
        for config_file in config_files:
            if config_file.name in [
                "config.yaml",
                "config.yml",
                "main.yaml",
                "main.yml",
            ]:
                main_config = self._load_single_config(config_file)
                break

        if not main_config:
            # Use first file as main
            main_config = self._load_single_config(config_files[0])

        return main_config

    def _process_includes(self, includes: List[str], base_path: Path):
        """Process configuration includes."""
        for include_path in includes:
            include_file = base_path / include_path
            if include_file.exists():
                included_config = self._load_single_config(include_file)
                self.context.includes.append(include_file)
                # Merge included config (templates, variables, etc.)
                self._merge_included_config(included_config)
            else:
                self.logger.warning(f"Included config file not found: {include_file}")

    def _merge_included_config(self, included_config: Dict[str, Any]):
        """Merge included configuration."""
        if "templates" in included_config:
            for name, template in included_config["templates"].items():
                self.context.add_template(name, template)

        if "variables" in included_config:
            for name, value in included_config["variables"].items():
                self.context.set_variable(name, value)

    def _process_config_section(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a configuration section with references and variables."""
        processed_config = {}

        for key, value in raw_config.items():
            if key in ["ucup_config", "config"]:  # Root config sections
                processed_config.update(self._process_config_section(value))
            elif key in ["includes", "templates", "variables"]:
                # Already processed
                continue
            else:
                processed_config[key] = self._resolve_value(value)

        return processed_config

    def _resolve_value(self, value: Any) -> Any:
        """Resolve references and variables in configuration values."""
        if isinstance(value, str):
            # Check for variable references
            if value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                return self.context.get_variable(var_name)
            # Check for template references
            elif value.startswith("!template "):
                template_name = value[10:].strip()
                return f"!template:{template_name}"  # Placeholder for later resolution
            # Check for plugin references
            elif value.startswith("!plugin "):
                plugin_ref = value[8:].strip()
                return f"!plugin:{plugin_ref}"  # Placeholder for later resolution
            # Check for config references
            elif value.startswith("!ref "):
                ref_path = value[5:].strip()
                return f"!ref:{ref_path}"  # Placeholder for later resolution
            else:
                return value

        elif isinstance(value, dict):
            # Handle template usage
            if "!template" in value:
                template_name = value["!template"]
                overrides = {k: v for k, v in value.items() if k != "!template"}
                return self.context.resolve_template(template_name, overrides)

            # Recursively process nested dictionaries
            return {k: self._resolve_value(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._resolve_value(item) for item in value]

        else:
            return value

    def _resolve_variables(self, value: Any) -> Any:
        """Resolve variables in configuration values."""
        if isinstance(value, str):
            if value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                return self.context.get_variable(var_name)
            return value
        elif isinstance(value, dict):
            return {k: self._resolve_variables(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_variables(item) for item in value]
        else:
            return value


class ConfigValidator:
    """Validates UCUP configuration against schemas."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate UCUP config version
        if "version" not in config:
            errors.append("Missing required 'version' field")

        # Validate agents section
        if "agents" in config:
            agent_errors = self._validate_agents(config["agents"])
            errors.extend(agent_errors)

        # Validate coordination section
        if "coordination" in config:
            coord_errors = self._validate_coordination(config["coordination"], config)
            errors.extend(coord_errors)

        # Validate monitoring section
        if "monitoring" in config:
            monitor_errors = self._validate_monitoring(config["monitoring"])
            errors.extend(monitor_errors)

        # Validate plugins section
        if "plugins" in config:
            plugin_errors = self._validate_plugins(config["plugins"])
            errors.extend(plugin_errors)

        return errors

    def _validate_agents(self, agents_config: Dict[str, Any]) -> List[str]:
        """Validate agents configuration."""
        errors = []

        for agent_name, agent_config in agents_config.items():
            if not isinstance(agent_config, dict):
                errors.append(f"Agent '{agent_name}' must be a dictionary")
                continue

            # Check required fields
            if "type" not in agent_config:
                errors.append(f"Agent '{agent_name}' missing required 'type' field")

            # Validate config section
            if "config" in agent_config:
                config_errors = self._validate_agent_config(agent_config["config"])
                errors.extend([f"Agent '{agent_name}': {err}" for err in config_errors])

        return errors

    def _validate_agent_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate agent configuration."""
        errors = []

        # Check for required LLM if not using plugin
        if "type" in config and not config["type"].startswith("!plugin"):
            if "llm" not in config:
                errors.append("Non-plugin agents require 'llm' configuration")

        return errors

    def _validate_coordination(
        self, coord_config: Dict[str, Any], full_config: Dict[str, Any]
    ) -> List[str]:
        """Validate coordination configuration."""
        errors = []

        if "type" not in coord_config:
            errors.append("Coordination missing required 'type' field")
            return errors

        coord_type = coord_config["type"]

        # Validate hierarchical coordination
        if coord_type == "hierarchical":
            if "manager" not in coord_config:
                errors.append("Hierarchical coordination requires 'manager' field")
            if "workers" not in coord_config:
                errors.append("Hierarchical coordination requires 'workers' field")

        # Validate debate coordination
        elif coord_type == "debate":
            if "agents" not in coord_config:
                errors.append("Debate coordination requires 'agents' field")

        # Validate market coordination
        elif coord_type == "market":
            if "agents" not in coord_config:
                errors.append("Market coordination requires 'agents' field")

        # Validate references
        for field_name, field_value in coord_config.items():
            if isinstance(field_value, str) and field_value.startswith("!ref:"):
                ref_path = field_value[5:]
                if not self._reference_exists(ref_path, full_config):
                    errors.append(
                        f"Reference '{ref_path}' not found for coordination.{field_name}"
                    )

        return errors

    def _validate_monitoring(self, monitor_config: Dict[str, Any]) -> List[str]:
        """Validate monitoring configuration."""
        errors = []

        # Validate traces
        if "traces" in monitor_config:
            for trace_config in monitor_config["traces"]:
                if "type" not in trace_config:
                    errors.append("Trace configuration missing 'type' field")

        # Validate alerts
        if "alerts" in monitor_config:
            for alert_config in monitor_config["alerts"]:
                if "condition" not in alert_config:
                    errors.append("Alert configuration missing 'condition' field")

        return errors

    def _validate_plugins(self, plugins_config: Dict[str, Any]) -> List[str]:
        """Validate plugins configuration."""
        errors = []

        # plugin_manager = get_plugin_manager()  # Plugin system not yet implemented

        for plugin_name, plugin_config in plugins_config.items():
            # Check if plugin exists (temporarily disabled)
            # if not plugin_manager.get_plugin(plugin_name):
            #     errors.append(f"Plugin '{plugin_name}' not found")
            pass  # Skip plugin validation for now

        return errors

    def _reference_exists(self, ref_path: str, config: Dict[str, Any]) -> bool:
        """Check if a reference path exists in configuration."""
        parts = ref_path.split(".")
        current = config

        for part in parts:
            if part in current:
                current = current[part]
            else:
                return False

        return True


class ConfigResolver:
    """Resolves configuration into actual UCUP components."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Temporarily use dict for plugin manager to avoid circular import
        self.plugin_manager = {}

    def resolve_agents(self) -> Dict[str, Any]:
        """Resolve agent configurations into actual agent instances."""
        agents = {}

        if "agents" not in self.config:
            return agents

        for agent_name, agent_config in self.config["agents"].items():
            try:
                agent = self._resolve_agent(agent_config)
                agents[agent_name] = agent
            except Exception as e:
                # self.logger.error(f"Failed to resolve agent '{agent_name}': {e}")
                print(f"Failed to resolve agent '{agent_name}': {e}")

        return agents

    def resolve_coordination(self, agents: Dict[str, Any]) -> Any:
        """Resolve coordination configuration into coordination instance."""
        if "coordination" not in self.config:
            return None

        coord_config = self.config["coordination"]
        coord_type = coord_config.get("type")

        if coord_type == "hierarchical":
            return self._resolve_hierarchical_coordination(coord_config, agents)
        elif coord_type == "debate":
            return self._resolve_debate_coordination(coord_config, agents)
        elif coord_type == "market":
            return self._resolve_market_coordination(coord_config, agents)
        elif coord_type == "swarm":
            return self._resolve_swarm_coordination(coord_config, agents)
        else:
            raise ValueError(f"Unknown coordination type: {coord_type}")

    def resolve_monitoring(self) -> Dict[str, Any]:
        """Resolve monitoring configuration into monitoring instances."""
        monitoring = {}

        if "monitoring" not in self.config:
            return monitoring

        monitor_config = self.config["monitoring"]

        # Resolve tracers
        if "traces" in monitor_config:
            monitoring["tracers"] = []
            for trace_config in monitor_config["traces"]:
                tracer = self._resolve_tracer(trace_config)
                monitoring["tracers"].append(tracer)

        # Resolve alerts
        if "alerts" in monitor_config:
            monitoring["alerts"] = monitor_config["alerts"]

        return monitoring

    def _resolve_agent(self, agent_config: Dict[str, Any]) -> Any:
        """Resolve a single agent configuration."""
        agent_type = agent_config.get("type")

        if agent_type.startswith("!plugin:"):
            # Plugin agent
            plugin_name = agent_type[8:]  # Remove '!plugin:' prefix
            config = agent_config.get("config", {})
            # return self.plugin_manager.create_agent_from_plugin(plugin_name, config)  # Plugin system not yet implemented
            raise NotImplementedError("Plugin system not yet implemented")

        elif agent_type == "ProbabilisticAgent":
            # Built-in agent
            from .probabilistic import ProbabilisticAgent

            config = agent_config.get("config", {})
            return ProbabilisticAgent(**config)

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _resolve_hierarchical_coordination(
        self, coord_config: Dict[str, Any], agents: Dict[str, Any]
    ) -> Any:
        """Resolve hierarchical coordination."""
        from .coordination import HierarchicalCoordination

        manager_ref = coord_config.get("manager")
        worker_refs = coord_config.get("workers", [])

        manager = self._resolve_reference(manager_ref, agents)
        workers = [self._resolve_reference(ref, agents) for ref in worker_refs]

        return HierarchicalCoordination(
            manager_agent=manager,
            worker_agents=workers,
            approval_workflow=coord_config.get("approval_workflow", False),
        )

    def _resolve_debate_coordination(
        self, coord_config: Dict[str, Any], agents: Dict[str, Any]
    ) -> Any:
        """Resolve debate coordination."""
        from .coordination import DebateCoordination

        agent_refs = coord_config.get("agents", [])
        resolved_agents = [self._resolve_reference(ref, agents) for ref in agent_refs]

        return DebateCoordination(
            agents=resolved_agents, max_rounds=coord_config.get("max_rounds", 5)
        )

    def _resolve_market_coordination(
        self, coord_config: Dict[str, Any], agents: Dict[str, Any]
    ) -> Any:
        """Resolve market coordination."""
        from .coordination import MarketBasedCoordination

        agent_refs = coord_config.get("agents", [])
        resolved_agents = [self._resolve_reference(ref, agents) for ref in agent_refs]

        return MarketBasedCoordination(agents=resolved_agents)

    def _resolve_swarm_coordination(
        self, coord_config: Dict[str, Any], agents: Dict[str, Any]
    ) -> Any:
        """Resolve swarm coordination."""
        from .coordination import SwarmCoordination

        agent_refs = coord_config.get("agents", [])
        resolved_agents = [self._resolve_reference(ref, agents) for ref in agent_refs]

        return SwarmCoordination(agents=resolved_agents)

    def _resolve_tracer(self, trace_config: Dict[str, Any]) -> Any:
        """Resolve tracer configuration."""
        from .observability import DecisionTracer

        tracer_type = trace_config.get("type", "DecisionTracer")

        if tracer_type == "DecisionTracer":
            return DecisionTracer(
                enable_detailed_tracing=trace_config.get("detailed", True)
            )
        else:
            raise ValueError(f"Unknown tracer type: {tracer_type}")

    def _resolve_reference(self, ref: str, agents: Dict[str, Any]) -> Any:
        """Resolve a reference to an agent or other component."""
        if ref.startswith("!ref:"):
            ref_path = ref[5:]  # Remove '!ref:' prefix

            # For now, assume it's an agent reference like "agents.agent_name"
            if ref_path.startswith("agents."):
                agent_name = ref_path[7:]  # Remove 'agents.' prefix
                if agent_name in agents:
                    return agents[agent_name]
                else:
                    raise ValueError(f"Agent '{agent_name}' not found")

        # If not a reference, assume it's a direct value
        return ref


def get_default_config() -> Dict[str, Any]:
    """Get default UCUP configuration."""
    return {
        "version": "3.0.0",
        "toon": {
            "enabled": True,
            "auto_optimize": True,
            "min_savings_threshold": 10.0,  # Minimum % savings to use TOON
            "default_format": "auto",  # 'auto', 'toon', or 'json'
            "schemas": {},  # Custom TOON schemas
            "cost_tracking": True,
        },
        "agents": {},
        "coordination": {},
        "monitoring": {},
        "plugins": {},
    }


def load_ucup_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and validate UCUP configuration."""
    if config_path is None:
        # Return default config if no path provided
        return get_default_config()

    loader = ConfigLoader()
    config = loader.load_config(config_path)

    validator = ConfigValidator()
    errors = validator.validate_config(config)

    if errors:
        error_msg = "Configuration validation errors:\n" + "\n".join(
            f"  - {err}" for err in errors
        )
        raise ValueError(error_msg)

    # Merge with defaults to ensure TOON settings are available
    default_config = get_default_config()
    merged_config = _deep_merge_configs(default_config, config)

    return merged_config


def _deep_merge_configs(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge override config into base config."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def create_ucup_system(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Create a complete UCUP system from configuration."""
    config = load_ucup_config(config_path)

    resolver = ConfigResolver(config)

    # Resolve all components
    agents = resolver.resolve_agents()
    coordination = resolver.resolve_coordination(agents)
    monitoring = resolver.resolve_monitoring()

    return {
        "config": config,
        "agents": agents,
        "coordination": coordination,
        "monitoring": monitoring,
        "resolver": resolver,
    }


# Import logging for the module
import logging
