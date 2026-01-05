"""
UCUP Plugin System Initialization

Provides extensible plugin interfaces for custom AI agents and tools.
"""

# Import all plugin classes and utilities
from .plugin_system import (
    PluginInterface,
    AIPlugin,
    AgentPlugin,
    ModelAdapterPlugin,
    ToolPlugin,
    WorkflowPlugin,
    PluginManager,
)

# Legacy support - also import from plugins.py if it exists
try:
    from .plugins import *  # noqa
except ImportError:
    pass

__all__ = [
    "PluginInterface",
    "AIPlugin",
    "AgentPlugin",
    "ModelAdapterPlugin",
    "ToolPlugin",
    "WorkflowPlugin",
    "PluginManager",
]
