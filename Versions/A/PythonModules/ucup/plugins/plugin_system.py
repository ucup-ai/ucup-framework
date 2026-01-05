"""
UCUP Plugin System

Provides extensible plugin interfaces for custom AI agents and tools.
Allows users to create, install, and execute custom plugins with specific features.

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

import importlib.util
import json
import os
import sys
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.request import urlretrieve

import requests


class PluginInterface(ABC):
    """Base interface for UCUP plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Plugin category (ai, agent, model, tool)."""
        pass

    @abstractmethod
    def execute(self, input_data: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin with input data."""
        pass

    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities."""
        return []

    def get_supported_inputs(self) -> List[str]:
        """Get supported input types."""
        return ["text"]

    def get_output_formats(self) -> List[str]:
        """Get supported output formats."""
        return ["text"]


class AIPlugin(PluginInterface):
    """Base class for AI agent plugins."""

    @property
    def category(self) -> str:
        return "ai"

    def execute(self, input_data: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute AI inference."""
        raise NotImplementedError("AI plugins must implement execute method")


class AgentPlugin(PluginInterface):
    """Base class for agent orchestration plugins."""

    @property
    def category(self) -> str:
        return "agent"

    def execute(self, input_data: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute agent orchestration."""
        raise NotImplementedError("Agent plugins must implement execute method")


class ModelAdapterPlugin(PluginInterface):
    """Base class for model adapter plugins."""

    @property
    def category(self) -> str:
        return "model"

    def execute(self, input_data: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute model adaptation."""
        raise NotImplementedError("Model adapter plugins must implement execute method")


class ToolPlugin(PluginInterface):
    """Base class for utility tool plugins."""

    @property
    def category(self) -> str:
        return "tool"

    def execute(self, input_data: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute utility tool."""
        raise NotImplementedError("Tool plugins must implement execute method")


class WorkflowPlugin(PluginInterface):
    """Base class for workflow orchestration plugins."""

    @property
    def category(self) -> str:
        return "workflow"

    def execute(self, input_data: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute workflow orchestration."""
        raise NotImplementedError("Workflow plugins must implement execute method")


class PluginManager:
    """Manages UCUP plugins."""

    def __init__(self):
        self.plugins_dir = Path.home() / ".ucup" / "plugins"
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.plugins: Dict[str, PluginInterface] = {}
        self._load_installed_plugins()

    def _load_installed_plugins(self):
        """Load all installed plugins."""
        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir():
                self._load_plugin_from_directory(plugin_dir)

    def _load_plugin_from_directory(self, plugin_dir: Path):
        """Load a plugin from its directory."""
        try:
            # Look for plugin.json manifest
            manifest_file = plugin_dir / "plugin.json"
            if not manifest_file.exists():
                return

            with open(manifest_file, "r") as f:
                manifest = json.load(f)

            # Load the main plugin module
            main_file = plugin_dir / manifest.get("main", "plugin.py")
            if not main_file.exists():
                return

            # Import the plugin module
            spec = importlib.util.spec_from_file_location(
                f"ucup_plugin_{manifest['name']}", main_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find the plugin class
                plugin_class_name = manifest.get("class", "Plugin")
                plugin_class = getattr(module, plugin_class_name, None)

                if plugin_class:
                    plugin_instance = plugin_class()
                    self.plugins[manifest["name"]] = plugin_instance

        except Exception as e:
            print(f"Failed to load plugin from {plugin_dir}: {e}")

    def list_plugins(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available plugins."""
        plugins = []

        for name, plugin in self.plugins.items():
            plugin_info = {
                "name": name,
                "version": getattr(plugin, "version", "1.0.0"),
                "description": getattr(plugin, "description", "No description"),
                "category": getattr(plugin, "category", "misc"),
                "enabled": True,
                "capabilities": plugin.get_capabilities(),
                "supported_inputs": plugin.get_supported_inputs(),
                "output_formats": plugin.get_output_formats(),
            }
            plugins.append(plugin_info)

        if category and category != "all":
            plugins = [p for p in plugins if p["category"] == category]

        return plugins

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin."""
        if name not in self.plugins:
            return None

        plugin = self.plugins[name]
        plugin_dir = self.plugins_dir / name

        # Read manifest if available
        manifest_file = plugin_dir / "plugin.json"
        manifest = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)
            except:
                pass

        # Read README if available
        readme_file = plugin_dir / "README.md"
        readme = None
        if readme_file.exists():
            try:
                with open(readme_file, "r") as f:
                    readme = f.read()
            except:
                pass

        return {
            "name": name,
            "version": getattr(plugin, "version", "1.0.0"),
            "description": getattr(plugin, "description", "No description"),
            "category": getattr(plugin, "category", "misc"),
            "enabled": True,
            "author": manifest.get("author", "Unknown"),
            "license": manifest.get("license", "MIT"),
            "dependencies": manifest.get("dependencies", []),
            "capabilities": plugin.get_capabilities(),
            "supported_inputs": plugin.get_supported_inputs(),
            "output_formats": plugin.get_output_formats(),
            "path": str(plugin_dir),
            "modified": manifest.get("modified", "Unknown"),
            "readme": readme,
        }

    def install_from_file(self, file_path: str, name: Optional[str] = None) -> bool:
        """Install a plugin from a local file."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return False

            # Determine plugin name
            plugin_name = name or file_path.stem

            # Create plugin directory
            plugin_dir = self.plugins_dir / plugin_name
            plugin_dir.mkdir(exist_ok=True)

            if file_path.suffix == ".zip":
                # Extract zip file
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(plugin_dir)
            else:
                # Assume it's a Python file, copy to plugin directory
                import shutil
                shutil.copy2(file_path, plugin_dir / "plugin.py")

                # Create basic manifest
                manifest = {
                    "name": plugin_name,
                    "version": "1.0.0",
                    "main": "plugin.py",
                    "class": "Plugin",
                    "description": f"Custom plugin: {plugin_name}",
                    "author": "User",
                    "license": "MIT",
                }

                with open(plugin_dir / "plugin.json", "w") as f:
                    json.dump(manifest, f, indent=2)

            # Reload plugins
            self._load_plugin_from_directory(plugin_dir)

            return True

        except Exception as e:
            print(f"Failed to install plugin from file: {e}")
            return False

    def install_from_url(self, url: str, name: Optional[str] = None) -> bool:
        """Install a plugin from a URL."""
        try:
            # Download to temporary file
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
                urlretrieve(url, temp_file.name)
                temp_file_path = temp_file.name

            # Install from downloaded file
            success = self.install_from_file(temp_file_path, name)

            # Clean up
            os.unlink(temp_file_path)

            return success

        except Exception as e:
            print(f"Failed to install plugin from URL: {e}")
            return False

    def uninstall_plugin(self, name: str) -> bool:
        """Uninstall a plugin."""
        try:
            plugin_dir = self.plugins_dir / name

            if not plugin_dir.exists():
                return False

            # Remove from loaded plugins
            if name in self.plugins:
                del self.plugins[name]

            # Remove directory
            import shutil
            shutil.rmtree(plugin_dir)

            return True

        except Exception as e:
            print(f"Failed to uninstall plugin: {e}")
            return False

    def execute_plugin(self, name: str, input_data: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a plugin."""
        if name not in self.plugins:
            raise ValueError(f"Plugin '{name}' not found")

        plugin = self.plugins[name]
        return plugin.execute(input_data, config)

    def create_plugin_template(self, name: str, plugin_type: str, output_dir: str) -> Optional[str]:
        """Create a plugin template."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Create plugin directory
            plugin_dir = output_path / name
            plugin_dir.mkdir(exist_ok=True)

            # Determine base class
            base_classes = {
                "ai_agent": "AIPlugin",
                "model_adapter": "ModelAdapterPlugin",
                "tool": "ToolPlugin",
                "workflow": "WorkflowPlugin",
            }

            base_class = base_classes.get(plugin_type, "PluginInterface")

            # Create plugin.py
            plugin_code = f'''"""
{name} Plugin

Custom {plugin_type} plugin for UCUP Framework.
"""

from ucup.plugins import {base_class}


class Plugin({base_class}):
    """Custom plugin implementation."""

    @property
    def name(self) -> str:
        return "{name}"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Custom {plugin_type} plugin for UCUP"

    def execute(self, input_data, config=None):
        """
        Execute the plugin.

        Args:
            input_data: Input data for the plugin
            config: Optional configuration dictionary

        Returns:
            Plugin execution result
        """
        # TODO: Implement your plugin logic here
        print(f"Executing {{self.name}} plugin")
        print(f"Input: {{input_data}}")
        print(f"Config: {{config}}")

        # Example implementation - replace with your logic
        if isinstance(input_data, str):
            result = f"Processed by {{self.name}}: {{input_data}}"
        elif isinstance(input_data, dict):
            result = {{
                "plugin": self.name,
                "processed_data": input_data,
                "config": config or {{}},
                "timestamp": "2025-01-01T00:00:00Z"
            }}
        else:
            result = f"Unsupported input type: {{type(input_data)}}"

        return result

    def get_capabilities(self):
        """Return plugin capabilities."""
        return ["custom_processing", "data_transformation"]

    def get_supported_inputs(self):
        """Return supported input types."""
        return ["text", "json", "dict"]

    def get_output_formats(self):
        """Return supported output formats."""
        return ["text", "json"]
'''

            with open(plugin_dir / "plugin.py", "w") as f:
                f.write(plugin_code)

            # Create manifest
            manifest = {
                "name": name,
                "version": "1.0.0",
                "main": "plugin.py",
                "class": "Plugin",
                "type": plugin_type,
                "description": f"Custom {plugin_type} plugin for UCUP",
                "author": "User",
                "license": "MIT",
                "dependencies": [],
                "created": "2025-01-01T00:00:00Z",
            }

            with open(plugin_dir / "plugin.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # Create README
            readme = f"""# {name} Plugin

Custom {plugin_type} plugin for UCUP Framework.

## Installation

```bash
ucup plugin install {plugin_dir}
```

## Usage

```bash
# Execute the plugin
ucup plugin execute {name} --input "Hello World"

# Get plugin info
ucup plugin info {name}
```

## Development

To modify this plugin:

1. Edit `plugin.py` to implement your logic
2. Update `plugin.json` with metadata
3. Test with `ucup plugin execute {name}`
4. Reinstall with `ucup plugin install {plugin_dir}`
"""

            with open(plugin_dir / "README.md", "w") as f:
                f.write(readme)

            return str(plugin_dir)

        except Exception as e:
            print(f"Failed to create plugin template: {e}")
            return None


# Export plugin base classes for easy importing
__all__ = [
    "PluginInterface",
    "AIPlugin",
    "AgentPlugin",
    "ModelAdapterPlugin",
    "ToolPlugin",
    "WorkflowPlugin",
    "PluginManager",
]
