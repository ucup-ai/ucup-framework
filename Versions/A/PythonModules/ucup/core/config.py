"""
UCUP Configuration System.

macOS-specific configuration with integration for system frameworks.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class UCUPConfig:
    """Configuration manager for UCUP framework."""

    def __init__(self):
        self.settings: Dict[str, Any] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load macOS-optimized default settings."""
        self.settings.update({
            # Core ML settings
            "core_ml": {
                "enabled": True,
                "model_cache_size": 100 * 1024 * 1024,  # 100MB
                "compute_units": "auto",  # CPU/GPU/Neural Engine
            },

            # Grand Central Dispatch settings
            "gcd": {
                "max_concurrent_tasks": os.cpu_count() or 4,
                "queue_priority": "default",
                "enable_threading": True,
            },

            # Vision framework settings
            "vision": {
                "enable_high_accuracy": True,
                "supported_formats": ["image/jpeg", "image/png", "image/heic"],
                "max_image_size": 4096,  # pixels
            },

            # AVFoundation settings
            "avfoundation": {
                "enable_audio_processing": True,
                "supported_audio_formats": ["aac", "mp3", "wav"],
                "speech_recognition_locale": "en-US",
            },

            # Security framework integration
            "security": {
                "enable_keychain": True,
                "encryption_algorithm": "AES256",
                "secure_enclave_enabled": True,
            },

            # Memory management (macOS specific)
            "memory": {
                "enable_pressure_monitoring": True,
                "memory_warning_threshold": 0.8,  # 80%
                "enable_compression": True,
            },

            # Networking
            "networking": {
                "enable_urlsession": True,
                "timeout_interval": 30.0,
                "enable_background_sessions": True,
            },

            # File system
            "filesystem": {
                "enable_filecoordination": True,
                "enable_ubiquity": False,  # iCloud integration
                "temporary_directory": "/tmp/ucup",
            },

            # Performance monitoring
            "performance": {
                "enable_instruments": True,
                "metrics_interval": 1.0,  # seconds
                "enable_energy_monitoring": True,
            },

            # Logging
            "logging": {
                "level": "INFO",
                "enable_oslog": True,
                "enable_console": False,
            }
        })

    @classmethod
    def default(cls) -> 'UCUPConfig':
        """Create a default configuration instance."""
        return cls()

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'UCUPConfig':
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration JSON file

        Returns:
            UCUPConfig: Configured instance
        """
        config = cls()

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    config._merge_config(user_config)
            except (json.JSONDecodeError, IOError) as e:
                # Use default config if loading fails
                print(f"Warning: Failed to load config from {config_path}: {e}")

        # Apply macOS-specific overrides
        config._apply_macos_overrides()

        return config

    def _merge_config(self, user_config: Dict[str, Any]):
        """Merge user configuration with defaults."""
        def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        self.settings = deep_merge(self.settings, user_config)

    def _apply_macos_overrides(self):
        """Apply macOS-specific configuration overrides."""
        import platform

        # Check macOS version
        macos_version = platform.mac_ver()[0]
        if macos_version:
            major_version = int(macos_version.split('.')[0])

            # macOS 13+ has better Neural Engine support
            if major_version >= 13:
                self.settings["core_ml"]["compute_units"] = "neural_engine"

            # macOS 12+ has improved Vision framework
            if major_version >= 12:
                self.settings["vision"]["enable_high_accuracy"] = True

        # Check available hardware
        if hasattr(os, 'sysctl'):
            try:
                # Get CPU info for optimization
                cpu_count = os.cpu_count()
                if cpu_count and cpu_count > 8:
                    self.settings["gcd"]["max_concurrent_tasks"] = min(cpu_count * 2, 32)
            except:
                pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        keys = key.split('.')
        value = self.settings

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set a configuration value."""
        keys = key.split('.')
        config = self.settings

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, config_path: str):
        """Save configuration to file."""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(self.settings, f, indent=2)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
