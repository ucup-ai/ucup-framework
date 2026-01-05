"""
UCUP Memory Management System

Advanced memory management for UCUP framework with provider caching,
TTL-based expiration, and automatic cleanup routines.

Copyright (c) 2025 UCUP Framework Contributors
"""

import gc
import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics for monitoring."""

    heap_used: int
    heap_total: int
    heap_free: int
    usage_percent: float
    pressure_level: str = "low"  # low, medium, high, critical
    tracked_objects: int = 0
    cache_size: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata."""

    value: Any
    timestamp: datetime
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_estimate: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() - self.timestamp > timedelta(seconds=self.ttl)

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class MemoryMonitor:
    """Memory monitoring and pressure detection."""

    def __init__(self, update_interval: int = 30):
        self.stats = MemoryStats(0, 0, 0, 0.0)
        self.update_interval = update_interval
        self.pressure_callbacks: List[Callable[[MemoryStats], None]] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.tracked_objects: weakref.WeakSet = weakref.WeakSet()

    def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")

    def update_stats(self) -> MemoryStats:
        """Update and return current memory statistics."""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()

            # Get heap memory info (approximate)
            heap_used = mem_info.rss
            total_memory = psutil.virtual_memory().total
            heap_free = total_memory - heap_used
            usage_percent = (heap_used / total_memory) * 100

            # Determine pressure level
            if usage_percent >= 90:
                pressure_level = "critical"
            elif usage_percent >= 75:
                pressure_level = "high"
            elif usage_percent >= 60:
                pressure_level = "medium"
            else:
                pressure_level = "low"

            self.stats = MemoryStats(
                heap_used=heap_used,
                heap_total=total_memory,
                heap_free=heap_free,
                usage_percent=usage_percent,
                pressure_level=pressure_level,
                tracked_objects=len(self.tracked_objects),
                cache_size=0,  # Will be set by CacheManager
                last_updated=datetime.now(),
            )

            # Trigger pressure callbacks if needed
            if pressure_level in ["high", "critical"]:
                self._trigger_pressure_callbacks()

            return self.stats

        except Exception as e:
            logger.warning(f"Failed to update memory stats: {e}")
            return self.stats

    def track_object(self, obj: Any) -> None:
        """Track an object for memory monitoring."""
        self.tracked_objects.add(obj)

    def untrack_object(self, obj: Any) -> None:
        """Stop tracking an object."""
        self.tracked_objects.discard(obj)

    def on_memory_pressure(self, callback: Callable[[MemoryStats], None]) -> None:
        """Register a callback for memory pressure events."""
        self.pressure_callbacks.append(callback)

    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        before = len(gc.get_objects())
        collected = gc.collect()
        after = len(gc.get_objects())

        stats = {
            "objects_before": before,
            "objects_after": after,
            "objects_collected": collected,
            "collections_performed": gc.get_count()[0],
        }

        logger.info(f"Garbage collection: {stats}")
        return stats

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        stats = self.update_stats()

        return {
            "current_stats": {
                "heap_used_mb": stats.heap_used / 1024 / 1024,
                "heap_total_mb": stats.heap_total / 1024 / 1024,
                "usage_percent": stats.usage_percent,
                "pressure_level": stats.pressure_level,
                "tracked_objects": stats.tracked_objects,
            },
            "gc_stats": {
                "collections": gc.get_count(),
                "objects": len(gc.get_objects()),
                "garbage": len(gc.garbage),
            },
            "recommendations": self._get_memory_recommendations(),
        }

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                self.update_stats()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.update_interval)

    def _trigger_pressure_callbacks(self) -> None:
        """Trigger all registered pressure callbacks."""
        for callback in self.pressure_callbacks:
            try:
                callback(self.stats)
            except Exception as e:
                logger.error(f"Memory pressure callback failed: {e}")

    def _get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []

        if self.stats.usage_percent > 80:
            recommendations.append(
                "High memory usage detected. Consider reducing cache sizes."
            )
        if self.stats.pressure_level == "critical":
            recommendations.append(
                "Critical memory pressure. Force garbage collection or restart."
            )
        if len(self.tracked_objects) > 1000:
            recommendations.append(
                "Many objects are being tracked. Consider reducing provider caching."
            )

        return recommendations


class CacheManager:
    """TTL-based caching with automatic cleanup."""

    def __init__(self, cleanup_interval: int = 300):
        self.cache: Dict[str, CacheEntry] = {}
        self.cleanup_interval = cleanup_interval
        self.memory_monitor: Optional[MemoryMonitor] = None
        self.cleanup_thread: Optional[threading.Thread] = None
        self.stop_cleanup = threading.Event()

    def set_memory_monitor(self, monitor: MemoryMonitor) -> None:
        """Set memory monitor for integration."""
        self.memory_monitor = monitor

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set a value in cache with TTL."""
        size_estimate = self._estimate_size(value)

        entry = CacheEntry(
            value=value, timestamp=datetime.now(), ttl=ttl, size_estimate=size_estimate
        )

        self.cache[key] = entry

        # Track with memory monitor if available
        if self.memory_monitor:
            self.memory_monitor.track_object(value)

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        entry = self.cache.get(key)

        if entry is None:
            return None

        if entry.is_expired():
            self.delete(key)
            return None

        entry.touch()
        return entry.value

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        entry = self.cache.get(key)
        return entry is not None and not entry.is_expired()

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        entry = self.cache.pop(key, None)
        if entry and self.memory_monitor:
            self.memory_monitor.untrack_object(entry.value)
        return entry is not None

    def clear(self) -> int:
        """Clear all cache entries."""
        count = len(self.cache)
        for key in list(self.cache.keys()):
            self.delete(key)
        return count

    def cleanup(self) -> int:
        """Remove expired entries."""
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]

        for key in expired_keys:
            self.delete(key)

        if expired_keys:
            logger.info(f"Cache cleanup: removed {len(expired_keys)} expired entries")

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        total_size = sum(entry.size_estimate for entry in self.cache.values())
        active_entries = [
            entry for entry in self.cache.values() if not entry.is_expired()
        ]

        return {
            "total_entries": len(self.cache),
            "active_entries": len(active_entries),
            "expired_entries": len(self.cache) - len(active_entries),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "hit_rate": self._calculate_hit_rate(),
            "oldest_entry_age": self._get_oldest_entry_age(now),
            "newest_entry_age": self._get_newest_entry_age(now),
        }

    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        self.stop_cleanup.clear()
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Cache cleanup thread started")

    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        self.stop_cleanup.set()
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        logger.info("Cache cleanup thread stopped")

    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self.stop_cleanup.is_set():
            try:
                removed = self.cleanup()
                if removed > 0:
                    logger.debug(f"Periodic cleanup: removed {removed} entries")
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(self.cleanup_interval)

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of an object."""
        try:
            # Simple estimation based on string representation
            if hasattr(obj, "__dict__"):
                # For objects with __dict__
                size = len(str(obj.__dict__).encode("utf-8"))
            else:
                # For other objects
                size = len(str(obj).encode("utf-8"))

            # Add some overhead
            return max(size, 1024)  # Minimum 1KB estimate
        except:
            return 1024  # Default estimate

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        cache_hits = sum(1 for entry in self.cache.values() if entry.access_count > 0)

        if total_accesses == 0:
            return 0.0

        return (cache_hits / len(self.cache)) * 100

    def _get_oldest_entry_age(self, now: datetime) -> Optional[float]:
        """Get age of oldest entry in seconds."""
        if not self.cache:
            return None

        oldest = min(entry.timestamp for entry in self.cache.values())
        return (now - oldest).total_seconds()

    def _get_newest_entry_age(self, now: datetime) -> Optional[float]:
        """Get age of newest entry in seconds."""
        if not self.cache:
            return None

        newest = max(entry.timestamp for entry in self.cache.values())
        return (now - newest).total_seconds()


class ProviderManager:
    """Provider management with lazy loading and lifecycle management."""

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        memory_monitor: Optional[MemoryMonitor] = None,
    ):
        self.providers: Dict[str, Any] = {}
        self.provider_factories: Dict[str, Callable[[], Any]] = {}
        self.cache_manager = cache_manager or CacheManager()
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.disposal_callbacks: Dict[str, Callable[[Any], None]] = {}
        self.access_stats: Dict[str, Dict[str, Any]] = {}

    def register_provider(
        self,
        name: str,
        factory: Callable[[], Any],
        dispose_callback: Optional[Callable[[Any], None]] = None,
        cache_ttl: int = 300,
    ) -> None:
        """Register a provider with factory function."""
        self.provider_factories[name] = factory
        if dispose_callback:
            self.disposal_callbacks[name] = dispose_callback

        # Initialize access stats
        self.access_stats[name] = {
            "created_count": 0,
            "access_count": 0,
            "last_accessed": None,
            "cache_ttl": cache_ttl,
            "memory_usage": 0,
        }

    def get_provider(self, name: str, use_cache: bool = True) -> Any:
        """Get or create a provider instance."""
        # Try cache first if enabled
        if use_cache:
            cached = self.cache_manager.get(f"provider_{name}")
            if cached is not None:
                self.access_stats[name]["access_count"] += 1
                self.access_stats[name]["last_accessed"] = datetime.now()
                return cached

        # Check if provider is already created
        if name in self.providers:
            provider = self.providers[name]
            self.access_stats[name]["access_count"] += 1
            self.access_stats[name]["last_accessed"] = datetime.now()
            return provider

        # Create new provider
        if name not in self.provider_factories:
            raise ValueError(f"Provider '{name}' not registered")

        try:
            provider = self.provider_factories[name]()
            self.providers[name] = provider

            # Update stats
            self.access_stats[name]["created_count"] += 1
            self.access_stats[name]["last_accessed"] = datetime.now()

            # Track memory usage
            if self.memory_monitor:
                self.memory_monitor.track_object(provider)
                self.access_stats[name][
                    "memory_usage"
                ] = self.memory_monitor.stats.heap_used

            # Cache if enabled
            if use_cache and self.cache_manager:
                self.cache_manager.set(
                    f"provider_{name}", provider, self.access_stats[name]["cache_ttl"]
                )

            logger.info(f"Provider '{name}' created and cached")
            return provider

        except Exception as e:
            logger.error(f"Failed to create provider '{name}': {e}")
            raise

    def dispose_provider(self, name: str) -> bool:
        """Dispose of a specific provider."""
        provider = self.providers.pop(name, None)
        if provider:
            # Call disposal callback if registered
            if name in self.disposal_callbacks:
                try:
                    self.disposal_callbacks[name](provider)
                except Exception as e:
                    logger.error(f"Error disposing provider '{name}': {e}")

            # Remove from cache
            self.cache_manager.delete(f"provider_{name}")

            # Untrack from memory monitor
            if self.memory_monitor:
                self.memory_monitor.untrack_object(provider)

            logger.info(f"Provider '{name}' disposed")
            return True

        return False

    def dispose_all(self) -> int:
        """Dispose of all providers."""
        count = 0
        for name in list(self.providers.keys()):
            if self.dispose_provider(name):
                count += 1
        return count

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        return {
            name: {
                **stats,
                "is_loaded": name in self.providers,
                "is_cached": self.cache_manager.has(f"provider_{name}"),
            }
            for name, stats in self.access_stats.items()
        }

    def cleanup_expired_providers(self, max_age_seconds: int = 3600) -> int:
        """Clean up providers that haven't been accessed recently."""
        now = datetime.now()
        expired_providers = []

        for name, stats in self.access_stats.items():
            if (
                stats["last_accessed"]
                and (now - stats["last_accessed"]).total_seconds() > max_age_seconds
            ):
                expired_providers.append(name)

        for name in expired_providers:
            self.dispose_provider(name)

        if expired_providers:
            logger.info(f"Cleaned up {len(expired_providers)} expired providers")

        return len(expired_providers)


# Global instances for easy access
_memory_monitor = MemoryMonitor()
_cache_manager = CacheManager()
_provider_manager = ProviderManager(_cache_manager, _memory_monitor)

# Set up integration between components
_cache_manager.set_memory_monitor(_memory_monitor)
_memory_monitor.on_memory_pressure(
    lambda stats: _provider_manager.cleanup_expired_providers(300)
)

# Start background services
_memory_monitor.start_monitoring()
_cache_manager.start_cleanup_thread()


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    return _memory_monitor


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    return _cache_manager


def get_provider_manager() -> ProviderManager:
    """Get global provider manager instance."""
    return _provider_manager


def force_memory_cleanup() -> Dict[str, Any]:
    """Force comprehensive memory cleanup."""
    results = {
        "garbage_collected": _memory_monitor.force_gc(),
        "cache_cleaned": _cache_manager.cleanup(),
        "providers_cleaned": _provider_manager.cleanup_expired_providers(600),
    }

    logger.info(f"Memory cleanup completed: {results}")
    return results


def get_memory_report() -> Dict[str, Any]:
    """Get comprehensive memory report."""
    memory_stats = _memory_monitor.get_memory_report()
    cache_stats = _cache_manager.get_stats()
    provider_stats = _provider_manager.get_provider_stats()

    return {
        "memory": memory_stats,
        "cache": cache_stats,
        "providers": provider_stats,
        "summary": {
            "total_memory_mb": memory_stats["current_stats"]["heap_used_mb"],
            "memory_pressure": memory_stats["current_stats"]["pressure_level"],
            "cache_entries": cache_stats["active_entries"],
            "loaded_providers": sum(
                1 for p in provider_stats.values() if p["is_loaded"]
            ),
        },
    }


# Convenience functions for common operations
def cached_operation(
    cache_key: str, operation: Callable[[], Any], ttl: int = 300
) -> Any:
    """Execute an operation with caching."""
    cached_result = _cache_manager.get(cache_key)
    if cached_result is not None:
        return cached_result

    result = operation()
    _cache_manager.set(cache_key, result, ttl)
    return result


def monitored_operation(operation_name: str, operation: Callable[[], Any]) -> Any:
    """Execute an operation with memory monitoring."""
    start_memory = _memory_monitor.stats.heap_used
    start_time = time.time()

    try:
        result = operation()

        end_memory = _memory_monitor.stats.heap_used
        end_time = time.time()

        logger.debug(".2f" f"memory_delta={end_memory - start_memory} bytes")

        return result

    except Exception as e:
        logger.error(f"Operation '{operation_name}' failed: {e}")
        raise
