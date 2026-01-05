"""
Memory Management Module

This module provides memory optimization capabilities for UCUP agents including
memory pools, state compression, and lazy loading mechanisms.
"""

import asyncio
import gc
import hashlib
import logging
import lzma
import pickle
import threading
import time
import weakref
import zlib
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar

import psutil

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_allocated: int = 0
    active_objects: int = 0
    compressed_objects: int = 0
    memory_pressure: float = 0.0
    gc_collections: int = 0
    last_gc_time: float = 0.0


@dataclass
class MemoryPoolConfig:
    """Configuration for memory pool."""

    max_size_mb: int = 100
    compression_threshold_kb: int = 50
    gc_interval_seconds: int = 60
    max_compression_ratio: float = 0.8
    enable_lazy_loading: bool = True


class AgentMemoryPool:
    """
    Memory pool for managing agent states with automatic compression and cleanup.

    Provides efficient memory management for large numbers of agent instances.
    """

    def __init__(self, config: Optional[MemoryPoolConfig] = None):
        self.config = config or MemoryPoolConfig()
        self.pool: OrderedDict[str, Any] = OrderedDict()  # agent_id -> agent_state
        self.compressed_pool: Dict[str, bytes] = {}  # agent_id -> compressed_state
        self.metadata: Dict[str, Dict[str, Any]] = {}  # agent_id -> metadata
        self.access_times: Dict[str, float] = {}  # agent_id -> last_access_time
        self.lock = threading.RLock()
        self.stats = MemoryStats()
        self.gc_thread: Optional[threading.Thread] = None
        self.running = False

        # Compression and serialization
        self.compressor = StateCompressor()
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="memory_pool"
        )

        # Start background GC if configured
        if self.config.gc_interval_seconds > 0:
            self._start_background_gc()

    def store(
        self, agent_id: str, agent_state: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store an agent state in the pool.

        Applies compression if the state exceeds the threshold.
        """
        with self.lock:
            state_size = self._estimate_size(agent_state)
            should_compress = state_size > (self.config.compression_threshold_kb * 1024)

            # Check if we need to evict old items
            self._ensure_capacity(state_size)

            # Store state
            if should_compress:
                # Compress in background thread
                future = self.executor.submit(self.compressor.compress, agent_state)
                try:
                    compressed_state = future.result(timeout=5.0)
                    self.compressed_pool[agent_id] = compressed_state
                    self.pool[agent_id] = None  # Placeholder
                    self.stats.compressed_objects += 1
                except Exception as e:
                    logger.warning(f"Compression failed for {agent_id}: {e}")
                    self.pool[agent_id] = agent_state
            else:
                self.pool[agent_id] = agent_state

            # Update metadata
            self.metadata[agent_id] = metadata or {}
            self.metadata[agent_id].update(
                {
                    "size_bytes": state_size,
                    "compressed": should_compress,
                    "stored_at": time.time(),
                }
            )

            self.access_times[agent_id] = time.time()
            self.stats.total_allocated += state_size
            self.stats.active_objects += 1

            logger.debug(
                f"Stored agent {agent_id} ({state_size} bytes, compressed={should_compress})"
            )

    def retrieve(self, agent_id: str) -> Optional[Any]:
        """Retrieve an agent state from the pool."""
        with self.lock:
            if agent_id not in self.pool and agent_id not in self.compressed_pool:
                return None

            self.access_times[agent_id] = time.time()

            # Check if state is compressed
            if agent_id in self.compressed_pool:
                # Decompress in background thread
                future = self.executor.submit(
                    self.compressor.decompress, self.compressed_pool[agent_id]
                )
                try:
                    agent_state = future.result(timeout=5.0)
                    # Move to active pool
                    self.pool[agent_id] = agent_state
                    del self.compressed_pool[agent_id]
                    self.stats.compressed_objects -= 1
                    logger.debug(f"Decompressed agent {agent_id}")
                except Exception as e:
                    logger.error(f"Decompression failed for {agent_id}: {e}")
                    return None
            else:
                agent_state = self.pool[agent_id]

            return agent_state

    def remove(self, agent_id: str) -> bool:
        """Remove an agent state from the pool."""
        with self.lock:
            removed = False
            size_removed = 0

            if agent_id in self.pool:
                state = self.pool[agent_id]
                if state is not None:
                    size_removed = self._estimate_size(state)
                del self.pool[agent_id]
                removed = True

            if agent_id in self.compressed_pool:
                # Estimate compressed size (rough approximation)
                size_removed = (
                    len(self.compressed_pool[agent_id]) * 3
                )  # Assume 3:1 compression ratio
                del self.compressed_pool[agent_id]
                self.stats.compressed_objects -= 1
                removed = True

            if removed:
                if agent_id in self.metadata:
                    del self.metadata[agent_id]
                if agent_id in self.access_times:
                    del self.access_times[agent_id]

                self.stats.total_allocated -= size_removed
                self.stats.active_objects -= 1

                logger.debug(f"Removed agent {agent_id}")
                return True

            return False

    def _ensure_capacity(self, required_size: int) -> None:
        """Ensure there's enough capacity for a new item."""
        max_size_bytes = self.config.max_size_mb * 1024 * 1024
        current_size = self._get_current_size()

        if current_size + required_size <= max_size_bytes:
            return

        # Need to evict items (LRU - least recently used)
        items_to_remove = []
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])

        for agent_id, _ in sorted_items:
            if current_size + required_size <= max_size_bytes:
                break

            if agent_id in self.pool:
                state = self.pool[agent_id]
                if state is not None:
                    current_size -= self._estimate_size(state)
                items_to_remove.append(agent_id)
            elif agent_id in self.compressed_pool:
                # Rough estimation
                current_size -= len(self.compressed_pool[agent_id]) * 3
                items_to_remove.append(agent_id)

        # Remove items
        for agent_id in items_to_remove:
            self.remove(agent_id)
            logger.debug(f"Evicted agent {agent_id} due to memory pressure")

    def _get_current_size(self) -> int:
        """Get current memory usage of the pool."""
        total = 0
        for agent_id in self.pool:
            state = self.pool[agent_id]
            if state is not None:
                total += self._estimate_size(state)

        # Add compressed objects (rough estimation)
        total += sum(len(data) * 3 for data in self.compressed_pool.values())

        return total

    def _estimate_size(self, obj: Any) -> int:
        """Estimate the memory size of an object."""
        try:
            # Use sys.getsizeof for basic estimation
            import sys

            size = sys.getsizeof(obj)

            # Add sizes of contained objects for common types
            if isinstance(obj, dict):
                size += sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in obj.items()
                )
            elif isinstance(obj, (list, tuple)):
                size += sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, set):
                size += sum(self._estimate_size(item) for item in obj)

            return size
        except Exception:
            # Fallback estimation
            return 1000  # 1KB default

    def _start_background_gc(self) -> None:
        """Start background garbage collection thread."""
        self.running = True
        self.gc_thread = threading.Thread(target=self._background_gc_loop, daemon=True)
        self.gc_thread.start()

    def _background_gc_loop(self) -> None:
        """Background garbage collection loop."""
        while self.running:
            try:
                time.sleep(self.config.gc_interval_seconds)
                self._perform_gc()
            except Exception as e:
                logger.error(f"Error in background GC: {e}")

    def _perform_gc(self) -> None:
        """Perform garbage collection and memory cleanup."""
        start_time = time.time()

        # Python GC
        collected = gc.collect()

        # Check memory pressure
        memory = psutil.virtual_memory()
        memory_pressure = memory.percent / 100.0

        # Compress old/unused objects if memory pressure is high
        if memory_pressure > 0.8:
            self._compress_old_objects()

        self.stats.memory_pressure = memory_pressure
        self.stats.gc_collections += 1
        self.stats.last_gc_time = time.time() - start_time

        logger.debug(
            f"GC completed: collected {collected} objects, memory pressure {memory_pressure:.2f}"
        )

    def _compress_old_objects(self) -> None:
        """Compress old/unused objects to free memory."""
        current_time = time.time()
        max_age = 300  # 5 minutes

        candidates = []
        for agent_id, last_access in self.access_times.items():
            if (current_time - last_access) > max_age and agent_id in self.pool:
                state = self.pool[agent_id]
                if state is not None and not self.metadata.get(agent_id, {}).get(
                    "compressed", False
                ):
                    candidates.append((agent_id, state))

        # Compress candidates
        for agent_id, state in candidates:
            try:
                compressed = self.compressor.compress(state)
                self.compressed_pool[agent_id] = compressed
                self.pool[agent_id] = None
                self.stats.compressed_objects += 1
                logger.debug(f"Compressed old agent {agent_id}")
            except Exception as e:
                logger.warning(f"Failed to compress agent {agent_id}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        return {
            "active_objects": self.stats.active_objects,
            "compressed_objects": self.stats.compressed_objects,
            "total_allocated_mb": self.stats.total_allocated / (1024 * 1024),
            "memory_pressure": self.stats.memory_pressure,
            "gc_collections": self.stats.gc_collections,
            "pool_size": len(self.pool),
            "compressed_pool_size": len(self.compressed_pool),
            "config": {
                "max_size_mb": self.config.max_size_mb,
                "compression_threshold_kb": self.config.compression_threshold_kb,
                "gc_interval_seconds": self.config.gc_interval_seconds,
            },
        }

    def shutdown(self) -> None:
        """Shutdown the memory pool."""
        self.running = False
        if self.gc_thread:
            self.gc_thread.join(timeout=5.0)

        self.executor.shutdown(wait=True)
        self.pool.clear()
        self.compressed_pool.clear()
        self.metadata.clear()
        self.access_times.clear()

        logger.info("AgentMemoryPool shutdown complete")


class StateCompressor:
    """
    Compression utility for agent states.

    Supports multiple compression algorithms and automatic algorithm selection.
    """

    def __init__(self, default_algorithm: str = "lzma"):
        self.default_algorithm = default_algorithm
        self.algorithms = {
            "zlib": self._compress_zlib,
            "lzma": self._compress_lzma,
            "none": self._compress_none,
        }
        self.decompress_algorithms = {
            "zlib": self._decompress_zlib,
            "lzma": self._decompress_lzma,
            "none": self._decompress_none,
        }

    def compress(self, data: Any) -> bytes:
        """Compress data using the best available algorithm."""
        # Serialize data first
        serialized = self._serialize(data)

        # Try different algorithms and pick the best
        best_compressed = serialized
        best_ratio = 1.0
        best_algorithm = "none"

        for algorithm_name, compress_func in self.algorithms.items():
            try:
                compressed = compress_func(serialized)
                ratio = (
                    len(compressed) / len(serialized) if len(serialized) > 0 else 1.0
                )

                if ratio < best_ratio:
                    best_compressed = compressed
                    best_ratio = ratio
                    best_algorithm = algorithm_name
            except Exception as e:
                logger.debug(f"Compression algorithm {algorithm_name} failed: {e}")
                continue

        # Store algorithm in the compressed data
        algorithm_byte = best_algorithm.encode("utf-8")[
            :1
        ]  # First byte of algorithm name
        result = algorithm_byte + best_compressed

        logger.debug(
            f"Compressed data: {len(serialized)} -> {len(result)} bytes "
            f"(ratio: {best_ratio:.2f}, algorithm: {best_algorithm})"
        )

        return result

    def decompress(self, data: bytes) -> Any:
        """Decompress data."""
        if len(data) < 1:
            raise ValueError("Compressed data too short")

        # Extract algorithm
        algorithm_byte = data[0:1].decode("utf-8", errors="ignore")
        algorithm = "lzma"  # default fallback

        if algorithm_byte == "z":
            algorithm = "zlib"
        elif algorithm_byte == "l":
            algorithm = "lzma"
        elif algorithm_byte == "n":
            algorithm = "none"

        compressed_data = data[1:]

        # Decompress
        decompressed = self.decompress_algorithms[algorithm](compressed_data)

        # Deserialize
        return self._deserialize(decompressed)

    def _serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from bytes."""
        return pickle.loads(data)

    def _compress_zlib(self, data: bytes) -> bytes:
        """Compress using zlib."""
        return zlib.compress(data, level=9)

    def _decompress_zlib(self, data: bytes) -> bytes:
        """Decompress using zlib."""
        return zlib.decompress(data)

    def _compress_lzma(self, data: bytes) -> bytes:
        """Compress using LZMA."""
        return lzma.compress(data, preset=9)

    def _decompress_lzma(self, data: bytes) -> bytes:
        """Decompress using LZMA."""
        return lzma.decompress(data)

    def _compress_none(self, data: bytes) -> bytes:
        """No compression."""
        return data

    def _decompress_none(self, data: bytes) -> bytes:
        """No decompression."""
        return data


class LazyAgentLoader:
    """
    Lazy loading system for agent states.

    Loads agents on-demand from persistent storage to reduce memory usage.
    """

    def __init__(
        self,
        storage_path: str = "./agent_storage",
        memory_pool: Optional[AgentMemoryPool] = None,
    ):
        self.storage_path = storage_path
        self.memory_pool = memory_pool or AgentMemoryPool()
        self.loaded_agents: Dict[str, weakref.ReferenceType] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="lazy_loader"
        )

        # Ensure storage directory exists
        import os

        os.makedirs(storage_path, exist_ok=True)

    def store_agent(
        self, agent_id: str, agent: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store an agent to persistent storage."""
        # Store in memory pool first
        self.memory_pool.store(agent_id, agent, metadata)

        # Also persist to disk
        self._persist_agent(agent_id, agent, metadata)

        # Create weak reference for tracking
        self.loaded_agents[agent_id] = weakref.ref(
            agent, lambda ref: self._on_agent_garbage_collected(agent_id)
        )

        logger.debug(f"Stored agent {agent_id} with lazy loading enabled")

    def load_agent(self, agent_id: str) -> Optional[Any]:
        """Load an agent, using memory pool cache if available."""
        # Check memory pool first
        agent = self.memory_pool.retrieve(agent_id)
        if agent is not None:
            # Update weak reference
            self.loaded_agents[agent_id] = weakref.ref(
                agent, lambda ref: self._on_agent_garbage_collected(agent_id)
            )
            return agent

        # Load from persistent storage
        agent = self._load_agent_from_storage(agent_id)
        if agent is not None:
            # Store in memory pool for future use
            metadata = self.agent_metadata.get(agent_id, {})
            self.memory_pool.store(agent_id, agent, metadata)

            # Create weak reference
            self.loaded_agents[agent_id] = weakref.ref(
                agent, lambda ref: self._on_agent_garbage_collected(agent_id)
            )

            logger.debug(f"Lazy loaded agent {agent_id} from storage")
            return agent

        return None

    def unload_agent(self, agent_id: str) -> bool:
        """Unload an agent from memory (but keep in persistent storage)."""
        # Remove from memory pool
        removed = self.memory_pool.remove(agent_id)

        # Clear weak reference
        if agent_id in self.loaded_agents:
            del self.loaded_agents[agent_id]

        if removed:
            logger.debug(f"Unloaded agent {agent_id} from memory")
        return removed

    def is_agent_loaded(self, agent_id: str) -> bool:
        """Check if an agent is currently loaded in memory."""
        if agent_id in self.loaded_agents:
            ref = self.loaded_agents[agent_id]
            return ref() is not None
        return False

    def get_agent_metadata(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an agent."""
        return self.agent_metadata.get(agent_id)

    def list_agents(self) -> List[str]:
        """List all stored agent IDs."""
        import glob
        import os

        pattern = os.path.join(self.storage_path, "*.agent")
        files = glob.glob(pattern)
        agent_ids = []

        for file_path in files:
            filename = os.path.basename(file_path)
            if filename.endswith(".agent"):
                agent_id = filename[:-6]  # Remove .agent extension
                agent_ids.append(agent_id)

        return agent_ids

    def _persist_agent(
        self, agent_id: str, agent: Any, metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Persist an agent to disk storage."""
        import os

        filename = f"{agent_id}.agent"
        filepath = os.path.join(self.storage_path, filename)

        data = {
            "agent": agent,
            "metadata": metadata or {},
            "stored_at": time.time(),
            "version": "1.0",
        }

        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Persisted agent {agent_id} to {filepath}")
        except Exception as e:
            logger.error(f"Failed to persist agent {agent_id}: {e}")

    def _load_agent_from_storage(self, agent_id: str) -> Optional[Any]:
        """Load an agent from persistent storage."""
        import os

        filename = f"{agent_id}.agent"
        filepath = os.path.join(self.storage_path, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            agent = data["agent"]
            self.agent_metadata[agent_id] = data.get("metadata", {})

            logger.debug(f"Loaded agent {agent_id} from {filepath}")
            return agent

        except Exception as e:
            logger.error(f"Failed to load agent {agent_id}: {e}")
            return None

    def _on_agent_garbage_collected(self, agent_id: str) -> None:
        """Callback when an agent is garbage collected."""
        logger.debug(f"Agent {agent_id} was garbage collected")
        if agent_id in self.loaded_agents:
            del self.loaded_agents[agent_id]

    def cleanup_storage(self, max_age_days: int = 30) -> int:
        """Clean up old agent files from storage."""
        import glob
        import os

        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0

        pattern = os.path.join(self.storage_path, "*.agent")
        files = glob.glob(pattern)

        for file_path in files:
            try:
                stat = os.stat(file_path)
                if stat.st_mtime < cutoff_time:
                    os.remove(file_path)
                    cleaned_count += 1
                    logger.debug(f"Cleaned up old agent file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

        return cleaned_count

    def get_stats(self) -> Dict[str, Any]:
        """Get lazy loader statistics."""
        loaded_count = sum(
            1 for ref in self.loaded_agents.values() if ref() is not None
        )
        stored_count = len(self.list_agents())

        return {
            "loaded_agents": loaded_count,
            "stored_agents": stored_count,
            "memory_pool_stats": self.memory_pool.get_stats(),
            "storage_path": self.storage_path,
        }

    def shutdown(self) -> None:
        """Shutdown the lazy loader."""
        self.memory_pool.shutdown()
        self.executor.shutdown(wait=True)
        self.loaded_agents.clear()
        self.agent_metadata.clear()

        logger.info("LazyAgentLoader shutdown complete")
