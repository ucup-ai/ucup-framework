"""
Distributed Computing Module

This module provides distributed computing capabilities for UCUP including
worker nodes, coordination protocols, consistent hashing, and distributed state storage.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""

    node_id: str
    host: str
    port: int
    capabilities: Set[str] = field(default_factory=set)
    load_factor: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def address(self) -> str:
        """Get the full address of the node."""
        return f"{self.host}:{self.port}"

    def is_alive(self, timeout_seconds: float = 30.0) -> bool:
        """Check if the node is alive based on heartbeat."""
        return (time.time() - self.last_heartbeat) < timeout_seconds

    def update_load(self, new_load: float) -> None:
        """Update the load factor of this node."""
        self.load_factor = max(0.0, min(1.0, new_load))
        self.last_heartbeat = time.time()

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "capabilities": list(self.capabilities),
            "load_factor": self.load_factor,
            "last_heartbeat": self.last_heartbeat,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerNode":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            capabilities=set(data.get("capabilities", [])),
            load_factor=data.get("load_factor", 0.0),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            status=data.get("status", "active"),
            metadata=data.get("metadata", {}),
        )


class DistributedCoordinatorProtocol(ABC):
    """Abstract base class for distributed coordination protocols."""

    @abstractmethod
    async def register_worker(self, worker: WorkerNode) -> bool:
        """Register a new worker node."""
        pass

    @abstractmethod
    async def unregister_worker(self, node_id: str) -> bool:
        """Unregister a worker node."""
        pass

    @abstractmethod
    async def get_worker(self, node_id: str) -> Optional[WorkerNode]:
        """Get information about a specific worker."""
        pass

    @abstractmethod
    async def list_workers(
        self, capability_filter: Optional[str] = None
    ) -> List[WorkerNode]:
        """List all registered workers, optionally filtered by capability."""
        pass

    @abstractmethod
    async def assign_task(
        self, task_data: Dict[str, Any], preferred_worker: Optional[str] = None
    ) -> Optional[str]:
        """Assign a task to a worker node."""
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task."""
        pass

    @abstractmethod
    async def broadcast_message(
        self, message: Dict[str, Any], target_workers: Optional[List[str]] = None
    ) -> None:
        """Broadcast a message to workers."""
        pass

    @abstractmethod
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about the cluster."""
        pass


class ConsistentHashRing:
    """
    Consistent hashing ring for distributed key-value storage and load balancing.

    Provides efficient distribution of keys across nodes with minimal redistribution
    when nodes are added or removed.
    """

    def __init__(self, replicas: int = 3):
        self.replicas = replicas
        self.ring: OrderedDict[int, str] = OrderedDict()  # hash -> node_id
        self.nodes: Dict[str, WorkerNode] = {}  # node_id -> node_info
        self._keys: Dict[str, str] = {}  # key -> assigned_node

    def add_node(self, node: WorkerNode) -> None:
        """Add a node to the hash ring."""
        self.nodes[node.node_id] = node

        # Add virtual nodes (replicas) to the ring
        for i in range(self.replicas):
            key = f"{node.node_id}:{i}"
            hash_value = self._hash(key)
            self.ring[hash_value] = node.node_id

        # Sort the ring
        self.ring = OrderedDict(sorted(self.ring.items()))
        logger.info(f"Added node {node.node_id} to hash ring")

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the hash ring."""
        if node_id not in self.nodes:
            return

        # Remove all replicas of this node
        keys_to_remove = []
        for hash_value, nid in self.ring.items():
            if nid == node_id:
                keys_to_remove.append(hash_value)

        for key in keys_to_remove:
            del self.ring[key]

        del self.nodes[node_id]

        # Clear key assignments that were on this node
        keys_to_reassign = [k for k, n in self._keys.items() if n == node_id]
        for key in keys_to_reassign:
            del self._keys[key]

        logger.info(f"Removed node {node_id} from hash ring")

    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key."""
        if not self.ring:
            return None

        hash_value = self._hash(key)

        # Find the first node with hash >= key_hash
        for ring_hash in self.ring:
            if ring_hash >= hash_value:
                node_id = self.ring[ring_hash]
                if node_id in self.nodes and self.nodes[node_id].is_alive():
                    self._keys[key] = node_id
                    return node_id

        # If no node found, wrap around to first node
        if self.ring:
            node_id = next(iter(self.ring.values()))
            if node_id in self.nodes and self.nodes[node_id].is_alive():
                self._keys[key] = node_id
                return node_id

        return None

    def get_nodes_for_key(self, key: str, count: int = 1) -> List[str]:
        """Get multiple nodes for a key (for replication)."""
        if not self.ring:
            return []

        hash_value = self._hash(key)
        result = []

        # Start from the key's position and collect nodes
        ring_items = list(self.ring.items())

        for i, (ring_hash, node_id) in enumerate(ring_items):
            if (
                ring_hash >= hash_value or not result
            ):  # Include first valid node even if hash < key_hash
                if node_id in self.nodes and self.nodes[node_id].is_alive():
                    if node_id not in result:  # Avoid duplicates
                        result.append(node_id)
                        if len(result) >= count:
                            break

        # If we need more nodes, wrap around
        if len(result) < count:
            for ring_hash, node_id in ring_items:
                if node_id in self.nodes and self.nodes[node_id].is_alive():
                    if node_id not in result:
                        result.append(node_id)
                        if len(result) >= count:
                            break

        return result[:count]

    def get_node_loads(self) -> Dict[str, float]:
        """Get load factors for all nodes."""
        return {node_id: node.load_factor for node_id, node in self.nodes.items()}

    def update_node_load(self, node_id: str, load: float) -> None:
        """Update the load factor for a node."""
        if node_id in self.nodes:
            self.nodes[node_id].update_load(load)

    def get_least_loaded_node(self) -> Optional[str]:
        """Get the least loaded node."""
        if not self.nodes:
            return None

        min_load = float("inf")
        selected_node = None

        for node_id, node in self.nodes.items():
            if node.is_alive() and node.load_factor < min_load:
                min_load = node.load_factor
                selected_node = node_id

        return selected_node

    def _hash(self, key: str) -> int:
        """Generate hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def get_ring_info(self) -> Dict[str, Any]:
        """Get information about the hash ring."""
        return {
            "total_nodes": len(self.nodes),
            "ring_size": len(self.ring),
            "replicas": self.replicas,
            "node_loads": self.get_node_loads(),
            "keys_assigned": len(self._keys),
        }


class DistributedStateStore(ABC):
    """Abstract base class for distributed state storage."""

    @abstractmethod
    async def store(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store a key-value pair."""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key-value pair."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass


class InMemoryDistributedStore(DistributedStateStore):
    """
    In-memory implementation of distributed state store.

    Suitable for single-node testing or small clusters.
    """

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def store(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store a key-value pair."""
        async with self.lock:
            self.store[key] = value
            self.metadata[key] = {
                "stored_at": time.time(),
                "ttl": ttl_seconds,
                "size": self._estimate_size(value),
            }

            # Schedule TTL cleanup if needed
            if ttl_seconds:
                asyncio.create_task(self._schedule_cleanup(key, ttl_seconds))

            return True

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        async with self.lock:
            if key not in self.store:
                return None

            # Check TTL
            if self._is_expired(key):
                await self.delete(key)
                return None

            return self.store[key]

    async def delete(self, key: str) -> bool:
        """Delete a key-value pair."""
        async with self.lock:
            if key in self.store:
                del self.store[key]
                if key in self.metadata:
                    del self.metadata[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        async with self.lock:
            if key not in self.store:
                return False
            if self._is_expired(key):
                await self.delete(key)
                return False
            return True

    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        async with self.lock:
            keys = [k for k in self.store.keys() if k.startswith(prefix)]
            # Filter out expired keys
            valid_keys = []
            for key in keys:
                if not self._is_expired(key):
                    valid_keys.append(key)
                else:
                    await self.delete(key)
            return valid_keys

    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        async with self.lock:
            total_size = sum(meta.get("size", 0) for meta in self.metadata.values())
            expired_count = sum(1 for key in self.store if self._is_expired(key))

            return {
                "total_keys": len(self.store),
                "total_size_bytes": total_size,
                "expired_keys": expired_count,
                "storage_type": "in_memory",
            }

    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired."""
        if key not in self.metadata:
            return False

        meta = self.metadata[key]
        ttl = meta.get("ttl")
        if ttl is None:
            return False

        stored_at = meta.get("stored_at", 0)
        return (time.time() - stored_at) > ttl

    async def _schedule_cleanup(self, key: str, ttl_seconds: int) -> None:
        """Schedule cleanup of an expired key."""
        await asyncio.sleep(ttl_seconds)
        await self.delete(key)

    def _estimate_size(self, obj: Any) -> int:
        """Estimate the size of an object."""
        try:
            import sys

            return sys.getsizeof(obj)
        except:
            return 1000  # Default estimate


class SimpleDistributedCoordinator(DistributedCoordinatorProtocol):
    """
    Simple distributed coordinator using consistent hashing and in-memory storage.

    Suitable for development and small-scale distributed deployments.
    """

    def __init__(self):
        self.hash_ring = ConsistentHashRing()
        self.state_store = InMemoryDistributedStore()
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.workers: Dict[str, WorkerNode] = {}
        self.task_counter = 0
        self.lock = asyncio.Lock()

    async def register_worker(self, worker: WorkerNode) -> bool:
        """Register a new worker node."""
        async with self.lock:
            if worker.node_id in self.workers:
                return False

            self.workers[worker.node_id] = worker
            self.hash_ring.add_node(worker)

            # Store worker info in distributed store
            await self.state_store.store(f"worker:{worker.node_id}", worker.to_dict())

            logger.info(f"Registered worker {worker.node_id} at {worker.address}")
            return True

    async def unregister_worker(self, node_id: str) -> bool:
        """Unregister a worker node."""
        async with self.lock:
            if node_id not in self.workers:
                return False

            self.hash_ring.remove_node(node_id)
            del self.workers[node_id]

            # Remove from distributed store
            await self.state_store.delete(f"worker:{node_id}")

            logger.info(f"Unregistered worker {node_id}")
            return True

    async def get_worker(self, node_id: str) -> Optional[WorkerNode]:
        """Get information about a specific worker."""
        async with self.lock:
            worker_data = await self.state_store.retrieve(f"worker:{node_id}")
            if worker_data:
                return WorkerNode.from_dict(worker_data)
            return None

    async def list_workers(
        self, capability_filter: Optional[str] = None
    ) -> List[WorkerNode]:
        """List all registered workers, optionally filtered by capability."""
        async with self.lock:
            workers = []
            keys = await self.state_store.list_keys("worker:")

            for key in keys:
                node_id = key.replace("worker:", "")
                worker_data = await self.state_store.retrieve(key)
                if worker_data:
                    worker = WorkerNode.from_dict(worker_data)
                    if (
                        capability_filter is None
                        or capability_filter in worker.capabilities
                    ):
                        workers.append(worker)

            return workers

    async def assign_task(
        self, task_data: Dict[str, Any], preferred_worker: Optional[str] = None
    ) -> Optional[str]:
        """Assign a task to a worker node."""
        async with self.lock:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"

            # Determine target worker
            if preferred_worker and preferred_worker in self.workers:
                target_worker = preferred_worker
            else:
                # Use hash ring to assign based on task properties
                task_key = task_data.get("key", task_id)
                target_worker = self.hash_ring.get_node(task_key)

            if not target_worker:
                logger.warning("No available workers for task assignment")
                return None

            # Create task record
            task_record = {
                "task_id": task_id,
                "task_data": task_data,
                "assigned_worker": target_worker,
                "status": "assigned",
                "created_at": time.time(),
                "updated_at": time.time(),
            }

            self.tasks[task_id] = task_record

            # Store task in distributed store
            await self.state_store.store(f"task:{task_id}", task_record)

            # Update worker load
            if target_worker in self.workers:
                current_load = self.workers[target_worker].load_factor
                self.hash_ring.update_node_load(target_worker, current_load + 0.1)

            logger.info(f"Assigned task {task_id} to worker {target_worker}")
            return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task."""
        async with self.lock:
            # Check local cache first
            if task_id in self.tasks:
                return self.tasks[task_id]

            # Check distributed store
            task_data = await self.state_store.retrieve(f"task:{task_id}")
            if task_data:
                self.tasks[task_id] = task_data
                return task_data

            return None

    async def broadcast_message(
        self, message: Dict[str, Any], target_workers: Optional[List[str]] = None
    ) -> None:
        """Broadcast a message to workers."""
        if target_workers is None:
            target_workers = list(self.workers.keys())

        message["timestamp"] = time.time()
        message["sender"] = "coordinator"

        # Store message for each target worker
        for worker_id in target_workers:
            if worker_id in self.workers:
                message_key = f"message:{worker_id}:{int(time.time() * 1000)}"
                await self.state_store.store(
                    message_key, message, ttl_seconds=300
                )  # 5 minute TTL

        logger.debug(f"Broadcasted message to {len(target_workers)} workers")

    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about the cluster."""
        async with self.lock:
            active_workers = [w for w in self.workers.values() if w.is_alive()]
            total_load = sum(w.load_factor for w in active_workers)
            avg_load = total_load / len(active_workers) if active_workers else 0

            store_stats = await self.state_store.get_stats()
            ring_stats = self.hash_ring.get_ring_info()

            return {
                "total_workers": len(self.workers),
                "active_workers": len(active_workers),
                "average_load": avg_load,
                "total_tasks": len(self.tasks),
                "hash_ring": ring_stats,
                "storage": store_stats,
            }
