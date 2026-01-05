"""
Self-Healing Coordinators Module

This module provides self-healing capabilities for UCUP coordinators including
health monitoring, automatic recovery, failover mechanisms, and state restoration.
"""

import asyncio
import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

import psutil

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RecoveryStrategy(Enum):
    """Types of recovery strategies."""

    RESTART = "restart"
    FAILOVER = "failover"
    SCALE_UP = "scale_up"
    STATE_RECOVERY = "state_recovery"
    DEGRADATION = "degradation"


@dataclass
class CoordinatorHealthMetrics:
    """Health metrics for coordinators."""

    coordinator_id: str
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0  # tasks per second
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    queue_depth: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    health_score: float = 1.0  # 0.0 (unhealthy) to 1.0 (healthy)

    def update_from_system(self) -> None:
        """Update metrics from system resources."""
        try:
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            self.cpu_usage_percent = process.cpu_percent()
            self.last_heartbeat = time.time()
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def calculate_health_score(self) -> float:
        """Calculate overall health score based on metrics."""
        # Weighted scoring based on different metrics
        weights = {
            "response_time": 0.2,
            "error_rate": 0.3,
            "memory_usage": 0.15,
            "cpu_usage": 0.15,
            "queue_depth": 0.1,
            "connections": 0.1,
        }

        scores = {
            "response_time": max(
                0, 1.0 - (self.response_time_ms / 5000.0)
            ),  # 5s threshold
            "error_rate": max(0, 1.0 - self.error_rate),  # Error rate penalty
            "memory_usage": max(
                0, 1.0 - (self.memory_usage_mb / 1000.0)
            ),  # 1GB threshold
            "cpu_usage": max(0, 1.0 - (self.cpu_usage_percent / 90.0)),  # 90% threshold
            "queue_depth": max(
                0, 1.0 - (self.queue_depth / 1000.0)
            ),  # 1000 queue threshold
            "connections": min(
                1.0, self.active_connections / 100.0
            ),  # More connections = healthier
        }

        self.health_score = sum(weights[k] * scores[k] for k in weights.keys())
        return self.health_score

    def is_healthy(self, threshold: float = 0.7) -> bool:
        """Check if coordinator is healthy."""
        return self.calculate_health_score() >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "coordinator_id": self.coordinator_id,
            "response_time_ms": self.response_time_ms,
            "error_rate": self.error_rate,
            "throughput": self.throughput,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "active_connections": self.active_connections,
            "queue_depth": self.queue_depth,
            "last_heartbeat": self.last_heartbeat,
            "health_score": self.health_score,
        }


@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken."""

    action_id: str
    strategy: RecoveryStrategy
    coordinator_id: str
    priority: int = 1  # 1 (low) to 5 (critical)
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    executed_at: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None

    def mark_executed(self, success: bool, error_message: Optional[str] = None) -> None:
        """Mark the action as executed."""
        self.executed_at = time.time()
        self.success = success
        self.error_message = error_message

    def is_pending(self) -> bool:
        """Check if action is still pending."""
        return self.executed_at is None

    def is_successful(self) -> bool:
        """Check if action was successful."""
        return self.success is True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "strategy": self.strategy.value,
            "coordinator_id": self.coordinator_id,
            "priority": self.priority,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "executed_at": self.executed_at,
            "success": self.success,
            "error_message": self.error_message,
        }


class CoordinatorHealthMonitor:
    """
    Monitors coordinator health and triggers recovery actions.
    """

    def __init__(self, check_interval: float = 30.0, unhealthy_threshold: float = 0.6):
        self.coordinators: Dict[str, CoordinatorHealthMetrics] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.recovery_callbacks: List[Callable[[RecoveryAction], None]] = []
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Health history for trend analysis
        self.health_history: Dict[
            str, List[Tuple[float, float]]
        ] = {}  # coordinator_id -> [(timestamp, health_score)]

    def register_coordinator(
        self,
        coordinator_id: str,
        initial_metrics: Optional[CoordinatorHealthMetrics] = None,
    ) -> None:
        """Register a coordinator for monitoring."""
        if initial_metrics:
            self.coordinators[coordinator_id] = initial_metrics
        else:
            self.coordinators[coordinator_id] = CoordinatorHealthMetrics(
                coordinator_id=coordinator_id
            )

        self.health_history[coordinator_id] = []
        logger.info(f"Registered coordinator {coordinator_id} for health monitoring")

    def unregister_coordinator(self, coordinator_id: str) -> None:
        """Unregister a coordinator from monitoring."""
        if coordinator_id in self.coordinators:
            del self.coordinators[coordinator_id]
        if coordinator_id in self.health_history:
            del self.health_history[coordinator_id]
        if coordinator_id in self.monitoring_tasks:
            task = self.monitoring_tasks[coordinator_id]
            task.cancel()
            del self.monitoring_tasks[coordinator_id]

        logger.info(f"Unregistered coordinator {coordinator_id} from health monitoring")

    def update_metrics(
        self, coordinator_id: str, metrics: CoordinatorHealthMetrics
    ) -> None:
        """Update health metrics for a coordinator."""
        if coordinator_id not in self.coordinators:
            self.register_coordinator(coordinator_id, metrics)
            return

        self.coordinators[coordinator_id] = metrics

        # Record health history
        health_score = metrics.calculate_health_score()
        self.health_history[coordinator_id].append((time.time(), health_score))

        # Keep only last 100 entries
        if len(self.health_history[coordinator_id]) > 100:
            self.health_history[coordinator_id] = self.health_history[coordinator_id][
                -100:
            ]

        # Check for health deterioration
        if health_score < self.unhealthy_threshold:
            self._trigger_recovery(coordinator_id, metrics)

    def add_recovery_callback(self, callback: Callable[[RecoveryAction], None]) -> None:
        """Add a callback for recovery actions."""
        self.recovery_callbacks.append(callback)

    def start_monitoring(self) -> None:
        """Start the health monitoring process."""
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started coordinator health monitoring")

    def stop_monitoring(self) -> None:
        """Stop the health monitoring process."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()

        self.monitoring_tasks.clear()
        logger.info("Stopped coordinator health monitoring")

    def get_coordinator_health(
        self, coordinator_id: str
    ) -> Optional[CoordinatorHealthMetrics]:
        """Get health metrics for a coordinator."""
        return self.coordinators.get(coordinator_id)

    def get_health_trend(
        self, coordinator_id: str, window_seconds: float = 300.0
    ) -> List[Tuple[float, float]]:
        """Get health trend for a coordinator over a time window."""
        if coordinator_id not in self.health_history:
            return []

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        return [
            (t, h) for t, h in self.health_history[coordinator_id] if t >= cutoff_time
        ]

    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall health status of all coordinators."""
        total_coordinators = len(self.coordinators)
        healthy_count = sum(
            1
            for m in self.coordinators.values()
            if m.is_healthy(self.unhealthy_threshold)
        )
        unhealthy_count = total_coordinators - healthy_count

        avg_health_score = (
            sum(m.health_score for m in self.coordinators.values()) / total_coordinators
            if total_coordinators > 0
            else 0.0
        )

        return {
            "total_coordinators": total_coordinators,
            "healthy_coordinators": healthy_count,
            "unhealthy_coordinators": unhealthy_count,
            "average_health_score": avg_health_score,
            "health_distribution": {
                "excellent": sum(
                    1 for m in self.coordinators.values() if m.health_score >= 0.9
                ),
                "good": sum(
                    1 for m in self.coordinators.values() if 0.7 <= m.health_score < 0.9
                ),
                "fair": sum(
                    1 for m in self.coordinators.values() if 0.5 <= m.health_score < 0.7
                ),
                "poor": sum(
                    1 for m in self.coordinators.values() if m.health_score < 0.5
                ),
            },
        }

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Update system metrics for all coordinators
                for coordinator_id, metrics in self.coordinators.items():
                    metrics.update_from_system()
                    self.update_metrics(coordinator_id, metrics)

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

    def _trigger_recovery(
        self, coordinator_id: str, metrics: CoordinatorHealthMetrics
    ) -> None:
        """Trigger recovery actions for an unhealthy coordinator."""
        logger.warning(
            f"Coordinator {coordinator_id} is unhealthy (score: {metrics.health_score:.2f})"
        )

        # Determine recovery strategy based on symptoms
        strategy = self._determine_recovery_strategy(metrics)

        # Create recovery action
        action = RecoveryAction(
            action_id=f"recovery_{coordinator_id}_{int(time.time())}",
            strategy=strategy,
            coordinator_id=coordinator_id,
            priority=self._calculate_priority(metrics),
            parameters=self._get_strategy_parameters(strategy, metrics),
        )

        # Notify callbacks
        for callback in self.recovery_callbacks:
            try:
                callback(action)
            except Exception as e:
                logger.error(f"Error in recovery callback: {e}")

    def _determine_recovery_strategy(
        self, metrics: CoordinatorHealthMetrics
    ) -> RecoveryStrategy:
        """Determine the best recovery strategy based on metrics."""
        if metrics.memory_usage_mb > 800:  # High memory usage
            return RecoveryStrategy.RESTART
        elif metrics.cpu_usage_percent > 85:  # High CPU usage
            return RecoveryStrategy.SCALE_UP
        elif metrics.error_rate > 0.5:  # High error rate
            return RecoveryStrategy.FAILOVER
        elif metrics.queue_depth > 500:  # High queue depth
            return RecoveryStrategy.DEGRADATION
        else:
            return RecoveryStrategy.STATE_RECOVERY

    def _calculate_priority(self, metrics: CoordinatorHealthMetrics) -> int:
        """Calculate recovery priority based on severity."""
        if metrics.health_score < 0.3:
            return 5  # Critical
        elif metrics.health_score < 0.5:
            return 4  # High
        elif metrics.health_score < 0.6:
            return 3  # Medium
        else:
            return 2  # Low

    def _get_strategy_parameters(
        self, strategy: RecoveryStrategy, metrics: CoordinatorHealthMetrics
    ) -> Dict[str, Any]:
        """Get parameters for a recovery strategy."""
        if strategy == RecoveryStrategy.RESTART:
            return {"graceful_shutdown": True, "backup_state": True}
        elif strategy == RecoveryStrategy.FAILOVER:
            return {"target_coordinator": None, "transfer_load": True}
        elif strategy == RecoveryStrategy.SCALE_UP:
            return {"additional_resources": 0.5, "auto_scale": True}
        elif strategy == RecoveryStrategy.STATE_RECOVERY:
            return {"checkpoint_id": "latest", "validate_state": True}
        elif strategy == RecoveryStrategy.DEGRADATION:
            return {"reduced_capacity": 0.7, "prioritize_critical": True}
        else:
            return {}


class RestartStrategy:
    """
    Handles coordinator restart with different strategies.
    """

    def __init__(self, max_restart_attempts: int = 3, backoff_multiplier: float = 2.0):
        self.max_restart_attempts = max_restart_attempts
        self.backoff_multiplier = backoff_multiplier
        self.restart_history: Dict[
            str, List[Tuple[float, bool]]
        ] = {}  # coordinator_id -> [(timestamp, success)]

    async def restart_coordinator(
        self, coordinator_id: str, coordinator_ref: Any, graceful: bool = True
    ) -> bool:
        """
        Restart a coordinator with exponential backoff.

        Args:
            coordinator_id: ID of coordinator to restart
            coordinator_ref: Reference to coordinator instance
            graceful: Whether to attempt graceful shutdown first

        Returns:
            True if restart was successful
        """
        if coordinator_id not in self.restart_history:
            self.restart_history[coordinator_id] = []

        restart_count = len(
            [
                r
                for r in self.restart_history[coordinator_id]
                if time.time() - r[0] < 3600
            ]
        )  # Last hour

        if restart_count >= self.max_restart_attempts:
            logger.error(
                f"Max restart attempts ({self.max_restart_attempts}) exceeded for {coordinator_id}"
            )
            return False

        try:
            # Calculate backoff delay
            delay = self.backoff_multiplier**restart_count
            if delay > 300:  # Max 5 minutes
                delay = 300

            logger.info(
                f"Restarting coordinator {coordinator_id} in {delay:.1f} seconds"
            )

            if delay > 0:
                await asyncio.sleep(delay)

            # Attempt graceful shutdown if requested
            if graceful and hasattr(coordinator_ref, "shutdown"):
                try:
                    await coordinator_ref.shutdown()
                    await asyncio.sleep(2.0)  # Wait for cleanup
                except Exception as e:
                    logger.warning(
                        f"Graceful shutdown failed for {coordinator_id}: {e}"
                    )

            # Restart the coordinator
            if hasattr(coordinator_ref, "start"):
                await coordinator_ref.start()
            elif hasattr(coordinator_ref, "initialize"):
                await coordinator_ref.initialize()
            else:
                # Assume callable restart
                await coordinator_ref()

            # Record successful restart
            self.restart_history[coordinator_id].append((time.time(), True))

            # Clean old history (keep last 24 hours)
            cutoff_time = time.time() - 86400
            self.restart_history[coordinator_id] = [
                (t, s)
                for t, s in self.restart_history[coordinator_id]
                if t > cutoff_time
            ]

            logger.info(f"Successfully restarted coordinator {coordinator_id}")
            return True

        except Exception as e:
            # Record failed restart
            self.restart_history[coordinator_id].append((time.time(), False))
            logger.error(f"Failed to restart coordinator {coordinator_id}: {e}")
            return False

    def get_restart_stats(self, coordinator_id: str) -> Dict[str, Any]:
        """Get restart statistics for a coordinator."""
        if coordinator_id not in self.restart_history:
            return {"total_restarts": 0, "successful_restarts": 0, "failed_restarts": 0}

        history = self.restart_history[coordinator_id]
        successful = sum(1 for _, success in history if success)
        failed = len(history) - successful

        return {
            "total_restarts": len(history),
            "successful_restarts": successful,
            "failed_restarts": failed,
            "success_rate": successful / len(history) if history else 0.0,
        }


class StateRecoveryManager:
    """
    Manages state capture and recovery for coordinators.
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints: Dict[
            str, Dict[str, Any]
        ] = {}  # coordinator_id -> checkpoint_id -> metadata
        import os

        os.makedirs(checkpoint_dir, exist_ok=True)

    def create_checkpoint(
        self, coordinator_id: str, state: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a checkpoint of coordinator state.

        Returns checkpoint ID.
        """
        checkpoint_id = f"checkpoint_{coordinator_id}_{int(time.time())}_{id(state)}"

        checkpoint_data = {
            "coordinator_id": coordinator_id,
            "checkpoint_id": checkpoint_id,
            "state": state,
            "metadata": metadata or {},
            "created_at": time.time(),
            "size_bytes": self._estimate_size(state),
        }

        # Store checkpoint
        self.checkpoints.setdefault(coordinator_id, {})[checkpoint_id] = checkpoint_data

        # Persist to disk for durability
        self._persist_checkpoint(checkpoint_data)

        logger.info(
            f"Created checkpoint {checkpoint_id} for coordinator {coordinator_id}"
        )
        return checkpoint_id

    def restore_from_checkpoint(
        self, coordinator_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Restore coordinator state from checkpoint.

        Args:
            coordinator_id: Coordinator to restore
            checkpoint_id: Specific checkpoint ID, or None for latest

        Returns:
            Restored state, or None if not found
        """
        if coordinator_id not in self.checkpoints:
            return None

        coordinator_checkpoints = self.checkpoints[coordinator_id]

        if not coordinator_checkpoints:
            return None

        # Find checkpoint to restore from
        if checkpoint_id:
            checkpoint_data = coordinator_checkpoints.get(checkpoint_id)
        else:
            # Use latest checkpoint
            latest_checkpoint = max(
                coordinator_checkpoints.values(), key=lambda x: x["created_at"]
            )
            checkpoint_data = latest_checkpoint

        if not checkpoint_data:
            return None

        state = checkpoint_data["state"]
        logger.info(
            f"Restored state from checkpoint {checkpoint_data['checkpoint_id']} for {coordinator_id}"
        )
        return state

    def list_checkpoints(self, coordinator_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a coordinator."""
        if coordinator_id not in self.checkpoints:
            return []

        return list(self.checkpoints[coordinator_id].values())

    def delete_old_checkpoints(self, coordinator_id: str, keep_last: int = 5) -> int:
        """Delete old checkpoints, keeping only the most recent ones."""
        if coordinator_id not in self.checkpoints:
            return 0

        coordinator_checkpoints = self.checkpoints[coordinator_id]

        # Sort by creation time (newest first)
        sorted_checkpoints = sorted(
            coordinator_checkpoints.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True,
        )

        # Keep only the most recent ones
        to_delete = sorted_checkpoints[keep_last:]
        deleted_count = 0

        for checkpoint_id, _ in to_delete:
            del coordinator_checkpoints[checkpoint_id]
            self._delete_persisted_checkpoint(checkpoint_id)
            deleted_count += 1

        logger.info(
            f"Deleted {deleted_count} old checkpoints for coordinator {coordinator_id}"
        )
        return deleted_count

    def _persist_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Persist checkpoint to disk."""
        import os
        import pickle

        checkpoint_id = checkpoint_data["checkpoint_id"]
        filename = f"{checkpoint_id}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)

        try:
            with open(filepath, "wb") as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to persist checkpoint {checkpoint_id}: {e}")

    def _delete_persisted_checkpoint(self, checkpoint_id: str) -> None:
        """Delete persisted checkpoint from disk."""
        import os

        filename = f"{checkpoint_id}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)

        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.warning(
                f"Failed to delete persisted checkpoint {checkpoint_id}: {e}"
            )

    def _estimate_size(self, obj: Any) -> int:
        """Estimate the size of an object."""
        try:
            import sys

            return sys.getsizeof(obj)
        except:
            return 1000


class FailoverManager:
    """
    Manages failover between coordinators.
    """

    def __init__(self):
        self.primary_coordinators: Dict[
            str, Any
        ] = {}  # coordinator_id -> coordinator_ref
        self.backup_coordinators: Dict[
            str, Dict[str, Any]
        ] = {}  # coordinator_id -> backup_id -> backup_ref
        self.active_failovers: Dict[
            str, asyncio.Task
        ] = {}  # coordinator_id -> failover_task

    def register_primary(self, coordinator_id: str, coordinator_ref: Any) -> None:
        """Register a primary coordinator."""
        self.primary_coordinators[coordinator_id] = coordinator_ref
        logger.info(f"Registered primary coordinator {coordinator_id}")

    def register_backup(self, primary_id: str, backup_id: str, backup_ref: Any) -> None:
        """Register a backup coordinator for a primary."""
        if primary_id not in self.backup_coordinators:
            self.backup_coordinators[primary_id] = {}

        self.backup_coordinators[primary_id][backup_id] = backup_ref
        logger.info(
            f"Registered backup coordinator {backup_id} for primary {primary_id}"
        )

    async def initiate_failover(self, failed_coordinator_id: str) -> bool:
        """
        Initiate failover from a failed coordinator to a backup.

        Returns True if failover was successful.
        """
        if failed_coordinator_id not in self.primary_coordinators:
            logger.warning(
                f"No primary coordinator registered for {failed_coordinator_id}"
            )
            return False

        if failed_coordinator_id in self.active_failovers:
            logger.warning(f"Failover already in progress for {failed_coordinator_id}")
            return False

        # Start failover process
        task = asyncio.create_task(self._perform_failover(failed_coordinator_id))
        self.active_failovers[failed_coordinator_id] = task

        try:
            success = await task
            return success
        except Exception as e:
            logger.error(f"Failover failed for {failed_coordinator_id}: {e}")
            return False
        finally:
            del self.active_failovers[failed_coordinator_id]

    async def _perform_failover(self, failed_coordinator_id: str) -> bool:
        """Perform the actual failover process."""
        backups = self.backup_coordinators.get(failed_coordinator_id, {})

        if not backups:
            logger.warning(
                f"No backup coordinators available for {failed_coordinator_id}"
            )
            return False

        # Try each backup in order
        for backup_id, backup_ref in backups.items():
            try:
                logger.info(
                    f"Attempting failover from {failed_coordinator_id} to backup {backup_id}"
                )

                # Transfer state if possible
                await self._transfer_state(failed_coordinator_id, backup_id)

                # Activate backup coordinator
                if hasattr(backup_ref, "activate"):
                    await backup_ref.activate()
                elif hasattr(backup_ref, "start"):
                    await backup_ref.start()

                # Update registration
                self.primary_coordinators[failed_coordinator_id] = backup_ref

                logger.info(
                    f"Successfully failed over {failed_coordinator_id} to {backup_id}"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to failover to backup {backup_id}: {e}")
                continue

        logger.error(f"All failover attempts failed for {failed_coordinator_id}")
        return False

    async def _transfer_state(
        self, from_coordinator_id: str, to_backup_id: str
    ) -> None:
        """Transfer state from failed coordinator to backup."""
        # This would integrate with StateRecoveryManager to transfer state
        # For now, just log the intent
        logger.info(
            f"Transferring state from {from_coordinator_id} to backup {to_backup_id}"
        )

    def get_failover_status(self, coordinator_id: str) -> Dict[str, Any]:
        """Get failover status for a coordinator."""
        backups = self.backup_coordinators.get(coordinator_id, {})

        return {
            "coordinator_id": coordinator_id,
            "has_primary": coordinator_id in self.primary_coordinators,
            "backup_count": len(backups),
            "failover_in_progress": coordinator_id in self.active_failovers,
            "backup_ids": list(backups.keys()),
        }


class SelfHealingCoordinator:
    """
    Main wrapper class that adds self-healing capabilities to any coordinator.
    """

    def __init__(self, coordinator_id: str, wrapped_coordinator: Any):
        self.coordinator_id = coordinator_id
        self.wrapped_coordinator = wrapped_coordinator

        # Initialize self-healing components
        self.health_monitor = CoordinatorHealthMonitor()
        self.restart_strategy = RestartStrategy()
        self.state_recovery = StateRecoveryManager()
        self.failover_manager = FailoverManager()

        # Register self for monitoring
        self.health_monitor.register_coordinator(coordinator_id)
        self.failover_manager.register_primary(coordinator_id, self)

        # Set up recovery callbacks
        self.health_monitor.add_recovery_callback(self._handle_recovery_action)

        # State
        self.is_active = False
        self.last_checkpoint: Optional[str] = None

    async def start(self) -> None:
        """Start the self-healing coordinator."""
        logger.info(f"Starting self-healing coordinator {self.coordinator_id}")

        # Create initial checkpoint
        self.last_checkpoint = self.state_recovery.create_checkpoint(
            self.coordinator_id, self.wrapped_coordinator, {"phase": "startup"}
        )

        # Start health monitoring
        self.health_monitor.start_monitoring()

        # Start wrapped coordinator
        if hasattr(self.wrapped_coordinator, "start"):
            await self.wrapped_coordinator.start()
        elif hasattr(self.wrapped_coordinator, "initialize"):
            await self.wrapped_coordinator.initialize()

        self.is_active = True
        logger.info(
            f"Self-healing coordinator {self.coordinator_id} started successfully"
        )

    async def shutdown(self) -> None:
        """Shutdown the self-healing coordinator."""
        logger.info(f"Shutting down self-healing coordinator {self.coordinator_id}")

        self.is_active = False

        # Stop health monitoring
        self.health_monitor.stop_monitoring()

        # Shutdown wrapped coordinator
        if hasattr(self.wrapped_coordinator, "shutdown"):
            await self.wrapped_coordinator.shutdown()

        logger.info(f"Self-healing coordinator {self.coordinator_id} shutdown complete")

    async def _handle_recovery_action(self, action: RecoveryAction) -> None:
        """Handle a recovery action triggered by health monitoring."""
        logger.info(
            f"Executing recovery action {action.action_id} for {action.coordinator_id}"
        )

        try:
            if action.strategy == RecoveryStrategy.RESTART:
                success = await self.restart_strategy.restart_coordinator(
                    action.coordinator_id,
                    self.wrapped_coordinator,
                    action.parameters.get("graceful_shutdown", True),
                )

            elif action.strategy == RecoveryStrategy.FAILOVER:
                success = await self.failover_manager.initiate_failover(
                    action.coordinator_id
                )

            elif action.strategy == RecoveryStrategy.STATE_RECOVERY:
                restored_state = self.state_recovery.restore_from_checkpoint(
                    action.coordinator_id, action.parameters.get("checkpoint_id")
                )
                success = restored_state is not None

            elif action.strategy == RecoveryStrategy.SCALE_UP:
                # Scaling would require integration with resource management
                logger.info("Scale-up recovery requested - not implemented")
                success = False

            elif action.strategy == RecoveryStrategy.DEGRADATION:
                # Graceful degradation would modify coordinator behavior
                logger.info("Degradation recovery requested - not implemented")
                success = False

            else:
                logger.warning(f"Unknown recovery strategy: {action.strategy}")
                success = False

            action.mark_executed(success)

            if success:
                logger.info(
                    f"Recovery action {action.action_id} completed successfully"
                )
            else:
                logger.error(f"Recovery action {action.action_id} failed")

        except Exception as e:
            logger.error(f"Error executing recovery action {action.action_id}: {e}")
            action.mark_executed(False, str(e))

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_metrics = self.health_monitor.get_coordinator_health(self.coordinator_id)
        failover_status = self.failover_manager.get_failover_status(self.coordinator_id)
        restart_stats = self.restart_strategy.get_restart_stats(self.coordinator_id)

        return {
            "coordinator_id": self.coordinator_id,
            "is_active": self.is_active,
            "health_metrics": health_metrics.to_dict() if health_metrics else None,
            "failover_status": failover_status,
            "restart_stats": restart_stats,
            "last_checkpoint": self.last_checkpoint,
            "checkpoints": self.state_recovery.list_checkpoints(self.coordinator_id),
        }

    def __getattr__(self, name):
        """Delegate attribute access to wrapped coordinator."""
        return getattr(self.wrapped_coordinator, name)


class SelfHealingCoordinatorFactory:
    """
    Factory for creating self-healing coordinator wrappers.
    """

    def __init__(self):
        self.created_coordinators: Dict[str, SelfHealingCoordinator] = {}

    def create_wrapped_coordinator(
        self, coordinator_id: str, coordinator: Any, enable_failover: bool = True
    ) -> SelfHealingCoordinator:
        """
        Create a self-healing wrapper for a coordinator.

        Args:
            coordinator_id: Unique ID for the coordinator
            coordinator: The coordinator instance to wrap
            enable_failover: Whether to enable failover capabilities

        Returns:
            SelfHealingCoordinator instance
        """
        wrapped = SelfHealingCoordinator(coordinator_id, coordinator)

        if enable_failover:
            # Register backups if available (this would be configurable)
            pass

        self.created_coordinators[coordinator_id] = wrapped
        return wrapped

    def get_coordinator(self, coordinator_id: str) -> Optional[SelfHealingCoordinator]:
        """Get a created coordinator by ID."""
        return self.created_coordinators.get(coordinator_id)

    def list_coordinators(self) -> List[str]:
        """List all created coordinator IDs."""
        return list(self.created_coordinators.keys())

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status of all coordinators."""
        coordinator_healths = {}
        for cid, coordinator in self.created_coordinators.items():
            coordinator_healths[cid] = coordinator.get_health_status()

        return {
            "total_coordinators": len(self.created_coordinators),
            "coordinator_healths": coordinator_healths,
            "overall_health": self._calculate_overall_health(coordinator_healths),
        }

    def _calculate_overall_health(
        self, coordinator_healths: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall system health."""
        if not coordinator_healths:
            return {"status": "no_coordinators", "score": 0.0}

        health_scores = []
        active_count = 0

        for health in coordinator_healths.values():
            if health.get("is_active", False):
                active_count += 1
                metrics = health.get("health_metrics")
                if metrics:
                    health_scores.append(metrics.get("health_score", 0.0))

        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0.0

        if active_count == 0:
            status = "all_inactive"
        elif avg_health >= 0.8:
            status = "healthy"
        elif avg_health >= 0.6:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "score": avg_health,
            "active_coordinators": active_count,
        }
