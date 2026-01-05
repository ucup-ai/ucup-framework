"""
Hierarchical Coordinator for UCUP.

Implements hierarchical coordination patterns using Grand Central Dispatch.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from ..core.manager import UCUPManager


class TaskNode:
    """Node in the hierarchical task tree."""

    def __init__(self, task_id: str, task_data: Dict[str, Any], priority: int = 0):
        self.task_id = task_id
        self.task_data = task_data
        self.priority = priority
        self.children: List['TaskNode'] = []
        self.parent: Optional['TaskNode'] = None
        self.status = "pending"  # pending, running, completed, failed
        self.result: Optional[Any] = None
        self.dependencies: List[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def add_child(self, child: 'TaskNode'):
        """Add a child task node."""
        child.parent = self
        self.children.append(child)

    def is_ready(self) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep.status == "completed" for dep in self.dependencies)

    def get_execution_time(self) -> Optional[float]:
        """Get task execution time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "priority": self.priority,
            "result": self.result,
            "children": [child.to_dict() for child in self.children],
            "dependencies": self.dependencies.copy(),
            "execution_time": self.get_execution_time()
        }


class HierarchicalCoordinator:
    """Hierarchical task coordinator using GCD patterns."""

    def __init__(self, manager: UCUPManager):
        self.manager = manager
        self.config = manager.config
        self.task_tree: Optional[TaskNode] = None
        self.task_map: Dict[str, TaskNode] = {}
        self.executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._lock = threading.RLock()

        self._initialize_executor()

    def _initialize_executor(self):
        """Initialize the thread pool executor."""
        max_workers = self.config.get("gcd.max_concurrent_tasks", 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="UCUP-Hierarchical")

    def coordinate_tasks(self, tasks: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Coordinate hierarchical task execution.

        Args:
            tasks: List of task definitions with hierarchy information
            **kwargs: Additional coordination parameters

        Returns:
            Coordination results
        """
        with self._lock:
            self._running = True
            start_time = time.time()

            try:
                # Build task hierarchy
                self._build_task_hierarchy(tasks)

                # Execute tasks hierarchically
                results = self._execute_hierarchical()

                execution_time = time.time() - start_time

                return {
                    "success": True,
                    "results": results,
                    "execution_time": execution_time,
                    "total_tasks": len(self.task_map),
                    "coordination_method": "hierarchical"
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
            finally:
                self._running = False

    def _build_task_hierarchy(self, tasks: List[Dict[str, Any]]):
        """Build the hierarchical task tree from task definitions."""
        self.task_map.clear()
        self.task_tree = None

        # Create task nodes
        for task_def in tasks:
            task_id = task_def["id"]
            node = TaskNode(
                task_id=task_id,
                task_data=task_def,
                priority=task_def.get("priority", 0)
            )
            node.dependencies = task_def.get("dependencies", [])
            self.task_map[task_id] = node

        # Build hierarchy relationships
        for task_def in tasks:
            task_id = task_def["id"]
            parent_id = task_def.get("parent_id")

            if parent_id:
                parent_node = self.task_map.get(parent_id)
                if parent_node:
                    parent_node.add_child(self.task_map[task_id])
            elif self.task_tree is None:
                self.task_tree = self.task_map[task_id]

    def _execute_hierarchical(self) -> List[Dict[str, Any]]:
        """Execute tasks in hierarchical order."""
        if not self.task_tree:
            return []

        results = []
        pending_futures = []

        # Start with root level tasks
        root_tasks = [self.task_tree] if self.task_tree else []
        root_tasks.extend(child for node in self.task_map.values()
                         for child in node.children if not child.parent)

        # Execute level by level
        current_level = root_tasks
        while current_level and self._running:
            level_futures = []

            # Submit all tasks in current level concurrently
            for task_node in current_level:
                if task_node.is_ready():
                    future = self.executor.submit(self._execute_task_node, task_node)
                    level_futures.append((task_node, future))
                    pending_futures.append(future)

            # Wait for current level to complete
            for task_node, future in level_futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append({
                        "task_id": task_node.task_id,
                        "result": result,
                        "status": "completed",
                        "execution_time": task_node.get_execution_time()
                    })
                except Exception as e:
                    task_node.status = "failed"
                    results.append({
                        "task_id": task_node.task_id,
                        "error": str(e),
                        "status": "failed"
                    })

            # Move to next level
            next_level = []
            for task_node in current_level:
                next_level.extend(task_node.children)

            current_level = [node for node in next_level if node.status == "pending"]

        return results

    def _execute_task_node(self, task_node: TaskNode) -> Any:
        """Execute a single task node."""
        task_node.status = "running"
        task_node.start_time = time.time()

        try:
            # Execute task using the manager
            task_data = task_node.task_data

            if task_data.get("type") == "probabilistic":
                # Probabilistic task
                result = self.manager.execute_probabilistic_task(task_data.get("parameters", {}))
            elif task_data.get("type") == "multimodal":
                # Multimodal analysis
                result = self.manager.analyze_multimodal_data(task_data.get("data", []))
            else:
                # Generic task execution
                result = self._execute_generic_task(task_data)

            task_node.result = result
            task_node.status = "completed"
            task_node.end_time = time.time()

            return result

        except Exception as e:
            task_node.status = "failed"
            task_node.end_time = time.time()
            raise e

    def _execute_generic_task(self, task_data: Dict[str, Any]) -> Any:
        """Execute a generic task."""
        # Simulate task execution
        task_type = task_data.get("task_type", "generic")

        if task_type == "computation":
            # Simulate computation
            import random
            time.sleep(random.uniform(0.1, 1.0))
            return {"computation_result": random.random()}

        elif task_type == "data_processing":
            # Simulate data processing
            input_data = task_data.get("input", [])
            return {"processed_items": len(input_data), "status": "processed"}

        else:
            # Default task
            return {"task_type": task_type, "status": "executed"}

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        task_node = self.task_map.get(task_id)
        if task_node:
            return task_node.to_dict()
        return None

    def get_hierarchy_status(self) -> Dict[str, Any]:
        """Get the overall hierarchy execution status."""
        if not self.task_tree:
            return {"error": "No task hierarchy available"}

        total_tasks = len(self.task_map)
        completed_tasks = sum(1 for node in self.task_map.values() if node.status == "completed")
        failed_tasks = sum(1 for node in self.task_map.values() if node.status == "failed")
        running_tasks = sum(1 for node in self.task_map.values() if node.status == "running")

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "pending_tasks": total_tasks - completed_tasks - failed_tasks - running_tasks,
            "completion_percentage": (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0,
            "hierarchy": self.task_tree.to_dict() if self.task_tree else None
        }

    def cancel_execution(self):
        """Cancel the current task execution."""
        with self._lock:
            self._running = False
            if self.executor:
                self.executor.shutdown(wait=False)

    def reset(self):
        """Reset the coordinator state."""
        with self._lock:
            self.task_tree = None
            self.task_map.clear()
            self._running = False

            # Reinitialize executor
            if self.executor:
                self.executor.shutdown(wait=True)
            self._initialize_executor()

    def __repr__(self) -> str:
        return f"HierarchicalCoordinator(tasks={len(self.task_map)}, running={self._running})"
