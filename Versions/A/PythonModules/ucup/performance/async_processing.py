"""
Asynchronous Processing Module

This module provides async processing capabilities for improved performance
including task pools, parallel execution, and streaming result aggregation.
"""

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Task:
    """Represents an async task with metadata."""

    id: str
    coroutine: Awaitable[Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    timeout: Optional[float] = None


@dataclass
class TaskResult:
    """Result of a completed task."""

    task_id: str
    result: Any
    error: Optional[Exception] = None
    execution_time: float = 0.0
    completed_at: float = field(default_factory=time.time)


class AsyncTaskPool:
    """
    Asynchronous task pool with backpressure and priority queuing.

    Provides efficient task scheduling and execution with configurable concurrency limits.
    """

    def __init__(self, max_concurrent: int = 10, queue_size: int = 1000):
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=queue_size
        )
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.running = False
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_execution_time": 0.0,
            "queue_size": 0,
            "active_tasks": 0,
        }

    async def start(self) -> None:
        """Start the task pool processing."""
        self.running = True
        logger.info(f"Starting AsyncTaskPool with max_concurrent={self.max_concurrent}")

        while self.running:
            try:
                # Get next task with priority
                priority, task = await self.task_queue.get()

                async with self.semaphore:
                    # Execute task
                    asyncio_task = asyncio.create_task(self._execute_task(task))
                    self.active_tasks[task.id] = asyncio_task

                    # Update stats
                    self.stats["active_tasks"] = len(self.active_tasks)

            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Error in task pool processing: {e}")

    def stop(self) -> None:
        """Stop the task pool."""
        self.running = False
        logger.info("Stopping AsyncTaskPool")

    async def submit_task(self, task: Task) -> str:
        """
        Submit a task to the pool.

        Returns task ID for result retrieval.
        """
        try:
            # Priority queue: higher priority number = higher priority
            # Use negative priority for max-heap behavior
            await self.task_queue.put((-task.priority, task))
            self.stats["tasks_submitted"] += 1
            self.stats["queue_size"] = self.task_queue.qsize()
            logger.debug(f"Submitted task {task.id} with priority {task.priority}")
            return task.id
        except asyncio.QueueFull:
            raise RuntimeError("Task queue is full")

    async def submit_callable(
        self,
        func: Callable[..., Awaitable[R]],
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Submit a callable as a task."""
        task_id = f"task_{int(time.time() * 1000000)}_{len(self.active_tasks)}"

        async def wrapper():
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                # Run in thread pool for sync functions
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)

        task = Task(id=task_id, coroutine=wrapper(), priority=priority, timeout=timeout)

        return await self.submit_task(task)

    async def _execute_task(self, task: Task) -> None:
        """Execute a single task with error handling and timing."""
        start_time = time.time()

        try:
            if task.timeout:
                result = await asyncio.wait_for(task.coroutine, timeout=task.timeout)
            else:
                result = await task.coroutine

            execution_time = time.time() - start_time

            task_result = TaskResult(
                task_id=task.id, result=result, execution_time=execution_time
            )

            self.completed_tasks[task.id] = task_result
            self.stats["tasks_completed"] += 1

            # Update average execution time
            self._update_average_execution_time(execution_time)

            logger.debug(
                f"Task {task.id} completed successfully in {execution_time:.4f}s"
            )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error = TimeoutError(
                f"Task {task.id} timed out after {execution_time:.4f}s"
            )

            task_result = TaskResult(
                task_id=task.id, result=None, error=error, execution_time=execution_time
            )

            self.completed_tasks[task.id] = task_result
            self.stats["tasks_failed"] += 1

            logger.warning(f"Task {task.id} timed out")

        except Exception as e:
            execution_time = time.time() - start_time

            task_result = TaskResult(
                task_id=task.id, result=None, error=e, execution_time=execution_time
            )

            self.completed_tasks[task.id] = task_result
            self.stats["tasks_failed"] += 1

            logger.error(f"Task {task.id} failed with error: {e}")

        finally:
            # Clean up active tasks
            self.active_tasks.pop(task.id, None)
            self.stats["active_tasks"] = len(self.active_tasks)

    def _update_average_execution_time(self, new_time: float) -> None:
        """Update rolling average execution time."""
        current_avg = self.stats["average_execution_time"]
        completed_count = self.stats["tasks_completed"]

        if completed_count == 1:
            self.stats["average_execution_time"] = new_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats["average_execution_time"] = (
                alpha * new_time + (1 - alpha) * current_avg
            )

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task."""
        return self.completed_tasks.get(task_id)

    async def wait_for_task(
        self, task_id: str, timeout: Optional[float] = None
    ) -> TaskResult:
        """Wait for a task to complete and return its result."""
        start_time = time.time()

        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")

            await asyncio.sleep(0.01)

        return self.completed_tasks[task_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self.stats,
            "queue_size": self.task_queue.qsize()
            if hasattr(self.task_queue, "qsize")
            else 0,
            "active_tasks": len(self.active_tasks),
            "completed_tasks_count": len(self.completed_tasks),
        }


class ParallelAgentExecutor:
    """
    Executor for running multiple agents in parallel with coordination.

    Manages agent lifecycle, load balancing, and result aggregation.
    """

    def __init__(
        self, max_parallel_agents: int = 5, task_pool: Optional[AsyncTaskPool] = None
    ):
        self.max_parallel_agents = max_parallel_agents
        self.task_pool = task_pool or AsyncTaskPool(
            max_concurrent=max_parallel_agents * 2
        )
        self.agent_registry: Dict[str, Any] = {}  # agent_id -> agent_instance
        self.execution_contexts: Dict[str, Dict[str, Any]] = {}
        self.load_balancer = self._create_load_balancer()

    def _create_load_balancer(self) -> "SimpleLoadBalancer":
        """Create a simple load balancer for agent execution."""
        return SimpleLoadBalancer(max_load=self.max_parallel_agents)

    def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """Register an agent for parallel execution."""
        self.agent_registry[agent_id] = agent_instance
        logger.info(f"Registered agent {agent_id}")

    async def execute_parallel(
        self,
        tasks: List[Dict[str, Any]],
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tasks in parallel across registered agents.

        Args:
            tasks: List of task dictionaries with 'agent_id', 'task_data', etc.
            execution_context: Shared context for all executions

        Returns:
            List of execution results
        """
        execution_context = execution_context or {}
        execution_id = f"exec_{int(time.time() * 1000000)}"

        # Distribute tasks to agents
        agent_tasks = self._distribute_tasks_to_agents(tasks)

        # Submit execution tasks
        submitted_tasks = []
        for agent_id, agent_task_list in agent_tasks.items():
            if agent_id in self.agent_registry:
                task_id = await self.task_pool.submit_callable(
                    self._execute_agent_tasks,
                    agent_id,
                    agent_task_list,
                    execution_context,
                    execution_id,
                    priority=1,
                )
                submitted_tasks.append(task_id)

        # Wait for all executions to complete
        results = []
        for task_id in submitted_tasks:
            try:
                task_result = await self.task_pool.wait_for_task(task_id, timeout=300.0)
                if task_result.error:
                    logger.error(f"Agent execution failed: {task_result.error}")
                    results.append(
                        {
                            "execution_id": execution_id,
                            "error": str(task_result.error),
                            "success": False,
                        }
                    )
                else:
                    results.extend(task_result.result)
            except asyncio.TimeoutError:
                logger.error(f"Agent execution timed out for task {task_id}")
                results.append(
                    {"execution_id": execution_id, "error": "timeout", "success": False}
                )

        return results

    def _distribute_tasks_to_agents(
        self, tasks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Distribute tasks to available agents based on load balancing."""
        agent_tasks: Dict[str, List[Dict[str, Any]]] = {}

        for task in tasks:
            agent_id = task.get("agent_id")
            if not agent_id or agent_id not in self.agent_registry:
                # Assign to least loaded agent
                agent_id = self.load_balancer.get_least_loaded_agent(
                    list(self.agent_registry.keys())
                )

            if agent_id not in agent_tasks:
                agent_tasks[agent_id] = []
            agent_tasks[agent_id].append(task)

        return agent_tasks

    async def _execute_agent_tasks(
        self,
        agent_id: str,
        tasks: List[Dict[str, Any]],
        execution_context: Dict[str, Any],
        execution_id: str,
    ) -> List[Dict[str, Any]]:
        """Execute tasks for a specific agent."""
        agent = self.agent_registry[agent_id]
        results = []

        for task in tasks:
            try:
                # Execute task using agent
                result = await self._execute_single_task(agent, task, execution_context)

                result["execution_id"] = execution_id
                result["agent_id"] = agent_id
                result["success"] = True
                results.append(result)

            except Exception as e:
                logger.error(f"Task execution failed for agent {agent_id}: {e}")
                results.append(
                    {
                        "execution_id": execution_id,
                        "agent_id": agent_id,
                        "task_id": task.get("task_id"),
                        "error": str(e),
                        "success": False,
                    }
                )

        return results

    async def _execute_single_task(
        self, agent: Any, task: Dict[str, Any], execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single task using an agent."""
        # This is a generic interface - specific agents should implement their execution logic
        if hasattr(agent, "execute_task"):
            return await agent.execute_task(task, execution_context)
        elif hasattr(agent, "process"):
            return await agent.process(task, execution_context)
        else:
            # Default execution - assume agent is callable
            return await agent(task, execution_context)


class SimpleLoadBalancer:
    """Simple load balancer for agent task distribution."""

    def __init__(self, max_load: int = 5):
        self.agent_loads: Dict[str, int] = {}
        self.max_load = max_load

    def get_least_loaded_agent(self, available_agents: List[str]) -> str:
        """Get the least loaded agent."""
        if not available_agents:
            raise ValueError("No agents available")

        min_load = float("inf")
        selected_agent = available_agents[0]

        for agent in available_agents:
            load = self.agent_loads.get(agent, 0)
            if load < min_load:
                min_load = load
                selected_agent = agent

        return selected_agent

    def increment_load(self, agent_id: str) -> None:
        """Increment load for an agent."""
        self.agent_loads[agent_id] = self.agent_loads.get(agent_id, 0) + 1

    def decrement_load(self, agent_id: str) -> None:
        """Decrement load for an agent."""
        current_load = self.agent_loads.get(agent_id, 0)
        if current_load > 0:
            self.agent_loads[agent_id] = current_load - 1


class StreamingResultAggregator:
    """
    Aggregates streaming results from multiple sources.

    Handles out-of-order results, batching, and real-time aggregation.
    """

    def __init__(self, batch_size: int = 10, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.result_buffer: Dict[str, List[Any]] = {}
        self.batch_queues: Dict[str, asyncio.Queue] = {}
        self.aggregation_tasks: Dict[str, asyncio.Task] = {}
        self.stats = {
            "total_results": 0,
            "batches_processed": 0,
            "average_batch_size": 0.0,
            "streams_active": 0,
        }

    def create_stream(self, stream_id: str) -> asyncio.Queue:
        """Create a new result stream."""
        queue = asyncio.Queue()
        self.batch_queues[stream_id] = queue
        self.result_buffer[stream_id] = []
        self.stats["streams_active"] += 1

        # Start aggregation task for this stream
        task = asyncio.create_task(self._aggregate_stream(stream_id))
        self.aggregation_tasks[stream_id] = task

        logger.info(f"Created result stream {stream_id}")
        return queue

    async def add_result(self, stream_id: str, result: Any) -> None:
        """Add a result to a stream."""
        if stream_id not in self.batch_queues:
            raise ValueError(f"Stream {stream_id} does not exist")

        await self.batch_queues[stream_id].put(result)
        self.stats["total_results"] += 1

    async def _aggregate_stream(self, stream_id: str) -> None:
        """Aggregate results for a stream into batches."""
        queue = self.batch_queues[stream_id]
        buffer = self.result_buffer[stream_id]

        while True:
            try:
                # Wait for first result with timeout
                result = await asyncio.wait_for(queue.get(), timeout=self.max_wait_time)
                buffer.append(result)

                # Try to get more results without waiting
                while len(buffer) < self.batch_size:
                    try:
                        result = queue.get_nowait()
                        buffer.append(result)
                    except asyncio.QueueEmpty:
                        break

                # Process batch if we have results
                if buffer:
                    await self._process_batch(stream_id, buffer.copy())
                    buffer.clear()
                    self.stats["batches_processed"] += 1

                    # Update average batch size
                    self._update_average_batch_size(len(buffer))

            except asyncio.TimeoutError:
                # Process any remaining results in buffer
                if buffer:
                    await self._process_batch(stream_id, buffer.copy())
                    buffer.clear()
                    self.stats["batches_processed"] += 1
            except Exception as e:
                logger.error(f"Error in stream aggregation for {stream_id}: {e}")
                break

    async def _process_batch(self, stream_id: str, batch: List[Any]) -> None:
        """Process a batch of results."""
        # Default processing - can be overridden by subclasses
        logger.debug(f"Processing batch of {len(batch)} results for stream {stream_id}")

        # Example aggregation: compute statistics
        if batch:
            try:
                # Assume results are numeric for demonstration
                numeric_results = [r for r in batch if isinstance(r, (int, float))]
                if numeric_results:
                    stats = {
                        "stream_id": stream_id,
                        "batch_size": len(batch),
                        "sum": sum(numeric_results),
                        "avg": sum(numeric_results) / len(numeric_results),
                        "min": min(numeric_results),
                        "max": max(numeric_results),
                        "timestamp": time.time(),
                    }
                    await self._emit_batch_result(stream_id, stats)
            except Exception as e:
                logger.warning(f"Could not compute batch statistics: {e}")

    async def _emit_batch_result(self, stream_id: str, result: Any) -> None:
        """Emit a processed batch result."""
        # Default: just log. Subclasses can override to send to callbacks, queues, etc.
        logger.info(f"Batch result for {stream_id}: {result}")

    def _update_average_batch_size(self, batch_size: int) -> None:
        """Update rolling average batch size."""
        current_avg = self.stats["average_batch_size"]
        batch_count = self.stats["batches_processed"]

        if batch_count == 1:
            self.stats["average_batch_size"] = batch_size
        else:
            alpha = 0.1
            self.stats["average_batch_size"] = (
                alpha * batch_size + (1 - alpha) * current_avg
            )

    def get_stream_stats(self, stream_id: str) -> Dict[str, Any]:
        """Get statistics for a specific stream."""
        buffer_size = len(self.result_buffer.get(stream_id, []))
        queue_size = (
            self.batch_queues.get(stream_id, asyncio.Queue()).qsize()
            if hasattr(asyncio.Queue, "qsize")
            else 0
        )

        return {
            "stream_id": stream_id,
            "buffer_size": buffer_size,
            "queue_size": queue_size,
            "is_active": stream_id in self.aggregation_tasks,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall aggregator statistics."""
        return {
            **self.stats,
            "active_streams": len(self.batch_queues),
            "stream_details": {
                sid: self.get_stream_stats(sid) for sid in self.batch_queues.keys()
            },
        }

    async def shutdown(self) -> None:
        """Shutdown all streams and clean up resources."""
        # Cancel all aggregation tasks
        for task in self.aggregation_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.aggregation_tasks.values(), return_exceptions=True)

        # Process any remaining buffered results
        for stream_id, buffer in self.result_buffer.items():
            if buffer:
                await self._process_batch(stream_id, buffer)

        self.batch_queues.clear()
        self.result_buffer.clear()
        self.aggregation_tasks.clear()

        logger.info("StreamingResultAggregator shutdown complete")
