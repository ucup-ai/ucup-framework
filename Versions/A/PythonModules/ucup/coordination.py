"""
Coordination patterns for UCUP Framework.

This module provides flexible coordination strategies for multi-agent systems,
ranging from simple hierarchical arrangements to complex market-based
coordination and emergent swarm behaviors.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Import validation utilities
from .validation import (
    TypeValidator,
    UCUPValidationError,
    UCUPValueError,
    create_error_message,
    sanitize_inputs,
    validate_non_empty_string,
    validate_positive_number,
    validate_probability,
    validate_types,
)


class CoordinationContext(Enum):
    """Different contexts that influence coordination strategy choice."""

    COMPLEX_PROBLEM = "complex_problem"
    TIME_SENSITIVE = "time_sensitive"
    EXPLORATORY_TASK = "exploratory_task"
    RESOURCE_CONSTRAINED = "resource_constrained"
    HIGH_RELIABILITY_NEEDED = "high_reliability_needed"
    CREATIVE_TASK = "creative_task"
    ANALYTICAL_TASK = "analytical_task"


class MessageType(Enum):
    """Standardized message types for agent communication."""

    TASK_DELEGATION = "task_delegation"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    CONSENSUS_BUILDING = "consensus_building"
    CONFLICT_RESOLUTION = "conflict_resolution"
    STATUS_UPDATE = "status_update"
    RESOURCE_REQUEST = "resource_request"
    COORDINATION_SIGNAL = "coordination_signal"


@dataclass
class AgentMessage:
    """Standardized message format for inter-agent communication."""

    message_id: str = field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    message_type: MessageType = MessageType.STATUS_UPDATE
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # For linking related messages
    ttl: int = 10  # Time-to-live in processing steps

    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.ttl <= 0

    def decrement_ttl(self):
        """Decrement time-to-live."""
        self.ttl -= 1


@dataclass
class TaskDelegationMessage(AgentMessage):
    """Message for delegating tasks to other agents."""

    def __post_init__(self):
        self.message_type = MessageType.TASK_DELEGATION
        super().__post_init__()


@dataclass
class KnowledgeUpdateMessage(AgentMessage):
    """Message for sharing knowledge or learned information."""

    def __post_init__(self):
        self.message_type = MessageType.KNOWLEDGE_SHARING
        super().__post_init__()


@dataclass
class VoteRequestMessage(AgentMessage):
    """Message for requesting votes in consensus-building."""

    def __post_init__(self):
        self.message_type = MessageType.CONSENSUS_BUILDING
        super().__post_init__()


@dataclass
class MediationRequestMessage(AgentMessage):
    """Message for requesting conflict mediation."""

    def __post_init__(self):
        self.message_type = MessageType.CONFLICT_RESOLUTION
        super().__post_init__()


class AgentId:
    """Identifier for agents in coordination systems."""

    def __init__(self, agent_id: str, agent_type: str = "generic"):
        """Initialize AgentId with validation."""
        try:
            validate_non_empty_string(agent_id, "agent_id")
            validate_non_empty_string(agent_type, "agent_type")
        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="AgentId Initialization",
                    action="Validating agent identifier",
                    details=str(e),
                    suggestion="Provide non-empty strings for both agent_id and agent_type",
                ),
                context={"agent_id": agent_id, "agent_type": agent_type},
            ) from e

        self.agent_id = agent_id
        self.agent_type = agent_type

    def __str__(self):
        return f"{self.agent_type}:{self.agent_id}"

    def __eq__(self, other):
        return isinstance(other, AgentId) and self.agent_id == other.agent_id

    def __hash__(self):
        return hash(self.agent_id)


class AgentBus(ABC):
    """Abstract base class for agent communication buses."""

    @abstractmethod
    async def broadcast(self, message: AgentMessage) -> List[AgentMessage]:
        """Broadcast a message to all agents."""
        pass

    @abstractmethod
    async def send_direct(
        self, recipient: AgentId, message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Send a message directly to a specific agent."""
        pass

    @abstractmethod
    async def subscribe_to_topic(self, topic: str, callback: Callable):
        """Subscribe to messages on a specific topic."""
        pass


class CoordinationAgent(ABC):
    """Abstract base class for agents that participate in coordination."""

    def __init__(self, agent_id: AgentId, capabilities: Set[str] = None):
        self.agent_id = agent_id
        self.capabilities = capabilities or set()
        self.coordination_state: Dict[str, Any] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def process_task(self, task: Any, context: Dict[str, Any]) -> Any:
        """Process a task assigned by the coordination system."""
        pass

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages."""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        return None

    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler


class InMemoryAgentBus(AgentBus):
    """Simple in-memory implementation of agent communication bus."""

    def __init__(self):
        self.agents: Dict[AgentId, CoordinationAgent] = {}
        self.topic_subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)

    def register_agent(self, agent: CoordinationAgent):
        """Register an agent with the bus."""
        self.agents[agent.agent_id] = agent

    async def broadcast(self, message: AgentMessage) -> List[AgentMessage]:
        """Broadcast message to all registered agents."""

        responses = []
        message.recipient_id = None  # Ensure it's a broadcast

        for agent_id, agent in self.agents.items():
            if agent_id != AgentId(message.sender_id):  # Don't send to self
                try:
                    response = await agent.handle_message(message)
                    if response:
                        responses.append(response)
                except Exception as e:
                    self.logger.warning(
                        f"Agent {agent_id} failed to handle broadcast: {e}"
                    )

        return responses

    async def send_direct(
        self, recipient: AgentId, message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Send message directly to a specific agent."""

        agent = self.agents.get(recipient)
        if not agent:
            self.logger.warning(f"Agent {recipient} not found")
            return None

        try:
            return await agent.handle_message(message)
        except Exception as e:
            self.logger.error(f"Direct message to {recipient} failed: {e}")
            return None

    async def subscribe_to_topic(self, topic: str, callback: Callable):
        """Subscribe to messages on a specific topic."""
        self.topic_subscriptions[topic].append(callback)

    async def publish_to_topic(self, topic: str, message: AgentMessage):
        """Publish a message to a specific topic."""
        callbacks = self.topic_subscriptions.get(topic, [])
        for callback in callbacks:
            try:
                await callback(message)
            except Exception as e:
                self.logger.warning(f"Topic callback failed for {topic}: {e}")


class AgentMessageBus:
    """
    High-level interface for agent communication and coordination.

    Provides a standardized API for sending messages between agents,
    with support for different delivery patterns and message types.
    """

    def __init__(self, bus: AgentBus = None):
        self.bus = bus or InMemoryAgentBus()

    async def broadcast(self, message: AgentMessage) -> List[AgentMessage]:
        """Broadcast a message to all agents."""
        return await self.bus.broadcast(message)

    async def direct_message(
        self, to_agent: AgentId, message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Send a direct message to a specific agent."""
        return await self.bus.send_direct(to_agent, message)

    async def subscribe_to_topic(self, topic: str, callback: Callable):
        """Subscribe to messages on a topic."""
        return await self.bus.subscribe_to_topic(topic, callback)

    async def publish_to_topic(self, topic: str, message: AgentMessage):
        """Publish a message to a topic."""
        if hasattr(self.bus, "publish_to_topic"):
            return await self.bus.publish_to_topic(topic, message)


class HierarchicalCoordination:
    """
    Hierarchical coordination pattern where agents are organized in a tree structure.

    A manager agent delegates tasks to worker agents, collects results,
    and makes final decisions. This pattern is effective for complex,
    decomposable problems where clear authority and structured workflow
    are beneficial.
    """

    def __init__(
        self,
        manager_agent: CoordinationAgent,
        worker_agents: List[CoordinationAgent],
        approval_workflow: bool = False,
    ):
        self.manager_agent = manager_agent
        self.worker_agents = worker_agents
        self.approval_workflow = approval_workflow
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_collector: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    async def coordinate_task(self, task: Any, context: Dict[str, Any]) -> Any:
        """
        Coordinate task execution using hierarchical delegation with enhanced capabilities.

        Enhanced Process:
        1. Manager analyzes task complexity and decomposes it intelligently
        2. Manager delegates subtasks to appropriate workers with load balancing
        3. Workers execute their subtasks with progress tracking and error recovery
        4. Manager collects results with quality validation and conflict resolution
        5. Manager synthesizes final answer with confidence scoring
        6. Optional: Manager approval workflow with escalation paths
        7. Learning: Update delegation strategies based on performance
        """

        start_time = asyncio.get_event_loop().time()

        # Enhanced Step 1: Intelligent task decomposition with complexity analysis
        decomposition_result = await self._analyze_and_decompose_task(task, context)
        subtasks = decomposition_result["subtasks"]
        task_complexity = decomposition_result["complexity"]
        estimated_duration = decomposition_result["estimated_duration"]

        # Step 2: Enhanced task delegation with load balancing and prioritization
        delegation_plan = await self._plan_delegations(subtasks, context)
        delegation_tasks = []

        for delegation in delegation_plan:
            delegation_tasks.append(
                self._delegate_subtask_enhanced(delegation, context)
            )

        # Execute delegations with enhanced monitoring
        delegation_results = await self._execute_with_monitoring(
            delegation_tasks, estimated_duration
        )

        # Step 3: Enhanced result collection with validation and conflict resolution
        subtask_results = await self._collect_and_validate_results(
            delegation_results, context
        )

        # Step 4: Intelligent result synthesis with confidence scoring
        synthesis_result = await self._synthesize_with_confidence(
            task, subtask_results, context
        )
        final_result = synthesis_result["result"]
        confidence_score = synthesis_result["confidence"]

        # Step 5: Enhanced approval workflow with escalation
        if self.approval_workflow:
            approval_result = await self._enhanced_approval_workflow(
                final_result, confidence_score, context
            )
            final_result = approval_result["result"]
            confidence_score = approval_result["confidence"]

        # Step 6: Performance learning and adaptation
        execution_time = asyncio.get_event_loop().time() - start_time
        await self._learn_from_execution(
            subtasks, subtask_results, execution_time, context
        )

        # Step 7: Comprehensive result packaging
        return {
            "result": final_result,
            "confidence": confidence_score,
            "execution_time": execution_time,
            "subtask_count": len(subtasks),
            "task_complexity": task_complexity,
            "delegation_efficiency": self._calculate_delegation_efficiency(
                subtask_results
            ),
            "coordination_metadata": {
                "strategy": "hierarchical_enhanced",
                "worker_utilization": self._calculate_worker_utilization(
                    subtask_results
                ),
                "error_recovery_count": sum(
                    1 for r in subtask_results.values() if r.get("recovered", False)
                ),
                "quality_score": self._calculate_overall_quality(subtask_results),
            },
        }

    async def _decompose_task(
        self, task: Any, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose the main task into subtasks."""
        # This would typically call the manager agent to decompose
        # For demonstration, create mock subtasks
        return [
            {"description": f"Analyze aspect 1 of {task}", "priority": 1},
            {"description": f"Research aspect 2 of {task}", "priority": 2},
            {"description": f"Synthesize findings for {task}", "priority": 3},
        ]

    async def _delegate_subtask(
        self, task_id: str, subtask: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        """Delegate a subtask to an appropriate worker agent."""

        # Select worker based on capabilities and availability
        worker = self._select_worker(subtask)

        if not worker:
            raise ValueError(f"No suitable worker found for subtask {task_id}")

        # Create delegation message
        delegation_message = TaskDelegationMessage(
            sender_id=str(self.manager_agent.agent_id),
            recipient_id=str(worker.agent_id),
            payload={
                "task_id": task_id,
                "subtask": subtask,
                "context": context,
                "coordinator": "hierarchical",
            },
            priority=subtask.get("priority", 0),
        )

        # Send delegation (in practice, this would use the message bus)
        result = await worker.process_task(subtask, context)
        return result

    def _select_worker(self, subtask: Dict[str, Any]) -> Optional[CoordinationAgent]:
        """Select the most appropriate worker for a subtask."""
        # Simple capability matching - in practice, more sophisticated selection
        for worker in self.worker_agents:
            if (
                "analysis" in worker.capabilities
                and "analyze" in subtask["description"].lower()
            ):
                return worker
            elif (
                "research" in worker.capabilities
                and "research" in subtask["description"].lower()
            ):
                return worker

        # Fallback to first available worker
        return self.worker_agents[0] if self.worker_agents else None

    async def _synthesize_results(
        self,
        original_task: Any,
        subtask_results: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """Synthesize subtask results into final answer."""
        # Manager agent would typically handle this
        # For demonstration, combine results
        combined = {
            "original_task": original_task,
            "subtask_results": subtask_results,
            "synthesis_method": "hierarchical_combination",
            "coordination_type": "hierarchical",
        }
        return combined

    async def _approval_workflow(self, result: Any, context: Dict[str, Any]) -> Any:
        """Execute approval workflow if required."""
        # In practice, this would involve human approval or additional validation
        self.logger.info("Approval workflow completed")
        return result


class DebateCoordination:
    """
    Debate coordination pattern where agents argue different viewpoints.

    Agents present arguments for and against different approaches,
    then reach consensus through structured debate. Effective for
    complex decisions where multiple perspectives need consideration.
    """

    def __init__(
        self,
        agents: List[CoordinationAgent],
        consensus_strategy: Callable = None,
        max_rounds: int = 5,
        debate_topic_focus: float = 0.8,  # How focused on topic to stay
    ):
        """Initialize DebateCoordination with comprehensive validation."""
        try:
            # Validate agents list
            if not agents:
                raise UCUPValueError(
                    "At least one agent is required for debate coordination"
                )
            if len(agents) < 2:
                raise UCUPValueError("Debate coordination requires at least 2 agents")

            # Validate max_rounds
            validate_positive_number(max_rounds, "max_rounds")
            if not isinstance(max_rounds, int):
                raise UCUPTypeError("max_rounds must be an integer")

            # Validate debate_topic_focus
            validate_probability(debate_topic_focus, "debate_topic_focus")

        except (UCUPValueError, UCUPTypeError) as e:
            raise UCUPValueError(
                create_error_message(
                    context="DebateCoordination Initialization",
                    action="Validating debate parameters",
                    details=str(e),
                    suggestion="Provide at least 2 agents, positive integer max_rounds, and debate_topic_focus between 0.0-1.0",
                ),
                context={
                    "agent_count": len(agents) if "agents" in locals() else 0,
                    "max_rounds": max_rounds,
                    "debate_topic_focus": debate_topic_focus,
                },
            ) from e

        self.agents = agents
        self.consensus_strategy = consensus_strategy or self._majority_vote_consensus
        self.max_rounds = max_rounds
        self.debate_topic_focus = debate_topic_focus
        self.debate_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    async def coordinate_task(self, task: Any, context: Dict[str, Any]) -> Any:
        """
        Coordinate through structured debate and consensus building.
        """

        # Initialize debate positions
        positions = await self._initialize_positions(task, context)

        # Conduct debate rounds
        for round_num in range(self.max_rounds):
            round_arguments = await self._conduct_debate_round(
                task, positions, round_num, context
            )

            # Update positions based on arguments
            positions = await self._update_positions(positions, round_arguments)

            # Check for early consensus
            consensus = await self._check_consensus(positions)
            if consensus:
                break

        # Reach final consensus
        final_decision = await self.consensus_strategy(positions, self.debate_history)

        return {
            "decision": final_decision,
            "debate_rounds": len(self.debate_history),
            "positions": positions,
            "debate_history": self.debate_history,
            "coordination_type": "debate",
        }

    async def _initialize_positions(
        self, task: Any, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize debate positions for each agent."""
        positions = {}

        for agent in self.agents:
            position = await self._get_agent_position(agent, task, "initial", context)
            positions[str(agent.agent_id)] = position

        return positions

    async def _conduct_debate_round(
        self,
        task: Any,
        positions: Dict[str, Any],
        round_num: int,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Conduct one round of debate."""

        arguments = []

        # Each agent makes an argument considering others' positions
        debate_tasks = []
        for agent in self.agents:
            task = self._make_argument(agent, task, positions, round_num, context)
            debate_tasks.append(task)

        argument_results = await asyncio.gather(*debate_tasks, return_exceptions=True)

        for i, result in enumerate(argument_results):
            agent = self.agents[i]
            if isinstance(result, Exception):
                self.logger.warning(f"Agent {agent.agent_id} failed to argue: {result}")
                continue

            arguments.append(
                {
                    "agent_id": str(agent.agent_id),
                    "argument": result,
                    "round": round_num,
                    "timestamp": datetime.now(),
                }
            )

        # Record in debate history
        self.debate_history.append(
            {"round": round_num, "arguments": arguments, "positions": positions.copy()}
        )

        return arguments

    async def _make_argument(
        self,
        agent: CoordinationAgent,
        task: Any,
        positions: Dict[str, Any],
        round_num: int,
        context: Dict[str, Any],
    ) -> Any:
        """Have an agent make an argument in the debate."""

        argument_context = {
            **context,
            "debate_round": round_num,
            "current_positions": positions.copy(),
            "agent_position": positions.get(str(agent.agent_id)),
            "task": task,
        }

        return await agent.process_task(
            f"Make argument in debate about: {task}", argument_context
        )

    async def _update_positions(
        self, positions: Dict[str, Any], arguments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update agent positions based on new arguments."""

        updated_positions = positions.copy()

        # In a real implementation, agents would analyze arguments and update positions
        # For demonstration, keep positions stable
        return updated_positions

    async def _check_consensus(self, positions: Dict[str, Any]) -> bool:
        """Check if consensus has been reached."""

        # Simple consensus check - if all positions agree
        position_values = [pos.get("position") for pos in positions.values()]
        return len(set(position_values)) == 1

    async def _majority_vote_consensus(
        self, positions: Dict[str, Any], debate_history: List[Dict[str, Any]]
    ) -> Any:
        """Reach consensus through majority voting."""

        # Count votes for each position
        vote_counts = defaultdict(int)
        for position in positions.values():
            vote = position.get("position", "unknown")
            vote_counts[vote] += 1

        # Find majority
        majority_vote = max(vote_counts.items(), key=lambda x: x[1])
        return majority_vote[0]

    async def _get_agent_position(
        self,
        agent: CoordinationAgent,
        task: Any,
        position_type: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get an agent's position on a debate topic."""
        # In practice, this would call the agent
        return {
            "position": f"Sample position by {agent.agent_id}",
            "confidence": 0.8,
            "rationale": f"Agent {agent.agent_id} reasoning about {task}",
        }


class MarketBasedCoordination:
    """
    Market-based coordination using bidding and auctions.

    Agents bid for tasks based on their capabilities and resource costs.
    Tasks are allocated to the most efficient agents through competitive bidding.
    Effective when resources are limited or costs need optimization.
    """

    def __init__(
        self,
        agents: List[CoordinationAgent],
        bidding_strategy: Callable = None,
        task_allocation: Callable = None,
    ):
        self.agents = agents
        self.bidding_strategy = bidding_strategy or self._cost_quality_bidding
        self.task_allocation = task_allocation or self._auctions_with_constraints
        self.market_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    async def coordinate_task(self, task: Any, context: Dict[str, Any]) -> Any:
        """
        Coordinate through market-based bidding and allocation.
        """

        # Step 1: Task decomposition
        subtasks = await self._decompose_task_market(task, context)

        # Step 2: Bidding rounds
        bids = await self._collect_bids(subtasks, context)

        # Step 3: Task allocation
        allocations = await self.task_allocation(bids, context)

        # Step 4: Execute allocated tasks
        execution_results = await self._execute_allocations(allocations, context)

        # Step 5: Compensate agents (conceptually)
        compensations = await self._calculate_compensations(
            allocations, execution_results
        )

        return {
            "task_result": execution_results,
            "allocations": allocations,
            "market_efficiency": self._calculate_market_efficiency(bids, allocations),
            "total_compensation": sum(compensations.values()),
            "coordination_type": "market_based",
        }

    async def _decompose_task_market(
        self, task: Any, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose task for market-based allocation."""
        # Similar to hierarchical decomposition but focused on parallelizable units
        return [
            {
                "id": "subtask_1",
                "description": f"Component 1 of {task}",
                "estimated_effort": 5,
            },
            {
                "id": "subtask_2",
                "description": f"Component 2 of {task}",
                "estimated_effort": 3,
            },
            {
                "id": "subtask_3",
                "description": f"Component 3 of {task}",
                "estimated_effort": 7,
            },
        ]

    async def _collect_bids(
        self, subtasks: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect bids from agents for each subtask."""

        bids = {}

        bid_collection_tasks = []
        for agent in self.agents:
            task = self._collect_agent_bids(agent, subtasks, context)
            bid_collection_tasks.append(task)

        bid_results = await asyncio.gather(
            *bid_collection_tasks, return_exceptions=True
        )

        for i, result in enumerate(bid_results):
            agent = self.agents[i]
            if isinstance(result, Exception):
                self.logger.warning(f"Bidding failed for {agent.agent_id}: {result}")
                continue

            for bid in result:
                subtask_id = bid["subtask_id"]
                if subtask_id not in bids:
                    bids[subtask_id] = []
                bids[subtask_id].append(
                    {
                        **bid,
                        "agent_id": str(agent.agent_id),
                        "timestamp": datetime.now(),
                    }
                )

        return bids

    async def _collect_agent_bids(
        self,
        agent: CoordinationAgent,
        subtasks: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Collect bids from a single agent."""

        agent_bids = []
        for subtask in subtasks:
            # In practice, agent would evaluate the subtask and bid
            mock_bid = {
                "subtask_id": subtask["id"],
                "cost": np.random.uniform(1, 10),  # Mock bidding logic
                "quality_estimate": np.random.uniform(0.5, 1.0),
                "time_estimate": np.random.uniform(1, 5),
                "capability_match": 0.8,
            }
            agent_bids.append(mock_bid)

        return agent_bids

    async def _auctions_with_constraints(
        self, bids: Dict[str, List[Dict[str, Any]]], context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Allocate tasks through auctions with constraints."""

        allocations = {}

        for subtask_id, subtask_bids in bids.items():
            if not subtask_bids:
                continue

            # Find winning bid (lowest cost with quality constraint)
            qualified_bids = [
                bid for bid in subtask_bids if bid.get("quality_estimate", 0) >= 0.7
            ]

            if qualified_bids:
                winner = min(qualified_bids, key=lambda x: x["cost"])
                allocations[subtask_id] = winner["agent_id"]

        return allocations

    async def _execute_allocations(
        self, allocations: Dict[str, str], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the allocated tasks."""

        execution_tasks = []
        execution_map = {}  # Track which agent is executing which subtask

        # Group by agent for efficiency
        agent_tasks = defaultdict(list)
        for subtask_id, agent_id in allocations.items():
            agent_tasks[agent_id].append(subtask_id)
            execution_map[subtask_id] = agent_id

        # Execute tasks per agent
        results = {}
        for agent_id, subtasks in agent_tasks.items():
            agent = next((a for a in self.agents if str(a.agent_id) == agent_id), None)
            if agent:
                # Each agent executes their allocated subtasks
                for subtask_id in subtasks:
                    result = await agent.process_task(
                        f"Execute subtask {subtask_id}", {"market_allocated": True}
                    )
                    results[subtask_id] = result

        return results

    async def _calculate_compensations(
        self, allocations: Dict[str, str], execution_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate compensation for agents."""
        # In practice, this would involve actual payment logic
        compensations = {}
        for agent_id in set(allocations.values()):
            task_count = sum(1 for a in allocations.values() if a == agent_id)
            compensations[agent_id] = task_count * 1.0  # Mock compensation

        return compensations

    def _calculate_market_efficiency(
        self, bids: Dict[str, List[Dict[str, Any]]], allocations: Dict[str, str]
    ) -> float:
        """Calculate market efficiency metric."""

        total_allocated_cost = 0
        total_minimum_cost = 0

        for subtask_id, subtask_bids in bids.items():
            if subtask_id in allocations:
                # Cost of actual allocation
                winner_id = allocations[subtask_id]
                winner_bid = next(
                    (b for b in subtask_bids if b["agent_id"] == winner_id), None
                )
                if winner_bid:
                    total_allocated_cost += winner_bid["cost"]

            # Theoretical minimum cost
            if subtask_bids:
                min_cost_bid = min(subtask_bids, key=lambda x: x["cost"])
                total_minimum_cost += min_cost_bid["cost"]

        if total_minimum_cost > 0:
            efficiency = total_minimum_cost / total_allocated_cost
            return min(efficiency, 1.0)  # Cap at 1.0
        else:
            return 0.0

    async def _cost_quality_bidding(
        self,
        agent: CoordinationAgent,
        subtasks: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Default bidding strategy balancing cost and quality."""

        bids = []
        for subtask in subtasks:
            # Mock bidding decision
            can_handle = any(
                cap in agent.capabilities
                for cap in ["analysis", "research", "synthesis"]
            )

            if can_handle:
                bid = {
                    "subtask_id": subtask["id"],
                    "cost": subtask["estimated_effort"] * np.random.uniform(0.8, 1.2),
                    "quality_estimate": np.random.uniform(0.7, 0.95),
                    "time_estimate": subtask["estimated_effort"]
                    * np.random.uniform(0.9, 1.1),
                }
                bids.append(bid)

        return bids


class SwarmCoordination:
    """
    Swarm coordination for emergent collective intelligence.

    Agents interact locally and globally to create emergent patterns.
    Effective for exploratory tasks where the solution emerges from
    collective behavior rather than being explicitly designed.
    """

    def __init__(
        self,
        agents: List[CoordinationAgent],
        emergence_detector: Callable = None,
        collective_decision_aggregator: Callable = None,
    ):
        self.agents = agents
        self.emergence_detector = (
            emergence_detector or self._pattern_recognition_emergence
        )
        self.collective_decision_aggregator = (
            collective_decision_aggregator or self._wisdom_of_crowds
        )
        self.swarm_state: Dict[str, Any] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    async def coordinate_task(self, task: Any, context: Dict[str, Any]) -> Any:
        """
        Coordinate through swarm intelligence patterns.
        """

        # Initialize swarm
        await self._initialize_swarm(task, context)

        # Run swarm iterations
        max_iterations = context.get("max_swarm_iterations", 20)
        convergence_threshold = context.get("convergence_threshold", 0.8)

        for iteration in range(max_iterations):
            # Agents interact locally
            local_interactions = await self._local_swarm_interactions(context)

            # Global information propagation
            await self._global_information_flow(local_interactions)

            # Check for emergent patterns
            emergence = await self.emergence_detector(
                self.swarm_state,
                self.interaction_history[-10:] if self.interaction_history else [],
            )

            # Check for convergence
            if emergence.get("convergence_confidence", 0) > convergence_threshold:
                break

        # Aggregate collective decision
        final_decision = await self.collective_decision_aggregator(
            self.swarm_state, self.agents
        )

        return {
            "decision": final_decision,
            "swarm_iterations": iteration + 1,
            "emergence_patterns": emergence,
            "interaction_count": len(self.interaction_history),
            "coordination_type": "swarm",
        }

    async def _initialize_swarm(self, task: Any, context: Dict[str, Any]):
        """Initialize swarm state and agent configurations."""

        self.swarm_state = {
            "task": task,
            "phase": "initialization",
            "agent_states": {},
            "global_signals": [],
            "emergent_patterns": [],
        }

        # Initialize each agent with starting state
        for agent in self.agents:
            initial_state = await self._get_agent_swarm_state(agent, task, context)
            self.swarm_state["agent_states"][str(agent.agent_id)] = initial_state

    async def _local_swarm_interactions(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute local interactions between nearby agents."""

        interactions = []

        # Create interaction pairs (simplified neighborhood)
        for i, agent1 in enumerate(self.agents):
            # Interact with next 2 agents (circular)
            for j in range(1, 3):
                agent2_idx = (i + j) % len(self.agents)
                agent2 = self.agents[agent2_idx]

                interaction = await self._agent_agent_interaction(
                    agent1, agent2, context
                )
                interactions.append(interaction)

        return interactions

    async def _agent_agent_interaction(
        self,
        agent1: CoordinationAgent,
        agent2: CoordinationAgent,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle interaction between two agents."""

        # Create interaction message
        interaction_message = KnowledgeUpdateMessage(
            sender_id=str(agent1.agent_id),
            recipient_id=str(agent2.agent_id),
            payload={
                "interaction_type": "local_exchange",
                "state_exchange": self.swarm_state["agent_states"].get(
                    str(agent1.agent_id)
                ),
                "context": context,
            },
        )

        # Send interaction
        response = await agent2.handle_message(interaction_message)

        interaction_record = {
            "agent1": str(agent1.agent_id),
            "agent2": str(agent2.agent_id),
            "message": interaction_message,
            "response": response,
            "timestamp": datetime.now(),
        }

        self.interaction_history.append(interaction_record)

        return interaction_record

    async def _global_information_flow(self, local_interactions: List[Dict[str, Any]]):
        """Propagate information globally through the swarm."""

        # Extract global signals from local interactions
        global_signals = []
        for interaction in local_interactions:
            if interaction.get("response"):
                signal = {
                    "source_agents": [interaction["agent1"], interaction["agent2"]],
                    "signal_type": interaction["response"].message_type.value,
                    "content": interaction["response"].payload,
                    "strength": np.random.uniform(0.1, 1.0),  # Mock signal strength
                }
                global_signals.append(signal)

        # Update swarm state with global signals
        self.swarm_state["global_signals"].extend(global_signals)

        # Limit global signal history
        if len(self.swarm_state["global_signals"]) > 50:
            self.swarm_state["global_signals"] = self.swarm_state["global_signals"][
                -50:
            ]

    async def _get_agent_swarm_state(
        self, agent: CoordinationAgent, task: Any, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get initial swarm state for an agent."""
        return {
            "position": np.random.rand(2),  # 2D position in solution space
            "velocity": np.zeros(2),
            "confidence": np.random.uniform(0.5, 0.9),
            "current_solution": f"Agent {agent.agent_id} initial approach",
            "last_interaction": None,
        }

    async def _pattern_recognition_emergence(
        self, swarm_state: Dict[str, Any], recent_interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect emergent patterns in swarm behavior."""

        # Simple pattern recognition
        agent_states = swarm_state.get("agent_states", {})

        # Check for clustering in solution space
        positions = [state["position"] for state in agent_states.values()]
        if positions:
            centroid = np.mean(positions, axis=0)
            distances_to_centroid = [
                np.linalg.norm(pos - centroid) for pos in positions
            ]
            avg_distance = np.mean(distances_to_centroid)

            # Low average distance indicates clustering/alignment
            convergence_confidence = max(0, 1 - avg_distance)
        else:
            convergence_confidence = 0.0

        # Check for information flow patterns
        global_signals = swarm_state.get("global_signals", [])
        signal_density = len(global_signals) / max(1, len(recent_interactions))

        return {
            "convergence_confidence": convergence_confidence,
            "signal_density": signal_density,
            "agent_alignment": 1
            - np.std([s["confidence"] for s in agent_states.values()]),
            "patterns_detected": ["clustering"] if convergence_confidence > 0.7 else [],
        }

    async def _wisdom_of_crowds(
        self, swarm_state: Dict[str, Any], agents: List[CoordinationAgent]
    ) -> Any:
        """Aggregate collective wisdom from swarm."""

        agent_states = swarm_state.get("agent_states", {})

        # Collect agent solutions weighted by confidence
        solutions = []
        weights = []

        for agent in agents:
            state = agent_states.get(str(agent.agent_id), {})
            solution = state.get("current_solution")
            confidence = state.get("confidence", 0.5)

            if solution:
                solutions.append(solution)
                weights.append(confidence)

        if not solutions:
            return "No solutions generated by swarm"

        # Weighted random selection or averaging
        if weights:
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            selected_idx = np.random.choice(len(solutions), p=normalized_weights)
            return solutions[selected_idx]
        else:
            return solutions[0]


class AdaptiveOrchestrator:
    """
    Dynamically selects and switches coordination strategies based on context.

    Monitors task progress and agent performance to seamlessly switch between
    different coordination patterns as conditions change.
    """

    def __init__(
        self,
        agents: List[CoordinationAgent],
        strategy_selector: Callable,
        transition_handler: Callable = None,
    ):
        self.agents = agents
        self.strategy_selector = strategy_selector
        self.transition_handler = transition_handler or self._incremental_state_transfer
        self.current_strategy = None
        self.coordination_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    async def coordinate_adaptive(self, task: Any, context: Dict[str, Any]) -> Any:
        """
        Adaptively coordinate using different strategies based on context.
        """

        # Initial strategy selection
        initial_strategy = await self.strategy_selector(context)
        self.current_strategy = self._create_coordination_strategy(initial_strategy)

        # Execute initial phases
        result = await self.current_strategy.coordinate_task(task, context)

        # Monitor progress and potentially switch strategies
        while not self._is_task_complete(result, context):
            # Assess current progress
            progress_assessment = await self._assess_progress(result, context)

            # Check if strategy switch is needed
            if self._should_switch_strategy(progress_assessment, context):
                new_strategy_type = await self.strategy_selector(
                    {
                        **context,
                        "current_progress": progress_assessment,
                        "failed_attempts": self.coordination_history,
                    }
                )

                await self._switch_strategy(new_strategy_type, result)
                result = await self.current_strategy.coordinate_task(task, context)

            await asyncio.sleep(0.1)  # Prevent tight looping

        return result

    def _create_coordination_strategy(self, strategy_type: str) -> Any:
        """Create coordination strategy instance based on type."""

        if strategy_type == "hierarchical":
            return HierarchicalCoordination(
                manager_agent=self.agents[0],
                worker_agents=self.agents[1:],
                approval_workflow=False,
            )
        elif strategy_type == "debate":
            return DebateCoordination(agents=self.agents, max_rounds=3)
        elif strategy_type == "market":
            return MarketBasedCoordination(agents=self.agents)
        elif strategy_type == "swarm":
            return SwarmCoordination(agents=self.agents)
        else:
            # Default to hierarchical
            return HierarchicalCoordination(
                manager_agent=self.agents[0], worker_agents=self.agents[1:]
            )

    async def _switch_strategy(self, new_strategy_type: str, current_result: Any):
        """Switch to a new coordination strategy with state transfer."""

        old_strategy = self.current_strategy
        new_strategy = self._create_coordination_strategy(new_strategy_type)

        # Perform state transfer
        await self.transition_handler(old_strategy, new_strategy, current_result)

        # Record strategy switch
        self.coordination_history.append(
            {
                "from_strategy": type(old_strategy).__name__,
                "to_strategy": type(new_strategy).__name__,
                "transition_reason": "adaptive_switch",
                "timestamp": datetime.now(),
            }
        )

        self.current_strategy = new_strategy
        self.logger.info(f"Switched coordination strategy to {new_strategy_type}")

    async def _incremental_state_transfer(
        self, old_strategy: Any, new_strategy: Any, current_result: Any
    ):
        """Perform incremental state transfer between strategies."""
        # In practice, this would transfer relevant state information
        # For demonstration, just log the transfer
        self.logger.info("Performing incremental state transfer")

    async def _assess_progress(
        self, result: Any, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess current task progress."""
        # Mock progress assessment
        return {
            "completion_percentage": np.random.uniform(0.3, 0.8),
            "stalled": np.random.choice([True, False], p=[0.2, 0.8]),
            "strategy_effectiveness": np.random.uniform(0.6, 0.9),
        }

    def _should_switch_strategy(
        self, progress_assessment: Dict[str, Any], context: Dict[str, Any]
    ) -> bool:
        """Determine if strategy switch is needed."""
        if progress_assessment.get("stalled", False):
            return True
        if progress_assessment.get("strategy_effectiveness", 0.8) < 0.7:
            return True
        return False

    def _is_task_complete(self, result: Any, context: Dict[str, Any]) -> bool:
        """Check if task is complete."""
        # Simple completion check - would be more sophisticated in practice
        return np.random.choice([True, False], p=[0.1, 0.9])


# Example context-aware strategy selector
class ContextAwareStrategySelector:
    """
    Selects coordination strategy based on task context and requirements.
    """

    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        self.rules = rules or [
            {
                "condition": lambda ctx: ctx.get("complexity", 0.5) > 0.8,
                "strategy": "hierarchical",
            },
            {
                "condition": lambda ctx: ctx.get("time_sensitive", False),
                "strategy": "hierarchical",
            },
            {
                "condition": lambda ctx: ctx.get("exploratory", False),
                "strategy": "swarm",
            },
            {
                "condition": lambda ctx: ctx.get("resource_constrained", False),
                "strategy": "market",
            },
            {
                "condition": lambda ctx: ctx.get("multiple_perspectives", False),
                "strategy": "debate",
            },
        ]

    async def __call__(self, context: Dict[str, Any]) -> str:
        """Select appropriate strategy based on context."""

        for rule in self.rules:
            if rule["condition"](context):
                return rule["strategy"]

        # Default strategy
        return "hierarchical"


class SeamlessStrategyTransition:
    """
    Handles smooth transitions between coordination strategies.
    """

    def __init__(
        self,
        state_migration: Callable = None,
        communication_forwarding: Callable = None,
    ):
        self.state_migration = state_migration or self._default_state_migration
        self.communication_forwarding = (
            communication_forwarding or self._default_communication_forwarding
        )

    async def __call__(self, old_strategy: Any, new_strategy: Any, current_state: Any):
        """Perform seamless strategy transition."""

        # Migrate state
        await self.state_migration(old_strategy, new_strategy, current_state)

        # Set up communication forwarding if needed
        await self.communication_forwarding(old_strategy, new_strategy)

    async def _default_state_migration(
        self, old_strategy: Any, new_strategy: Any, current_state: Any
    ):
        """Default state migration between strategies."""
        # Implementation would transfer relevant state
        pass

    async def _default_communication_forwarding(
        self, old_strategy: Any, new_strategy: Any
    ):
        """Forward existing communications to new strategy."""
        # Implementation would handle message forwarding
        pass
