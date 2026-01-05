"""
Swarm Coordinator for UCUP.

Implements swarm intelligence coordination patterns using distributed agent networks.
"""

import time
import random
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from ..core.manager import UCUPManager


class SwarmAgent:
    """Individual agent in the swarm."""

    def __init__(self, agent_id: str, capabilities: List[str], initial_position: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.capabilities = set(capabilities)
        self.position = initial_position or {"x": 0.0, "y": 0.0, "z": 0.0}
        self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.fitness = 0.0
        self.neighbors: Set[str] = set()
        self.messages: List[Dict[str, Any]] = []
        self.last_update = time.time()
        self.status = "active"

    def update_position(self, new_position: Dict[str, Any]):
        """Update agent position."""
        self.position.update(new_position)
        self.last_update = time.time()

    def send_message(self, target_id: str, message: Dict[str, Any]):
        """Send message to another agent."""
        message.update({
            "sender": self.agent_id,
            "timestamp": time.time(),
            "type": "direct"
        })
        return {
            "target": target_id,
            "message": message
        }

    def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all neighbors."""
        message.update({
            "sender": self.agent_id,
            "timestamp": time.time(),
            "type": "broadcast"
        })
        return {
            "targets": list(self.neighbors),
            "message": message
        }

    def receive_message(self, message: Dict[str, Any]):
        """Receive message from another agent."""
        self.messages.append(message)
        # Keep only recent messages
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

    def calculate_fitness(self, task_requirements: Dict[str, Any]) -> float:
        """Calculate fitness for a given task."""
        fitness = 0.0

        # Capability matching
        required_caps = set(task_requirements.get("capabilities", []))
        if required_caps:
            matched_caps = len(self.capabilities.intersection(required_caps))
            fitness += matched_caps / len(required_caps) * 0.6

        # Distance penalty (prefer closer agents)
        if "target_position" in task_requirements:
            target = task_requirements["target_position"]
            distance = ((self.position["x"] - target["x"]) ** 2 +
                       (self.position["y"] - target["y"]) ** 2 +
                       (self.position["z"] - target["z"]) ** 2) ** 0.5
            fitness += max(0, 1.0 - distance / 100.0) * 0.3

        # Load balancing (prefer less busy agents)
        fitness += (1.0 - min(1.0, len(self.messages) / 10.0)) * 0.1

        self.fitness = fitness
        return fitness

    def __repr__(self) -> str:
        return f"SwarmAgent(id={self.agent_id}, fitness={self.fitness:.2f}, neighbors={len(self.neighbors)})"


class SwarmCoordinator:
    """Swarm intelligence coordinator using distributed agent networks."""

    def __init__(self, manager: UCUPManager, num_agents: int = 10):
        self.manager = manager
        self.config = manager.config
        self.agents: Dict[str, SwarmAgent] = {}
        self._executor = ThreadPoolExecutor(max_workers=num_agents)
        self._running = False
        self._lock = threading.RLock()

        # Swarm parameters
        self.cohesion_weight = 0.5
        self.separation_weight = 0.3
        self.alignment_weight = 0.2
        self.neighbor_radius = 50.0
        self.max_speed = 5.0

        self._initialize_swarm(num_agents)

    def _initialize_swarm(self, num_agents: int):
        """Initialize the swarm with agents."""
        base_capabilities = ["computation", "communication", "sensing"]

        for i in range(num_agents):
            agent_id = f"swarm_agent_{i:03d}"

            # Random initial position
            position = {
                "x": random.uniform(-100, 100),
                "y": random.uniform(-100, 100),
                "z": random.uniform(-50, 50)
            }

            # Random capabilities
            capabilities = base_capabilities.copy()
            if random.random() > 0.5:
                capabilities.append("vision")
            if random.random() > 0.5:
                capabilities.append("audio")
            if random.random() > 0.6:
                capabilities.append("nlp")

            self.agents[agent_id] = SwarmAgent(agent_id, capabilities, position)

        # Establish initial neighbor relationships
        self._update_neighbor_relationships()

    def coordinate_tasks(self, tasks: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Coordinate tasks using swarm intelligence.

        Args:
            tasks: List of task definitions
            **kwargs: Additional coordination parameters

        Returns:
            Coordination results
        """
        with self._lock:
            self._running = True
            start_time = time.time()

            try:
                # Initialize swarm behavior
                self._swarm_initialization(tasks)

                # Execute swarm coordination
                results = self._execute_swarm_coordination(tasks)

                # Cleanup and analysis
                self._swarm_cleanup()

                execution_time = time.time() - start_time

                return {
                    "success": True,
                    "results": results,
                    "execution_time": execution_time,
                    "total_tasks": len(tasks),
                    "swarm_size": len(self.agents),
                    "coordination_method": "swarm_intelligence"
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
            finally:
                self._running = False

    def _swarm_initialization(self, tasks: List[Dict[str, Any]]):
        """Initialize swarm behavior for task coordination."""
        # Update agent fitness for each task
        for task in tasks:
            for agent in self.agents.values():
                agent.calculate_fitness(task)

        # Update neighbor relationships
        self._update_neighbor_relationships()

    def _execute_swarm_coordination(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute swarm-based task coordination."""
        results = []

        for task in tasks:
            if not self._running:
                break

            # Find best agents for this task
            best_agents = self._select_best_agents_for_task(task, max_agents=3)

            if not best_agents:
                results.append({
                    "task_id": task.get("id"),
                    "status": "failed",
                    "error": "No suitable agents found"
                })
                continue

            # Coordinate task execution among selected agents
            task_result = self._coordinate_task_execution(task, best_agents)
            results.append(task_result)

        return results

    def _select_best_agents_for_task(self, task: Dict[str, Any], max_agents: int = 3) -> List[SwarmAgent]:
        """Select best agents for a given task."""
        # Calculate fitness for all agents
        agent_fitness = []
        for agent in self.agents.values():
            fitness = agent.calculate_fitness(task)
            agent_fitness.append((agent, fitness))

        # Sort by fitness and select top agents
        agent_fitness.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, fitness in agent_fitness[:max_agents]]

    def _coordinate_task_execution(self, task: Dict[str, Any], agents: List[SwarmAgent]) -> Dict[str, Any]:
        """Coordinate task execution among selected agents."""
        task_id = task.get("id", "unknown_task")

        # Submit task to agents concurrently
        futures = []
        for agent in agents:
            future = self._executor.submit(self._execute_task_on_agent, task, agent)
            futures.append((agent, future))

        # Wait for results and implement consensus
        completed_results = []
        for agent, future in futures:
            try:
                result = future.result(timeout=60)  # 1 minute timeout
                completed_results.append({
                    "agent_id": agent.agent_id,
                    "result": result,
                    "fitness": agent.fitness
                })
            except Exception as e:
                completed_results.append({
                    "agent_id": agent.agent_id,
                    "error": str(e),
                    "fitness": agent.fitness
                })

        # Consensus decision
        consensus_result = self._reach_consensus(completed_results)

        return {
            "task_id": task_id,
            "status": "completed",
            "consensus_result": consensus_result,
            "agent_results": completed_results,
            "agents_used": len(agents)
        }

    def _execute_task_on_agent(self, task: Dict[str, Any], agent: SwarmAgent) -> Any:
        """Execute task on a specific agent."""
        task_type = task.get("type", "generic")

        if task_type == "probabilistic":
            return self.manager.execute_probabilistic_task(task.get("parameters", {}))
        elif task_type == "multimodal":
            return self.manager.analyze_multimodal_data(task.get("data", []))
        else:
            # Simulate agent-specific processing
            time.sleep(random.uniform(0.1, 1.0))

            # Add some agent-specific variation
            base_result = random.random()
            agent_factor = agent.fitness * 0.1
            return base_result + agent_factor

    def _reach_consensus(self, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reach consensus from multiple agent results."""
        successful_results = [r for r in agent_results if "result" in r and "error" not in r]

        if not successful_results:
            return {"error": "No successful results from agents"}

        # Weighted average based on agent fitness
        total_weight = sum(r["fitness"] for r in successful_results)
        if total_weight == 0:
            # Equal weights if all fitnesses are 0
            consensus_value = sum(r["result"] for r in successful_results) / len(successful_results)
        else:
            consensus_value = sum(r["result"] * r["fitness"] for r in successful_results) / total_weight

        # Calculate confidence based on agreement
        values = [r["result"] for r in successful_results]
        variance = sum((v - consensus_value) ** 2 for v in values) / len(values)
        confidence = max(0, 1.0 - variance)  # Higher agreement = higher confidence

        return {
            "consensus_value": consensus_value,
            "confidence": confidence,
            "variance": variance,
            "agents_contributed": len(successful_results),
            "total_agents": len(agent_results)
        }

    def _update_neighbor_relationships(self):
        """Update neighbor relationships between agents."""
        for agent in self.agents.values():
            agent.neighbors.clear()

        # Calculate distances and establish neighbors
        agent_list = list(self.agents.values())
        for i, agent1 in enumerate(agent_list):
            for j, agent2 in enumerate(agent_list[i+1:], i+1):
                distance = self._calculate_distance(agent1.position, agent2.position)

                if distance <= self.neighbor_radius:
                    agent1.neighbors.add(agent2.agent_id)
                    agent2.neighbors.add(agent1.agent_id)

    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return ((pos1["x"] - pos2["x"]) ** 2 +
                (pos1["y"] - pos2["y"]) ** 2 +
                (pos1["z"] - pos2["z"]) ** 2) ** 0.5

    def _swarm_cleanup(self):
        """Clean up after swarm coordination."""
        # Clear old messages and update agent states
        for agent in self.agents.values():
            # Keep only recent messages
            recent_messages = [msg for msg in agent.messages
                             if time.time() - msg.get("timestamp", 0) < 300]  # 5 minutes
            agent.messages = recent_messages

    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status."""
        total_neighbors = sum(len(agent.neighbors) for agent in self.agents.values())
        avg_fitness = sum(agent.fitness for agent in self.agents.values()) / len(self.agents)

        return {
            "swarm_size": len(self.agents),
            "total_connections": total_neighbors // 2,  # Divide by 2 since undirected
            "average_fitness": avg_fitness,
            "average_neighbors": total_neighbors / len(self.agents),
            "cohesion_weight": self.cohesion_weight,
            "separation_weight": self.separation_weight,
            "alignment_weight": self.alignment_weight
        }

    def update_swarm_parameters(self, **params):
        """Update swarm behavior parameters."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)

        # Reinitialize relationships if radius changed
        if "neighbor_radius" in params:
            self._update_neighbor_relationships()

    def shutdown(self):
        """Shutdown the swarm coordinator."""
        with self._lock:
            self._running = False
            if self._executor:
                self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        status = self.get_swarm_status()
        return f"SwarmCoordinator(agents={status['swarm_size']}, avg_fitness={status['average_fitness']:.2f})"
