"""
Advanced Probabilistic Models for UCUP Framework

This module implements sophisticated probabilistic reasoning capabilities including
Bayesian Networks, Markov Decision Processes, and Monte Carlo Tree Search.
"""

import asyncio
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .probabilistic import AgentState, ProbabilisticAgent, ProbabilisticResult
from .validation import (
    UCUPValidationError,
    validate_positive_number,
    validate_probability,
)


@dataclass
class ConditionalProbabilityTable:
    """Conditional Probability Table for Bayesian networks."""

    variables: List[str] = field(default_factory=list)
    probabilities: Dict[Tuple, float] = field(default_factory=dict)
    evidence: Dict[str, str] = field(default_factory=dict)

    def get_probability(self, assignment: Dict[str, str]) -> float:
        """Get probability for a variable assignment."""
        key = tuple(assignment[var] for var in self.variables)
        return self.probabilities.get(key, 0.0)

    def set_evidence(self, evidence: Dict[str, str]):
        """Set evidence variables."""
        self.evidence = evidence.copy()

    def marginalize(self, variable: str) -> "ConditionalProbabilityTable":
        """Marginalize out a variable."""
        remaining_vars = [v for v in self.variables if v != variable]
        new_probabilities = defaultdict(float)

        for assignment, prob in self.probabilities.items():
            assignment_dict = dict(zip(self.variables, assignment))
            if variable in assignment_dict:
                key = tuple(assignment_dict[v] for v in remaining_vars)
                new_probabilities[key] += prob

        return ConditionalProbabilityTable(
            variables=remaining_vars, probabilities=dict(new_probabilities)
        )

    def reduce(self, evidence: Dict[str, str]) -> "ConditionalProbabilityTable":
        """Reduce CPT given evidence."""
        new_probabilities = {}

        for assignment, prob in self.probabilities.items():
            assignment_dict = dict(zip(self.variables, assignment))

            # Check if assignment is consistent with evidence
            consistent = True
            for var, value in evidence.items():
                if var in assignment_dict and assignment_dict[var] != value:
                    consistent = False
                    break

            if consistent:
                # Remove evidence variables from assignment
                remaining_assignment = tuple(
                    assignment_dict[var]
                    for var in self.variables
                    if var not in evidence
                )
                remaining_vars = [v for v in self.variables if v not in evidence]
                key = tuple(remaining_assignment)
                new_probabilities[key] = prob

        return ConditionalProbabilityTable(
            variables=remaining_vars, probabilities=new_probabilities
        )


@dataclass
class BayesianNode:
    """A node in a Bayesian Network representing a random variable."""

    name: str
    parents: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    cpt: Dict[Tuple, float] = field(
        default_factory=dict
    )  # Conditional Probability Table

    def __post_init__(self):
        if not self.states:
            raise UCUPValidationError("BayesianNode must have at least one state")

        # Validate CPT probabilities sum to 1 for each parent configuration
        self._validate_cpt()

    def _validate_cpt(self):
        """Validate that CPT probabilities are valid."""
        parent_states = [self.parents] if self.parents else [()]

        for parent_combo in self._generate_parent_combinations():
            total_prob = 0.0
            for state in self.states:
                key = parent_combo + (state,)
                prob = self.cpt.get(key, 0.0)
                validate_probability(prob, f"cpt[{key}]")
                total_prob += prob

            if not math.isclose(total_prob, 1.0, abs_tol=1e-6):
                raise UCUPValidationError(
                    f"Probabilities for node {self.name} with parents {parent_combo} "
                    f"must sum to 1.0, got {total_prob}"
                )

    def _generate_parent_combinations(self) -> List[Tuple]:
        """Generate all possible parent state combinations."""
        if not self.parents:
            return [()]

        # This would need to be implemented based on parent node states
        # For simplicity, we'll assume parents are provided when needed
        return [()]


@dataclass
class BayesianNetwork:
    """A Bayesian Network for probabilistic inference."""

    nodes: Dict[str, BayesianNode] = field(default_factory=dict)
    evidence: Dict[str, str] = field(default_factory=dict)

    def add_node(self, node: BayesianNode):
        """Add a node to the network."""
        # Validate parent nodes exist
        for parent in node.parents:
            if parent not in self.nodes:
                raise UCUPValidationError(f"Parent node {parent} not found in network")

        self.nodes[node.name] = node

    def set_evidence(self, evidence: Dict[str, str]):
        """Set evidence for inference."""
        for node_name, state in evidence.items():
            if node_name not in self.nodes:
                raise UCUPValidationError(f"Node {node_name} not found in network")
            if state not in self.nodes[node_name].states:
                raise UCUPValidationError(
                    f"State {state} not valid for node {node_name}"
                )

        self.evidence = evidence.copy()

    def compute_marginal(self, query_node: str) -> Dict[str, float]:
        """Compute marginal probability distribution for a node given evidence."""
        if query_node not in self.nodes:
            raise UCUPValidationError(f"Query node {query_node} not found")

        # Simple enumeration for now - could be optimized with variable elimination
        return self._enumerate_all(query_node, set())

    def _enumerate_all(self, query_node: str, visited: Set[str]) -> Dict[str, float]:
        """Recursive enumeration for marginal computation."""
        if query_node in visited:
            return {}

        visited.add(query_node)
        node = self.nodes[query_node]

        if query_node in self.evidence:
            # Node has evidence, return delta function
            result = {state: 0.0 for state in node.states}
            result[self.evidence[query_node]] = 1.0
            return result

        # No evidence, sum over all possible values
        result = {state: 0.0 for state in node.states}

        for state in node.states:
            prob = 1.0

            # Multiply by CPT probability
            parent_states = tuple(self.evidence.get(p, "unknown") for p in node.parents)
            cpt_key = parent_states + (state,)
            prob *= node.cpt.get(cpt_key, 0.0)

            # Multiply by parent probabilities
            for parent in node.parents:
                if parent not in visited:
                    parent_dist = self._enumerate_all(parent, visited.copy())
                    if parent in self.evidence:
                        parent_prob = (
                            1.0
                            if parent_dist.get(self.evidence[parent], 0) > 0
                            else 0.0
                        )
                    else:
                        parent_prob = parent_dist.get(
                            self.evidence.get(parent, "unknown"), 0.0
                        )
                    prob *= parent_prob

            result[state] = prob

        # Normalize
        total = sum(result.values())
        if total > 0:
            for state in result:
                result[state] /= total

        return result


@dataclass
class MDPState:
    """A state in a Markov Decision Process."""

    name: str
    reward: float = 0.0
    is_terminal: bool = False
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MDPAction:
    """An action in a Markov Decision Process."""

    name: str
    cost: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MDPTransition:
    """A transition in a Markov Decision Process."""

    from_state: str
    action: str
    to_state: str
    probability: float
    reward: float = 0.0


class MarkovDecisionProcess:
    """A Markov Decision Process for decision making under uncertainty."""

    def __init__(self, discount_factor: float = 0.9):
        validate_probability(discount_factor, "discount_factor")
        self.discount_factor = discount_factor
        self.states: Dict[str, MDPState] = {}
        self.actions: Dict[str, MDPAction] = {}
        self.transitions: List[MDPTransition] = []
        self._transition_map: Dict[
            Tuple[str, str], List[Tuple[str, float, float]]
        ] = defaultdict(list)

    def add_state(self, state: MDPState):
        """Add a state to the MDP."""
        self.states[state.name] = state

    def add_action(self, action: MDPAction):
        """Add an action to the MDP."""
        self.actions[action.name] = action

    def add_transition(self, transition: MDPTransition):
        """Add a transition to the MDP."""
        validate_probability(transition.probability, "transition.probability")

        if transition.from_state not in self.states:
            raise UCUPValidationError(f"From state {transition.from_state} not found")
        if transition.to_state not in self.states:
            raise UCUPValidationError(f"To state {transition.to_state} not found")
        if transition.action not in self.actions:
            raise UCUPValidationError(f"Action {transition.action} not found")

        self.transitions.append(transition)
        self._transition_map[(transition.from_state, transition.action)].append(
            (transition.to_state, transition.probability, transition.reward)
        )

    def get_transitions(
        self, state: str, action: str
    ) -> List[Tuple[str, float, float]]:
        """Get all transitions for a state-action pair."""
        return self._transition_map.get((state, action), [])

    def value_iteration(
        self, epsilon: float = 1e-6, max_iterations: int = 1000
    ) -> Dict[str, float]:
        """Perform value iteration to compute optimal value function."""
        values = {
            state_name: (state.reward if state.is_terminal else 0.0)
            for state_name, state in self.states.items()
        }
        iteration = 0

        while iteration < max_iterations:
            delta = 0.0
            new_values = values.copy()

            for state_name, state in self.states.items():
                if state.is_terminal:
                    # Terminal states keep their reward values
                    continue

                max_value = float("-inf")
                for action_name in self.actions:
                    action_value = 0.0
                    transitions = self.get_transitions(state_name, action_name)

                    for next_state, prob, reward in transitions:
                        action_value += prob * (
                            reward + self.discount_factor * values[next_state]
                        )

                    max_value = max(max_value, action_value)

                if max_value != float("-inf"):
                    new_values[state_name] = max_value
                    delta = max(delta, abs(values[state_name] - new_values[state_name]))

            values = new_values
            iteration += 1

            if delta < epsilon:
                break

        return values

    def extract_policy(self, values: Dict[str, float]) -> Dict[str, str]:
        """Extract optimal policy from value function."""
        policy = {}

        for state_name, state in self.states.items():
            if state.is_terminal:
                policy[state_name] = None  # No action needed for terminal states
                continue

            best_action = None
            best_value = float("-inf")

            for action_name in self.actions:
                action_value = 0.0
                transitions = self.get_transitions(state_name, action_name)

                for next_state, prob, reward in transitions:
                    action_value += prob * (
                        reward + self.discount_factor * values[next_state]
                    )

                if action_value > best_value:
                    best_value = action_value
                    best_action = action_name

            policy[state_name] = best_action

        return policy


@dataclass
class MCTSNode:
    """A node in the Monte Carlo Tree Search."""

    state: Any
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action: Optional[Any] = None
    is_terminal: bool = False


class MonteCarloTreeSearch:
    """Monte Carlo Tree Search for probabilistic reasoning."""

    def __init__(self, exploration_constant: float = 1.4):
        validate_positive_number(exploration_constant, "exploration_constant")
        self.exploration_constant = exploration_constant

    async def search(self, root_state: Any, iterations: int = 1000) -> MCTSNode:
        """Perform MCTS search from root state."""
        root = MCTSNode(state=root_state)

        for _ in range(iterations):
            # Selection
            node = self._select(root)

            # Expansion
            if not node.is_terminal:
                node = self._expand(node)

            # Simulation
            reward = await self._simulate(node)

            # Backpropagation
            self._backpropagate(node, reward)

        return root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCB1."""
        while node.children:
            if not all(child.visits > 0 for child in node.children):
                # Select unvisited child
                for child in node.children:
                    if child.visits == 0:
                        return child

            # Select best child using UCB1
            best_child = max(node.children, key=self._ucb1_value)
            node = best_child

        return node

    def _ucb1_value(self, node: MCTSNode) -> float:
        """Calculate UCB1 value for node selection."""
        if node.visits == 0:
            return float("inf")

        exploitation = node.value / node.visits
        exploration = self.exploration_constant * math.sqrt(
            math.log(node.parent.visits) / node.visits
        )

        return exploitation + exploration

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child."""
        # This should be implemented based on the specific problem domain
        # For now, return the node itself
        return node

    async def _simulate(self, node: MCTSNode) -> float:
        """Run a simulation from the given node."""
        # This should be implemented based on the specific problem domain
        # For now, return a random reward
        return random.random()

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate the reward up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def get_best_action(self, root: MCTSNode) -> Any:
        """Get the best action from root node."""
        if not root.children:
            return None

        return max(root.children, key=lambda c: c.visits).action


class BayesianAgentNetwork:
    """Multi-agent system with probabilistic dependencies using Bayesian Networks."""

    def __init__(
        self, agents: List[ProbabilisticAgent], dependencies: Dict[str, List[str]]
    ):
        self.agents = {agent.__class__.__name__: agent for agent in agents}
        self.dependencies = dependencies
        self.bayesian_network = BayesianNetwork()

        # Build Bayesian network from agent dependencies
        self._build_network()

    def _build_network(self):
        """Build Bayesian network from agent dependencies."""
        # Create nodes for each agent
        for agent_name in self.agents:
            node = BayesianNode(
                name=agent_name,
                parents=self.dependencies.get(agent_name, []),
                states=["success", "failure"],
            )
            # Initialize simple CPT (could be learned from data)
            self._initialize_cpt(node)
            self.bayesian_network.add_node(node)

    def _initialize_cpt(self, node: BayesianNode):
        """Initialize conditional probability table for a node."""
        # Simple initialization - could be made more sophisticated
        if not node.parents:
            # No parents - assume 70% success rate
            node.cpt[("success",)] = 0.7
            node.cpt[("failure",)] = 0.3
        else:
            # With parents - success depends on parent success
            parent_combinations = [
                ("success",) * len(node.parents),
                ("failure",) * len(node.parents),
            ]

            for combo in parent_combinations:
                if all(state == "success" for state in combo):
                    node.cpt[combo + ("success",)] = 0.8
                    node.cpt[combo + ("failure",)] = 0.2
                else:
                    node.cpt[combo + ("success",)] = 0.4
                    node.cpt[combo + ("failure",)] = 0.6

    async def coordinate_with_uncertainty(self, task: Any) -> ProbabilisticResult:
        """Coordinate agents considering probabilistic dependencies."""
        # Set evidence based on current agent states (simplified)
        evidence = {}

        # Execute agents in topological order respecting dependencies
        results = {}
        executed = set()

        while len(executed) < len(self.agents):
            # Find agents whose dependencies are satisfied
            ready_agents = [
                name
                for name in self.agents
                if name not in executed
                and all(dep in executed for dep in self.dependencies.get(name, []))
            ]

            if not ready_agents:
                break  # Circular dependency or other issue

            # Execute ready agents
            tasks = []
            for agent_name in ready_agents:
                agent = self.agents[agent_name]
                task_coro = agent.execute(task)
                tasks.append((agent_name, task_coro))

            # Execute agents concurrently
            agent_results = await asyncio.gather(*[t[1] for t in tasks])

            # Process results
            for (agent_name, _), result in zip(tasks, agent_results):
                results[agent_name] = result
                executed.add(agent_name)

                # Update evidence for Bayesian network
                evidence[agent_name] = (
                    "success" if result.confidence > 0.5 else "failure"
                )

        # Compute overall confidence using Bayesian network
        if results:
            # Use Bayesian network to compute joint probability
            overall_confidence = self._compute_overall_confidence(evidence)
        else:
            overall_confidence = 0.0

        return ProbabilisticResult(
            success=len([r for r in results.values() if r.success]) > len(results) // 2,
            confidence=overall_confidence,
            alternatives=[],
            metadata={"agent_results": results, "evidence": evidence},
        )

    def _compute_overall_confidence(self, evidence: Dict[str, str]) -> float:
        """Compute overall confidence using Bayesian network."""
        self.bayesian_network.set_evidence(evidence)

        # Compute marginal for a representative agent or use joint probability
        # Simplified: average of individual agent confidences weighted by dependencies
        total_weight = 0.0
        weighted_confidence = 0.0

        for agent_name, state in evidence.items():
            weight = 1.0 + len(self.dependencies.get(agent_name, []))
            total_weight += weight

            if state == "success":
                weighted_confidence += (
                    weight * 0.8
                )  # Assume high confidence for success
            else:
                weighted_confidence += weight * 0.2  # Low confidence for failure

        return weighted_confidence / total_weight if total_weight > 0 else 0.0


class MDPBasedCoordinator:
    """Decision making under uncertainty with state transitions."""

    def __init__(self, agents: List[ProbabilisticAgent]):
        self.agents = agents
        self.mdp = MarkovDecisionProcess()
        self._build_mdp()

    def _build_mdp(self):
        """Build MDP from agent capabilities."""
        # Create states based on agent coordination scenarios
        scenarios = [
            "coordination_needed",
            "independent_execution",
            "conflict_resolution",
            "success",
            "failure",
        ]

        for scenario in scenarios:
            is_terminal = scenario in ["success", "failure"]
            reward = (
                1.0 if scenario == "success" else -1.0 if scenario == "failure" else 0.0
            )
            self.mdp.add_state(
                MDPState(scenario, reward=reward, is_terminal=is_terminal)
            )

        # Create actions based on coordination strategies
        strategies = [
            "delegate_to_best",
            "parallel_execution",
            "sequential_execution",
            "consensus_voting",
        ]
        for strategy in strategies:
            self.mdp.add_action(
                MDPAction(strategy, cost=0.1)
            )  # Small cost for coordination

        # Add transitions (simplified)
        transitions = [
            MDPTransition(
                "coordination_needed", "delegate_to_best", "success", 0.7, 0.5
            ),
            MDPTransition(
                "coordination_needed", "delegate_to_best", "failure", 0.3, -0.5
            ),
            MDPTransition(
                "coordination_needed", "parallel_execution", "success", 0.6, 0.4
            ),
            MDPTransition(
                "coordination_needed",
                "parallel_execution",
                "conflict_resolution",
                0.4,
                0.0,
            ),
            MDPTransition(
                "coordination_needed", "sequential_execution", "success", 0.8, 0.3
            ),
            MDPTransition(
                "coordination_needed", "sequential_execution", "failure", 0.2, -0.3
            ),
            MDPTransition(
                "coordination_needed", "consensus_voting", "success", 0.5, 0.6
            ),
            MDPTransition(
                "coordination_needed",
                "consensus_voting",
                "conflict_resolution",
                0.5,
                0.0,
            ),
        ]

        for transition in transitions:
            self.mdp.add_transition(transition)

    async def optimize_policy(self, environment_model: Dict[str, Any]) -> str:
        """Optimize coordination policy using MDP."""
        # Compute optimal value function
        values = self.mdp.value_iteration()

        # Extract optimal policy
        policy = self.mdp.extract_policy(values)

        # Return best action for current state
        current_state = self._assess_current_state(environment_model)
        return policy.get(current_state, "delegate_to_best")

    def _assess_current_state(self, environment_model: Dict[str, Any]) -> str:
        """Assess current coordination state from environment model."""
        # Simplified state assessment
        agent_count = len(self.agents)
        task_complexity = environment_model.get("complexity", 0.5)

        if agent_count > 1 and task_complexity > 0.7:
            return "coordination_needed"
        else:
            return "independent_execution"


class MCTSReasoner:
    """Probabilistic reasoning with tree search and rollout."""

    def __init__(self, max_rollout_depth: int = 10):
        self.max_rollout_depth = max_rollout_depth
        self.mcts = MonteCarloTreeSearch()

    async def search(self, root_state: Any, iterations: int = 1000) -> Any:
        """Perform MCTS-based reasoning."""
        root = await self.mcts.search(root_state, iterations)
        return self.mcts.get_best_action(root)

    async def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node with domain-specific actions."""
        # This should be implemented based on the specific reasoning domain
        # For now, create a dummy child
        child_state = f"{node.state}_expanded"
        child = MCTSNode(state=child_state, parent=node)
        node.children.append(child)
        return child

    async def _simulate(self, node: MCTSNode) -> float:
        """Run simulation with domain-specific rollout."""
        # This should be implemented based on the specific reasoning domain
        # For now, return random reward
        return random.random()


@dataclass
class PUCTNode:
    """
    MCTS Node with Polynomial Upper Confidence Trees (PUCT) algorithm support.

    PUCT uses neural network value and policy predictions for better tree search.
    """

    state: Any
    parent: Optional["PUCTNode"] = None
    children: List["PUCTNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    action: Optional[Any] = None
    is_terminal: bool = False
    prior_probability: float = 0.0  # From policy network
    state_value: Optional[float] = None  # From value network

    @property
    def value(self) -> float:
        """Average value of this node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_expanded(self) -> bool:
        """Check if this node has been expanded."""
        return len(self.children) > 0

    def select_child(self, c_puct: float = 1.0) -> "PUCTNode":
        """
        Select child using PUCT formula.

        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if not self.children:
            raise ValueError("Node has no children to select")

        best_child = None
        best_puct_value = float("-inf")

        sqrt_parent_visits = math.sqrt(self.visits)

        for child in self.children:
            if child.visits == 0:
                # Prioritize unvisited children
                return child

            # PUCT calculation
            q_value = child.value if child.visits > 0 else 0.0
            exploration_term = (
                c_puct
                * child.prior_probability
                * sqrt_parent_visits
                / (1 + child.visits)
            )
            puct_value = q_value + exploration_term

            if puct_value > best_puct_value:
                best_puct_value = puct_value
                best_child = child

        return best_child

    def expand(
        self, actions: List[Any], policy_probabilities: List[float]
    ) -> List["PUCTNode"]:
        """Expand node with given actions and policy probabilities."""
        if self.is_expanded():
            raise ValueError("Node is already expanded")

        for action, prior_prob in zip(actions, policy_probabilities):
            child = PUCTNode(
                state=None,  # Will be set when action is taken
                parent=self,
                action=action,
                prior_probability=prior_prob,
            )
            self.children.append(child)

        return self.children

    def backup(self, leaf_value: float):
        """Backup the leaf value up the tree."""
        current = self
        while current is not None:
            current.visits += 1
            current.value_sum += leaf_value
            current = current.parent


class AlphaZeroMCTS:
    """
    AlphaZero-style Monte Carlo Tree Search with neural network guidance.

    Combines MCTS with policy and value networks for superior performance.
    """

    def __init__(self, c_puct: float = 1.0, policy_network=None, value_network=None):
        self.c_puct = c_puct
        self.policy_network = policy_network  # Should predict action probabilities
        self.value_network = value_network  # Should predict state values

    async def search(self, root_state: Any, num_simulations: int = 800) -> PUCTNode:
        """
        Perform AlphaZero MCTS search.

        Args:
            root_state: The root state to search from
            num_simulations: Number of MCTS simulations to run

        Returns:
            Root node with search results
        """
        root = PUCTNode(state=root_state)

        # Get initial policy and value for root
        policy_probs, root_value = await self._evaluate_state(root_state)
        root.state_value = root_value

        # Expand root with policy
        actions = await self._get_possible_actions(root_state)
        root.expand(actions, policy_probs)

        # Run simulations
        for _ in range(num_simulations):
            await self._run_simulation(root)

        return root

    async def _run_simulation(self, root: PUCTNode):
        """Run a single MCTS simulation."""
        path = []
        current = root

        # Selection: traverse tree to leaf
        while current.is_expanded() and not current.is_terminal:
            current = current.select_child(self.c_puct)
            path.append(current)

        # Expansion: if node is not terminal, expand it
        if not current.is_terminal:
            await self._expand_node(current)
            if current.children:
                current = random.choice(current.children)
                path.append(current)

        # Evaluation: evaluate leaf node
        leaf_value = await self._evaluate_leaf(current)

        # Backup: propagate value back up the path
        for node in reversed(path):
            node.backup(leaf_value)

    async def _expand_node(self, node: PUCTNode):
        """Expand a node by getting its children."""
        if node.is_expanded():
            return

        # Get possible actions from this state
        actions = await self._get_possible_actions(node.state)

        if not actions:
            node.is_terminal = True
            return

        # Get policy probabilities for actions
        policy_probs, _ = await self._evaluate_state(node.state)

        # Ensure policy probs match actions
        if len(policy_probs) != len(actions):
            # Fallback to uniform policy
            policy_probs = [1.0 / len(actions)] * len(actions)

        node.expand(actions, policy_probs)

    async def _evaluate_leaf(self, node: PUCTNode) -> float:
        """Evaluate a leaf node."""
        if node.is_terminal:
            # Terminal nodes get exact value (assuming +1 for win, -1 for loss, 0 for draw)
            return await self._get_terminal_value(node.state)

        # Use value network or rollout
        if self.value_network is not None:
            _, value = await self._evaluate_state(node.state)
            return value
        else:
            # Simple rollout
            return await self._rollout(node.state)

    async def _evaluate_state(self, state: Any) -> Tuple[List[float], float]:
        """
        Evaluate state using neural networks.

        Returns:
            Tuple of (policy_probabilities, state_value)
        """
        if self.policy_network is not None and self.value_network is not None:
            # Use neural networks
            policy_output = await self._call_policy_network(state)
            value_output = await self._call_value_network(state)
            return policy_output, value_output
        else:
            # Fallback to uniform policy and zero value
            actions = await self._get_possible_actions(state)
            uniform_policy = [1.0 / len(actions)] * len(actions) if actions else []
            return uniform_policy, 0.0

    async def _call_policy_network(self, state: Any) -> List[float]:
        """Call policy network (placeholder - implement based on your NN)."""
        # This should interface with your actual policy network
        actions = await self._get_possible_actions(state)
        return [random.random() for _ in actions]  # Placeholder

    async def _call_value_network(self, state: Any) -> float:
        """Call value network (placeholder - implement based on your NN)."""
        # This should interface with your actual value network
        return random.random() * 2 - 1  # Random value between -1 and 1

    async def _get_possible_actions(self, state: Any) -> List[Any]:
        """Get possible actions from state (domain-specific)."""
        # This should be implemented based on your problem domain
        return ["action1", "action2", "action3"]  # Placeholder

    async def _rollout(self, state: Any) -> float:
        """Run a random rollout from state."""
        current_state = state
        depth = 0
        max_depth = 50  # Prevent infinite rollouts

        while depth < max_depth:
            if await self._is_terminal(current_state):
                return await self._get_terminal_value(current_state)

            actions = await self._get_possible_actions(current_state)
            if not actions:
                return 0.0  # No actions available

            # Random action selection
            action = random.choice(actions)
            current_state = await self._take_action(current_state, action)
            depth += 1

        return 0.0  # Max depth reached

    async def _is_terminal(self, state: Any) -> bool:
        """Check if state is terminal."""
        # Domain-specific implementation
        return False  # Placeholder

    async def _get_terminal_value(self, state: Any) -> float:
        """Get value of terminal state."""
        # Domain-specific implementation
        return 0.0  # Placeholder

    async def _take_action(self, state: Any, action: Any) -> Any:
        """Take action and return new state."""
        # Domain-specific implementation
        return state  # Placeholder

    def get_best_action(self, root: PUCTNode, temperature: float = 1.0) -> Any:
        """
        Get best action from search results using visit counts.

        Args:
            root: Root node after search
            temperature: Temperature for action selection (0 = greedy, higher = more random)

        Returns:
            Best action to take
        """
        if not root.children:
            return None

        if temperature == 0:
            # Greedy selection
            best_child = max(root.children, key=lambda c: c.visits)
        else:
            # Proportional selection with temperature
            visit_counts = [child.visits for child in root.children]
            if temperature != 1.0:
                visit_counts = [count ** (1.0 / temperature) for count in visit_counts]

            total = sum(visit_counts)
            probabilities = [count / total for count in visit_counts]

            # Sample from distribution
            r = random.random()
            cumulative = 0.0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    best_child = root.children[i]
                    break
            else:
                best_child = root.children[-1]

        return best_child.action


class QLearningMDP(MarkovDecisionProcess):
    """
    Q-Learning implementation for model-free reinforcement learning in MDPs.

    Learns optimal policy through interaction with environment without knowing transition dynamics.
    """

    def __init__(
        self,
        discount_factor: float = 0.9,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.1,
        q_table: Optional[Dict] = None,
    ):
        super().__init__(discount_factor)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table: Dict[Tuple[str, str], float] = q_table or {}

    def learn(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool = False,
    ) -> None:
        """
        Update Q-value using Q-learning update rule.

        Q(s,a) = Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        state_action = (state, action)

        # Initialize Q-value if not seen before
        if state_action not in self.q_table:
            self.q_table[state_action] = 0.0

        # Get current Q-value
        current_q = self.q_table[state_action]

        # Get max Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            next_actions = [a for a in self.actions.keys()]
            next_q_values = [
                self.q_table.get((next_state, a), 0.0) for a in next_actions
            ]
            max_next_q = max(next_q_values) if next_q_values else 0.0

        # Q-learning update
        target = reward + self.discount_factor * max_next_q
        self.q_table[state_action] = current_q + self.learning_rate * (
            target - current_q
        )

    def get_action(
        self, state: str, available_actions: Optional[List[str]] = None
    ) -> str:
        """
        Get action using epsilon-greedy policy.

        Args:
            state: Current state
            available_actions: List of available actions (if None, use all actions)

        Returns:
            Selected action
        """
        if available_actions is None:
            available_actions = list(self.actions.keys())

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Greedy exploitation
        best_action = None
        best_q_value = float("-inf")

        for action in available_actions:
            q_value = self.q_table.get((state, action), 0.0)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action or random.choice(available_actions)

    def get_policy(self) -> Dict[str, str]:
        """Extract deterministic policy from Q-table."""
        policy = {}

        # Group actions by state
        state_actions = defaultdict(list)
        for state, action in self.q_table.keys():
            state_actions[state].append(action)

        # For each state, pick action with highest Q-value
        for state, actions in state_actions.items():
            best_action = None
            best_q = float("-inf")

            for action in actions:
                q_value = self.q_table.get((state, action), 0.0)
                if q_value > best_q:
                    best_q = q_value
                    best_action = action

            if best_action:
                policy[state] = best_action

        return policy

    def get_value_function(self) -> Dict[str, float]:
        """Get state values from Q-table (V(s) = max_a Q(s,a))."""
        values = {}

        for state_action, q_value in self.q_table.items():
            state, _ = state_action
            if state not in values or q_value > values[state]:
                values[state] = q_value

        return values

    def decay_exploration(
        self, decay_rate: float = 0.995, min_rate: float = 0.01
    ) -> None:
        """Decay exploration rate over time."""
        self.exploration_rate = max(min_rate, self.exploration_rate * decay_rate)

    def save_q_table(self, filepath: str) -> None:
        """Save Q-table to file."""
        import json

        with open(filepath, "w") as f:
            # Convert tuple keys to strings for JSON serialization
            serializable_q = {
                f"{state}#{action}": value
                for (state, action), value in self.q_table.items()
            }
            json.dump(serializable_q, f)

    def load_q_table(self, filepath: str) -> None:
        """Load Q-table from file."""
        import json

        with open(filepath, "r") as f:
            serializable_q = json.load(f)
            self.q_table = {tuple(k.split("#")): v for k, v in serializable_q.items()}

    def get_q_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the Q-table."""
        if not self.q_table:
            return {"states_learned": 0, "state_action_pairs": 0, "avg_q_value": 0.0}

        states = set(state for state, _ in self.q_table.keys())
        q_values = list(self.q_table.values())

        return {
            "states_learned": len(states),
            "state_action_pairs": len(self.q_table),
            "avg_q_value": sum(q_values) / len(q_values),
            "max_q_value": max(q_values),
            "min_q_value": min(q_values),
        }


class DeepQLearningMDP(QLearningMDP):
    """
    Deep Q-Learning implementation using neural networks for Q-value approximation.

    Suitable for problems with large or continuous state spaces.
    """

    def __init__(
        self,
        discount_factor: float = 0.9,
        learning_rate: float = 0.001,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ):
        super().__init__(discount_factor, learning_rate, exploration_rate)

        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Experience replay buffer
        self.replay_memory: deque = deque(maxlen=memory_size)

        # Neural network components (placeholders - implement with your NN framework)
        self.q_network = None  # Main Q-network
        self.target_network = None  # Target Q-network for stability
        self.optimizer = None
        self.loss_fn = None

        # Training stats
        self.training_steps = 0

    def store_experience(
        self, state: str, action: str, reward: float, next_state: str, done: bool
    ) -> None:
        """Store experience tuple in replay memory."""
        experience = (state, action, reward, next_state, done)
        self.replay_memory.append(experience)

    def get_action(
        self, state: str, available_actions: Optional[List[str]] = None
    ) -> str:
        """
        Get action using epsilon-greedy with neural network Q-values.

        Args:
            state: Current state
            available_actions: Available actions (if None, use all)

        Returns:
            Selected action
        """
        if available_actions is None:
            available_actions = list(self.actions.keys())

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Use neural network to get Q-values
        if self.q_network is not None:
            q_values = self._get_q_values(state, available_actions)
            best_action_idx = q_values.argmax()
            return available_actions[best_action_idx]
        else:
            # Fallback to tabular Q-learning
            return super().get_action(state, available_actions)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_memory) < self.batch_size or self.q_network is None:
            return None

        # Sample batch from replay memory
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors (implementation depends on your NN framework)
        loss = self._train_network(states, actions, rewards, next_states, dones)

        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self._update_target_network()

        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration, self.exploration_rate * self.exploration_decay
        )

        return loss

    def _get_q_values(self, state: str, actions: List[str]) -> List[float]:
        """Get Q-values for state-action pairs (placeholder)."""
        # This should interface with your neural network
        # For now, return random values
        return [random.random() for _ in actions]

    def _train_network(self, states, actions, rewards, next_states, dones) -> float:
        """Train the neural network (placeholder)."""
        # This should implement the DQN training loop
        # Return dummy loss
        return random.random()

    def _update_target_network(self) -> None:
        """Update target network weights (placeholder)."""
        # Copy weights from main network to target network
        pass

    def save_model(self, filepath: str) -> None:
        """Save neural network model."""
        # Implementation depends on your NN framework
        pass

    def load_model(self, filepath: str) -> None:
        """Load neural network model."""
        # Implementation depends on your NN framework
        pass

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "training_steps": self.training_steps,
            "replay_memory_size": len(self.replay_memory),
            "exploration_rate": self.exploration_rate,
            "q_table_stats": self.get_q_table_stats(),
        }
