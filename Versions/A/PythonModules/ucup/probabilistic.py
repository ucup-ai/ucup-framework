"""
Probabilistic reasoning core for UCUP Framework.

This module provides base classes and utilities for building agents that embrace
uncertainty rather than fighting it.

Key Components:
- ProbabilisticAgent: Base class for uncertainty-aware agents
- ProbabilisticResult: Result container with confidence quantification
- AlternativePath: Alternative reasoning paths with confidence scores
- Reasoning strategies: Chain-of-thought, Tree-of-thought, Bayesian updates
- Advanced models: Bayesian networks, MDPs, Monte Carlo Tree Search
"""

# Enhanced __all__ exports with comprehensive metadata
__all__ = [
    # Core Classes
    "ProbabilisticAgent",
    "ProbabilisticResult",
    "AlternativePath",
    "AgentState",
    # Reasoning Strategies
    "ReasoningStrategy",
    # LLM Interface
    "LLMInterface",
    "DummyLLM",
    # Context and Prompt Management
    "ContextAwareSelector",
    "DynamicPromptRegistry",
    # Advanced Models
    "BayesianNetwork",
    "BayesianNetworkNode",
    "MarkovDecisionProcess",
    "MDPState",
    "MDPAction",
    "MonteCarloTreeSearch",
    "MCTSNode",
    # Agent Implementations
    "BayesianAgent",
    "MDPAgent",
    "MCTSReasoner",
    # Enhanced Uncertainty Quantification (ADK Integration)
    "AdkProbabilisticResult",
    "ConfidenceDistribution",
    "AlternativePrediction",
    "ConfidenceRanges",
    "VolatilityMetrics",
    "AdkUncertaintyAnalyzer",
]

# Component metadata for API discovery
__component_metadata__ = {
    "ProbabilisticAgent": {
        "description": "Base class for probabilistic agents with uncertainty quantification",
        "category": "probabilistic",
        "stability": "stable",
        "examples": [
            "agent = ProbabilisticAgent()",
            'result = await agent.execute("task")',
            'print(f"Confidence: {result.confidence:.2f}")',
        ],
        "related_components": [
            "ProbabilisticResult",
            "DecisionTracer",
            "AlternativePath",
        ],
        "tags": {"class", "agent", "probabilistic", "core"},
    },
    "ProbabilisticResult": {
        "description": "Container for probabilistic computation results with confidence",
        "category": "probabilistic",
        "stability": "stable",
        "examples": [
            'result = ProbabilisticResult(value="answer", confidence=0.85)',
            "alternatives = result.alternatives",
        ],
        "related_components": ["AlternativePath", "ProbabilisticAgent"],
        "tags": {"class", "result", "probabilistic"},
    },
    "BayesianNetwork": {
        "description": "Bayesian network for probabilistic inference and reasoning",
        "category": "advanced_probabilistic",
        "stability": "stable",
        "examples": [
            "network = BayesianNetwork()",
            'node = network.add_node("variable")',
            "beliefs = network.infer(evidence, queries)",
        ],
        "related_components": ["BayesianNetworkNode", "BayesianAgent"],
        "tags": {"class", "bayesian", "inference", "advanced"},
    },
    "MarkovDecisionProcess": {
        "description": "MDP for sequential decision making under uncertainty",
        "category": "advanced_probabilistic",
        "stability": "stable",
        "examples": [
            "mdp = MarkovDecisionProcess()",
            'mdp.add_state("state1")',
            "values = mdp.value_iteration()",
        ],
        "related_components": ["MDPState", "MDPAction", "MDPAgent"],
        "tags": {"class", "mdp", "decision_making", "advanced"},
    },
    "MonteCarloTreeSearch": {
        "description": "Monte Carlo Tree Search for complex decision exploration",
        "category": "advanced_probabilistic",
        "stability": "stable",
        "examples": [
            "mcts = MonteCarloTreeSearch()",
            "root = mcts.search(root_state, available_actions_func, ...)",
        ],
        "related_components": ["MCTSNode", "MCTSReasoner"],
        "tags": {"class", "mcts", "search", "advanced"},
    },
    "AdkUncertaintyAnalyzer": {
        "description": "Enhanced uncertainty analyzer for ADK operations",
        "category": "advanced_probabilistic",
        "stability": "experimental",
        "examples": [
            "analyzer = AdkUncertaintyAnalyzer()",
            "result = await analyzer.analyze_adk_result(adk_output)",
        ],
        "related_components": ["AdkProbabilisticResult", "ConfidenceDistribution"],
        "tags": {"class", "analyzer", "adk", "experimental"},
    },
}

import asyncio
import logging
import math
import random
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np


class ReasoningStrategy(Enum):
    """Built-in reasoning strategies for probabilistic exploration."""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    STEP_BACK_QUESTIONING = "step_back_questioning"
    TOTAL_PROBABILITY = "total_probability"
    BAYESIAN_UPDATES = "bayesian_updates"


@dataclass
class ProbabilisticResult:
    """Result from a probabilistic agent execution."""

    value: Any
    confidence: float
    alternatives: List["AlternativePath"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class AlternativePath:
    """An alternative path considered during reasoning."""

    value: Any
    confidence: float
    reasoning_steps: List[str] = field(default_factory=list)
    exploration_cost: float = 0.0


@dataclass
class AgentState:
    """Full probabilistic state of an agent execution."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Task completion tracking
    task_completion_confidence: float = 0.0
    subgoal_confidence_scores: Dict[str, float] = field(default_factory=dict)
    subgoals_completed: List[str] = field(default_factory=list)

    # Alternative considerations
    alternative_paths: List[AlternativePath] = field(default_factory=list)
    uncertainty_heatmap: Dict[str, float] = field(default_factory=dict)

    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    current_step: str = ""

    def update_confidence(self, subtask: str, confidence: float):
        """Update confidence score for a specific subtask."""
        self.subgoal_confidence_scores[subtask] = confidence
        self.last_updated = datetime.now()
        # Recalculate overall confidence
        if self.subgoal_confidence_scores:
            self.task_completion_confidence = np.mean(
                list(self.subgoal_confidence_scores.values())
            )


class LLMInterface(Protocol):
    """Protocol for LLM interfaces that support confidence scoring."""

    async def generate_with_confidence(self, prompt: str) -> Tuple[str, float]:
        """Generate a response with confidence score."""
        ...


class DummyLLM(LLMInterface):
    """Dummy LLM for testing - replace with actual LLM integration."""

    def __init__(self, base_confidence: float = 0.8, variance: float = 0.1):
        self.base_confidence = base_confidence
        self.variance = variance

    async def generate_with_confidence(self, prompt: str) -> Tuple[str, float]:
        """Generate dummy response with random confidence."""
        # Simulate processing delay
        await asyncio.sleep(0.1)

        # Generate response based on prompt length
        confidence = np.clip(
            np.random.normal(self.base_confidence, self.variance), 0.0, 1.0
        )

        response = f"Response to: {prompt[:50]}..."
        return response, confidence


class ProbabilisticAgent(ABC):
    """
    Base class for probabilistic agents.

    Agents built with this framework embrace uncertainty by:
    - Tracking confidence at every step
    - Exploring multiple reasoning paths
    - Adapting behavior based on confidence levels
    - Providing alternative interpretations
    """

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        reasoning_strategies: List[Union[str, ReasoningStrategy]] = None,
        exploration_budget: float = 0.2,
        fallback_strategy: str = "ensemble_voting",
        min_confidence_threshold: float = 0.7,
        max_alternatives: int = 3,
    ):
        self.llm = llm or DummyLLM()
        self.reasoning_strategies = [
            ReasoningStrategy(s) if isinstance(s, str) else s
            for s in (reasoning_strategies or [ReasoningStrategy.CHAIN_OF_THOUGHT])
        ]
        self.exploration_budget = exploration_budget
        self.fallback_strategy = fallback_strategy
        self.min_confidence_threshold = min_confidence_threshold
        self.max_alternatives = max_alternatives

        self.state = AgentState()
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def execute(self, task: str, **kwargs) -> ProbabilisticResult:
        """
        Execute the agent's task with probabilistic reasoning.

        Subclasses should implement their specific logic while using
        the framework's probabilistic utilities.
        """
        pass

    async def _generate_with_confidence(
        self, prompt: str, reasoning_strategy: Optional[ReasoningStrategy] = None
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Generate response with confidence using specified reasoning strategy."""

        # Select reasoning strategy if not provided
        strategy = reasoning_strategy or self.reasoning_strategies[0]

        enhanced_prompt = self._enhance_prompt_with_strategy(prompt, strategy)

        result, confidence = await self.llm.generate_with_confidence(enhanced_prompt)

        # Generate fake reasoning trace for demonstration
        reasoning_trace = [
            {
                "step": 1,
                "thinking": f"Analyzed the prompt using {strategy.value}",
                "confidence": confidence,
            }
        ]

        return result, confidence, reasoning_trace

    def _enhance_prompt_with_strategy(
        self, prompt: str, strategy: ReasoningStrategy
    ) -> str:
        """Enhance prompt based on reasoning strategy."""

        strategy_prompts = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: """
Break down your reasoning step by step:
1. Understand the core problem
2. Break it into components
3. Reason through each component systematically
4. Synthesize the final answer with confidence assessment
""",
            ReasoningStrategy.TREE_OF_THOUGHT: """
Explore multiple reasoning paths simultaneously:
- Consider different approaches
- Evaluate each path's likelihood
- Combine the strongest elements
""",
            ReasoningStrategy.STEP_BACK_QUESTIONING: """
Ask: What are the fundamental principles here?
Ask: What related problems have I seen before?
Connect to broader understanding.
""",
            ReasoningStrategy.TOTAL_PROBABILITY: """
Consider conditional probabilities:
- What are the mutually exclusive outcomes?
- What are the probabilities of reaching those outcomes?
- Use total probability theorem if applicable.
""",
            ReasoningStrategy.BAYESIAN_UPDATES: """
Start with prior beliefs.
Consider new evidence.
Update beliefs using Bayes' theorem.
Condition on relevant information.
""",
        }

        enhanced = f"""{strategy_prompts[strategy]}

Original Query: {prompt}

Please provide your answer with a confidence score between 0 and 1.
"""
        return enhanced

    async def low_confidence_workflow(
        self, initial_result: str, initial_confidence: float, task: str
    ) -> ProbabilisticResult:
        """
        Handle cases where initial confidence is too low.

        This could involve:
        - Seeking clarification from user
        - Exploring alternative approaches
        - Using fallback strategies
        - Escalating to more sophisticated agents
        """

        alternatives = []

        # Explore alternative reasoning strategies
        if len(self.reasoning_strategies) > 1:
            for strategy in self.reasoning_strategies[1 : self.max_alternatives + 1]:
                try:
                    (
                        alt_result,
                        alt_confidence,
                        _,
                    ) = await self._generate_with_confidence(
                        f"Alternative approach using {strategy.value}: {task}", strategy
                    )

                    if alt_confidence > initial_confidence:
                        alternatives.append(
                            AlternativePath(
                                value=alt_result,
                                confidence=alt_confidence,
                                reasoning_steps=[f"Used {strategy.value} reasoning"],
                                exploration_cost=self.exploration_budget * 0.3,
                            )
                        )
                except Exception as e:
                    self.logger.warning(f"Alternative strategy failed: {e}")

        # Use ensemble voting if we have alternatives
        if alternatives and self.fallback_strategy == "ensemble_voting":
            # Simple weighted average
            weights = np.array([alt.confidence for alt in alternatives])
            weights = weights / weights.sum()

            # For text results, pick the highest confidence alternative
            best_alt = max(alternatives, key=lambda x: x.confidence)

            final_result = best_alt.value
            final_confidence = best_alt.confidence

        else:
            # Stick with initial result but mark as low confidence
            final_result = initial_result
            final_confidence = initial_confidence

        return ProbabilisticResult(
            value=final_result,
            confidence=final_confidence,
            alternatives=alternatives,
            metadata={
                "low_confidence_handled": True,
                "fallback_strategy": self.fallback_strategy,
                "exploration_attempts": len(alternatives),
            },
        )

    def get_alternative_interpretations(self, task: str) -> List[AlternativePath]:
        """Get alternative interpretations of the task."""
        # This would typically involve NLP/ML analysis
        # For demonstration, return mock alternatives
        return [
            AlternativePath(
                value="Alternative interpretation 1",
                confidence=0.6,
                reasoning_steps=["Considered different assumptions"],
            ),
            AlternativePath(
                value="Alternative interpretation 2",
                confidence=0.4,
                reasoning_steps=["Explored edge cases"],
            ),
        ]

    def get_reasoning_steps(self) -> List[Dict[str, Any]]:
        """Get the current reasoning trace."""
        return [
            {
                "timestamp": self.state.last_updated,
                "step": self.state.current_step,
                "confidence": self.state.task_completion_confidence,
                "metadata": self.state.uncertainty_heatmap,
            }
        ]


class ContextAwareSelector:
    """
    Selects prompts and strategies based on context analysis.
    """

    def __init__(
        self,
        context_features: List[str] = None,
        selection_rules: Dict[str, Callable] = None,
    ):
        self.context_features = context_features or [
            "task_complexity",
            "domain",
            "past_success_rates",
        ]
        self.selection_rules = selection_rules or {}

    def select_prompt(self, context: Dict[str, Any]) -> str:
        """Select the best prompt based on context."""

        complexity = context.get("task_complexity", 0.5)

        if complexity > 0.8:
            return "analytical"
        elif complexity > 0.5:
            return "creative"
        else:
            return "cautious"

    def select_strategy(self, context: Dict[str, Any]) -> ReasoningStrategy:
        """Select the best reasoning strategy based on context."""

        success_rate = context.get("past_success_rate", 0.5)

        if success_rate > 0.8:
            return ReasoningStrategy.CHAIN_OF_THOUGHT
        elif success_rate > 0.5:
            return ReasoningStrategy.TREE_OF_THOUGHT
        else:
            return ReasoningStrategy.STEP_BACK_QUESTIONING


class DynamicPromptRegistry:
    """
    Registry for adaptive prompt management.
    """

    def __init__(
        self,
        prompts: Dict[str, str] = None,
        selector: Optional[ContextAwareSelector] = None,
    ):
        self.prompts = prompts or {
            "analytical": "You are an analytical thinker...",
            "creative": "You are a creative problem solver...",
            "cautious": "You carefully verify each step...",
        }
        self.selector = selector or ContextAwareSelector()
        self.performance_history: Dict[str, List[float]] = {}

    def get_best_prompt(self, context: Dict[str, Any]) -> str:
        """Get the best prompt for the given context."""
        prompt_key = self.selector.select_prompt(context)
        return self.prompts[prompt_key]

    def update_performance(self, prompt_key: str, success_rate: float):
        """Update performance tracking for prompt selection."""
        if prompt_key not in self.performance_history:
            self.performance_history[prompt_key] = []

        self.performance_history[prompt_key].append(success_rate)

        # Keep only recent history
        if len(self.performance_history[prompt_key]) > 10:
            self.performance_history[prompt_key] = self.performance_history[prompt_key][
                -10:
            ]


# Advanced Probabilistic Models


@dataclass
class BayesianNetworkNode:
    """Node in a Bayesian network."""

    name: str
    parents: List[str] = field(default_factory=list)
    cpt: Dict[Tuple, float] = field(
        default_factory=dict
    )  # Conditional Probability Table

    def add_parent(self, parent: str):
        """Add a parent node."""
        if parent not in self.parents:
            self.parents.append(parent)

    def set_cpt(self, conditions: Dict[str, Any], probability: float):
        """Set conditional probability for given parent conditions."""
        key = tuple(sorted(conditions.items()))
        self.cpt[key] = probability


class BayesianNetwork:
    """
    Bayesian Network for uncertainty modeling and probabilistic inference.

    Allows modeling complex probabilistic relationships between variables
    and performing inference to update beliefs based on evidence.
    """

    def __init__(self):
        self.nodes: Dict[str, BayesianNetworkNode] = {}
        self.node_order: List[str] = []  # Topological order

    def add_node(self, name: str, parents: List[str] = None) -> BayesianNetworkNode:
        """Add a node to the network."""
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")

        node = BayesianNetworkNode(name=name, parents=parents or [])
        self.nodes[name] = node

        # Validate that all parents exist
        for parent in node.parents:
            if parent not in self.nodes:
                raise ValueError(f"Parent node {parent} does not exist")

        # Update topological order (simple approach - may need refinement)
        if name not in self.node_order:
            self.node_order.append(name)

        return node

    def set_probability(
        self, node_name: str, conditions: Dict[str, Any], probability: float
    ):
        """Set conditional probability for a node given its parents."""
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist")

        node = self.nodes[node_name]
        node.set_cpt(conditions, probability)

    def infer(
        self, evidence: Dict[str, Any], query_nodes: List[str]
    ) -> Dict[str, float]:
        """
        Perform probabilistic inference using variable elimination.

        Simplified implementation - production version would use more sophisticated algorithms.
        """
        # This is a simplified implementation
        # Real Bayesian inference is computationally complex

        results = {}

        for query_node in query_nodes:
            if query_node in evidence:
                results[query_node] = 1.0 if evidence[query_node] else 0.0
            else:
                # Simple prior probability estimation
                results[query_node] = self._estimate_probability(query_node, evidence)

        return results

    def _estimate_probability(self, node_name: str, evidence: Dict[str, Any]) -> float:
        """Estimate probability for a node given evidence (simplified)."""
        node = self.nodes[node_name]

        # Count compatible conditions in CPT
        total_prob = 0.0
        count = 0

        for conditions, prob in node.cpt.items():
            compatible = True
            for cond_node, cond_value in conditions:
                if cond_node in evidence and evidence[cond_node] != cond_value:
                    compatible = False
                    break

            if compatible:
                total_prob += prob
                count += 1

        return total_prob / count if count > 0 else 0.5


@dataclass
class MDPState:
    """State in a Markov Decision Process."""

    name: str
    reward: float = 0.0
    is_terminal: bool = False


@dataclass
class MDPAction:
    """Action in a Markov Decision Process."""

    name: str
    transitions: Dict[str, float] = field(default_factory=dict)  # state -> probability


class MarkovDecisionProcess:
    """
    Markov Decision Process for decision making under uncertainty.

    Models sequential decision problems where outcomes are uncertain
    and rewards are received over time.
    """

    def __init__(self, discount_factor: float = 0.9):
        self.states: Dict[str, MDPState] = {}
        self.actions: Dict[str, MDPAction] = {}
        self.transitions: Dict[
            Tuple[str, str], Dict[str, float]
        ] = {}  # (state, action) -> {next_state: prob}
        self.rewards: Dict[
            Tuple[str, str, str], float
        ] = {}  # (state, action, next_state) -> reward
        self.discount_factor = discount_factor

    def add_state(self, name: str, reward: float = 0.0, is_terminal: bool = False):
        """Add a state to the MDP."""
        self.states[name] = MDPState(name=name, reward=reward, is_terminal=is_terminal)

    def add_action(self, name: str):
        """Add an action to the MDP."""
        self.actions[name] = MDPAction(name=name)

    def set_transition(
        self, state: str, action: str, next_state: str, probability: float
    ):
        """Set transition probability from state via action to next state."""
        if (state, action) not in self.transitions:
            self.transitions[(state, action)] = {}

        self.transitions[(state, action)][next_state] = probability

    def set_reward(self, state: str, action: str, next_state: str, reward: float):
        """Set reward for transitioning from state via action to next state."""
        self.rewards[(state, action, next_state)] = reward

    def value_iteration(
        self, max_iterations: int = 100, tolerance: float = 1e-6
    ) -> Dict[str, float]:
        """
        Solve the MDP using value iteration.

        Returns state values (expected discounted sum of future rewards).
        """
        # Initialize value function
        values = {state: 0.0 for state in self.states}

        for iteration in range(max_iterations):
            new_values = {}
            max_change = 0.0

            for state_name, state in self.states.items():
                if state.is_terminal:
                    new_values[state_name] = state.reward
                    continue

                # Find best action for this state
                best_value = float("-inf")

                for action_name in self.actions:
                    action_value = 0.0

                    if (state_name, action_name) in self.transitions:
                        for next_state, prob in self.transitions[
                            (state_name, action_name)
                        ].items():
                            reward = self.rewards.get(
                                (state_name, action_name, next_state), 0.0
                            )
                            action_value += prob * (
                                reward
                                + self.discount_factor * values.get(next_state, 0.0)
                            )

                    best_value = max(best_value, action_value)

                new_values[state_name] = (
                    best_value if best_value != float("-inf") else 0.0
                )
                max_change = max(
                    max_change, abs(new_values[state_name] - values[state_name])
                )

            values = new_values

            if max_change < tolerance:
                break

        return values

    def get_optimal_policy(self, values: Dict[str, float]) -> Dict[str, str]:
        """Extract optimal policy from value function."""
        policy = {}

        for state_name in self.states:
            if self.states[state_name].is_terminal:
                policy[state_name] = None
                continue

            best_action = None
            best_value = float("-inf")

            for action_name in self.actions:
                action_value = 0.0

                if (state_name, action_name) in self.transitions:
                    for next_state, prob in self.transitions[
                        (state_name, action_name)
                    ].items():
                        reward = self.rewards.get(
                            (state_name, action_name, next_state), 0.0
                        )
                        action_value += prob * (
                            reward + self.discount_factor * values.get(next_state, 0.0)
                        )

                if action_value > best_value:
                    best_value = action_value
                    best_action = action_name

            policy[state_name] = best_action

        return policy


@dataclass
class MCTSNode:
    """Node in Monte Carlo Tree Search."""

    state: Any
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    actions: List[Any] = field(
        default_factory=list
    )  # Available actions from this state

    @property
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been explored."""
        return len(self.children) == len(self.actions)

    def best_child(self, exploration_weight: float = 1.4) -> "MCTSNode":
        """Select best child using UCB1 formula."""
        if not self.children:
            return None

        def ucb_score(child: MCTSNode) -> float:
            if child.visits == 0:
                return float("inf")
            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )
            return exploitation + exploration

        return max(self.children, key=ucb_score)


class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search for probabilistic reasoning and planning.

    Uses random sampling to explore decision trees and find optimal paths
    through uncertain environments.
    """

    def __init__(self, exploration_weight: float = 1.4):
        self.exploration_weight = exploration_weight
        self.root: Optional[MCTSNode] = None

    def search(
        self,
        root_state: Any,
        available_actions_func: Callable,
        transition_func: Callable,
        reward_func: Callable,
        iterations: int = 1000,
    ) -> MCTSNode:
        """
        Perform MCTS search.

        Args:
            root_state: Initial state
            available_actions_func: Function that returns available actions for a state
            transition_func: Function that returns (next_state, reward) for (state, action)
            reward_func: Function that returns reward for a state
            iterations: Number of MCTS iterations

        Returns:
            Root node with search results
        """
        self.root = MCTSNode(
            state=root_state, actions=available_actions_func(root_state)
        )

        for _ in range(iterations):
            # Selection
            leaf = self._select(self.root)

            # Expansion
            if not leaf.is_fully_expanded:
                leaf = self._expand(leaf, available_actions_func, transition_func)

            # Simulation
            reward = self._simulate(
                leaf, available_actions_func, transition_func, reward_func
            )

            # Backpropagation
            self._backpropagate(leaf, reward)

        return self.root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB1."""
        current = node
        while current.children and current.is_fully_expanded:
            current = current.best_child(self.exploration_weight)
        return current

    def _expand(
        self,
        node: MCTSNode,
        available_actions_func: Callable,
        transition_func: Callable,
    ) -> MCTSNode:
        """Expand node by adding a new child."""
        # Find untried actions
        tried_actions = {child.state for child in node.children}
        available_actions = available_actions_func(node.state)

        for action in available_actions:
            if action not in tried_actions:
                next_state, _ = transition_func(node.state, action)
                child = MCTSNode(
                    state=next_state,
                    parent=node,
                    actions=available_actions_func(next_state),
                )
                node.children.append(child)
                return child

        return node

    def _simulate(
        self,
        node: MCTSNode,
        available_actions_func: Callable,
        transition_func: Callable,
        reward_func: Callable,
    ) -> float:
        """Run simulation from node to terminal state."""
        state = node.state
        total_reward = 0.0
        depth = 0
        max_depth = 50  # Prevent infinite simulations

        while depth < max_depth:
            actions = available_actions_func(state)
            if not actions:
                break

            # Random action selection
            action = random.choice(actions)
            next_state, reward = transition_func(state, action)

            total_reward += reward
            state = next_state
            depth += 1

        # Add terminal reward
        total_reward += reward_func(state)

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Propagate reward back up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def get_best_action(self, root: MCTSNode) -> Any:
        """Get the best action from root node."""
        if not root.children:
            return None

        # Return action that leads to most visited child
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.state  # This should be the action that led to this state


class BayesianAgent(ProbabilisticAgent):
    """
    Agent that uses Bayesian networks for uncertainty modeling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bayesian_network = BayesianNetwork()

    async def execute(self, task: str, **kwargs) -> ProbabilisticResult:
        """Execute task using Bayesian reasoning."""
        # Build or use existing Bayesian network for the task domain
        evidence = kwargs.get("evidence", {})
        query_variables = kwargs.get("query_variables", [])

        if not query_variables:
            # Default query
            query_variables = ["outcome"]

        beliefs = self.bayesian_network.infer(evidence, query_variables)

        # Convert to probabilistic result
        best_outcome = (
            max(beliefs.items(), key=lambda x: x[1]) if beliefs else ("unknown", 0.5)
        )

        return ProbabilisticResult(
            value=best_outcome[0],
            confidence=best_outcome[1],
            metadata={"inference_method": "bayesian_network", "all_beliefs": beliefs},
        )


class MDPAgent(ProbabilisticAgent):
    """
    Agent that uses Markov Decision Processes for sequential decision making.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mdp = MarkovDecisionProcess()

    async def execute(self, task: str, **kwargs) -> ProbabilisticResult:
        """Execute task using MDP-based planning."""
        current_state = kwargs.get("current_state", "start")
        max_iterations = kwargs.get("max_iterations", 100)

        # Solve MDP
        values = self.mdp.value_iteration(max_iterations)
        policy = self.mdp.get_optimal_policy(values)

        next_action = policy.get(current_state)

        confidence = values.get(current_state, 0.5)
        if confidence > 1.0:  # Normalize if needed
            confidence = min(confidence / 10.0, 1.0)  # Rough normalization

        return ProbabilisticResult(
            value=next_action or "no_action",
            confidence=confidence,
            metadata={
                "planning_method": "mdp",
                "state_values": values,
                "optimal_policy": policy,
            },
        )


class MCTSReasoner(ProbabilisticAgent):
    """
    Agent that uses Monte Carlo Tree Search for complex reasoning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcts = MonteCarloTreeSearch()

    async def execute(self, task: str, **kwargs) -> ProbabilisticResult:
        """Execute task using MCTS exploration."""
        root_state = kwargs.get("root_state", task)
        available_actions_func = kwargs.get(
            "available_actions_func", lambda s: ["explore", "analyze", "conclude"]
        )
        transition_func = kwargs.get("transition_func", lambda s, a: (f"{s}_{a}", 0.1))
        reward_func = kwargs.get("reward_func", lambda s: 0.5)
        iterations = kwargs.get("iterations", 100)

        root = self.mcts.search(
            root_state, available_actions_func, transition_func, reward_func, iterations
        )

        best_action = self.mcts.get_best_action(root)
        confidence = root.value / root.visits if root.visits > 0 else 0.5

        return ProbabilisticResult(
            value=best_action or "no_solution",
            confidence=min(confidence, 1.0),
            metadata={
                "reasoning_method": "mcts",
                "iterations": iterations,
                "root_visits": root.visits,
                "root_value": root.value,
            },
        )


# Enhanced Uncertainty Quantification for ADK Operations


@dataclass
class AdkProbabilisticResult(ProbabilisticResult):
    """Enhanced result for ADK operations with comprehensive uncertainty quantification."""

    confidence_distribution: "ConfidenceDistribution" = None
    uncertainty_score: float = 0.0
    robustness_score: float = 0.0
    confidence_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    volatility_metrics: "VolatilityMetrics" = None
    alternative_predictions: List["AlternativePrediction"] = field(default_factory=list)


@dataclass
class ConfidenceDistribution:
    """Statistical distribution of confidence scores for uncertainty quantification."""

    mean_confidence: float
    variance: float
    standard_deviation: float
    skewness: float = 0.0
    kurtosis: float = 0.0
    confidence_interval_95: Tuple[float, float] = (0.0, 1.0)
    distribution_type: str = "normal"
    sample_size: int = 1

    @property
    def uncertainty_score(self) -> float:
        """Calculate uncertainty score based on distribution characteristics."""
        # Higher variance and deviation from normality indicate more uncertainty
        variance_score = min(self.variance * 100, 1.0)
        normality_score = (
            abs(self.skewness) * 0.1 + max(0, abs(self.kurtosis) - 3) * 0.1
        )
        return min(variance_score + normality_score, 1.0)


@dataclass
class AlternativePrediction:
    """Alternative prediction with associated uncertainty."""

    value: Any
    confidence: float
    probability_mass: float
    robustness_metrics: Dict[str, float] = field(default_factory=dict)
    reasoning_context: List[str] = field(default_factory=list)


@dataclass
class ConfidenceRanges:
    """Confidence intervals and prediction bands for decision-making."""

    point_estimate: float
    confidence_interval_95: Tuple[float, float]
    prediction_interval: Tuple[float, float]
    tolerance_interval: Tuple[float, float]
    reliability_score: float

    def contains_value(self, value: float) -> bool:
        """Check if a value falls within the 95% confidence interval."""
        return self.confidence_interval_95[0] <= value <= self.confidence_interval_95[1]

    def get_decision_confidence(self, threshold: float = 0.7) -> float:
        """Calculate confidence level for decision-making above a threshold."""
        if self.confidence_interval_95[1] >= threshold:
            # Upper bound is above threshold
            overlap = min(
                1.0,
                (self.confidence_interval_95[1] - threshold)
                / (self.confidence_interval_95[1] - self.confidence_interval_95[0]),
            )
            return self.point_estimate * overlap
        else:
            # Upper bound is below threshold - very low decision confidence
            return max(0.01, self.point_estimate * 0.1)


@dataclass
class VolatilityMetrics:
    """Real-time confidence volatility and stability tracking."""

    current_volatility: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    stability_score: float  # 0.0 (highly volatile) to 1.0 (very stable)
    change_rate: float  # Rate of change per time unit
    rolling_std_dev: float
    outlier_count: int
    last_updated: datetime

    def get_stability_assessment(self) -> Dict[str, Any]:
        """Provide stability assessment for decision-making."""
        return {
            "stability_level": "high"
            if self.stability_score > 0.8
            else "medium"
            if self.stability_score > 0.6
            else "low",
            "volatility_risk": "high"
            if self.current_volatility > 0.3
            else "medium"
            if self.current_volatility > 0.15
            else "low",
            "trend_risk": "increasing"
            if self.trend_direction == "increasing" and self.change_rate > 0.1
            else "stable",
            "recommendation": self._get_recommendation(),
        }

    def _get_recommendation(self) -> str:
        """Generate recommendation based on current volatility metrics."""
        if self.stability_score > 0.8:
            return "High confidence in stability - proceed with decision"
        elif self.current_volatility > 0.3:
            return "High volatility detected - consider delaying decision or gathering more data"
        elif self.trend_direction == "decreasing" and self.change_rate > 0.1:
            return "Confidence trending down - reassess decision criteria"
        else:
            return "Monitor confidence closely before finalizing decision"


class AdkUncertaintyAnalyzer(ProbabilisticAgent):
    """
    Enhanced uncertainty analyzer specifically for ADK operations.

    Provides comprehensive probabilistic analysis including:
    - Confidence distribution estimation with uncertainty scoring
    - Alternative predictions handling for decision robustness
    - Confidence range calculations for better decision-making
    - Real-time confidence volatility and stability tracking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_history: List[Tuple[datetime, float]] = []
        self.alternative_history: List[List[AlternativePrediction]] = []
        self.volatility_window_size = kwargs.get("volatility_window_size", 20)
        self.adaptive_thresholds = kwargs.get("adaptive_thresholds", True)

    async def analyze_adk_result(
        self,
        adk_result: Any,
        operation_context: Dict[str, Any],
        historical_results: List[Any] = None,
    ) -> AdkProbabilisticResult:
        """
        Perform comprehensive probabilistic analysis of ADK operation results.

        Args:
            adk_result: Raw ADK operation result
            operation_context: Context information about the ADK operation
            historical_results: Previous results for trend analysis

        Returns:
            Enhanced probabilistic result with comprehensive uncertainty quantification
        """
        # Extract primary confidence from ADK result
        primary_confidence = self._extract_confidence(adk_result, operation_context)

        # Generate confidence distribution
        confidence_distribution = await self._estimate_confidence_distribution(
            primary_confidence, operation_context, historical_results
        )

        # Generate alternative predictions
        alternative_predictions = await self._generate_alternative_predictions(
            adk_result, operation_context, confidence_distribution
        )

        # Calculate confidence ranges
        confidence_ranges = self._calculate_confidence_ranges(
            confidence_distribution, alternative_predictions, operation_context
        )

        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(primary_confidence)

        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(
            primary_confidence, alternative_predictions, confidence_distribution
        )

        # Update time-series tracking
        self._update_confidence_history(primary_confidence)
        self._update_alternative_history(alternative_predictions)

        return AdkProbabilisticResult(
            value=adk_result,
            confidence=primary_confidence,
            confidence_distribution=confidence_distribution,
            uncertainty_score=confidence_distribution.uncertainty_score,
            robustness_score=robustness_score,
            confidence_ranges=confidence_ranges,
            volatility_metrics=volatility_metrics,
            alternative_predictions=alternative_predictions,
            metadata={
                "analysis_method": "adk_uncertainty_quantification",
                "operation_context": operation_context,
                "distribution_type": confidence_distribution.distribution_type,
                "robustness_factors": list(operation_context.keys()),
            },
        )

    async def _estimate_confidence_distribution(
        self,
        primary_confidence: float,
        context: Dict[str, Any],
        historical_results: List[Any] = None,
    ) -> ConfidenceDistribution:
        """
        Estimate confidence distribution using Bayesian methods and historical data.

        Uses a combination of:
        - Empirical distribution from historical data
        - Bayesian updating with context factors
        - Ensemble estimation for robustness
        """
        sample_confidences = [primary_confidence]

        # Add historical confidences if available
        if historical_results:
            historical_confidences = [
                self._extract_confidence(r, context) for r in historical_results
            ]
            sample_confidences.extend(historical_confidences)

        # Add contextually-derived confidence samples
        context_samples = self._generate_context_samples(primary_confidence, context)
        sample_confidences.extend(context_samples)

        # Calculate distribution statistics
        if len(sample_confidences) >= 3:
            mean_conf = np.mean(sample_confidences)
            variance = np.var(sample_confidences, ddof=1)
            std_dev = np.sqrt(variance)

            # Calculate confidence interval (95%)
            confidence_interval = self._calculate_confidence_interval(
                sample_confidences, 0.95
            )

            # Estimate skewness and kurtosis
            skewness = np.mean(((sample_confidences - mean_conf) / std_dev) ** 3)
            kurtosis = np.mean(((sample_confidences - mean_conf) / std_dev) ** 4) - 3

            return ConfidenceDistribution(
                mean_confidence=mean_conf,
                variance=variance,
                standard_deviation=std_dev,
                skewness=skewness,
                kurtosis=kurtosis,
                confidence_interval_95=confidence_interval,
                distribution_type=self._classify_distribution(skewness, kurtosis),
                sample_size=len(sample_confidences),
            )
        else:
            # Minimal data - return simple distribution
            return ConfidenceDistribution(
                mean_confidence=primary_confidence,
                variance=0.05,  # Default variance
                standard_deviation=0.2236,
                confidence_interval_95=(
                    max(0, primary_confidence - 0.1),
                    min(1.0, primary_confidence + 0.1),
                ),
                sample_size=1,
            )

    def _generate_context_samples(
        self, primary_confidence: float, context: Dict[str, Any]
    ) -> List[float]:
        """Generate additional confidence samples based on operational context."""
        samples = []

        # Environmental factors can introduce variance
        if "lighting_conditions" in context:
            lighting_factor = (
                1.0 if context["lighting_conditions"] == "well_lit" else 0.9
            )
            samples.append(primary_confidence * lighting_factor)

        if "network_state" in context and context["network_state"] == "offline":
            samples.append(primary_confidence * 0.95)  # Offline reduces confidence

        if "battery_level" in context:
            battery_factor = min(
                1.0, context["battery_level"] + 0.1
            )  # Battery affects reliability
            samples.append(primary_confidence * battery_factor)

        # Operation complexity adds uncertainty
        if "operation_complexity" in context:
            complexity_penalty = context["operation_complexity"] * 0.05
            samples.append(max(0.1, primary_confidence - complexity_penalty))

        return samples if samples else [primary_confidence]

    async def _generate_alternative_predictions(
        self,
        adk_result: Any,
        context: Dict[str, Any],
        distribution: ConfidenceDistribution,
    ) -> List[AlternativePrediction]:
        """
        Generate alternative predictions using ensemble methods and robustness analysis.
        """
        alternatives = []

        # Generate diverse alternatives based on operation type
        operation_type = context.get("operation_type", "unknown")
        base_alternatives = self._get_operation_alternatives(operation_type, adk_result)

        for alt_value in base_alternatives:
            # Calculate probability mass for this alternative
            prob_mass = self._calculate_alternative_probability(
                alt_value, adk_result, distribution, context
            )

            # Calculate robustness metrics
            robustness_metrics = self._assess_alternative_robustness(alt_value, context)

            # Generate reasoning context
            reasoning = self._generate_alternative_reasoning(
                alt_value, context, robustness_metrics
            )

            alternatives.append(
                AlternativePrediction(
                    value=alt_value,
                    confidence=prob_mass,
                    probability_mass=prob_mass,
                    robustness_metrics=robustness_metrics,
                    reasoning_context=reasoning,
                )
            )

        # Sort by confidence
        alternatives.sort(key=lambda x: x.confidence, reverse=True)

        # Limit to top 5 alternatives
        return alternatives[:5]

    def _get_operation_alternatives(
        self, operation_type: str, adk_result: Any
    ) -> List[Any]:
        """Get operation-specific alternatives."""
        if operation_type == "text_recognition":
            return [
                "unclear_text_detected",
                "partial_recognition_successful",
                "high_confidence_match",
                "low_quality_input",
                "processing_timeout",
            ]
        elif operation_type == "object_detection":
            return [
                "no_objects_detected",
                "single_object_found",
                "multiple_objects_found",
                "occlusion_detected",
                "lighting_issues",
            ]
        elif operation_type == "speech_to_text":
            return [
                "speech_recognized_clearly",
                "partial_recognition_with_noise",
                "no_speech_detected",
                "accent_recognition_issues",
                "background_noise_interference",
            ]
        else:
            return [
                "alternative_result_1",
                "alternative_result_2",
                "alternative_result_3",
            ]

    def _calculate_alternative_probability(
        self,
        alternative: Any,
        primary_result: Any,
        distribution: ConfidenceDistribution,
        context: Dict[str, Any],
    ) -> float:
        """Calculate the probability mass for an alternative prediction."""
        # Base probability from distribution
        base_prob = np.random.beta(2, 5)  # Conservative prior

        # Adjust based on distance from primary result
        if isinstance(primary_result, str) and isinstance(alternative, str):
            similarity = self._calculate_text_similarity(primary_result, alternative)
            base_prob *= (
                1.0 - similarity
            ) * 0.3  # Less similar = more alternative probability

        # Adjust based on context factors
        if context.get("environmental_conditions") == "challenging":
            base_prob *= 1.2  # More alternatives in challenging conditions

        if context.get("operation_complexity", 0) > 0.7:
            base_prob *= 1.1  # More alternatives for complex operations

        return min(base_prob, 0.8)  # Cap at 80%

    def _assess_alternative_robustness(
        self, alternative: Any, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess robustness of an alternative prediction."""
        robustness = {
            "environmental_stability": 0.8,
            "data_quality_resistance": 0.7,
            "processing_reliability": 0.85,
            "context_sensitivity": 0.6,
        }

        # Adjust based on context
        if context.get("lighting_conditions") == "poor":
            robustness["environmental_stability"] *= 0.8

        if context.get("network_state") == "unstable":
            robustness["processing_reliability"] *= 0.9

        if context.get("sensor_noise") == "high":
            robustness["data_quality_resistance"] *= 0.7

        return robustness

    def _calculate_confidence_ranges(
        self,
        distribution: ConfidenceDistribution,
        alternatives: List[AlternativePrediction],
        context: Dict[str, Any],
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate various confidence ranges for decision-making."""
        mean = distribution.mean_confidence
        std = distribution.standard_deviation

        # 95% Confidence Interval
        ci_95 = distribution.confidence_interval_95

        # Prediction Interval (wider than confidence interval)
        pred_interval = (max(0, mean - 2.5 * std), min(1.0, mean + 2.5 * std))

        # Tolerance Interval (very conservative)
        tolerance_factor = 3.5 if len(alternatives) > 3 else 2.5
        tolerance_interval = (
            max(0, mean - tolerance_factor * std),
            min(1.0, mean + tolerance_factor * std),
        )

        # Robustness range based on alternatives
        alt_min = min((alt.confidence for alt in alternatives), default=0.0)
        alt_max = max((alt.confidence for alt in alternatives), default=1.0)
        robustness_range = (alt_min * 0.8, min(1.0, alt_max * 1.2))

        return {
            "confidence_interval_95": ci_95,
            "prediction_interval": pred_interval,
            "tolerance_interval": tolerance_interval,
            "robustness_range": robustness_range,
        }

    def _calculate_volatility_metrics(
        self, current_confidence: float
    ) -> VolatilityMetrics:
        """Calculate real-time confidence volatility metrics."""
        recent_confidences = [
            c for _, c in self.confidence_history[-self.volatility_window_size :]
        ]

        if len(recent_confidences) < 2:
            return VolatilityMetrics(
                current_volatility=0.0,
                trend_direction="stable",
                stability_score=1.0,
                change_rate=0.0,
                rolling_std_dev=0.0,
                outlier_count=0,
                last_updated=datetime.now(),
            )

        # Calculate rolling statistics
        rolling_mean = np.mean(recent_confidences)
        rolling_std = np.std(recent_confidences, ddof=1)
        rolling_variance = rolling_std**2

        # Trend analysis
        if len(recent_confidences) >= 3:
            trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
            trend_direction = (
                "increasing"
                if trend > 0.001
                else "decreasing"
                if trend < -0.001
                else "stable"
            )
            change_rate = abs(trend) * 100  # Convert to percentage per step
        else:
            trend_direction = "stable"
            change_rate = 0.0

        # Current volatility
        current_volatility = rolling_std

        # Stability score (inverse of normalized volatility)
        stability_score = max(
            0.0, 1.0 - (current_volatility / 0.5)
        )  # Normalize against max expected volatility

        # Outlier detection (values beyond 2 sigma)
        z_scores = [(c - rolling_mean) / rolling_std for c in recent_confidences]
        outlier_count = sum(1 for z in z_scores if abs(z) > 2.0)

        return VolatilityMetrics(
            current_volatility=current_volatility,
            trend_direction=trend_direction,
            stability_score=stability_score,
            change_rate=change_rate,
            rolling_std_dev=rolling_std,
            outlier_count=outlier_count,
            last_updated=datetime.now(),
        )

    def _calculate_robustness_score(
        self,
        primary_confidence: float,
        alternatives: List[AlternativePrediction],
        distribution: ConfidenceDistribution,
    ) -> float:
        """Calculate robustness score considering multiple factors."""
        # Base robustness from primary confidence
        base_robustness = primary_confidence

        # Adjust for distribution characteristics
        distribution_factor = 1.0 - distribution.uncertainty_score

        # Adjust for alternative predictions
        if alternatives:
            alt_weight = (
                min(len(alternatives), 3) / 3.0
            )  # More alternatives = potentially more robust
            alt_avg_robustness = np.mean(
                [
                    alt.robustness_metrics.get("overall_robustness", 0.5)
                    for alt in alternatives
                ]
            )
            alternative_factor = 0.5 * alt_weight + 0.5 * alt_avg_robustness
        else:
            alternative_factor = 0.5

        # Context-sensitive adjustments
        context_factor = 0.8  # Could be made dynamic based on context

        robustness_score = (
            base_robustness * 0.4
            + distribution_factor * 0.3
            + alternative_factor * 0.2
            + context_factor * 0.1
        )

        return min(robustness_score, 1.0)

    def _extract_confidence(self, adk_result: Any, context: Dict[str, Any]) -> float:
        """Extract confidence score from ADK result."""
        if hasattr(adk_result, "confidence"):
            return float(adk_result.confidence)
        elif isinstance(adk_result, dict) and "confidence" in adk_result:
            return float(adk_result["confidence"])
        elif hasattr(adk_result, "__getitem__") and "confidence" in adk_result:
            return float(adk_result["confidence"])
        else:
            # Default confidence based on operation type
            operation_type = context.get("operation_type", "")
            defaults = {
                "text_recognition": 0.8,
                "object_detection": 0.75,
                "speech_to_text": 0.7,
                "scene_description": 0.65,
            }
            return defaults.get(operation_type, 0.7)

    def _calculate_confidence_interval(
        self, samples: List[float], confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a sample."""
        if len(samples) < 2:
            mean = samples[0] if samples else 0.5
            margin = 0.1  # Default margin
            return (max(0, mean - margin), min(1.0, mean + margin))

        mean = np.mean(samples)
        std = np.std(samples, ddof=1)

        # t-distribution critical value approximation
        if len(samples) < 30:
            t_critical = 2.0 if len(samples) > 10 else 2.5  # Conservative approach
        else:
            t_critical = 1.96  # z-score for 95%

        margin = t_critical * std / np.sqrt(len(samples))
        return (max(0, mean - margin), min(1.0, mean + margin))

    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify the type of distribution based on shape parameters."""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "normal"
        elif skewness > 1.0:
            return "right_skewed"
        elif skewness < -1.0:
            return "left_skewed"
        elif kurtosis > 1.0:
            return "leptokurtic"
        else:
            return "non_normal"

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity for alternative prediction analysis."""
        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())

        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))

        return intersection / union if union > 0 else 0.0

    def _generate_alternative_reasoning(
        self, alternative: Any, context: Dict[str, Any], robustness: Dict[str, float]
    ) -> List[str]:
        """Generate reasoning context for an alternative prediction."""
        reasoning = []

        if context.get("environmental_conditions") == "challenging":
            reasoning.append("Environmental factors suggest higher uncertainty")

        if robustness.get("data_quality_resistance", 0.5) < 0.7:
            reasoning.append("Lower resistance to data quality issues")

        if context.get("operation_complexity", 0) > 0.7:
            reasoning.append("Complex operation increases alternative likelihood")

        if not reasoning:
            reasoning.append(
                "Standard alternative prediction based on probabilistic analysis"
            )

        return reasoning

    def _update_confidence_history(self, confidence: float):
        """Update the confidence time-series history."""
        timestamp = datetime.now()
        self.confidence_history.append((timestamp, confidence))

        # Limit history size
        max_history = self.volatility_window_size * 2
        if len(self.confidence_history) > max_history:
            self.confidence_history = self.confidence_history[-max_history:]

    def _update_alternative_history(self, alternatives: List[AlternativePrediction]):
        """Update the alternatives history."""
        self.alternative_history.append(alternatives)

        # Limit history size
        if len(self.alternative_history) > 100:
            self.alternative_history = self.alternative_history[-100:]

    def get_uncertainty_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """Get uncertainty trends for monitoring and alerting."""
        if len(self.confidence_history) < window_size:
            return {"insufficient_data": True}

        recent_confidences = [c for _, c in self.confidence_history[-window_size:]]
        recent_alternatives = self.alternative_history[-window_size:]

        return {
            "confidence_volatility": np.std(recent_confidences),
            "average_alternatives_count": np.mean(
                [len(alts) for alts in recent_alternatives]
            ),
            "uncertainty_trend": self._analyze_uncertainty_trend(recent_confidences),
            "most_common_alternatives": self._find_common_alternatives(
                recent_alternatives
            ),
            "risk_assessment": self._assess_risk_level(
                recent_confidences, recent_alternatives
            ),
        }

    def _analyze_uncertainty_trend(self, confidences: List[float]) -> str:
        """Analyze the trend in uncertainty over time."""
        if len(confidences) < 3:
            return "insufficient_data"

        # Calculate trend in uncertainty (1 - confidence)
        uncertainties = [1.0 - c for c in confidences]
        trend = np.polyfit(range(len(uncertainties)), uncertainties, 1)[0]

        if trend > 0.01:
            return "increasing_uncertainty"
        elif trend < -0.01:
            return "decreasing_uncertainty"
        else:
            return "stable_uncertainty"

    def _find_common_alternatives(
        self, alternative_history: List[List[AlternativePrediction]]
    ) -> List[str]:
        """Find most common alternative predictions."""
        from collections import Counter

        all_alternatives = []
        for alternatives in alternative_history:
            all_alternatives.extend([str(alt.value) for alt in alternatives])

        common = Counter(all_alternatives).most_common(3)
        return [alt[0] for alt in common]

    def _assess_risk_level(
        self, confidences: List[float], alternatives: List[List[AlternativePrediction]]
    ) -> str:
        """Assess overall risk level based on confidence and alternatives."""
        avg_confidence = np.mean(confidences)

        if avg_confidence > 0.8 and len(alternatives) < 2:
            return "low_risk"
        elif avg_confidence > 0.6 or len(alternatives) < 5:
            return "medium_risk"
        else:
            return "high_risk"
