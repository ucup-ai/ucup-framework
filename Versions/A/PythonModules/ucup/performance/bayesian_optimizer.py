"""
Bayesian Network Optimization Module

This module provides optimized Bayesian network inference algorithms including
variable elimination and caching mechanisms for improved performance.
"""

import hashlib
import logging
import pickle
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from cachetools import LRUCache, TTLCache

from ..advanced_probabilistic import BayesianNetwork, ConditionalProbabilityTable

logger = logging.getLogger(__name__)


class VariableElimination:
    """
    Optimized variable elimination algorithm for Bayesian network inference.

    This implementation uses efficient data structures and caching to improve
    performance over the basic enumeration approach.
    """

    def __init__(self, network: BayesianNetwork, use_cache: bool = True):
        self.network = network
        self.use_cache = use_cache
        self.factor_cache: Dict[str, Any] = {}
        self.elimination_order_cache: Dict[str, List[str]] = {}

    def query(
        self, query_vars: List[str], evidence: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Perform probabilistic query using variable elimination.

        Args:
            query_vars: Variables to query
            evidence: Observed evidence variables

        Returns:
            Dictionary mapping query variable values to probabilities
        """
        start_time = time.time()

        # Get optimal elimination order
        all_vars = set(self.network.nodes.keys())
        evidence_vars = set(evidence.keys())
        query_vars_set = set(query_vars)
        hidden_vars = all_vars - evidence_vars - query_vars_set

        elimination_order = self._get_elimination_order(
            hidden_vars, query_vars_set, evidence_vars
        )

        # Initialize factors
        factors = self._initialize_factors(evidence)

        # Eliminate hidden variables
        for var in elimination_order:
            if var in factors:
                # Sum out the variable
                factors = self._sum_out_variable(factors, var)

        # Normalize and return result
        result = self._normalize_factors(factors, query_vars)

        elapsed_time = time.time() - start_time
        logger.debug(f"Variable elimination query completed in {elapsed_time:.4f}s")

        return result

    def _get_elimination_order(
        self, hidden_vars: Set[str], query_vars: Set[str], evidence_vars: Set[str]
    ) -> List[str]:
        """Get optimal variable elimination order using min-fill heuristic."""
        cache_key = (
            f"{sorted(hidden_vars)}_{sorted(query_vars)}_{sorted(evidence_vars)}"
        )

        if cache_key in self.elimination_order_cache:
            return self.elimination_order_cache[cache_key]

        # Simple min-fill heuristic: minimize new edges created when eliminating
        remaining_vars = list(hidden_vars)
        order = []

        while remaining_vars:
            # Find variable that creates fewest new edges when eliminated
            best_var = None
            best_score = float("inf")

            for var in remaining_vars:
                neighbors = self._get_neighbors(var, remaining_vars)
                score = (
                    len(neighbors) * (len(neighbors) - 1) // 2
                )  # Number of edges that would be added
                if score < best_score:
                    best_score = score
                    best_var = var

            if best_var:
                order.append(best_var)
                remaining_vars.remove(best_var)

        self.elimination_order_cache[cache_key] = order
        return order

    def _get_neighbors(self, var: str, remaining_vars: List[str]) -> Set[str]:
        """Get neighbors of a variable among remaining variables."""
        neighbors = set()

        # Check parent-child relationships
        node = self.network.nodes.get(var)
        if node:
            # Parents
            for parent in node.parents:
                if parent in remaining_vars:
                    neighbors.add(parent)
            # Children
            for child_name, child_node in self.network.nodes.items():
                if child_name in remaining_vars and var in child_node.parents:
                    neighbors.add(child_name)

        return neighbors

    def _initialize_factors(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize factors from network CPTs and evidence."""
        factors = {}

        for var_name, node in self.network.nodes.items():
            if var_name in evidence:
                # Evidence variable - create factor with observed value
                observed_value = evidence[var_name]
                factor = {observed_value: 1.0}
            else:
                # Hidden or query variable - use CPT
                factor = self._get_factor_from_cpt(node.cpt, evidence)

            factors[var_name] = factor

        return factors

    def _get_factor_from_cpt(
        self, cpt: ConditionalProbabilityTable, evidence: Dict[str, Any]
    ) -> Dict[Any, float]:
        """Extract factor from CPT given current evidence."""
        factor = {}

        # Get all possible values for this variable
        var_values = cpt.get_variable_values(cpt.variable)

        for value in var_values:
            # Check if all parents are observed
            parent_values = {}
            all_parents_observed = True

            for parent in cpt.parents:
                if parent in evidence:
                    parent_values[parent] = evidence[parent]
                else:
                    all_parents_observed = False
                    break

            if all_parents_observed:
                # Look up probability in CPT
                conditions = {cpt.variable: value}
                conditions.update(parent_values)
                prob = cpt.get_probability(conditions)
                factor[value] = prob

        return factor

    def _sum_out_variable(self, factors: Dict[str, Any], var: str) -> Dict[str, Any]:
        """Sum out a variable from the factors."""
        # Find factors that contain this variable
        relevant_factors = []
        remaining_factors = []

        for factor_name, factor in factors.items():
            if var in self._get_factor_variables(factor_name):
                relevant_factors.append((factor_name, factor))
            else:
                remaining_factors.append((factor_name, factor))

        if not relevant_factors:
            return factors

        # Multiply relevant factors
        combined_factor = self._multiply_factors(relevant_factors)

        # Sum out the variable
        summed_factor = self._sum_factor(combined_factor, var)

        # Update factors
        new_factors = dict(remaining_factors)
        new_factors[f"summed_{var}"] = summed_factor

        return new_factors

    def _get_factor_variables(self, factor_name: str) -> Set[str]:
        """Get variables in a factor (simplified - assumes factor name indicates variables)."""
        # This is a simplification - in practice, you'd track variables per factor
        if "_" in factor_name:
            return set(factor_name.split("_"))
        return {factor_name}

    def _multiply_factors(self, factors: List[Tuple[str, Any]]) -> Dict[Tuple, float]:
        """Multiply multiple factors together."""
        if not factors:
            return {}

        result = factors[0][1].copy()

        for factor_name, factor in factors[1:]:
            new_result = {}
            for key1, prob1 in result.items():
                for key2, prob2 in factor.items():
                    # Combine keys (assuming tuple keys for multiple variables)
                    if isinstance(key1, tuple) and isinstance(key2, tuple):
                        combined_key = key1 + key2
                    elif isinstance(key1, tuple):
                        combined_key = key1 + (key2,)
                    elif isinstance(key2, tuple):
                        combined_key = (key1,) + key2
                    else:
                        combined_key = (key1, key2)

                    new_result[combined_key] = prob1 * prob2

            result = new_result

        return result

    def _sum_factor(self, factor: Dict[Any, float], var: str) -> Dict[Any, float]:
        """Sum out a variable from a factor."""
        result = defaultdict(float)

        for key, prob in factor.items():
            if isinstance(key, tuple):
                # Find variable index in tuple
                var_index = None
                for i, k in enumerate(key):
                    if str(k).startswith(var) or str(k) == var:
                        var_index = i
                        break

                if var_index is not None:
                    # Sum over this variable
                    remaining_key = key[:var_index] + key[var_index + 1 :]
                    if len(remaining_key) == 1:
                        remaining_key = remaining_key[0]
                    elif len(remaining_key) == 0:
                        remaining_key = "constant"
                    result[remaining_key] += prob
            else:
                # Single variable factor
                result["constant"] += prob

        return dict(result)

    def _normalize_factors(
        self, factors: Dict[str, Any], query_vars: List[str]
    ) -> Dict[str, float]:
        """Normalize final factors to get probabilities."""
        # Simplified normalization - assumes we have the joint distribution
        result = {}

        # Find the final factor
        final_factor = None
        for factor in factors.values():
            if isinstance(factor, dict):
                final_factor = factor
                break

        if not final_factor:
            return result

        # Normalize
        total = sum(final_factor.values())
        if total > 0:
            for key, prob in final_factor.items():
                result[str(key)] = prob / total

        return result


class BayesianInferenceCache:
    """
    Cache for Bayesian network inference results.

    Uses TTL and LRU caching strategies to improve performance for repeated queries.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.query_cache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self.factor_cache = LRUCache(maxsize=max_size * 2)
        self.hit_count = 0
        self.miss_count = 0

    def get_query_result(self, query_key: str) -> Optional[Dict[str, float]]:
        """Get cached query result."""
        result = self.query_cache.get(query_key)
        if result is not None:
            self.hit_count += 1
        else:
            self.miss_count += 1
        return result

    def store_query_result(self, query_key: str, result: Dict[str, float]) -> None:
        """Store query result in cache."""
        self.query_cache[query_key] = result

    def get_factor(self, factor_key: str) -> Optional[Any]:
        """Get cached factor."""
        return self.factor_cache.get(factor_key)

    def store_factor(self, factor_key: str, factor: Any) -> None:
        """Store factor in cache."""
        self.factor_cache[factor_key] = factor

    def generate_query_key(
        self, network_hash: str, query_vars: List[str], evidence: Dict[str, Any]
    ) -> str:
        """Generate a unique key for a query."""
        key_components = [
            network_hash,
            ",".join(sorted(query_vars)),
            str(sorted(evidence.items())),
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def generate_factor_key(
        self, network_hash: str, variables: List[str], evidence: Dict[str, Any]
    ) -> str:
        """Generate a unique key for a factor."""
        key_components = [
            network_hash,
            ",".join(sorted(variables)),
            str(sorted(evidence.items())),
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def clear(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()
        self.factor_cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        return {
            "query_cache_size": len(self.query_cache),
            "factor_cache_size": len(self.factor_cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }


class OptimizedBayesianNetwork(BayesianNetwork):
    """
    Enhanced Bayesian network with performance optimizations.

    Includes caching, optimized inference algorithms, and performance monitoring.
    """

    def __init__(self, name: str = "OptimizedBayesianNetwork"):
        # Initialize dataclass fields
        self.nodes = {}
        self.evidence = {}
        self.variable_eliminator = VariableElimination(self, use_cache=True)
        self.cache = BayesianInferenceCache()
        self.network_hash = self._compute_network_hash()
        self.query_stats = {
            "total_queries": 0,
            "cached_queries": 0,
            "average_query_time": 0.0,
            "cache_hit_rate": 0.0,
        }

    def _compute_network_hash(self) -> str:
        """Compute hash of network structure for caching."""
        network_repr = str(sorted(self.nodes.keys()))
        return hashlib.md5(network_repr.encode()).hexdigest()

    def query(
        self, query_vars: List[str], evidence: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Perform optimized probabilistic query.

        Uses caching and variable elimination for improved performance.
        """
        start_time = time.time()
        evidence = evidence or {}

        # Generate cache key
        query_key = self.cache.generate_query_key(
            self.network_hash, query_vars, evidence
        )

        # Check cache first
        cached_result = self.cache.get_query_result(query_key)
        if cached_result is not None:
            self.query_stats["cached_queries"] += 1
            self.query_stats["total_queries"] += 1
            return cached_result

        # Perform inference
        result = self.variable_eliminator.query(query_vars, evidence)

        # Cache result
        self.cache.store_query_result(query_key, result)

        # Update stats
        query_time = time.time() - start_time
        self.query_stats["total_queries"] += 1
        self._update_average_query_time(query_time)

        cache_stats = self.cache.get_stats()
        self.query_stats["cache_hit_rate"] = cache_stats["hit_rate"]

        return result

    def _update_average_query_time(self, new_time: float) -> None:
        """Update rolling average query time."""
        current_avg = self.query_stats["average_query_time"]
        total_queries = self.query_stats["total_queries"]

        if total_queries == 1:
            self.query_stats["average_query_time"] = new_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.query_stats["average_query_time"] = (
                alpha * new_time + (1 - alpha) * current_avg
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_stats = self.cache.get_stats()
        return {
            **self.query_stats,
            "network_hash": self.network_hash,
            "cache_stats": cache_stats,
        }

    def clear_cache(self) -> None:
        """Clear inference caches."""
        self.cache.clear()

    def optimize_for_query_pattern(
        self, common_queries: List[Tuple[List[str], Dict[str, Any]]]
    ) -> None:
        """
        Pre-compute and cache common query patterns.

        Args:
            common_queries: List of (query_vars, evidence) tuples representing common queries
        """
        logger.info(f"Optimizing for {len(common_queries)} common query patterns")

        for query_vars, evidence in common_queries:
            try:
                # Pre-compute and cache result
                result = self.variable_eliminator.query(query_vars, evidence)
                query_key = self.cache.generate_query_key(
                    self.network_hash, query_vars, evidence
                )
                self.cache.store_query_result(query_key, result)
                logger.debug(f"Pre-cached query: {query_vars} with evidence {evidence}")
            except Exception as e:
                logger.warning(f"Failed to pre-cache query {query_vars}: {e}")

        logger.info("Query pattern optimization completed")
