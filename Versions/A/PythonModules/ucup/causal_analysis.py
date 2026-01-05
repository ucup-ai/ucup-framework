"""
Causal Analysis Module

This module provides causal reasoning and failure analysis capabilities for UCUP,
enabling root cause identification and causal chain analysis.
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional import for NetworkX - causal analysis is unavailable if NetworkX is not compatible
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"NetworkX not available: {e}. Causal analysis features will be disabled."
    )
    NETWORKX_AVAILABLE = False
    nx = None


@dataclass
class CausalFactor:
    """Represents a causal factor in a failure or event."""

    factor_id: str
    name: str
    description: str = ""
    factor_type: str = (
        "unknown"  # 'root_cause', 'contributing_factor', 'symptom', 'mitigating_factor'
    )
    severity: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def is_root_cause(self) -> bool:
        """Check if this is a root cause factor."""
        return self.factor_type == "root_cause"

    def is_contributing(self) -> bool:
        """Check if this is a contributing factor."""
        return self.factor_type == "contributing_factor"

    def add_evidence(self, evidence_item: str) -> None:
        """Add evidence supporting this causal factor."""
        if evidence_item not in self.evidence:
            self.evidence.append(evidence_item)

    def calculate_effectiveness_score(self) -> float:
        """Calculate effectiveness score based on confidence and evidence."""
        evidence_score = min(
            1.0, len(self.evidence) / 5.0
        )  # Cap at 5 pieces of evidence
        return (self.confidence + evidence_score) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "description": self.description,
            "factor_type": self.factor_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence": self.evidence.copy(),
            "metadata": self.metadata.copy(),
            "timestamp": self.timestamp.isoformat(),
            "effectiveness_score": self.calculate_effectiveness_score(),
        }


@dataclass
class CausalRelationship:
    """Represents a causal relationship between factors."""

    source_factor_id: str
    target_factor_id: str
    relationship_type: str  # 'causes', 'contributes_to', 'mitigates', 'correlates_with'
    strength: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False

    def is_causal(self) -> bool:
        """Check if this is a direct causal relationship."""
        return self.relationship_type in ["causes", "contributes_to"]

    def is_mitigating(self) -> bool:
        """Check if this relationship mitigates effects."""
        return self.relationship_type == "mitigates"

    def add_evidence(self, evidence_item: str) -> None:
        """Add evidence supporting this relationship."""
        if evidence_item not in self.evidence:
            self.evidence.append(evidence_item)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_factor_id": self.source_factor_id,
            "target_factor_id": self.target_factor_id,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "evidence": self.evidence.copy(),
            "metadata": self.metadata.copy(),
            "bidirectional": self.bidirectional,
        }


@dataclass
class CausalChain:
    """Represents a chain of causal relationships."""

    chain_id: str
    factors: List[CausalFactor] = field(default_factory=list)
    relationships: List[CausalRelationship] = field(default_factory=list)
    root_cause: Optional[CausalFactor] = None
    final_effect: Optional[CausalFactor] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_factor(self, factor: CausalFactor) -> None:
        """Add a factor to the causal chain."""
        if factor not in self.factors:
            self.factors.append(factor)

            # Update root cause if this is a root cause factor
            if factor.is_root_cause() and self.root_cause is None:
                self.root_cause = factor

            # Update final effect if this is a symptom or final outcome
            if factor.factor_type == "symptom":
                self.final_effect = factor

    def add_relationship(self, relationship: CausalRelationship) -> None:
        """Add a relationship to the causal chain."""
        self.relationships.append(relationship)

    def get_chain_length(self) -> int:
        """Get the length of the causal chain."""
        return len(self.factors)

    def get_root_causes(self) -> List[CausalFactor]:
        """Get all root cause factors in the chain."""
        return [f for f in self.factors if f.is_root_cause()]

    def calculate_chain_confidence(self) -> float:
        """Calculate overall confidence of the causal chain."""
        if not self.factors:
            return 0.0

        factor_confidences = [f.calculate_effectiveness_score() for f in self.factors]
        relationship_strengths = [r.strength for r in self.relationships]

        # Combine factor and relationship confidences
        factor_avg = np.mean(factor_confidences) if factor_confidences else 0.0
        relationship_avg = (
            np.mean(relationship_strengths) if relationship_strengths else 1.0
        )

        self.confidence = (factor_avg + relationship_avg) / 2.0
        return self.confidence

    def get_chain_summary(self) -> Dict[str, Any]:
        """Get a summary of the causal chain."""
        return {
            "chain_id": self.chain_id,
            "length": self.get_chain_length(),
            "root_causes": len(self.get_root_causes()),
            "total_factors": len(self.factors),
            "total_relationships": len(self.relationships),
            "confidence": self.calculate_chain_confidence(),
            "root_cause_name": self.root_cause.name if self.root_cause else None,
            "final_effect_name": self.final_effect.name if self.final_effect else None,
            "created_at": self.created_at.isoformat(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chain_id": self.chain_id,
            "factors": [f.to_dict() for f in self.factors],
            "relationships": [r.to_dict() for r in self.relationships],
            "root_cause": self.root_cause.to_dict() if self.root_cause else None,
            "final_effect": self.final_effect.to_dict() if self.final_effect else None,
            "confidence": self.confidence,
            "metadata": self.metadata.copy(),
            "created_at": self.created_at.isoformat(),
            "summary": self.get_chain_summary(),
        }


class CausalGraph:
    """Graph-based representation of causal relationships."""

    def __init__(self, graph_id: str = "causal_graph"):
        self.graph_id = graph_id
        self.graph = nx.DiGraph()
        self.factors: Dict[str, CausalFactor] = {}
        self.relationships: List[CausalRelationship] = []
        self.metadata: Dict[str, Any] = {}

    def add_factor(self, factor: CausalFactor) -> None:
        """Add a factor to the causal graph."""
        self.factors[factor.factor_id] = factor
        self.graph.add_node(factor.factor_id, **factor.to_dict())

    def add_relationship(self, relationship: CausalRelationship) -> None:
        """Add a relationship to the causal graph."""
        self.relationships.append(relationship)

        # Add edge to networkx graph
        self.graph.add_edge(
            relationship.source_factor_id,
            relationship.target_factor_id,
            **relationship.to_dict(),
        )

        # Add bidirectional edge if specified
        if relationship.bidirectional:
            self.graph.add_edge(
                relationship.target_factor_id,
                relationship.source_factor_id,
                **relationship.to_dict(),
            )

    def get_factor(self, factor_id: str) -> Optional[CausalFactor]:
        """Get a factor by ID."""
        return self.factors.get(factor_id)

    def get_relationships_for_factor(self, factor_id: str) -> List[CausalRelationship]:
        """Get all relationships involving a specific factor."""
        return [
            r
            for r in self.relationships
            if r.source_factor_id == factor_id or r.target_factor_id == factor_id
        ]

    def find_root_causes(self, target_factor_id: str) -> List[CausalFactor]:
        """
        Find root causes that lead to a target factor.

        Uses graph traversal to find all paths from root causes to the target.
        """
        if target_factor_id not in self.graph:
            return []

        # Find all nodes that have no incoming edges (potential root causes)
        root_candidates = [
            node for node in self.graph.nodes() if self.graph.in_degree(node) == 0
        ]

        # Find paths from root candidates to target
        root_causes = []
        for root_id in root_candidates:
            try:
                # Check if there's a path from root to target
                if nx.has_path(self.graph, root_id, target_factor_id):
                    factor = self.get_factor(root_id)
                    if factor and factor.is_root_cause():
                        root_causes.append(factor)
            except nx.NetworkXNoPath:
                continue

        return root_causes

    def find_causal_paths(
        self, source_id: str, target_id: str, max_length: int = 10
    ) -> List[List[str]]:
        """Find all causal paths from source to target."""
        try:
            paths = list(
                nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_length)
            )
            return paths
        except nx.NetworkXNoPath:
            return []

    def get_shortest_causal_path(
        self, source_id: str, target_id: str
    ) -> Optional[List[str]]:
        """Get the shortest causal path from source to target."""
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return None

    def analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze the structure of the causal graph."""
        analysis = {
            "total_factors": len(self.factors),
            "total_relationships": len(self.relationships),
            "graph_density": nx.density(self.graph) if len(self.graph) > 0 else 0,
            "is_connected": nx.is_weakly_connected(self.graph)
            if len(self.graph) > 0
            else False,
            "average_clustering": nx.average_clustering(self.graph.to_undirected())
            if len(self.graph) > 0
            else 0.0,
            "degree_centrality": dict(nx.degree_centrality(self.graph)),
            "betweenness_centrality": dict(nx.betweenness_centrality(self.graph)),
            "root_causes": len([f for f in self.factors.values() if f.is_root_cause()]),
            "contributing_factors": len(
                [f for f in self.factors.values() if f.is_contributing()]
            ),
            "symptoms": len(
                [f for f in self.factors.values() if f.factor_type == "symptom"]
            ),
        }

        return analysis

    def get_subgraph_for_factor(self, factor_id: str, depth: int = 2) -> "CausalGraph":
        """Get a subgraph centered on a specific factor."""
        if factor_id not in self.graph:
            return CausalGraph(f"subgraph_{factor_id}")

        # Get nodes within specified depth
        nodes_within_depth = set()
        nodes_within_depth.add(factor_id)

        # BFS to get nodes at each depth level
        current_level = {factor_id}
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Add predecessors (causes)
                next_level.update(self.graph.predecessors(node))
                # Add successors (effects)
                next_level.update(self.graph.successors(node))
            nodes_within_depth.update(next_level)
            current_level = next_level

        # Create subgraph
        subgraph = CausalGraph(f"subgraph_{factor_id}")
        subgraph.graph = self.graph.subgraph(nodes_within_depth).copy()

        # Copy factors
        for node_id in nodes_within_depth:
            if node_id in self.factors:
                subgraph.factors[node_id] = self.factors[node_id]

        # Copy relationships
        for rel in self.relationships:
            if (
                rel.source_factor_id in nodes_within_depth
                and rel.target_factor_id in nodes_within_depth
            ):
                subgraph.relationships.append(rel)

        subgraph.metadata = {
            "original_graph": self.graph_id,
            "center_factor": factor_id,
            "depth": depth,
            "node_count": len(nodes_within_depth),
        }

        return subgraph

    def export_to_dict(self) -> Dict[str, Any]:
        """Export the entire causal graph to a dictionary."""
        return {
            "graph_id": self.graph_id,
            "factors": {fid: f.to_dict() for fid, f in self.factors.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "metadata": self.metadata.copy(),
            "structure_analysis": self.analyze_graph_structure(),
        }

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import a causal graph from a dictionary."""
        self.graph_id = data.get("graph_id", "imported_graph")
        self.metadata = data.get("metadata", {})

        # Import factors
        for factor_id, factor_data in data.get("factors", {}).items():
            # Convert dict back to CausalFactor
            factor = CausalFactor(
                factor_id=factor_data["factor_id"],
                name=factor_data["name"],
                description=factor_data["description"],
                factor_type=factor_data["factor_type"],
                severity=factor_data["severity"],
                confidence=factor_data["confidence"],
                evidence=factor_data.get("evidence", []),
                metadata=factor_data.get("metadata", {}),
                timestamp=datetime.fromisoformat(factor_data["timestamp"]),
            )
            self.add_factor(factor)

        # Import relationships
        for rel_data in data.get("relationships", []):
            relationship = CausalRelationship(
                source_factor_id=rel_data["source_factor_id"],
                target_factor_id=rel_data["target_factor_id"],
                relationship_type=rel_data["relationship_type"],
                strength=rel_data["strength"],
                evidence=rel_data.get("evidence", []),
                metadata=rel_data.get("metadata", {}),
                bidirectional=rel_data.get("bidirectional", False),
            )
            self.add_relationship(relationship)


class CausalFailureAnalyzer:
    """
    Main analyzer for performing causal analysis of failures and events.
    """

    def __init__(self):
        self.causal_graph = CausalGraph("failure_analysis")
        self.failure_chains: Dict[str, CausalChain] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.confidence_threshold: float = 0.6

    async def analyze_failure(
        self,
        failure_data: Dict[str, Any],
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform causal analysis of a failure.

        Args:
            failure_data: Information about the failure
            context_data: Additional context information

        Returns:
            Analysis results including root causes and causal chains
        """
        analysis_id = f"analysis_{int(asyncio.get_event_loop().time() * 1000000)}"

        # Extract causal factors from failure data
        factors = await self._extract_causal_factors(failure_data, context_data)

        # Build causal relationships
        relationships = await self._identify_causal_relationships(factors, failure_data)

        # Create causal chain
        causal_chain = await self._build_causal_chain(
            factors, relationships, failure_data
        )

        # Store results
        self.failure_chains[analysis_id] = causal_chain

        # Add to causal graph
        for factor in factors:
            self.causal_graph.add_factor(factor)
        for relationship in relationships:
            self.causal_graph.add_relationship(relationship)

        # Perform analysis
        analysis_result = {
            "analysis_id": analysis_id,
            "causal_chain": causal_chain.to_dict(),
            "root_causes": [f.to_dict() for f in causal_chain.get_root_causes()],
            "contributing_factors": [
                f.to_dict() for f in factors if f.is_contributing()
            ],
            "recommendations": await self._generate_recommendations(causal_chain),
            "confidence_score": causal_chain.calculate_chain_confidence(),
            "graph_analysis": self.causal_graph.analyze_graph_structure(),
            "timestamp": datetime.now().isoformat(),
        }

        # Store analysis in history
        self.analysis_history.append(analysis_result)

        logger.info(
            f"Completed causal analysis {analysis_id} with confidence {causal_chain.confidence:.2f}"
        )
        return analysis_result

    async def _extract_causal_factors(
        self, failure_data: Dict[str, Any], context_data: Optional[Dict[str, Any]]
    ) -> List[CausalFactor]:
        """Extract causal factors from failure and context data."""
        factors = []

        # Extract from failure symptoms
        symptoms = failure_data.get("symptoms", [])
        for symptom in symptoms:
            factor = CausalFactor(
                factor_id=f"symptom_{len(factors)}",
                name=f"Symptom: {symptom}",
                description=f"Observed symptom: {symptom}",
                factor_type="symptom",
                severity=0.8,
                confidence=0.9,
                evidence=[f"Failure report: {symptom}"],
                metadata={"source": "failure_data"},
            )
            factors.append(factor)

        # Extract from error logs
        error_logs = failure_data.get("error_logs", [])
        for error in error_logs:
            factor = CausalFactor(
                factor_id=f"error_{len(factors)}",
                name=f"Error: {error.get('message', 'Unknown error')}",
                description=f"Error occurred: {error}",
                factor_type="contributing_factor",
                severity=0.7,
                confidence=0.8,
                evidence=[f"Error log: {error}"],
                metadata={"source": "error_logs", "error_code": error.get("code")},
            )
            factors.append(factor)

        # Extract from system metrics
        metrics = context_data.get("system_metrics", {}) if context_data else {}
        for metric_name, metric_value in metrics.items():
            # Check for abnormal metrics
            if self._is_metric_abnormal(metric_name, metric_value):
                factor = CausalFactor(
                    factor_id=f"metric_{len(factors)}",
                    name=f"Abnormal {metric_name}",
                    description=f"System metric {metric_name} shows abnormal value: {metric_value}",
                    factor_type="contributing_factor",
                    severity=0.6,
                    confidence=0.7,
                    evidence=[f"Metric {metric_name} = {metric_value}"],
                    metadata={"source": "system_metrics", "metric_name": metric_name},
                )
                factors.append(factor)

        # Extract from configuration issues
        config_issues = (
            context_data.get("configuration_issues", []) if context_data else []
        )
        for issue in config_issues:
            factor = CausalFactor(
                factor_id=f"config_{len(factors)}",
                name=f"Configuration Issue: {issue}",
                description=f"Configuration problem: {issue}",
                factor_type="root_cause",
                severity=0.9,
                confidence=0.85,
                evidence=[f"Configuration analysis: {issue}"],
                metadata={"source": "configuration"},
            )
            factors.append(factor)

        return factors

    async def _identify_causal_relationships(
        self, factors: List[CausalFactor], failure_data: Dict[str, Any]
    ) -> List[CausalRelationship]:
        """Identify causal relationships between factors."""
        relationships = []

        # Simple rule-based relationship identification
        symptoms = [f for f in factors if f.factor_type == "symptom"]
        contributing = [f for f in factors if f.is_contributing()]
        root_causes = [f for f in factors if f.is_root_cause()]

        # Root causes contribute to contributing factors
        for root in root_causes:
            for contrib in contributing:
                if self._are_related(root, contrib):
                    relationship = CausalRelationship(
                        source_factor_id=root.factor_id,
                        target_factor_id=contrib.factor_id,
                        relationship_type="causes",
                        strength=0.8,
                        evidence=[f"Causal link: {root.name} -> {contrib.name}"],
                        metadata={"relationship_type": "direct_causation"},
                    )
                    relationships.append(relationship)

        # Contributing factors lead to symptoms
        for contrib in contributing:
            for symptom in symptoms:
                if self._are_related(contrib, symptom):
                    relationship = CausalRelationship(
                        source_factor_id=contrib.factor_id,
                        target_factor_id=symptom.factor_id,
                        relationship_type="contributes_to",
                        strength=0.6,
                        evidence=[f"Contribution: {contrib.name} -> {symptom.name}"],
                        metadata={"relationship_type": "contribution"},
                    )
                    relationships.append(relationship)

        return relationships

    async def _build_causal_chain(
        self,
        factors: List[CausalFactor],
        relationships: List[CausalRelationship],
        failure_data: Dict[str, Any],
    ) -> CausalChain:
        """Build a complete causal chain from factors and relationships."""
        chain_id = f"chain_{int(asyncio.get_event_loop().time() * 1000000)}"

        chain = CausalChain(
            chain_id=chain_id,
            metadata={
                "failure_type": failure_data.get("failure_type", "unknown"),
                "severity": failure_data.get("severity", "unknown"),
            },
        )

        # Add all factors to chain
        for factor in factors:
            chain.add_factor(factor)

        # Add relationships to chain
        for relationship in relationships:
            chain.add_relationship(relationship)

        # Calculate chain confidence
        chain.calculate_chain_confidence()

        return chain

    async def _generate_recommendations(self, causal_chain: CausalChain) -> List[str]:
        """Generate recommendations based on causal analysis."""
        recommendations = []

        root_causes = causal_chain.get_root_causes()

        for root_cause in root_causes:
            if "configuration" in root_cause.name.lower():
                recommendations.append(
                    f"Review and fix configuration settings related to: {root_cause.description}"
                )
            elif "resource" in root_cause.name.lower():
                recommendations.append(
                    f"Scale up or optimize resource allocation for: {root_cause.description}"
                )
            elif "network" in root_cause.name.lower():
                recommendations.append(
                    f"Check network connectivity and configuration for: {root_cause.description}"
                )
            else:
                recommendations.append(f"Address root cause: {root_cause.description}")

        # Add general recommendations
        if causal_chain.confidence < 0.7:
            recommendations.append(
                "Consider gathering more diagnostic information for more accurate analysis"
            )

        if len(root_causes) > 3:
            recommendations.append(
                "Multiple root causes identified - prioritize fixes based on severity and impact"
            )

        return recommendations

    def _is_metric_abnormal(self, metric_name: str, value: Any) -> bool:
        """Check if a metric value is abnormal."""
        # Simple threshold-based abnormality detection
        if not isinstance(value, (int, float)):
            return False

        # Define thresholds for common metrics
        thresholds = {
            "cpu_usage": 90.0,
            "memory_usage": 95.0,
            "disk_usage": 95.0,
            "error_rate": 0.1,
            "response_time": 5.0,  # seconds
        }

        threshold = thresholds.get(metric_name)
        if threshold:
            if "error_rate" in metric_name or "response_time" in metric_name:
                return value > threshold  # Higher is worse
            else:
                return value > threshold  # Higher usage is worse

        return False

    def _are_related(self, factor1: CausalFactor, factor2: CausalFactor) -> bool:
        """Check if two factors are related based on their content."""
        # Simple keyword-based relatedness check
        text1 = (factor1.name + " " + factor1.description).lower()
        text2 = (factor2.name + " " + factor2.description).lower()

        # Common keywords that indicate relationships
        common_keywords = set(text1.split()) & set(text2.split())

        # Check for causal keywords
        causal_keywords = {"caused", "due", "because", "result", "led", "triggered"}

        return len(common_keywords) > 2 or any(
            keyword in text1 + text2 for keyword in causal_keywords
        )

    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis history."""
        return self.analysis_history[-limit:] if self.analysis_history else []

    def get_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across all failure analyses."""
        if not self.analysis_history:
            return {"status": "no_analyses_available"}

        # Analyze root cause patterns
        root_cause_counts = defaultdict(int)
        failure_type_counts = defaultdict(int)

        for analysis in self.analysis_history:
            failure_type = (
                analysis.get("causal_chain", {})
                .get("metadata", {})
                .get("failure_type", "unknown")
            )
            failure_type_counts[failure_type] += 1

            for root_cause in analysis.get("root_causes", []):
                root_cause_name = root_cause.get("name", "unknown")
                # Simplify root cause names for pattern analysis
                if "config" in root_cause_name.lower():
                    root_cause_counts["configuration_issue"] += 1
                elif "resource" in root_cause_name.lower():
                    root_cause_counts["resource_issue"] += 1
                elif "network" in root_cause_name.lower():
                    root_cause_counts["network_issue"] += 1
                elif "memory" in root_cause_name.lower():
                    root_cause_counts["memory_issue"] += 1
                else:
                    root_cause_counts["other"] += 1

        return {
            "total_analyses": len(self.analysis_history),
            "root_cause_patterns": dict(root_cause_counts),
            "failure_type_distribution": dict(failure_type_counts),
            "most_common_root_cause": max(
                root_cause_counts.keys(), key=lambda k: root_cause_counts[k]
            )
            if root_cause_counts
            else None,
            "most_common_failure_type": max(
                failure_type_counts.keys(), key=lambda k: failure_type_counts[k]
            )
            if failure_type_counts
            else None,
        }

    def export_analysis_results(self, filepath: str) -> None:
        """Export all analysis results to a file."""
        import json

        export_data = {
            "causal_graph": self.causal_graph.export_to_dict(),
            "failure_chains": {
                cid: chain.to_dict() for cid, chain in self.failure_chains.items()
            },
            "analysis_history": self.analysis_history,
            "failure_patterns": self.get_failure_patterns(),
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported causal analysis results to {filepath}")
