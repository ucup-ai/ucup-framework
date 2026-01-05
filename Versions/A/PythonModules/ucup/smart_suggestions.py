"""
UCUP 4.0 - Intelligent API Suggestions and Auto-Completion System

This module provides smart import discovery, context-aware suggestions,
and intelligent help systems for UCUP Framework components.
"""

import inspect
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ComponentRelationship:
    """Relationship between UCUP components."""

    component: str
    related_components: List[str] = field(default_factory=list)
    confidence: float = 1.0
    use_case: str = ""
    category: str = "core"


@dataclass
class SmartSuggestion:
    """Intelligent suggestion for UCUP components."""

    component_name: str
    category: str
    confidence_score: float
    reasoning: str
    related_imports: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    help_text: str = ""


class SmartImportSuggester:
    """
    Intelligent import suggester for UCUP Framework.

    Provides context-aware suggestions for imports and related components
    based on usage patterns and component relationships.
    """

    def __init__(self):
        self.component_relationships = self._build_component_relationships()
        self.usage_patterns = self._load_usage_patterns()
        self.component_metadata = self._build_component_metadata()

    def _build_component_relationships(self) -> Dict[str, ComponentRelationship]:
        """Build comprehensive component relationship map."""
        return {
            # Core probabilistic components
            "ProbabilisticAgent": ComponentRelationship(
                component="ProbabilisticAgent",
                related_components=[
                    "DecisionTracer",
                    "ConfidenceScorer",
                    "FailureDetector",
                    "AlternativePath",
                    "ProbabilisticResult",
                    "StateCheckpointer",
                ],
                confidence=0.95,
                use_case="Core probabilistic reasoning with full observability",
                category="probabilistic",
            ),
            "DecisionTracer": ComponentRelationship(
                component="DecisionTracer",
                related_components=[
                    "ProbabilisticAgent",
                    "DecisionExplorer",
                    "DecisionVisualization",
                    "ReasoningVisualizer",
                    "LiveAgentMonitor",
                ],
                confidence=0.9,
                use_case="Decision tracing and debugging",
                category="observability",
            ),
            "ConfidenceScorer": ComponentRelationship(
                component="ConfidenceScorer",
                related_components=[
                    "ProbabilisticAgent",
                    "ProbabilisticResult",
                    "AlternativePath",
                ],
                confidence=0.85,
                use_case="Confidence scoring and calibration",
                category="probabilistic",
            ),
            # Coordination components
            "HierarchicalCoordination": ComponentRelationship(
                component="HierarchicalCoordination",
                related_components=[
                    "DecisionTracer",
                    "AgentMonitor",
                    "StateCheckpointer",
                    "MessageBus",
                    "TaskQueue",
                ],
                confidence=0.9,
                use_case="Multi-agent hierarchical coordination",
                category="coordination",
            ),
            "DebateCoordination": ComponentRelationship(
                component="DebateCoordination",
                related_components=[
                    "ProbabilisticAgent",
                    "DecisionTracer",
                    "ConfidenceScorer",
                    "AlternativePath",
                    "ConsensusTracker",
                ],
                confidence=0.85,
                use_case="Debate-based decision making",
                category="coordination",
            ),
            # Multimodal components
            "MultimodalFusionEngine": ComponentRelationship(
                component="MultimodalFusionEngine",
                related_components=[
                    "VisionLanguageAgent",
                    "AudioAnalysisAgent",
                    "MultimodalInputs",
                    "FusedAnalysis",
                    "RealTimeStreamingProcessor",
                ],
                confidence=0.9,
                use_case="Multimodal data fusion and processing",
                category="multimodal",
            ),
            "VisionLanguageAgent": ComponentRelationship(
                component="VisionLanguageAgent",
                related_components=[
                    "MultimodalFusionEngine",
                    "MultimodalInputs",
                    "FusedAnalysis",
                    "RealTimeStreamingProcessor",
                    "StreamChunk",
                ],
                confidence=0.9,
                use_case="Vision-language understanding and processing",
                category="multimodal",
            ),
            # Reliability components
            "FailureDetector": ComponentRelationship(
                component="FailureDetector",
                related_components=[
                    "AutomatedRecoveryPipeline",
                    "StateCheckpointer",
                    "SelfHealingAgent",
                    "AgentMonitor",
                    "ReliabilityTracker",
                ],
                confidence=0.9,
                use_case="Failure detection and automated recovery",
                category="reliability",
            ),
            # Advanced probabilistic
            "BayesianAgent": ComponentRelationship(
                component="BayesianAgent",
                related_components=[
                    "BayesianNetwork",
                    "ConditionalProbabilityTable",
                    "BayesianNode",
                    "DecisionTracer",
                    "ConfidenceScorer",
                ],
                confidence=0.9,
                use_case="Bayesian network-based reasoning",
                category="advanced_probabilistic",
            ),
            "MDPAgent": ComponentRelationship(
                component="MDPAgent",
                related_components=[
                    "MarkovDecisionProcess",
                    "MDPState",
                    "MDPAction",
                    "MDPTransition",
                    "DecisionTracer",
                    "StateCheckpointer",
                ],
                confidence=0.85,
                use_case="Markov Decision Process-based planning",
                category="advanced_probabilistic",
            ),
            "MCTSReasoner": ComponentRelationship(
                component="MCTSReasoner",
                related_components=[
                    "MonteCarloTreeSearch",
                    "MCTSNode",
                    "PUCTNode",
                    "AlphaZeroMCTS",
                    "DecisionTracer",
                    "ConfidenceScorer",
                ],
                confidence=0.85,
                use_case="Monte Carlo Tree Search reasoning",
                category="advanced_probabilistic",
            ),
            # Memory and performance
            "MemoryMonitor": ComponentRelationship(
                component="MemoryMonitor",
                related_components=[
                    "CacheManager",
                    "ProviderManager",
                    "IntelligentCache",
                    "PerformanceProfiler",
                    "ResourceTracker",
                ],
                confidence=0.8,
                use_case="Memory monitoring and optimization",
                category="performance",
            ),
            # TOON optimization
            "TOONOptimizer": ComponentRelationship(
                component="TOONOptimizer",
                related_components=[
                    "ToonFormatter",
                    "ToonSchema",
                    "TokenMetrics",
                    "ToonConversionResult",
                ],
                confidence=0.9,
                use_case="Token optimization and formatting",
                category="optimization",
            ),
            # Testing components
            "AgentTestSuite": ComponentRelationship(
                component="AgentTestSuite",
                related_components=[
                    "ProbabilisticAssert",
                    "Scenario",
                    "ExpectedOutcome",
                    "TestRun",
                    "AdversarialTestGenerator",
                    "IntelligentTestGenerator",
                ],
                confidence=0.85,
                use_case="Comprehensive agent testing",
                category="testing",
            ),
            # Deployment components
            "CloudDeploymentManager": ComponentRelationship(
                component="CloudDeploymentManager",
                related_components=[
                    "CloudConfig",
                    "DeploymentSpec",
                    "DeploymentResult",
                    "AWSDeploymentProvider",
                    "AzureDeploymentProvider",
                    "GCPDeploymentProvider",
                ],
                confidence=0.9,
                use_case="Cloud deployment and management",
                category="deployment",
            ),
        }

    def _load_usage_patterns(self) -> Dict[str, Any]:
        """Load common usage patterns for intelligent suggestions."""
        return {
            "customer_service": {
                "primary": [
                    "ProbabilisticAgent",
                    "DecisionTracer",
                    "SentimentAnalyzer",
                ],
                "coordination": ["HierarchicalCoordination"],
                "reliability": ["FailureDetector", "AutomatedRecoveryPipeline"],
            },
            "multimodal": {
                "primary": ["VisionLanguageAgent", "MultimodalFusionEngine"],
                "processing": ["RealTimeStreamingProcessor", "FusedAnalysis"],
                "coordination": ["MultimodalCoordinator"],
            },
            "research": {
                "primary": ["BayesianAgent", "MDPAgent", "MCTSReasoner"],
                "analysis": ["DecisionTracer", "ConfidenceScorer"],
                "optimization": ["TOONOptimizer"],
            },
            "production": {
                "monitoring": ["AgentMonitor", "MemoryMonitor", "PerformanceProfiler"],
                "reliability": ["FailureDetector", "StateCheckpointer"],
                "deployment": ["CloudDeploymentManager"],
            },
        }

    def _build_component_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive metadata for all UCUP components."""
        return {
            "ProbabilisticAgent": {
                "description": "Base class for probabilistic agents with uncertainty quantification",
                "examples": [
                    "agent = ProbabilisticAgent()",
                    "result = await agent.execute('task')",
                    "print(f'Confidence: {result.confidence:.2f}')",
                ],
                "help": "Core agent class that embraces uncertainty rather than fighting it",
            },
            "DecisionTracer": {
                "description": "Comprehensive decision tracing and debugging",
                "examples": [
                    "tracer = DecisionTracer()",
                    "tracer.start_session('session_1')",
                    "trace = tracer.end_session('session_1')",
                ],
                "help": "Track every decision with detailed context for debugging",
            },
            "HierarchicalCoordination": {
                "description": "Multi-agent hierarchical coordination system",
                "examples": [
                    "coord = HierarchicalCoordination()",
                    "await coord.route_to_level('supervisor', task)",
                ],
                "help": "Coordinate multiple agents in hierarchical structures",
            },
            "MultimodalFusionEngine": {
                "description": "Fuse multiple data modalities (text, image, audio)",
                "examples": [
                    "fusion = MultimodalFusionEngine()",
                    "result = fusion.fuse([text_data, image_data])",
                ],
                "help": "Combine different types of data for richer understanding",
            },
        }

    def get_smart_suggestions(
        self, current_imports: List[str] = None, project_context: str = ""
    ) -> List[SmartSuggestion]:
        """
        Get intelligent suggestions based on current imports and context.

        Args:
            current_imports: List of already imported UCUP components
            project_context: Description of project/use case

        Returns:
            List of smart suggestions with confidence scores
        """
        suggestions = []
        current_imports = current_imports or []

        # Analyze project context
        context_patterns = self._analyze_project_context(project_context)

        # Get suggestions based on current imports
        for imported_component in current_imports:
            if imported_component in self.component_relationships:
                relationship = self.component_relationships[imported_component]

                for related in relationship.related_components:
                    if related not in current_imports:  # Don't suggest already imported
                        confidence = self._calculate_suggestion_confidence(
                            related, imported_component, context_patterns
                        )

                        if (
                            confidence > 0.6
                        ):  # Only suggest high-confidence recommendations
                            suggestion = SmartSuggestion(
                                component_name=related,
                                category=self.component_relationships.get(
                                    related, ComponentRelationship(related)
                                ).category,
                                confidence_score=confidence,
                                reasoning=f"Works well with {imported_component}",
                                related_imports=self._get_related_imports(related),
                                examples=self._get_component_examples(related),
                                help_text=self._get_component_help(related),
                            )
                            suggestions.append(suggestion)

        # Add context-based suggestions
        for pattern_type, components in context_patterns.items():
            for component in components:
                if component not in current_imports:
                    suggestion = SmartSuggestion(
                        component_name=component,
                        category=self.component_relationships.get(
                            component, ComponentRelationship(component)
                        ).category,
                        confidence_score=0.7,
                        reasoning=f"Recommended for {pattern_type} use cases",
                        related_imports=self._get_related_imports(component),
                        examples=self._get_component_examples(component),
                        help_text=self._get_component_help(component),
                    )
                    suggestions.append(suggestion)

        # Remove duplicates and sort by confidence
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion.component_name not in seen:
                seen.add(suggestion.component_name)
                unique_suggestions.append(suggestion)

        unique_suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
        return unique_suggestions[:10]  # Top 10 suggestions

    def _analyze_project_context(self, project_context: str) -> Dict[str, List[str]]:
        """Analyze project description to identify relevant patterns."""
        patterns = {}
        context_lower = project_context.lower()

        # Map context keywords to usage patterns
        context_mappings = {
            "customer service": "customer_service",
            "chatbot": "customer_service",
            "support": "customer_service",
            "vision": "multimodal",
            "image": "multimodal",
            "audio": "multimodal",
            "multimodal": "multimodal",
            "research": "research",
            "scientific": "research",
            "experiment": "research",
            "production": "production",
            "deploy": "production",
            "enterprise": "production",
        }

        for keyword, pattern_type in context_mappings.items():
            if keyword in context_lower:
                patterns[pattern_type] = self.usage_patterns.get(pattern_type, {}).get(
                    "primary", []
                )

        return patterns

    def _calculate_suggestion_confidence(
        self, component: str, imported_from: str, context_patterns: Dict[str, List[str]]
    ) -> float:
        """Calculate confidence score for a component suggestion."""
        base_confidence = 0.7

        # Boost confidence if component is in context patterns
        for pattern_components in context_patterns.values():
            if component in pattern_components:
                base_confidence += 0.2
                break

        # Boost confidence based on relationship strength
        if component in self.component_relationships:
            relationship = self.component_relationships[component]
            base_confidence *= relationship.confidence

        return min(base_confidence, 1.0)

    def _get_related_imports(self, component: str) -> List[str]:
        """Get list of imports needed for a component."""
        related_imports = []

        if component in self.component_relationships:
            relationship = self.component_relationships[component]
            related_imports.extend(relationship.related_components[:3])  # Top 3 related

        # Add common supporting imports
        if "Agent" in component:
            related_imports.extend(["ProbabilisticResult", "AlternativePath"])
        elif "Coordination" in component:
            related_imports.extend(["DecisionTracer", "AgentMonitor"])

        return list(set(related_imports))  # Remove duplicates

    def _get_component_examples(self, component: str) -> List[str]:
        """Get usage examples for a component."""
        if component in self.component_metadata:
            return self.component_metadata[component].get("examples", [])

        # Generate basic examples based on component type
        if "Agent" in component:
            return [
                f"{component.lower()} = {component}()",
                f"result = await {component.lower()}.execute('task')",
                f"print(f'Result: {{result.value}}')",
            ]
        elif "Engine" in component:
            return [
                f"{component.lower()} = {component}()",
                f"result = {component.lower()}.process(data)",
            ]
        else:
            return [f"{component.lower()} = {component}()"]

    def _get_component_help(self, component: str) -> str:
        """Get help text for a component."""
        if component in self.component_metadata:
            return self.component_metadata[component].get("help", "")

        # Generate basic help based on component type
        if "Agent" in component:
            return f"{component} provides intelligent agent capabilities with uncertainty quantification."
        elif "Engine" in component:
            return f"{component} processes and analyzes data for intelligent decision making."
        elif "Coordinator" in component:
            return f"{component} manages coordination between multiple agents or components."
        else:
            return f"{component} provides specialized functionality for UCUP Framework."


# Global instance for easy access
_suggester_instance = None


def get_smart_suggestions(
    current_imports: List[str] = None, project_context: str = ""
) -> List[SmartSuggestion]:
    """
    Get intelligent suggestions for UCUP components.

    This is the main entry point for the smart suggestions system.

    Args:
        current_imports: List of currently imported UCUP components
        project_context: Description of your project/use case

    Returns:
        List of smart suggestions ordered by confidence

    Example:
        >>> suggestions = get_smart_suggestions(['ProbabilisticAgent'])
        >>> for suggestion in suggestions:
        ...     print(f"Consider importing: {suggestion.component_name}")
        ...     print(f"Reason: {suggestion.reasoning}")
    """
    global _suggester_instance
    if _suggester_instance is None:
        _suggester_instance = SmartImportSuggester()

    return _suggester_instance.get_smart_suggestions(current_imports, project_context)


def suggest_related_components(component_name: str) -> List[str]:
    """
    Suggest components that work well with the given component.

    Args:
        component_name: Name of the UCUP component

    Returns:
        List of related component names

    Example:
        >>> related = suggest_related_components('ProbabilisticAgent')
        >>> print(related)
        ['DecisionTracer', 'ConfidenceScorer', 'FailureDetector', ...]
    """
    global _suggester_instance
    if _suggester_instance is None:
        _suggester_instance = SmartImportSuggester()

    if component_name in _suggester_instance.component_relationships:
        relationship = _suggester_instance.component_relationships[component_name]
        return relationship.related_components

    return []


def get_component_help(component_name: str) -> str:
    """
    Get detailed help and usage information for a UCUP component.

    Args:
        component_name: Name of the UCUP component

    Returns:
        Help text with usage examples

    Example:
        >>> help_text = get_component_help('ProbabilisticAgent')
        >>> print(help_text)
        ProbabilisticAgent provides intelligent agent capabilities...
    """
    global _suggester_instance
    if _suggester_instance is None:
        _suggester_instance = SmartImportSuggester()

    return _suggester_instance._get_component_help(component_name)


# Enhanced help system integration
def enhanced_help(obj):
    """
    Enhanced help function that provides intelligent UCUP-specific help.

    Args:
        obj: Object or string to get help for

    Returns:
        Enhanced help text with examples and related components
    """
    if isinstance(obj, str):
        component_name = obj
    else:
        component_name = obj.__class__.__name__

    global _suggester_instance
    if _suggester_instance is None:
        _suggester_instance = SmartImportSuggester()

    # Get standard help
    try:
        standard_help = help(obj)
    except:
        standard_help = f"No standard help available for {component_name}"

    # Add UCUP-specific enhancements
    if component_name in _suggester_instance.component_metadata:
        metadata = _suggester_instance.component_metadata[component_name]

        enhanced_text = f"""
{component_name} - UCUP Framework Component
{'=' * (len(component_name) + 25)}

Description: {metadata.get('description', 'No description available')}

Examples:
{chr(10).join(f'  {example}' for example in metadata.get('examples', []))}

Related Components:
{suggest_related_components(component_name)[:5]}

Help: {metadata.get('help', 'No additional help available')}

{'=' * 50}
Standard Python help follows:
{'=' * 50}

{standard_help}
"""
        print(enhanced_text)
    else:
        # Fallback to standard help
        help(obj)


# Context-aware auto-completion hints
def get_method_completion_hints(
    component_name: str, method_name: str = ""
) -> Dict[str, Any]:
    """
    Get intelligent completion hints for component methods.

    Args:
        component_name: Name of the UCUP component
        method_name: Specific method name (optional)

    Returns:
        Dictionary with completion hints and examples
    """
    hints = {
        "ProbabilisticAgent": {
            "execute": {
                "signature": "execute(self, task: str, **kwargs) -> ProbabilisticResult",
                "description": "Execute probabilistic reasoning task",
                "example": "result = await agent.execute('Analyze this data', context=data)",
                "parameters": {
                    "task": "String description of the task to perform",
                    "kwargs": "Additional context and configuration options",
                },
            },
            "get_coordination_strategies": {
                "signature": "get_coordination_strategies(self) -> List[str]",
                "description": "Get available coordination strategies",
                "example": "strategies = agent.get_coordination_strategies()",
                "returns": "List of available coordination strategy names",
            },
        },
        "DecisionTracer": {
            "start_session": {
                "signature": "start_session(self, session_id: str) -> str",
                "description": "Start a new decision tracing session",
                "example": "tracer.start_session('analysis_session_1')",
                "parameters": {"session_id": "Unique identifier for the session"},
            },
            "record_decision": {
                "signature": "record_decision(self, session_id: str, ...) -> DecisionNode",
                "description": "Record a decision with full context",
                "example": "decision = tracer.record_decision('session_1', actions, 'chosen_action', scores)",
                "parameters": {
                    "session_id": "Session identifier",
                    "available_actions": "List of possible actions",
                    "chosen_action": "The selected action",
                    "confidence_scores": "Confidence scores for each action",
                },
            },
        },
        "MultimodalFusionEngine": {
            "fuse": {
                "signature": "fuse(self, inputs: List[MultimodalInputs]) -> FusedAnalysis",
                "description": "Fuse multiple multimodal inputs",
                "example": "result = fusion.fuse([text_input, image_input, audio_input])",
                "parameters": {"inputs": "List of MultimodalInputs to fuse"},
            }
        },
    }

    if component_name in hints:
        if method_name:
            return hints[component_name].get(method_name, {})
        return hints[component_name]

    return {}


# Auto-completion integration helper
def get_completion_context(code_context: str) -> Dict[str, Any]:
    """
    Analyze code context and provide completion suggestions.

    Args:
        code_context: Current code context around cursor

    Returns:
        Dictionary with completion suggestions and metadata
    """
    suggestions = []

    # Analyze the code context
    lines = code_context.split("\n")
    current_line = ""
    imported_components = []

    for line in lines:
        line = line.strip()
        if line.startswith("from ucup import") or line.startswith("import ucup"):
            # Extract imported components
            if "from ucup import" in line:
                imports_part = line.split("from ucup import")[1].split("#")[0].strip()
                if imports_part != "*":
                    imported_components.extend(
                        [imp.strip() for imp in imports_part.split(",")]
                    )

        if "agent." in line or "tracer." in line or "fusion." in line:
            current_line = line

    # Provide context-aware suggestions
    for component in imported_components:
        if component in [
            "ProbabilisticAgent",
            "DecisionTracer",
            "MultimodalFusionEngine",
        ]:
            method_hints = get_method_completion_hints(component)

            for method_name, hint_info in method_hints.items():
                suggestions.append(
                    {
                        "text": method_name,
                        "detail": hint_info.get("signature", method_name),
                        "documentation": hint_info.get("description", ""),
                        "example": hint_info.get("example", ""),
                    }
                )

    return {
        "suggestions": suggestions,
        "imported_components": imported_components,
        "context_analysis": {
            "has_agent_usage": any("agent." in line for line in lines),
            "has_tracing": any("tracer." in line for line in lines),
            "has_fusion": any("fusion." in line for line in lines),
        },
    }


# Initialize the suggester when module is imported
_suggester_instance = SmartImportSuggester()
