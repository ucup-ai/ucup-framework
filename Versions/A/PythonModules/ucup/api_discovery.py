"""
UCUP Smart API Discovery & Auto-Import System

This module provides intelligent API discovery, metadata-enhanced exports,
and automatic import suggestions for the UCUP Framework.

Key Features:
- Metadata-enhanced __all__ exports with rich component information
- Intelligent API discovery utilities
- Auto-import suggestions based on usage patterns
- Type hint enhancement and validation
- Component relationship mapping
"""

import importlib
import inspect
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, get_type_hints


@dataclass
class ComponentMetadata:
    """Rich metadata for UCUP components."""

    name: str
    module: str
    category: str
    description: str
    version: str = "1.0.0"
    stability: str = "stable"  # stable, experimental, deprecated
    dependencies: List[str] = field(default_factory=list)
    related_components: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    help_text: str = ""
    type_hints: Dict[str, str] = field(default_factory=dict)
    confidence_score: float = 1.0
    usage_patterns: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)


@dataclass
class ExportMetadata:
    """Enhanced export information with metadata."""

    component: ComponentMetadata
    export_name: str
    is_class: bool = False
    is_function: bool = False
    is_constant: bool = False
    is_type: bool = False
    signature: Optional[str] = None
    docstring: Optional[str] = None
    source_lines: Tuple[int, int] = (0, 0)


class APIDiscoveryEngine:
    """
    Intelligent API discovery engine for UCUP components.

    Provides comprehensive discovery, metadata extraction, and
    intelligent suggestions for component usage.
    """

    def __init__(self, base_module: str = "ucup"):
        self.base_module = base_module
        self.discovered_components: Dict[str, ComponentMetadata] = {}
        self.export_registry: Dict[str, ExportMetadata] = {}
        self.module_cache: Dict[str, Any] = {}

    def discover_all_components(self) -> Dict[str, ComponentMetadata]:
        """
        Discover all UCUP components with rich metadata.

        Returns comprehensive component registry with metadata.
        """
        self._discover_core_components()
        self._discover_submodules()
        self._build_relationships()
        self._enhance_type_hints()

        return self.discovered_components

    def _discover_core_components(self):
        """Discover core UCUP components from main module."""
        try:
            main_module = importlib.import_module(self.base_module)
            self._analyze_module(main_module, "core")
        except ImportError as e:
            warnings.warn(f"Could not import main module {self.base_module}: {e}")

    def _discover_submodules(self):
        """Discover components from submodules."""
        submodules = [
            "probabilistic",
            "coordination",
            "validation",
            "errors",
            "feature_flags",
            "memory_management",
            "metrics",
            "multimodal",
            "observability",
            "plugins",
            "pytest_plugin",
            "reliability",
            "smart_recommendations",
            "smart_suggestions",
            "testing",
            "test_environments",
            "toon",
            "advanced_probabilistic",
            "agent_monitoring",
            "build_wrapper",
            "cli",
            "cloud_deployment",
            "config",
            "deployment_automation",
            "email_reporting",
            "performance",
            "self_healing",
            "time_travel_debugger",
            "uncertainty_visualizer",
            "visualization",
        ]

        for submodule in submodules:
            try:
                module_path = f"{self.base_module}.{submodule}"
                module = importlib.import_module(module_path)
                category = self._infer_category_from_module(submodule)
                self._analyze_module(module, category)
            except ImportError:
                continue

    def _analyze_module(self, module: Any, category: str):
        """Analyze a module and extract component metadata."""
        module_name = module.__name__

        # Get __all__ exports if available
        exports = getattr(module, "__all__", None)
        if exports is None:
            # Fallback to public attributes
            exports = [name for name in dir(module) if not name.startswith("_")]

        for export_name in exports:
            try:
                obj = getattr(module, export_name, None)
                if obj is None:
                    continue

                metadata = self._extract_component_metadata(
                    obj, export_name, module_name, category
                )

                if metadata:
                    self.discovered_components[export_name] = metadata
                    export_meta = self._create_export_metadata(
                        obj, metadata, export_name
                    )
                    self.export_registry[export_name] = export_meta

            except Exception as e:
                warnings.warn(f"Error analyzing {export_name} in {module_name}: {e}")

    def _extract_component_metadata(
        self, obj: Any, name: str, module: str, category: str
    ) -> Optional[ComponentMetadata]:
        """Extract rich metadata from a component."""
        try:
            # Get basic information
            description = self._get_description(obj)
            version = getattr(obj, "__version__", "1.0.0")
            stability = self._infer_stability(obj)

            # Get type hints
            type_hints = {}
            try:
                hints = get_type_hints(obj)
                type_hints = {k: str(v) for k, v in hints.items()}
            except:
                pass

            # Get usage patterns and examples
            examples = self._extract_examples(obj)
            usage_patterns = self._infer_usage_patterns(obj, category)

            # Get tags
            tags = self._generate_tags(obj, category)

            return ComponentMetadata(
                name=name,
                module=module,
                category=category,
                description=description,
                version=version,
                stability=stability,
                type_hints=type_hints,
                examples=examples,
                usage_patterns=usage_patterns,
                tags=tags,
            )

        except Exception as e:
            warnings.warn(f"Error extracting metadata for {name}: {e}")
            return None

    def _create_export_metadata(
        self, obj: Any, component_meta: ComponentMetadata, export_name: str
    ) -> ExportMetadata:
        """Create detailed export metadata."""
        is_class = inspect.isclass(obj)
        is_function = callable(obj) and not is_class
        is_constant = not callable(obj)
        is_type = hasattr(obj, "__annotations__") or inspect.isclass(obj)

        signature = None
        if is_function or is_class:
            try:
                signature = str(inspect.signature(obj))
            except:
                pass

        docstring = inspect.getdoc(obj)

        return ExportMetadata(
            component=component_meta,
            export_name=export_name,
            is_class=is_class,
            is_function=is_function,
            is_constant=is_constant,
            is_type=is_type,
            signature=signature,
            docstring=docstring,
        )

    def _get_description(self, obj: Any) -> str:
        """Extract description from docstring or other sources."""
        doc = inspect.getdoc(obj)
        if doc:
            # Take first line of docstring
            return doc.split("\n")[0].strip()

        # Fallback descriptions based on type
        if inspect.isclass(obj):
            return f"{obj.__name__} class for UCUP framework"
        elif callable(obj):
            return f"{obj.__name__} function for UCUP framework"
        else:
            return f"{obj.__name__} component"

    def _infer_stability(self, obj: Any) -> str:
        """Infer stability level of component."""
        # Check for stability indicators
        if hasattr(obj, "__version__"):
            version = getattr(obj, "__version__", "")
            if "dev" in version or "alpha" in version or "beta" in version:
                return "experimental"
            return "stable"

        # Check class name for experimental indicators
        name = getattr(obj, "__name__", "")
        if "experimental" in name.lower() or "beta" in name.lower():
            return "experimental"

        return "stable"

    def _extract_examples(self, obj: Any) -> List[str]:
        """Extract usage examples from docstring."""
        examples = []
        doc = inspect.getdoc(obj)

        if doc:
            # Look for Examples section
            example_match = re.search(
                r"Examples?:?\s*\n(.*?)(?:\n\n|\n[A-Z]|$)",
                doc,
                re.DOTALL | re.IGNORECASE,
            )
            if example_match:
                example_text = example_match.group(1)
                # Extract code examples
                code_blocks = re.findall(
                    r"```python\s*(.*?)\s*```|`([^`]+)`", example_text, re.DOTALL
                )
                for block in code_blocks:
                    examples.append(block[0] or block[1])

        return examples[:3]  # Limit to 3 examples

    def _infer_usage_patterns(self, obj: Any, category: str) -> List[str]:
        """Infer common usage patterns for component."""
        patterns = []

        if category == "probabilistic":
            patterns.extend(["uncertainty_quantification", "decision_making"])
        elif category == "coordination":
            patterns.extend(["multi_agent_systems", "task_delegation"])
        elif category == "testing":
            patterns.extend(["test_automation", "validation"])
        elif category == "multimodal":
            patterns.extend(["data_fusion", "cross_modal_analysis"])
        elif category == "deployment":
            patterns.extend(["cloud_deployment", "infrastructure"])
        elif category == "monitoring":
            patterns.extend(["observability", "performance_tracking"])

        return patterns

    def _generate_tags(self, obj: Any, category: str) -> Set[str]:
        """Generate relevant tags for component."""
        tags = {category}

        name = getattr(obj, "__name__", "").lower()

        # Add type-based tags
        if inspect.isclass(obj):
            tags.add("class")
            if issubclass(obj, Exception):
                tags.add("exception")
        elif callable(obj):
            tags.add("function")
        else:
            tags.add("constant")

        # Add domain-specific tags
        if "agent" in name:
            tags.add("agent")
        if "test" in name:
            tags.add("testing")
        if "monitor" in name:
            tags.add("monitoring")
        if "deploy" in name:
            tags.add("deployment")

        return tags

    def _infer_category_from_module(self, module_name: str) -> str:
        """Infer category from module name."""
        category_map = {
            "probabilistic": "probabilistic",
            "coordination": "coordination",
            "validation": "validation",
            "testing": "testing",
            "multimodal": "multimodal",
            "deployment": "deployment",
            "monitoring": "monitoring",
            "observability": "observability",
            "plugins": "plugins",
            "config": "configuration",
            "metrics": "metrics",
            "reliability": "reliability",
            "feature_flags": "features",
            "memory_management": "performance",
            "cloud_deployment": "deployment",
            "smart_recommendations": "intelligence",
            "smart_suggestions": "intelligence",
            "advanced_probabilistic": "probabilistic",
            "agent_monitoring": "monitoring",
            "build_wrapper": "build",
            "cli": "cli",
            "deployment_automation": "deployment",
            "email_reporting": "reporting",
            "self_healing": "reliability",
            "time_travel_debugger": "debugging",
            "uncertainty_visualizer": "visualization",
            "toon": "optimization",
        }

        return category_map.get(module_name, "core")

    def _build_relationships(self):
        """Build component relationships and dependencies."""
        # Define known relationships
        relationships = {
            "ProbabilisticAgent": [
                "DecisionTracer",
                "ProbabilisticResult",
                "AlternativePath",
            ],
            "DecisionTracer": ["ProbabilisticAgent", "DecisionVisualization"],
            "HierarchicalCoordination": ["AgentMessage", "CoordinationAgent"],
            "MultimodalFusionEngine": ["MultimodalInputs", "FusedAnalysis"],
            "FailureDetector": ["AutomatedRecoveryPipeline", "StateCheckpointer"],
            "AgentTestSuite": ["ProbabilisticAssert", "Scenario", "TestRun"],
            "CloudDeploymentManager": ["CloudConfig", "DeploymentSpec"],
            "MemoryMonitor": ["CacheManager", "ProviderManager"],
            "FeatureFlagManager": ["FeatureFlag", "FeatureFlagState"],
            "ToonFormatter": ["ToonSchema", "TokenMetrics"],
            "BayesianAgent": ["BayesianNetwork", "ConditionalProbabilityTable"],
            "MDPAgent": ["MarkovDecisionProcess", "MDPState"],
            "MCTSReasoner": ["MonteCarloTreeSearch", "MCTSNode"],
        }

        for component_name, related in relationships.items():
            if component_name in self.discovered_components:
                component = self.discovered_components[component_name]
                component.related_components = related

    def _enhance_type_hints(self):
        """Enhance type hints across components."""
        for name, component in self.discovered_components.items():
            try:
                obj = self._get_component_object(name)
                if obj:
                    hints = get_type_hints(obj)
                    component.type_hints = {k: str(v) for k, v in hints.items()}
            except:
                continue

    def _get_component_object(self, name: str) -> Optional[Any]:
        """Get component object by name."""
        if name in self.export_registry:
            export_meta = self.export_registry[name]
            module_name = export_meta.component.module

            try:
                if module_name not in self.module_cache:
                    self.module_cache[module_name] = importlib.import_module(
                        module_name
                    )
                module = self.module_cache[module_name]
                return getattr(module, name, None)
            except:
                pass
        return None

    def get_components_by_category(self, category: str) -> List[ComponentMetadata]:
        """Get all components in a specific category."""
        return [
            comp
            for comp in self.discovered_components.values()
            if comp.category == category
        ]

    def get_components_by_tag(self, tag: str) -> List[ComponentMetadata]:
        """Get all components with a specific tag."""
        return [
            comp for comp in self.discovered_components.values() if tag in comp.tags
        ]

    def search_components(self, query: str) -> List[ComponentMetadata]:
        """Search components by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for component in self.discovered_components.values():
            if (
                query_lower in component.name.lower()
                or query_lower in component.description.lower()
                or any(query_lower in tag for tag in component.tags)
            ):
                results.append(component)

        return results

    def get_import_suggestions(
        self, current_imports: List[str] = None, context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Get intelligent import suggestions based on current imports and context.

        Args:
            current_imports: List of currently imported component names
            context: Usage context description

        Returns:
            List of suggestion dictionaries with metadata
        """
        suggestions = []
        current_imports = current_imports or []

        # Find relationships for current imports
        related_components = set()
        for import_name in current_imports:
            if import_name in self.discovered_components:
                component = self.discovered_components[import_name]
                related_components.update(component.related_components)

        # Filter out already imported components
        new_suggestions = related_components - set(current_imports)

        # Convert to suggestion format
        for comp_name in new_suggestions:
            if comp_name in self.discovered_components:
                component = self.discovered_components[comp_name]
                suggestions.append(
                    {
                        "component": comp_name,
                        "category": component.category,
                        "description": component.description,
                        "confidence": component.confidence_score,
                        "reason": f"Related to {', '.join(current_imports[:2])}",
                    }
                )

        # Sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)

        return suggestions[:5]  # Top 5 suggestions

    def generate_enhanced_all_exports(self, module_path: str) -> Dict[str, Any]:
        """
        Generate enhanced __all__ exports with metadata for a module.

        Args:
            module_path: Path to the module file

        Returns:
            Dictionary with __all__ list and metadata
        """
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("temp_module", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get current exports
            current_all = getattr(module, "__all__", None)
            if current_all is None:
                # Generate __all__ from public attributes
                public_attrs = [
                    name for name in dir(module) if not name.startswith("_")
                ]
                current_all = public_attrs

            # Generate metadata for each export
            metadata = {}
            for export_name in current_all:
                obj = getattr(module, export_name, None)
                if obj is not None:
                    component_meta = self._extract_component_metadata(
                        obj, export_name, module.__name__, "generated"
                    )
                    if component_meta:
                        metadata[export_name] = component_meta

            return {
                "__all__": current_all,
                "metadata": metadata,
                "generated_at": str(Path(module_path).name),
            }

        except Exception as e:
            warnings.warn(f"Error generating enhanced exports for {module_path}: {e}")
            return {"__all__": [], "metadata": {}, "error": str(e)}


# Global discovery engine instance
_discovery_engine = None


def get_discovery_engine() -> APIDiscoveryEngine:
    """Get the global API discovery engine instance."""
    global _discovery_engine
    if _discovery_engine is None:
        _discovery_engine = APIDiscoveryEngine()
    return _discovery_engine


def discover_ucup_api() -> Dict[str, ComponentMetadata]:
    """
    Discover all UCUP API components with rich metadata.

    Returns:
        Comprehensive registry of UCUP components with metadata
    """
    engine = get_discovery_engine()
    return engine.discover_all_components()


def get_smart_import_suggestions(
    current_imports: List[str] = None, context: str = ""
) -> List[Dict[str, Any]]:
    """
    Get intelligent import suggestions for UCUP components.

    Args:
        current_imports: List of currently imported component names
        context: Usage context (e.g., 'customer_service', 'multimodal')

    Returns:
        List of suggestion dictionaries with component details
    """
    engine = get_discovery_engine()
    return engine.get_import_suggestions(current_imports, context)


def search_ucup_components(query: str) -> List[ComponentMetadata]:
    """
    Search UCUP components by name, description, or functionality.

    Args:
        query: Search query string

    Returns:
        List of matching components with metadata
    """
    engine = get_discovery_engine()
    return engine.search_components(query)


def get_components_by_category(category: str) -> List[ComponentMetadata]:
    """
    Get all UCUP components in a specific category.

    Args:
        category: Component category (e.g., 'probabilistic', 'coordination')

    Returns:
        List of components in the category
    """
    engine = get_discovery_engine()
    return engine.get_components_by_category(category)


def generate_module_exports(module_path: str) -> Dict[str, Any]:
    """
    Generate enhanced __all__ exports with metadata for a module.

    Args:
        module_path: Path to the Python module file

    Returns:
        Dictionary containing __all__ list and component metadata
    """
    engine = get_discovery_engine()
    return engine.generate_enhanced_all_exports(module_path)


# Auto-import helpers
def create_auto_import_map() -> Dict[str, Dict[str, Any]]:
    """
    Create a comprehensive auto-import mapping for UCUP components.

    Returns:
        Dictionary mapping component names to import information
    """
    engine = get_discovery_engine()
    components = engine.discover_all_components()

    import_map = {}
    for name, component in components.items():
        import_map[name] = {
            "module": component.module,
            "category": component.category,
            "description": component.description,
            "tags": list(component.tags),
            "examples": component.examples,
        }

    return import_map


# Type hint enhancement utilities
def enhance_type_hints_for_module(module_name: str) -> Dict[str, Dict[str, str]]:
    """
    Enhance and validate type hints for all components in a module.

    Args:
        module_name: Name of the module to enhance

    Returns:
        Dictionary of enhanced type hints by component
    """
    try:
        module = importlib.import_module(module_name)
        enhanced_hints = {}

        for name in getattr(module, "__all__", dir(module)):
            obj = getattr(module, name, None)
            if obj and (callable(obj) or inspect.isclass(obj)):
                try:
                    hints = get_type_hints(obj)
                    enhanced_hints[name] = {k: str(v) for k, v in hints.items()}
                except Exception as e:
                    enhanced_hints[name] = {"error": str(e)}

        return enhanced_hints

    except ImportError:
        return {}


# Export the enhanced API discovery interface
__all__ = [
    # Core discovery classes
    "APIDiscoveryEngine",
    "ComponentMetadata",
    "ExportMetadata",
    # Main discovery functions
    "discover_ucup_api",
    "get_smart_import_suggestions",
    "search_ucup_components",
    "get_components_by_category",
    # Utility functions
    "generate_module_exports",
    "create_auto_import_map",
    "enhance_type_hints_for_module",
    "get_discovery_engine",
]
