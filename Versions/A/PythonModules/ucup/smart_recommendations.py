"""
UCUP Smart Recommendations Engine
==================================

Intelligent system that analyzes project requirements and suggests optimal
UCUP components, configurations, and architectures based on learning from
successful builds and deployments.
"""

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProjectProfile:
    """Profile of a UCUP project with requirements and characteristics."""

    project_type: str  # 'agent', 'service', 'pipeline', 'multimodal', 'enterprise'
    domain: str  # 'customer-service', 'data-analysis', 'research', 'multimodal', etc.
    complexity: str  # 'simple', 'medium', 'complex'
    scale_requirements: str  # 'single', 'multi-agent', 'distributed'
    performance_needs: str  # 'low-latency', 'high-throughput', 'balanced'
    reliability_requirements: str  # 'basic', 'high', 'mission-critical'
    multimodal_needs: List[str]  # ['text', 'image', 'audio', 'sensor']
    coordination_pattern: Optional[str] = None
    deployment_target: str = "local"


@dataclass
class ComponentRecommendation:
    """Recommendation for a specific UCUP component."""

    component_type: str
    component_name: str
    confidence_score: float
    reasoning: str
    configuration_suggestions: Dict[str, Any]
    alternatives: List[str]
    benefits: List[str]


@dataclass
class ArchitectureRecommendation:
    """Complete architecture recommendation for a project."""

    project_profile: ProjectProfile
    primary_agent: ComponentRecommendation
    supporting_components: List[ComponentRecommendation]
    coordination_strategy: ComponentRecommendation
    deployment_config: Dict[str, Any]
    estimated_benefits: Dict[str, float]
    confidence_score: float
    learning_metadata: Dict[str, Any]


class SmartRecommendationEngine:
    """Intelligent recommendation engine for UCUP components."""

    def __init__(self, learning_data_path: str = None):
        self.learning_data_path = learning_data_path or os.path.expanduser(
            "~/.ucup/learning_data.json"
        )
        self.learning_data = self._load_learning_data()
        self.component_knowledge_base = self._build_component_knowledge_base()

    def _load_learning_data(self) -> Dict[str, Any]:
        """Load historical learning data from successful builds."""
        try:
            if os.path.exists(self.learning_data_path):
                with open(self.learning_data_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load learning data: {e}")

        return {
            "successful_builds": [],
            "project_patterns": {},
            "component_effectiveness": {},
            "last_updated": datetime.now().isoformat(),
        }

    def _build_component_knowledge_base(self) -> Dict[str, Any]:
        """Build knowledge base of UCUP components and their use cases."""
        return {
            "agents": {
                "ProbabilisticAgent": {
                    "domains": ["general", "customer-service", "data-analysis"],
                    "complexity_levels": ["simple", "medium"],
                    "scale_support": ["single", "multi-agent"],
                    "performance_profile": "balanced",
                    "reliability_profile": "high",
                    "multimodal_support": ["text"],
                    "strengths": [
                        "uncertainty_quantification",
                        "probabilistic_reasoning",
                    ],
                    "use_cases": [
                        "decision_making",
                        "risk_assessment",
                        "recommendation_systems",
                    ],
                },
                "BayesianAgent": {
                    "domains": ["data-analysis", "research", "financial"],
                    "complexity_levels": ["medium", "complex"],
                    "scale_support": ["single", "multi-agent"],
                    "performance_profile": "balanced",
                    "reliability_profile": "high",
                    "multimodal_support": ["text", "structured"],
                    "strengths": [
                        "causal_reasoning",
                        "probabilistic_inference",
                        "belief_updates",
                        "uncertainty_quantification",
                    ],
                    "use_cases": [
                        "diagnostic_systems",
                        "predictive_modeling",
                        "risk_analysis",
                        "human_in_the_loop_decisions",
                    ],
                },
                "MDPAgent": {
                    "domains": ["research", "planning", "optimization"],
                    "complexity_levels": ["complex"],
                    "scale_support": ["single"],
                    "performance_profile": "high-throughput",
                    "reliability_profile": "high",
                    "multimodal_support": ["text"],
                    "strengths": [
                        "sequential_decision_making",
                        "optimal_planning",
                        "value_iteration",
                    ],
                    "use_cases": [
                        "resource_planning",
                        "process_optimization",
                        "strategy_games",
                    ],
                },
                "MCTSReasoner": {
                    "domains": ["research", "planning", "games"],
                    "complexity_levels": ["complex"],
                    "scale_support": ["single"],
                    "performance_profile": "high-throughput",
                    "reliability_profile": "high",
                    "multimodal_support": ["text"],
                    "strengths": ["exploration", "tree_search", "uncertainty_handling"],
                    "use_cases": ["game_playing", "planning", "decision_trees"],
                },
                "VisionLanguageAgent": {
                    "domains": ["multimodal", "computer-vision", "content-analysis"],
                    "complexity_levels": ["medium", "complex"],
                    "scale_support": ["single", "multi-agent"],
                    "performance_profile": "balanced",
                    "reliability_profile": "high",
                    "multimodal_support": ["text", "image", "video"],
                    "strengths": [
                        "multimodal_fusion",
                        "visual_understanding",
                        "content_analysis",
                    ],
                    "use_cases": [
                        "image_captioning",
                        "visual_qa",
                        "content_moderation",
                    ],
                },
                "StructuredDataAgent": {
                    "domains": ["data-analysis", "business-intelligence", "reporting"],
                    "complexity_levels": ["medium"],
                    "scale_support": ["single", "multi-agent"],
                    "performance_profile": "high-throughput",
                    "reliability_profile": "high",
                    "multimodal_support": ["structured", "text"],
                    "strengths": [
                        "data_processing",
                        "pattern_recognition",
                        "insights_generation",
                    ],
                    "use_cases": [
                        "data_analysis",
                        "reporting",
                        "business_intelligence",
                    ],
                },
            },
            "coordinators": {
                "HierarchicalCoordination": {
                    "scale_support": ["multi-agent", "distributed"],
                    "complexity_levels": ["medium", "complex"],
                    "performance_profile": "balanced",
                    "reliability_profile": "high",
                    "use_cases": [
                        "team_management",
                        "workflow_orchestration",
                        "decision_hierarchy",
                    ],
                },
                "DebateCoordination": {
                    "scale_support": ["multi-agent"],
                    "complexity_levels": ["complex"],
                    "performance_profile": "low-latency",
                    "reliability_profile": "high",
                    "use_cases": [
                        "decision_debates",
                        "consensus_building",
                        "alternative_evaluation",
                    ],
                },
                "MarketBasedCoordination": {
                    "scale_support": ["multi-agent", "distributed"],
                    "complexity_levels": ["complex"],
                    "performance_profile": "high-throughput",
                    "reliability_profile": "medium",
                    "use_cases": [
                        "resource_allocation",
                        "task_assignment",
                        "economic_optimization",
                    ],
                },
                "SwarmCoordination": {
                    "scale_support": ["distributed"],
                    "complexity_levels": ["complex"],
                    "performance_profile": "high-throughput",
                    "reliability_profile": "high",
                    "use_cases": [
                        "distributed_processing",
                        "emergent_behavior",
                        "adaptive_systems",
                    ],
                },
            },
            "deployment_targets": {
                "local": {
                    "performance": "balanced",
                    "scalability": "limited",
                    "cost": "free",
                },
                "docker": {
                    "performance": "good",
                    "scalability": "medium",
                    "cost": "low",
                },
                "kubernetes": {
                    "performance": "excellent",
                    "scalability": "high",
                    "cost": "medium",
                },
                "aws": {
                    "performance": "excellent",
                    "scalability": "elastic",
                    "cost": "variable",
                },
                "gcp": {
                    "performance": "excellent",
                    "scalability": "elastic",
                    "cost": "variable",
                },
                "azure": {
                    "performance": "excellent",
                    "scalability": "elastic",
                    "cost": "variable",
                },
            },
        }

    def analyze_project_requirements(
        self,
        project_description: str,
        existing_code: Optional[str] = None,
        requirements_file: Optional[str] = None,
    ) -> ProjectProfile:
        """
        Analyze project requirements from description, code, and requirements files.
        Returns a structured project profile.
        """
        # Extract key information from project description
        description_lower = project_description.lower()

        # Determine project type
        if any(
            word in description_lower
            for word in ["agent", "ai", "intelligent", "cognitive"]
        ):
            project_type = "agent"
        elif any(
            word in description_lower for word in ["service", "api", "microservice"]
        ):
            project_type = "service"
        elif any(
            word in description_lower for word in ["pipeline", "workflow", "automation"]
        ):
            project_type = "pipeline"
        elif any(
            word in description_lower
            for word in ["multimodal", "vision", "audio", "sensor"]
        ):
            project_type = "multimodal"
        else:
            project_type = "agent"  # default

        # Determine domain
        domain_keywords = {
            "customer-service": ["customer", "support", "chatbot", "service"],
            "data-analysis": ["data", "analysis", "analytics", "processing"],
            "research": ["research", "scientific", "experiment", "study"],
            "multimodal": ["multimodal", "vision", "audio", "sensor", "image"],
            "financial": ["financial", "trading", "market", "finance"],
            "healthcare": ["health", "medical", "patient", "diagnosis"],
            "education": ["education", "learning", "teaching", "student"],
            "gaming": ["game", "gaming", "player", "entertainment"],
        }

        domain = "general"
        for domain_name, keywords in domain_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                domain = domain_name
                break

        # Determine complexity
        complexity_indicators = {
            "simple": ["basic", "simple", "starter", "beginner", "prototype"],
            "medium": ["intermediate", "standard", "moderate", "typical"],
            "complex": [
                "advanced",
                "complex",
                "enterprise",
                "large-scale",
                "sophisticated",
            ],
        }

        complexity = "medium"  # default
        for level, indicators in complexity_indicators.items():
            if any(indicator in description_lower for indicator in indicators):
                complexity = level
                break

        # Determine scale requirements
        if any(
            word in description_lower
            for word in ["distributed", "cluster", "swarm", "fleet"]
        ):
            scale_requirements = "distributed"
        elif any(
            word in description_lower
            for word in ["multi-agent", "team", "coordination"]
        ):
            scale_requirements = "multi-agent"
        else:
            scale_requirements = "single"

        # Determine performance needs
        if any(
            word in description_lower
            for word in ["real-time", "fast", "low-latency", "immediate"]
        ):
            performance_needs = "low-latency"
        elif any(
            word in description_lower
            for word in ["high-throughput", "batch", "processing", "volume"]
        ):
            performance_needs = "high-throughput"
        else:
            performance_needs = "balanced"

        # Determine reliability requirements
        if any(
            word in description_lower
            for word in [
                "mission-critical",
                "life-critical",
                "enterprise",
                "production",
            ]
        ):
            reliability_requirements = "mission-critical"
        elif any(
            word in description_lower for word in ["reliable", "robust", "production"]
        ):
            reliability_requirements = "high"
        else:
            reliability_requirements = "basic"

        # Determine multimodal needs
        multimodal_needs = []
        multimodal_types = ["text", "image", "audio", "video", "sensor", "structured"]
        for modal_type in multimodal_types:
            if modal_type in description_lower:
                multimodal_needs.append(modal_type)

        # Analyze existing code if provided
        if existing_code:
            code_analysis = self._analyze_code_patterns(existing_code)
            if code_analysis.get("multimodal_indicators"):
                multimodal_needs.extend(code_analysis["multimodal_indicators"])

        return ProjectProfile(
            project_type=project_type,
            domain=domain,
            complexity=complexity,
            scale_requirements=scale_requirements,
            performance_needs=performance_needs,
            reliability_requirements=reliability_requirements,
            multimodal_needs=list(set(multimodal_needs)),
        )

    def _analyze_code_patterns(self, code: str) -> Dict[str, Any]:
        """Analyze existing code to understand patterns and requirements."""
        analysis = {
            "multimodal_indicators": [],
            "complexity_indicators": [],
            "framework_imports": [],
            "design_patterns": [],
        }

        code_lower = code.lower()

        # Check for multimodal indicators
        if "image" in code_lower or "cv2" in code_lower or "pillow" in code_lower:
            analysis["multimodal_indicators"].append("image")
        if "audio" in code_lower or "speech" in code_lower or "wav" in code_lower:
            analysis["multimodal_indicators"].append("audio")
        if "sensor" in code_lower or "iot" in code_lower:
            analysis["multimodal_indicators"].append("sensor")

        # Check for complexity indicators
        if "async" in code_lower or "await" in code_lower:
            analysis["complexity_indicators"].append("async")
        if "class" in code_lower and code.count("class") > 3:
            analysis["complexity_indicators"].append("multiple_classes")

        return analysis

    def generate_recommendations(
        self, project_profile: ProjectProfile
    ) -> ArchitectureRecommendation:
        """
        Generate comprehensive architecture recommendations based on project profile.
        """
        # Find best primary agent
        primary_agent = self._recommend_primary_agent(project_profile)

        # Find supporting components
        supporting_components = self._recommend_supporting_components(
            project_profile, primary_agent
        )

        # Recommend coordination strategy
        coordination_strategy = self._recommend_coordination_strategy(project_profile)

        # Generate deployment configuration
        deployment_config = self._recommend_deployment_config(project_profile)

        # Calculate estimated benefits
        estimated_benefits = self._calculate_estimated_benefits(
            project_profile, primary_agent, supporting_components
        )

        # Calculate overall confidence
        confidence_score = self._calculate_recommendation_confidence(
            project_profile, primary_agent, supporting_components
        )

        return ArchitectureRecommendation(
            project_profile=project_profile,
            primary_agent=primary_agent,
            supporting_components=supporting_components,
            coordination_strategy=coordination_strategy,
            deployment_config=deployment_config,
            estimated_benefits=estimated_benefits,
            confidence_score=confidence_score,
            learning_metadata={
                "generated_at": datetime.now().isoformat(),
                "learning_data_version": self.learning_data.get(
                    "last_updated", "unknown"
                ),
                "recommendation_engine_version": "1.0.0",
            },
        )

    def _recommend_primary_agent(
        self, profile: ProjectProfile
    ) -> ComponentRecommendation:
        """Recommend the best primary agent for the project."""
        best_agent = None
        best_score = 0
        best_reasoning = ""

        agent_kb = self.component_knowledge_base["agents"]

        for agent_name, agent_info in agent_kb.items():
            score = 0
            reasoning_parts = []

            # Domain match
            if profile.domain in agent_info["domains"]:
                score += 30
                reasoning_parts.append(f"Excellent domain match for {profile.domain}")
            elif "general" in agent_info["domains"]:
                score += 15
                reasoning_parts.append("Good general-purpose capabilities")

            # Complexity match
            if profile.complexity in agent_info["complexity_levels"]:
                score += 25
                reasoning_parts.append(
                    f"Well-suited for {profile.complexity} complexity"
                )
            elif len(agent_info["complexity_levels"]) > 0:
                complexity_diff = abs(
                    ["simple", "medium", "complex"].index(profile.complexity)
                    - ["simple", "medium", "complex"].index(
                        agent_info["complexity_levels"][0]
                    )
                )
                score += max(0, 15 - complexity_diff * 5)

            # Multimodal support
            multimodal_overlap = len(
                set(profile.multimodal_needs)
                & set(agent_info.get("multimodal_support", []))
            )
            if multimodal_overlap > 0:
                score += multimodal_overlap * 15
                reasoning_parts.append(
                    f"Supports {multimodal_overlap} required multimodal types"
                )

            # Scale support
            if profile.scale_requirements in agent_info.get("scale_support", []):
                score += 20
                reasoning_parts.append(
                    f"Supports {profile.scale_requirements} scale requirements"
                )

            # Performance alignment
            if profile.performance_needs == agent_info.get("performance_profile"):
                score += 10
                reasoning_parts.append(
                    f"Performance profile matches {profile.performance_needs} needs"
                )

            # Learning data boost
            agent_key = f"{profile.domain}_{agent_name}"
            effectiveness = self.learning_data.get("component_effectiveness", {}).get(
                agent_key, 0
            )
            score += effectiveness * 10

            if score > best_score:
                best_score = score
                best_agent = agent_name
                best_reasoning = ". ".join(reasoning_parts)

        if not best_agent:
            best_agent = "ProbabilisticAgent"  # fallback
            best_reasoning = "General-purpose agent suitable for most use cases"

        # Get agent info
        agent_info = agent_kb.get(best_agent, agent_kb["ProbabilisticAgent"])

        return ComponentRecommendation(
            component_type="agent",
            component_name=best_agent,
            confidence_score=min(best_score / 100, 1.0),
            reasoning=best_reasoning,
            configuration_suggestions=self._generate_agent_config(best_agent, profile),
            alternatives=self._get_agent_alternatives(best_agent, profile),
            benefits=agent_info.get("strengths", []),
        )

    def _recommend_supporting_components(
        self, profile: ProjectProfile, primary_agent: ComponentRecommendation
    ) -> List[ComponentRecommendation]:
        """Recommend supporting components for the primary agent."""
        components = []

        # Add uncertainty quantification for complex projects
        if profile.complexity in ["medium", "complex"]:
            uncertainty_reasoning = "Unlike standard agents that may hallucinate with 'confidence,' UCUP-based agents quantify their own epistemic uncertainty. This enables better 'human-in-the-loop' triggers when an agent is unsure."
            components.append(
                ComponentRecommendation(
                    component_type="analysis",
                    component_name="AdkUncertaintyAnalyzer",
                    confidence_score=0.95,
                    reasoning=uncertainty_reasoning,
                    configuration_suggestions={
                        "volatility_window_size": 20,
                        "adaptive_thresholds": True,
                        "alternative_predictions_count": 5,
                        "uncertainty_thresholds": {
                            "high": 0.8,
                            "medium": 0.6,
                            "low": 0.4,
                        },
                    },
                    alternatives=[
                        "BasicUncertaintyAnalyzer",
                        "ProbabilisticUncertaintyAnalyzer",
                    ],
                    benefits=[
                        "Epistemic uncertainty quantification",
                        "Human-in-the-loop triggers",
                        "Confidence calibration",
                        "Risk-aware decision making",
                    ],
                )
            )

        # Add hierarchical coordination for multi-agent systems
        if profile.scale_requirements in ["multi-agent", "distributed"]:
            hierarchy_reasoning = "Hierarchical Coordination excels in multi-agent systems by using a unified framework to synchronize tasks across different cognitive layers (from low-level tool calling to high-level reasoning)."
            components.append(
                ComponentRecommendation(
                    component_type="coordination",
                    component_name="HierarchicalCoordination",
                    confidence_score=0.9,
                    reasoning=hierarchy_reasoning,
                    configuration_suggestions={
                        "hierarchy_levels": 3,
                        "coordination_timeout": 30000,
                        "task_prioritization": "adaptive",
                        "cognitive_layer_mapping": {
                            "low_level": ["tool_calling", "data_processing"],
                            "mid_level": ["reasoning", "planning"],
                            "high_level": ["strategy", "decision_making"],
                        },
                    },
                    alternatives=["DebateCoordination", "MarketBasedCoordination"],
                    benefits=[
                        "Multi-layer cognitive synchronization",
                        "Unified task coordination",
                        "Adaptive task prioritization",
                        "Cross-layer communication",
                    ],
                )
            )

        # Add multimodal fusion if multimodal needs exist
        if len(profile.multimodal_needs) > 1:
            components.append(
                ComponentRecommendation(
                    component_type="fusion",
                    component_name="AdaptiveFusionEngine",
                    confidence_score=0.85,
                    reasoning=f"Required for fusing {len(profile.multimodal_needs)} multimodal inputs",
                    configuration_suggestions={
                        "modalities": profile.multimodal_needs,
                        "fusion_strategy": "weighted_average",
                        "learning_rate": 0.01,
                    },
                    alternatives=["SimpleFusionEngine", "AttentionFusionEngine"],
                    benefits=[
                        "Multimodal integration",
                        "Adaptive weighting",
                        "Improved accuracy",
                    ],
                )
            )

        # Add developer tooling for complex projects
        if profile.complexity == "complex" or profile.reliability_requirements in [
            "high",
            "mission-critical",
        ]:
            tooling_reasoning = "The availability of a dedicated UCUP VS Code extension makes it easier to visualize agent hierarchies and debug the 'uncertainty phase boundaries' during a run."
            components.append(
                ComponentRecommendation(
                    component_type="tooling",
                    component_name="UCUP_VSCode_Extension",
                    confidence_score=0.9,
                    reasoning=tooling_reasoning,
                    configuration_suggestions={
                        "visualization_enabled": True,
                        "debug_mode": "advanced",
                        "uncertainty_boundaries_tracking": True,
                        "hierarchy_visualization": "interactive",
                    },
                    alternatives=["BasicDebugger", "AgentInspector"],
                    benefits=[
                        "Agent hierarchy visualization",
                        "Uncertainty phase boundary debugging",
                        "Interactive debugging tools",
                        "Performance monitoring dashboard",
                    ],
                )
            )

        # Add monitoring for production deployments
        if profile.reliability_requirements in ["high", "mission-critical"]:
            components.append(
                ComponentRecommendation(
                    component_type="monitoring",
                    component_name="AgentMonitor",
                    confidence_score=0.95,
                    reasoning="Critical for maintaining reliability in production systems",
                    configuration_suggestions={
                        "health_check_interval": 30,
                        "performance_tracking": True,
                        "alert_thresholds": {"confidence": 0.7, "latency": 1000},
                    },
                    alternatives=["BasicMonitor", "EnterpriseMonitor"],
                    benefits=[
                        "Health monitoring",
                        "Performance tracking",
                        "Automatic alerts",
                    ],
                )
            )

        return components

    def _recommend_coordination_strategy(
        self, profile: ProjectProfile
    ) -> ComponentRecommendation:
        """Recommend coordination strategy based on project needs."""
        coord_kb = self.component_knowledge_base["coordinators"]

        if profile.scale_requirements == "distributed":
            coord_name = "SwarmCoordination"
            reasoning = "Best for distributed systems with emergent behavior"
        elif profile.scale_requirements == "multi-agent":
            if profile.complexity == "complex":
                coord_name = "DebateCoordination"
                reasoning = "Excellent for complex multi-agent decision making"
            else:
                coord_name = "HierarchicalCoordination"
                reasoning = (
                    "Efficient for multi-agent coordination with clear hierarchy"
                )
        else:
            coord_name = "BasicCoordination"
            reasoning = "Simple coordination for single-agent systems"

        coord_info = coord_kb.get(
            coord_name, coord_kb.get("HierarchicalCoordination", {})
        )

        return ComponentRecommendation(
            component_type="coordinator",
            component_name=coord_name,
            confidence_score=0.8,
            reasoning=reasoning,
            configuration_suggestions={
                "max_participants": 10,
                "coordination_timeout": 30000,
                "consensus_threshold": 0.7,
            },
            alternatives=list(coord_kb.keys()),
            benefits=coord_info.get("use_cases", []),
        )

    def _recommend_deployment_config(self, profile: ProjectProfile) -> Dict[str, Any]:
        """Recommend deployment configuration."""
        deployment_kb = self.component_knowledge_base["deployment_targets"]

        # Default to local for development
        if profile.complexity == "simple":
            target = "local"
        elif profile.reliability_requirements == "mission-critical":
            target = "kubernetes"
        elif profile.scale_requirements == "distributed":
            target = "kubernetes"
        else:
            target = "docker"

        target_info = deployment_kb.get(target, deployment_kb["local"])

        return {
            "target": target,
            "performance_profile": target_info["performance"],
            "scalability": target_info["scalability"],
            "estimated_cost": target_info["cost"],
            "configuration": {
                "replicas": 1 if profile.scale_requirements == "single" else 3,
                "resource_limits": {
                    "cpu": "500m" if profile.complexity == "simple" else "1000m",
                    "memory": "512Mi" if profile.complexity == "simple" else "1Gi",
                },
            },
        }

    def _calculate_estimated_benefits(
        self,
        profile: ProjectProfile,
        primary_agent: ComponentRecommendation,
        supporting_components: List[ComponentRecommendation],
    ) -> Dict[str, float]:
        """Calculate estimated benefits of the recommended architecture."""
        benefits = {
            "development_speed": 0.0,
            "runtime_performance": 0.0,
            "reliability": 0.0,
            "maintainability": 0.0,
            "cost_savings": 0.0,
        }

        # Base benefits from primary agent
        agent_benefits = {
            "ProbabilisticAgent": {
                "development_speed": 0.8,
                "runtime_performance": 0.7,
                "reliability": 0.8,
                "maintainability": 0.8,
                "cost_savings": 0.6,
            },
            "BayesianAgent": {
                "development_speed": 0.6,
                "runtime_performance": 0.8,
                "reliability": 0.9,
                "maintainability": 0.7,
                "cost_savings": 0.7,
            },
            "VisionLanguageAgent": {
                "development_speed": 0.7,
                "runtime_performance": 0.6,
                "reliability": 0.7,
                "maintainability": 0.6,
                "cost_savings": 0.8,
            },
        }

        agent_name = primary_agent.component_name
        if agent_name in agent_benefits:
            for benefit, value in agent_benefits[agent_name].items():
                benefits[benefit] += value

        # Benefits from complexity alignment
        complexity_multiplier = {"simple": 1.2, "medium": 1.0, "complex": 0.8}
        for benefit in benefits:
            benefits[benefit] *= complexity_multiplier.get(profile.complexity, 1.0)

        # Benefits from supporting components
        for component in supporting_components:
            if component.component_name == "AdkUncertaintyAnalyzer":
                benefits["reliability"] += 0.2
                benefits["runtime_performance"] += 0.1
            elif "Fusion" in component.component_name:
                benefits["runtime_performance"] += 0.15
                benefits["development_speed"] += 0.1
            elif "Monitor" in component.component_name:
                benefits["reliability"] += 0.3
                benefits["maintainability"] += 0.2

        # Normalize benefits
        for benefit in benefits:
            benefits[benefit] = min(benefits[benefit], 1.0)

        return benefits

    def _calculate_recommendation_confidence(
        self,
        profile: ProjectProfile,
        primary_agent: ComponentRecommendation,
        supporting_components: List[ComponentRecommendation],
    ) -> float:
        """Calculate overall confidence score for recommendations."""
        base_confidence = primary_agent.confidence_score

        # Boost confidence based on supporting components
        component_boost = len(supporting_components) * 0.05
        base_confidence += component_boost

        # Boost confidence based on learning data matches
        profile_key = (
            f"{profile.domain}_{profile.complexity}_{profile.scale_requirements}"
        )
        if profile_key in self.learning_data.get("project_patterns", {}):
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _generate_agent_config(
        self, agent_name: str, profile: ProjectProfile
    ) -> Dict[str, Any]:
        """Generate configuration suggestions for an agent."""
        base_configs = {
            "ProbabilisticAgent": {
                "reasoning_strategies": ["chain_of_thought"],
                "exploration_budget": 0.2,
                "min_confidence_threshold": 0.7,
            },
            "BayesianAgent": {
                "network_type": "dynamic",
                "inference_method": "junction_tree",
                "evidence_integration": "incremental",
            },
            "VisionLanguageAgent": {
                "vision_encoder": "clip",
                "language_model": "gpt-3.5-turbo",
                "fusion_method": "cross_attention",
            },
        }

        config = base_configs.get(agent_name, {})

        # Adjust based on profile
        if profile.complexity == "complex":
            config["exploration_budget"] = 0.4
        if profile.performance_needs == "low-latency":
            config["min_confidence_threshold"] = 0.8

        return config

    def _get_agent_alternatives(
        self, agent_name: str, profile: ProjectProfile
    ) -> List[str]:
        """Get alternative agents for a given profile."""
        alternatives = []

        if agent_name == "ProbabilisticAgent":
            alternatives = ["BayesianAgent", "MDPAgent"]
        elif agent_name == "BayesianAgent":
            alternatives = ["ProbabilisticAgent", "StructuredDataAgent"]
        elif agent_name == "VisionLanguageAgent":
            alternatives = ["ProbabilisticAgent", "AudioAnalysisAgent"]

        return alternatives

    def save_build_success(
        self,
        project_profile: ProjectProfile,
        recommendations: ArchitectureRecommendation,
        build_metrics: Dict[str, Any],
    ):
        """
        Save successful build information for learning.
        """
        build_record = {
            "timestamp": datetime.now().isoformat(),
            "project_profile": asdict(project_profile),
            "recommendations": asdict(recommendations),
            "build_metrics": build_metrics,
            "success_score": self._calculate_success_score(build_metrics),
        }

        # Add to learning data
        if "successful_builds" not in self.learning_data:
            self.learning_data["successful_builds"] = []

        self.learning_data["successful_builds"].append(build_record)

        # Update project patterns
        profile_key = f"{project_profile.domain}_{project_profile.complexity}_{project_profile.scale_requirements}"
        if profile_key not in self.learning_data["project_patterns"]:
            self.learning_data["project_patterns"][profile_key] = []

        self.learning_data["project_patterns"][profile_key].append(
            {
                "primary_agent": recommendations.primary_agent.component_name,
                "coordinator": recommendations.coordination_strategy.component_name,
                "success_score": build_record["success_score"],
                "timestamp": build_record["timestamp"],
            }
        )

        # Update component effectiveness
        agent_key = (
            f"{project_profile.domain}_{recommendations.primary_agent.component_name}"
        )
        if agent_key not in self.learning_data["component_effectiveness"]:
            self.learning_data["component_effectiveness"][agent_key] = 0

        self.learning_data["component_effectiveness"][agent_key] += build_record[
            "success_score"
        ]

        # Save learning data
        self._save_learning_data()

    def _calculate_success_score(self, build_metrics: Dict[str, Any]) -> float:
        """Calculate success score from build metrics."""
        score = 0.5  # base score

        # Performance metrics
        if build_metrics.get("test_pass_rate", 0) > 0.8:
            score += 0.2
        if build_metrics.get("performance_score", 0) > 0.7:
            score += 0.15
        if build_metrics.get("reliability_score", 0) > 0.8:
            score += 0.15

        return min(score, 1.0)

    def _save_learning_data(self):
        """Save learning data to disk."""
        try:
            os.makedirs(os.path.dirname(self.learning_data_path), exist_ok=True)
            with open(self.learning_data_path, "w") as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save learning data: {e}")

    def get_project_insights(self, project_path: str) -> Dict[str, Any]:
        """Get insights about a specific project based on learning data."""
        insights = {
            "similar_projects": [],
            "recommended_improvements": [],
            "performance_predictions": {},
            "risk_assessment": {},
        }

        # Analyze project files to understand current setup
        project_files = self._scan_project_files(project_path)
        current_setup = self._analyze_current_setup(project_files)

        # Find similar successful projects
        for build in self.learning_data.get("successful_builds", []):
            similarity_score = self._calculate_project_similarity(current_setup, build)
            if similarity_score > 0.7:
                insights["similar_projects"].append(
                    {
                        "project": build["project_profile"],
                        "similarity_score": similarity_score,
                        "success_score": build.get("success_score", 0.5),
                    }
                )

        # Generate improvement recommendations
        insights["recommended_improvements"] = self._generate_improvement_suggestions(
            current_setup
        )

        return insights

    def _scan_project_files(self, project_path: str) -> List[str]:
        """Scan project files for analysis."""
        files = []
        try:
            for root, dirs, filenames in os.walk(project_path):
                for filename in filenames:
                    if filename.endswith((".py", ".yaml", ".yml", ".json")):
                        filepath = os.path.join(root, filename)
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                files.append(f.read())
                        except:
                            pass
        except:
            pass
        return files

    def _analyze_current_setup(self, project_files: List[str]) -> Dict[str, Any]:
        """Analyze current project setup."""
        setup = {
            "agents_used": [],
            "coordinators_used": [],
            "multimodal_components": [],
            "monitoring_components": [],
            "complexity_indicators": [],
        }

        for file_content in project_files:
            content_lower = file_content.lower()

            # Check for agents
            agent_types = [
                "ProbabilisticAgent",
                "BayesianAgent",
                "MDPAgent",
                "MCTSReasoner",
                "VisionLanguageAgent",
            ]
            for agent_type in agent_types:
                if agent_type.lower() in content_lower:
                    setup["agents_used"].append(agent_type)

            # Check for coordinators
            coord_types = [
                "HierarchicalCoordination",
                "DebateCoordination",
                "MarketBasedCoordination",
            ]
            for coord_type in coord_types:
                if coord_type.lower() in content_lower:
                    setup["coordinators_used"].append(coord_type)

            # Check for multimodal components
            if "fusion" in content_lower or "multimodal" in content_lower:
                setup["multimodal_components"].append("fusion_engine")

            # Check for monitoring
            if "monitor" in content_lower or "health" in content_lower:
                setup["monitoring_components"].append("monitoring")

        return setup

    def _calculate_project_similarity(
        self, current_setup: Dict[str, Any], build_record: Dict[str, Any]
    ) -> float:
        """Calculate similarity between current setup and successful build."""
        similarity = 0.0

        # Agent similarity
        current_agents = set(current_setup.get("agents_used", []))
        build_agents = set(
            build_record.get("recommendations", {})
            .get("primary_agent", {})
            .get("component_name", "")
        )
        if build_agents:
            agent_overlap = (
                len(current_agents & build_agents) / len(current_agents | build_agents)
                if current_agents or build_agents
                else 0
            )
            similarity += agent_overlap * 0.4

        # Coordinator similarity
        current_coords = set(current_setup.get("coordinators_used", []))
        build_coords = set(
            build_record.get("recommendations", {})
            .get("coordination_strategy", {})
            .get("component_name", "")
        )
        if build_coords:
            coord_overlap = (
                len(current_coords & build_coords) / len(current_coords | build_coords)
                if current_coords or build_coords
                else 0
            )
            similarity += coord_overlap * 0.3

        # Component similarity
        current_components = set(
            current_setup.get("multimodal_components", [])
            + current_setup.get("monitoring_components", [])
        )
        build_components = []
        for comp in build_record.get("recommendations", {}).get(
            "supporting_components", []
        ):
            build_components.append(comp.get("component_name", ""))

        component_overlap = (
            len(current_components & set(build_components))
            / len(current_components | set(build_components))
            if current_components or build_components
            else 0
        )
        similarity += component_overlap * 0.3

        return similarity

    def _generate_improvement_suggestions(
        self, current_setup: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement suggestions based on current setup."""
        suggestions = []

        # Check for missing components
        if not current_setup.get("monitoring_components"):
            suggestions.append("Add monitoring component for production reliability")

        if (
            not current_setup.get("multimodal_components")
            and len(current_setup.get("agents_used", [])) > 0
        ):
            agent_types = current_setup.get("agents_used", [])
            multimodal_agents = ["VisionLanguageAgent", "AudioAnalysisAgent"]
            if any(agent in multimodal_agents for agent in agent_types):
                suggestions.append(
                    "Consider adding multimodal fusion engine for better integration"
                )

        if (
            not current_setup.get("coordinators_used")
            and len(current_setup.get("agents_used", [])) > 1
        ):
            suggestions.append("Add coordination strategy for multi-agent systems")

        # Performance suggestions based on learning data
        if current_setup.get("agents_used"):
            agent_type = current_setup["agents_used"][0]
            effectiveness = self.learning_data.get("component_effectiveness", {}).get(
                agent_type, 0
            )
            if effectiveness < 0.7:
                suggestions.append(
                    f"Consider alternative agents - current choice has {effectiveness:.1f} effectiveness score"
                )

        return suggestions


# CLI Integration
async def analyze_project_smart(
    project_description: str, output_format: str = "json"
) -> str:
    """
    Smart project analysis and recommendations via CLI.
    """
    engine = SmartRecommendationEngine()

    # Analyze project requirements
    profile = engine.analyze_project_requirements(project_description)

    # Generate recommendations
    recommendations = engine.generate_recommendations(profile)

    # Format output
    if output_format == "json":
        result = {
            "project_profile": asdict(profile),
            "recommendations": asdict(recommendations),
        }
        return json.dumps(result, indent=2)
    else:
        # Text format
        output = f"""
UCUP Smart Project Analysis
===========================

Project Profile:
- Type: {profile.project_type}
- Domain: {profile.domain}
- Complexity: {profile.complexity}
- Scale: {profile.scale_requirements}
- Performance: {profile.performance_needs}
- Reliability: {profile.reliability_requirements}
- Multimodal: {', '.join(profile.multimodal_needs) if profile.multimodal_needs else 'None'}

Recommended Architecture:
========================

Primary Agent: {recommendations.primary_agent.component_name}
Confidence: {recommendations.primary_agent.confidence_score:.1%}
Reasoning: {recommendations.primary_agent.reasoning}

Supporting Components:
{chr(10).join(f"- {comp.component_name}: {comp.reasoning}" for comp in recommendations.supporting_components)}

Coordination: {recommendations.coordination_strategy.component_name}
Reasoning: {recommendations.coordination_strategy.reasoning}

Deployment: {recommendations.deployment_config['target']}
Performance Profile: {recommendations.deployment_config['performance_profile']}

Estimated Benefits:
{chr(10).join(f"- {benefit.replace('_', ' ').title()}: {value:.1%}" for benefit, value in recommendations.estimated_benefits.items())}

Overall Confidence: {recommendations.confidence_score:.1%}
"""
        return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        project_desc = sys.argv[1]
        output_format = sys.argv[2] if len(sys.argv) > 2 else "text"
        result = asyncio.run(analyze_project_smart(project_desc, output_format))
        print(result)
