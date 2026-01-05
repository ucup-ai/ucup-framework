# UCUP/src/ucup/__init__.py
"""
UCUP Framework - Unified Cognitive Uncertainty Processing.

Copyright (c) 2025 UCUP Framework Contributors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from .advanced_probabilistic import (
    AlphaZeroMCTS,
    BayesianAgentNetwork,
    BayesianNetwork,
    BayesianNode,
    ConditionalProbabilityTable,
    DeepQLearningMDP,
    MarkovDecisionProcess,
    MCTSNode,
    MCTSReasoner,
    MDPAction,
    MDPBasedCoordinator,
    MDPState,
    MDPTransition,
    MonteCarloTreeSearch,
    PUCTNode,
    QLearningMDP,
)

# Smart API Discovery & Auto-Import System
from .api_discovery import (
    APIDiscoveryEngine,
    ComponentMetadata,
    ExportMetadata,
    create_auto_import_map,
    discover_ucup_api,
    enhance_type_hints_for_module,
    generate_module_exports,
    get_components_by_category,
    get_discovery_engine,
    get_smart_import_suggestions,
    search_ucup_components,
)
from .cloud_deployment import (
    AWSDeploymentProvider,
    AzureDeploymentProvider,
    CloudConfig,
    CloudDeploymentManager,
    CloudDeploymentProvider,
    DeploymentResult,
    DeploymentSpec,
    GCPDeploymentProvider,
    create_cloud_config_from_env,
    create_cloud_deployment_manager,
    create_deployment_spec_from_config,
    deploy_to_cloud,
    destroy_cloud_deployment,
    get_cloud_status,
)
from .config import create_ucup_system, load_ucup_config
from .errors import ErrorHandler, ProbabilisticError, ValidationError, get_error_handler
from .feature_flags import (
    FeatureFlag,
    FeatureFlagManager,
    FeatureFlagState,
    get_feature_manager,
    is_feature_enabled,
    require_feature,
)
from .gemini_adapter import (
    GeminiAdapter,
    create_gemini_adapter,
)
from .memory_management import (
    CacheManager,
    MemoryMonitor,
    ProviderManager,
    cached_operation,
    force_memory_cleanup,
    get_cache_manager,
    get_memory_monitor,
    get_memory_report,
    get_provider_manager,
    monitored_operation,
)
from .metrics import (
    BenefitsDisplay,
    MetricsTracker,
    get_global_tracker,
    record_build_metrics,
    show_build_benefits,
    show_test_benefits,
)

# Import from multimodal submodules
from .multimodal.fusion_engine import (
    FusedAnalysis,
    MultimodalFusionEngine,
    MultimodalInputs,
    create_fusion_engine,
    fuse_multimodal,
)
from .multimodal.streaming_processor import (
    RealTimeStreamingProcessor,
    StreamChunk,
    StreamingAnalysis,
)
from .observability import (
    DecisionExplorer,
    DecisionTracer,
    DecisionVisualization,
    LiveAgentMonitor,
    ReasoningVisualizer,
)
from .agent_monitoring import (
    AgentMonitor,
    PerformanceValidator,
    BehaviorValidator,
    AgentHealth,
    ValidationSeverity,
    AgentMetrics,
    ValidationIssue,
    AgentHealthReport,
    get_global_monitor,
    record_agent_metrics,
    validate_agent,
    get_agent_health_report,
)
from .probabilistic import AlternativePath, ProbabilisticAgent, ProbabilisticResult
from .reliability import (
    AutomatedRecoveryPipeline,
    FailureDetector,
    StateCheckpointer,
)
from .coordination import (
    HierarchicalCoordination,
    DebateCoordination,
    MarketBasedCoordination,
    SwarmCoordination,
    AdaptiveOrchestrator,
    ContextAwareStrategySelector,
    SeamlessStrategyTransition,
)

# UCUP 4.0: Intelligent API Suggestions System
from .smart_suggestions import (
    SmartImportSuggester,
    get_component_help,
    get_smart_suggestions,
    suggest_related_components,
)
from .test_environments import (
    CondaEnvironmentManager,
    EnvironmentManager,
    TestEnvironment,
    TestEnvironmentManager,
    TestResult,
    TestRunner,
    TestSuite,
    VenvEnvironmentManager,
    create_default_test_environments,
    generate_test_template,
    run_ucup_tests,
    setup_ucup_test_environment,
)
from .testing import (
    AdversarialTestGenerator,
    AgentNetworkIntegrationTester,
    AgentTestSuite,
    APITestingHarness,
    BenchmarkIntegration,
    ComparativeModelTester,
    CustomerServiceContext,
    DynamicScenarioGenerator,
    ExpectedOutcome,
    IntelligentTestGenerator,
    ModelPrediction,
    PerformanceDegradationTester,
    ProbabilisticAssert,
    Scenario,
    ScenarioContext,
    ScenarioGenerationResult,
    TestRun,
    TestScenario,
    UniversalInput,
    UniversalModelInterface,
    UserSimulationTester,
)
from .toon.toon_formatter import (
    TokenMetrics,
    ToonConversionResult,
    ToonFormatter,
    TOONOptimizer,
    ToonSchema,
)
from .validation import ValidationReport, validate_data

# Plugin System
from .plugins import (
    PluginInterface,
    AIPlugin,
    AgentPlugin,
    ModelAdapterPlugin,
    ToolPlugin,
    WorkflowPlugin,
    PluginManager,
)

# MultimodalAgentTester is optional - import it separately if needed
# Note: testing/ dir contains additional testing utilities but is not a package
MultimodalAgentTester = None

__version__ = "4.0.1"

__all__ = [
    "load_ucup_config",
    "create_ucup_system",
    "ProbabilisticResult",
    "AlternativePath",
    "ProbabilisticAgent",
    "show_build_benefits",
    "show_test_benefits",
    "record_build_metrics",
    "get_global_tracker",
    "MetricsTracker",
    "BenefitsDisplay",
    "ProbabilisticError",
    "ValidationError",
    "ErrorHandler",
    "get_error_handler",
    "ValidationReport",
    "validate_data",
    "AgentTestSuite",
    "Scenario",
    "ExpectedOutcome",
    "TestRun",
    "ScenarioContext",
    "CustomerServiceContext",
    "AdversarialTestGenerator",
    "ProbabilisticAssert",
    "BenchmarkIntegration",
    "APITestingHarness",
    "DynamicScenarioGenerator",
    "AgentNetworkIntegrationTester",
    "PerformanceDegradationTester",
    "ComparativeModelTester",
    "UserSimulationTester",
    "MultimodalFusionEngine",
    "MultimodalInputs",
    "FusedAnalysis",
    "fuse_multimodal",
    "create_fusion_engine",
    "RealTimeStreamingProcessor",
    "StreamChunk",
    "StreamingAnalysis",
    "MultimodalAgentTester",
    "IntelligentTestGenerator",
    "TestScenario",
    "ScenarioGenerationResult",
    "UniversalInput",
    "ModelPrediction",
    "UniversalModelInterface",
    # Reliability and Recovery
    "FailureDetector",
    "AutomatedRecoveryPipeline",
    "StateCheckpointer",
    # TOON Token Optimization
    "BayesianNetwork",
    "BayesianNode",
    "ConditionalProbabilityTable",
    "MarkovDecisionProcess",
    "MDPState",
    "MDPAction",
    "MDPTransition",
    "MonteCarloTreeSearch",
    "MCTSNode",
    "BayesianAgentNetwork",
    "MDPBasedCoordinator",
    "MCTSReasoner",
    "PUCTNode",
    "AlphaZeroMCTS",
    "QLearningMDP",
    "DeepQLearningMDP",
    # TOON Token Optimization
    "ToonFormatter",
    "ToonSchema",
    "ToonConversionResult",
    "TokenMetrics",
    "TOONOptimizer",
    # Observability
    "DecisionTracer",
    "DecisionExplorer",
    "DecisionVisualization",
    "ReasoningVisualizer",
    "LiveAgentMonitor",
    # Memory Management
    "MemoryMonitor",
    "CacheManager",
    "ProviderManager",
    "get_memory_monitor",
    "get_cache_manager",
    "get_provider_manager",
    "force_memory_cleanup",
    "get_memory_report",
    "cached_operation",
    "monitored_operation",
    # Feature Flags
    "FeatureFlag",
    "FeatureFlagState",
    "FeatureFlagManager",
    "get_feature_manager",
    "is_feature_enabled",
    "require_feature",
    # Gemini Adapter
    "GeminiAdapter",
    "create_gemini_adapter",
    # Cloud Deployment
    "CloudConfig",
    "DeploymentSpec",
    "DeploymentResult",
    "CloudDeploymentProvider",
    "AWSDeploymentProvider",
    "AzureDeploymentProvider",
    "GCPDeploymentProvider",
    "CloudDeploymentManager",
    "create_cloud_deployment_manager",
    "create_deployment_spec_from_config",
    "create_cloud_config_from_env",
    "deploy_to_cloud",
    "get_cloud_status",
    "destroy_cloud_deployment",
    # Test Environments
    "TestEnvironment",
    "TestResult",
    "TestSuite",
    "EnvironmentManager",
    "CondaEnvironmentManager",
    "VenvEnvironmentManager",
    "TestRunner",
    "TestEnvironmentManager",
    "create_default_test_environments",
    "setup_ucup_test_environment",
    "run_ucup_tests",
    "generate_test_template",
    # Agent Monitoring
    "AgentMonitor",
    "PerformanceValidator",
    "BehaviorValidator",
    "AgentHealth",
    "ValidationSeverity",
    "AgentMetrics",
    "ValidationIssue",
    "AgentHealthReport",
    "get_global_monitor",
    "record_agent_metrics",
    "validate_agent",
    "get_agent_health_report",
    # Coordination
    "HierarchicalCoordination",
    "DebateCoordination",
    "MarketBasedCoordination",
    "SwarmCoordination",
    "AdaptiveOrchestrator",
    "ContextAwareStrategySelector",
    "SeamlessStrategyTransition",
    # Smart API Discovery & Auto-Import System
    "APIDiscoveryEngine",
    "ComponentMetadata",
    "ExportMetadata",
    "discover_ucup_api",
    "get_smart_import_suggestions",
    "search_ucup_components",
    "get_components_by_category",
    "generate_module_exports",
    "create_auto_import_map",
    "enhance_type_hints_for_module",
    "get_discovery_engine",
    # Plugin System
    "PluginInterface",
    "AIPlugin",
    "AgentPlugin",
    "ModelAdapterPlugin",
    "ToolPlugin",
    "WorkflowPlugin",
    "PluginManager",
]
