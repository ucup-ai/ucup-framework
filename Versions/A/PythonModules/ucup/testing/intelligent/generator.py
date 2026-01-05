import asyncio
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class TestScenario:
    """Represents a generated test scenario."""

    scenario_id: str
    description: str
    test_type: str
    modalities: List[str]
    complexity_level: int
    inputs: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    risk_level: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioGenerationResult:
    """Result of scenario generation process."""

    scenarios: List[TestScenario]
    generation_stats: Dict[str, Any]
    coverage_analysis: Dict[str, Any]
    quality_metrics: Dict[str, float]


class IntelligentTestGenerator:
    """
    AI-powered test scenario generator that creates diverse,
    realistic test cases using machine learning and rule-based approaches.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generation_models = {
            "llm_based": self._init_llm_generator(),
            "pattern_based": self._init_pattern_generator(),
            "mutation_based": self._init_mutation_generator(),
            "adversarial": self._init_adversarial_generator(),
        }

        # Scenario templates and patterns
        self.scenario_templates = self._load_scenario_templates()
        self.risk_patterns = self._load_risk_patterns()
        self.coverage_tracker = {}

    def _init_llm_generator(self):
        return "llm_generator_placeholder"

    def _init_pattern_generator(self):
        return "pattern_generator_placeholder"

    def _init_mutation_generator(self):
        return "mutation_generator_placeholder"

    def _init_adversarial_generator(self):
        return "adversarial_generator_placeholder"

    async def generate_diverse_scenarios(
        self,
        agent_behavior_spec: str,
        num_scenarios: int = 100,
        complexity_distribution: Dict[int, float] = None,
    ) -> ScenarioGenerationResult:
        """
        Generate diverse test scenarios based on agent behavior specification.

        Args:
            agent_behavior_spec: Description of agent capabilities and behavior
            num_scenarios: Number of scenarios to generate
            complexity_distribution: Distribution of complexity levels (1-5)

        Returns:
            ScenarioGenerationResult with generated scenarios and statistics
        """
        start_time = datetime.now()

        if complexity_distribution is None:
            complexity_distribution = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.15, 5: 0.05}

        # Phase 1: Generate base scenarios using multiple approaches
        raw_scenarios = await self._generate_raw_scenarios(
            agent_behavior_spec, num_scenarios, complexity_distribution
        )

        # Phase 2: Diversify and enrich scenarios
        diversified_scenarios = await self._diversify_scenarios(raw_scenarios)

        # Phase 3: Validate and filter scenarios
        validated_scenarios = await self._validate_scenarios(diversified_scenarios)

        # Phase 4: Analyze coverage and quality
        coverage_analysis = await self._analyze_scenario_coverage(validated_scenarios)
        quality_metrics = await self._assess_scenario_quality(validated_scenarios)

        generation_time = (datetime.now() - start_time).total_seconds()

        result = ScenarioGenerationResult(
            scenarios=validated_scenarios,
            generation_stats={
                "total_generated": len(raw_scenarios),
                "diversified": len(diversified_scenarios),
                "validated": len(validated_scenarios),
                "generation_time_seconds": generation_time,
                "scenarios_per_second": len(validated_scenarios)
                / max(generation_time, 1),
                "complexity_distribution": await self._analyze_complexity_distribution(
                    validated_scenarios
                ),
                "modality_distribution": await self._analyze_modality_distribution(
                    validated_scenarios
                ),
            },
            coverage_analysis=coverage_analysis,
            quality_metrics=quality_metrics,
        )

        return result

    async def _generate_raw_scenarios(
        self, behavior_spec: str, num_scenarios: int, complexity_dist: Dict[int, float]
    ) -> List[TestScenario]:
        """Generate raw scenarios using multiple generation approaches."""
        scenarios = []

        # Calculate target scenarios per complexity level
        targets = {}
        for complexity, proportion in complexity_dist.items():
            targets[complexity] = int(num_scenarios * proportion)

        # Generate scenarios using different models
        generation_tasks = []

        for complexity, target_count in targets.items():
            if target_count > 0:
                # Distribute across generation methods
                methods = ["llm_based", "pattern_based", "mutation_based"]
                if complexity >= 4:
                    methods.append("adversarial")

                scenarios_per_method = max(1, target_count // len(methods))

                for method in methods:
                    generation_tasks.append(
                        self._generate_with_method(
                            method, behavior_spec, complexity, scenarios_per_method
                        )
                    )

        # Execute generation tasks
        method_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        # Process results and flatten
        for result in method_results:
            if isinstance(result, Exception):
                print(f"Generation task failed: {result}")
            elif isinstance(result, list):
                scenarios.extend(result)

        return scenarios[:num_scenarios]  # Trim to requested number

    async def _generate_with_method(
        self, method: str, behavior_spec: str, complexity: int, count: int
    ) -> List[TestScenario]:
        """Generate scenarios using specific generation method."""

        if method == "llm_based":
            return await self._generate_llm_scenarios(behavior_spec, complexity, count)
        elif method == "pattern_based":
            return await self._generate_pattern_scenarios(
                behavior_spec, complexity, count
            )
        elif method == "mutation_based":
            return await self._generate_mutation_scenarios(
                behavior_spec, complexity, count
            )
        elif method == "adversarial":
            return await self._generate_adversarial_scenarios(
                behavior_spec, complexity, count
            )

        return []

    async def _generate_llm_scenarios(
        self, behavior_spec: str, complexity: int, count: int
    ) -> List[TestScenario]:
        """Generate scenarios using LLM for creative test case creation."""
        prompts = self._create_llm_prompts(behavior_spec, complexity, count)

        # Note: This is a placeholder - would integrate with actual LLM API
        scenarios = []
        for i in range(count):
            # Mock LLM generation
            llm_response = await self._mock_llm_generate(prompts[i % len(prompts)])

            scenario = self._parse_llm_response(llm_response, complexity)
            scenario.scenario_id = f"llm_{complexity}_{i}_{random.randint(1000, 9999)}"
            scenario.metadata["generation_method"] = "llm"
            scenarios.append(scenario)

        return scenarios

    async def _generate_pattern_scenarios(
        self, behavior_spec: str, complexity: int, count: int
    ) -> List[TestScenario]:
        """Generate scenarios using pattern matching and templates."""
        applicable_templates = self._find_applicable_templates(
            behavior_spec, complexity
        )

        scenarios = []
        if not applicable_templates:
            return scenarios

        for i in range(count):
            template = random.choice(applicable_templates)

            # Fill template with variations
            scenario_data = self._fill_template(
                template, behavior_spec, f"pattern_{complexity}_{i}"
            )
            scenarios.append(scenario_data)

        return scenarios

    async def _generate_mutation_scenarios(
        self, behavior_spec: str, complexity: int, count: int
    ) -> List[TestScenario]:
        """Generate scenarios through mutation of existing cases."""
        base_scenarios = await self._get_base_scenarios_for_mutation()
        mutation_operators = self._get_mutation_operators(complexity)

        if not base_scenarios:
            return []

        scenarios = []
        for i in range(count):
            base_scenario = random.choice(base_scenarios)
            mutated_scenario = await self._apply_mutations(
                base_scenario, mutation_operators
            )
            mutated_scenario.scenario_id = (
                f"mutant_{complexity}_{i}_{random.randint(1000, 9999)}"
            )
            mutated_scenario.metadata["generation_method"] = "mutation"
            mutated_scenario.metadata["parent_scenario"] = base_scenario.scenario_id
            scenarios.append(mutated_scenario)

        return scenarios

    async def _generate_adversarial_scenarios(
        self, behavior_spec: str, complexity: int, count: int
    ) -> List[TestScenario]:
        """Generate adversarial scenarios for edge case testing."""
        adversarial_patterns = self._get_adversarial_patterns()
        vulnerability_types = await self._analyze_behavior_vulnerabilities(
            behavior_spec
        )

        scenarios = []
        if not adversarial_patterns:
            return scenarios

        for i in range(count):
            vulnerability = (
                random.choice(vulnerability_types) if vulnerability_types else "general"
            )
            adversarial_pattern = random.choice(adversarial_patterns)

            scenario = await self._generate_adversarial_case(
                vulnerability, adversarial_pattern, complexity, f"adv_{complexity}_{i}"
            )
            scenarios.append(scenario)

        return scenarios

    async def _diversify_scenarios(
        self, scenarios: List[TestScenario]
    ) -> List[TestScenario]:
        """Apply diversification techniques to increase scenario variety."""
        diversified = []

        for scenario in scenarios:
            # Simple diversification: keep as is for now, could add noise or variations
            diversified.append(scenario)

        # Remove duplicates and near-duplicates
        diversified = await self._deduplicate_scenarios(diversified)

        return diversified

    async def _validate_scenarios(
        self, scenarios: List[TestScenario]
    ) -> List[TestScenario]:
        """Validate generated scenarios for quality and feasibility."""
        validated = []

        for scenario in scenarios:
            # Check basic validity
            is_valid = await self._check_scenario_validity(scenario)

            if is_valid:
                # Enrich with additional metadata
                scenario.metadata["validation_timestamp"] = datetime.now().isoformat()
                scenario.metadata[
                    "quality_score"
                ] = await self._calculate_scenario_quality(scenario)

                validated.append(scenario)

        return validated

    async def _analyze_scenario_coverage(
        self, scenarios: List[TestScenario]
    ) -> Dict[str, Any]:
        """Analyze test coverage across different dimensions."""
        coverage = {
            "modality_coverage": self._calculate_modality_coverage(scenarios),
            "complexity_coverage": self._calculate_complexity_coverage(scenarios),
            "risk_coverage": self._calculate_risk_coverage(scenarios),
            "edge_case_coverage": self._calculate_edge_case_coverage(scenarios),
            "gap_analysis": await self._identify_coverage_gaps(scenarios),
        }

        return coverage

    async def _assess_scenario_quality(
        self, scenarios: List[TestScenario]
    ) -> Dict[str, float]:
        """Assess overall quality of generated scenarios."""
        if not scenarios:
            return {"overall_quality": 0.0}

        quality_scores = [await self._calculate_scenario_quality(s) for s in scenarios]

        return {
            "average_quality": float(np.mean(quality_scores)),
            "median_quality": float(np.median(quality_scores)),
            "quality_variance": float(np.var(quality_scores)),
            "quality_distribution": {
                "high_quality": len([s for s in quality_scores if s >= 0.8]),
                "medium_quality": len([s for s in quality_scores if 0.6 <= s < 0.8]),
                "low_quality": len([s for s in quality_scores if s < 0.6]),
            },
        }

    async def _analyze_complexity_distribution(
        self, scenarios: List[TestScenario]
    ) -> Dict[int, float]:
        if not scenarios:
            return {}
        counts = {}
        for s in scenarios:
            counts[s.complexity_level] = counts.get(s.complexity_level, 0) + 1
        return {k: v / len(scenarios) for k, v in counts.items()}

    async def _analyze_modality_distribution(
        self, scenarios: List[TestScenario]
    ) -> Dict[str, float]:
        if not scenarios:
            return {}
        counts = {}
        total = 0
        for s in scenarios:
            for m in s.modalities:
                counts[m] = counts.get(m, 0) + 1
                total += 1
        return {k: v / max(1, len(scenarios)) for k, v in counts.items()}

    # Helper methods
    def _load_scenario_templates(self) -> Dict[str, Any]:
        """Load scenario templates from configuration."""
        return {
            "api_call": {
                "description": "Test {agent} behavior with {input_type} API call",
                "complexity": 2,
                "tags": ["api", "integration"],
            },
            "edge_case": {
                "description": "Test {agent} handling of {edge_condition} edge case",
                "complexity": 3,
                "tags": ["edge_case", "robustness"],
            },
        }

    def _load_risk_patterns(self) -> Dict[str, List[str]]:
        """Load risk assessment patterns."""
        return {
            "high_risk": ["security_violation", "data_breach", "system_crash"],
            "medium_risk": ["performance_degradation", "inaccurate_response"],
            "low_risk": ["cosmetic_issue", "minor_delay"],
        }

    async def _mock_llm_generate(self, prompt: str) -> str:
        """Mock LLM generation for testing."""
        # Placeholder for actual LLM integration
        # Deterministic generation based on prompt hash to be consistent
        return json.dumps(
            {
                "description": f"Generated scenario",
                "test_type": "functional",
                "modalities": ["text"],
                "complexity_level": 2,
                "inputs": {"content": "test input"},
                "expected_outcomes": {"result": "expected output"},
                "risk_level": "low",
                "tags": ["auto-generated"],
            }
        )

    def _create_llm_prompts(
        self, behavior_spec: str, complexity: int, count: int
    ) -> List[str]:
        """Create prompts for LLM generation."""
        prompts = []
        prompt_template = f"""
        Generate a test scenario for an AI agent with this behavior specification:
        {behavior_spec}

        Requirements:
        - Complexity level: {complexity}/5 (1=simple, 5=very complex)
        """
        for i in range(count):
            prompts.append(f"{prompt_template} Variation {i}")
        return prompts

    def _parse_llm_response(self, llm_response: str, complexity: int) -> TestScenario:
        """Parse LLM response into TestScenario object."""
        try:
            data = json.loads(llm_response)
            return TestScenario(
                scenario_id=f"llm_scenario_{complexity}",
                description=data.get("description", ""),
                test_type=data.get("test_type", "functional"),
                modalities=data.get("modalities", ["text"]),
                complexity_level=complexity,
                inputs=data.get("inputs", {}),
                expected_outcomes=data.get("expected_outcomes", {}),
                risk_level=data.get("risk_level", "low"),
                tags=data.get("tags", []),
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback
            return TestScenario(
                scenario_id=f"llm_scenario_{complexity}_fallback",
                description="Fallback scenario due to parsing error",
                test_type="functional",
                modalities=["text"],
                complexity_level=complexity,
                inputs={"content": "test content"},
                expected_outcomes={"result": "expected result"},
                risk_level="low",
                tags=["fallback"],
            )

    def _find_applicable_templates(
        self, behavior_spec: str, complexity: int
    ) -> List[Dict]:
        """Find scenario templates applicable to the behavior spec."""
        # Simplified logic
        return list(self.scenario_templates.values())

    def _fill_template(
        self, template: Dict, behavior_spec: str, id_prefix: str
    ) -> TestScenario:
        return TestScenario(
            scenario_id=f"{id_prefix}_{random.randint(1000,9999)}",
            description=template.get("description", "Templated scenario"),
            test_type="templated",
            modalities=["text"],
            complexity_level=template.get("complexity", 2),
            inputs={"type": "template_input"},
            expected_outcomes={"result": "success"},
            risk_level="low",
            tags=template.get("tags", []),
        )

    async def _get_base_scenarios_for_mutation(self) -> List[TestScenario]:
        """Return some base scenarios to mutate."""
        # In a real system, this would fetch from a database or existing test suite
        return [
            TestScenario(
                scenario_id="base_1",
                description="Base test scenario",
                test_type="functional",
                modalities=["text"],
                complexity_level=2,
                inputs={"text": "Hello world"},
                expected_outcomes={"response": "Hello"},
                risk_level="low",
            )
        ]

    def _get_mutation_operators(self, complexity: int) -> List[str]:
        return ["fuzz_input", "noise_injection", "field_deletion"]

    async def _apply_mutations(
        self, base: TestScenario, operators: List[str]
    ) -> TestScenario:
        # Clone and modify
        new_scenario = TestScenario(
            scenario_id=f"mutant_{base.scenario_id}",
            description=f"Mutated version of {base.description}",
            test_type=base.test_type,
            modalities=base.modalities,
            complexity_level=base.complexity_level,
            inputs=base.inputs.copy(),
            expected_outcomes=base.expected_outcomes.copy(),
            risk_level=base.risk_level,
            tags=base.tags + ["mutated"],
        )

        # Apply random mutation
        if "fuzz_input" in operators:
            for k in new_scenario.inputs:
                if isinstance(new_scenario.inputs[k], str):
                    new_scenario.inputs[k] += "_fuzzed"

        return new_scenario

    def _get_adversarial_patterns(self) -> List[str]:
        return ["injection", "denial_of_service", "confusion"]

    async def _analyze_behavior_vulnerabilities(self, spec: str) -> List[str]:
        return ["input_validation", "resource_exhaustion"]

    async def _generate_adversarial_case(
        self, vulnerability: str, pattern: str, complexity: int, id_val: str
    ) -> TestScenario:
        return TestScenario(
            scenario_id=id_val,
            description=f"Adversarial test for {vulnerability} using {pattern}",
            test_type="adversarial",
            modalities=["text"],
            complexity_level=complexity,
            inputs={"attack_vector": pattern},
            expected_outcomes={"secure": True},
            risk_level="high",
            tags=["adversarial", vulnerability],
        )

    async def _deduplicate_scenarios(
        self, scenarios: List[TestScenario]
    ) -> List[TestScenario]:
        # Simple deduplication by ID
        seen = set()
        unique = []
        for s in scenarios:
            if s.scenario_id not in seen:
                seen.add(s.scenario_id)
                unique.append(s)
        return unique

    async def _check_scenario_validity(self, scenario: TestScenario) -> bool:
        return True

    async def _calculate_scenario_quality(self, scenario: TestScenario) -> float:
        # Mock quality calculation
        score = 0.5
        if scenario.inputs:
            score += 0.2
        if scenario.expected_outcomes:
            score += 0.2
        if scenario.description:
            score += 0.1
        return min(score, 1.0)

    def _calculate_modality_coverage(self, scenarios: List[TestScenario]) -> float:
        return 0.8  # Mock

    def _calculate_complexity_coverage(self, scenarios: List[TestScenario]) -> float:
        return 0.7  # Mock

    def _calculate_risk_coverage(self, scenarios: List[TestScenario]) -> float:
        return 0.6  # Mock

    def _calculate_edge_case_coverage(self, scenarios: List[TestScenario]) -> float:
        return 0.5  # Mock

    async def _identify_coverage_gaps(self, scenarios: List[TestScenario]) -> List[str]:
        return ["audio_modality", "high_concurrency"]
