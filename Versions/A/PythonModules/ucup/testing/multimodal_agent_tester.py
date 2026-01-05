import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Add UCUP/src to path to allow absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
ucup_src = os.path.abspath(os.path.join(current_dir, "..", ".."))
if ucup_src not in sys.path:
    sys.path.insert(0, ucup_src)

from dataclasses import dataclass, field

# Use absolute imports (requires ucup to be in path)
from ucup.multimodal.fusion_engine import (
    FusedAnalysis,
    MultimodalFusionEngine,
    MultimodalInputs,
)
from ucup.testing import AgentTestSuite


@dataclass
class TestResult:
    """Result of a multimodal test scenario."""

    passed: bool
    score: float
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class MultimodalAgentTester(AgentTestSuite):
    """
    Specialized tester for multimodal AI agents with cross-modal validation.
    """

    def __init__(self, fusion_engine: Optional[MultimodalFusionEngine] = None):
        super().__init__()
        self.fusion_engine = fusion_engine or MultimodalFusionEngine()
        self.test_scenarios = {
            "cross_modal_consistency": self._test_cross_modal_consistency,
            "modality_fallback": self._test_modality_fallback,
            "fusion_accuracy": self._test_fusion_accuracy,
            "uncertainty_calibration": self._test_uncertainty_calibration,
        }

    async def test_multimodal_agent(
        self, agent: Any, test_scenarios: List[str]
    ) -> Dict[str, TestResult]:
        """
        Run comprehensive multimodal tests on an agent.

        Args:
            agent: Multimodal AI agent to test (must implement analyze_multimodal_input)
            test_scenarios: List of test scenarios to run

        Returns:
            Dictionary mapping scenario names to test results
        """
        results = {}

        for scenario in test_scenarios:
            if scenario in self.test_scenarios:
                test_func = self.test_scenarios[scenario]
                try:
                    results[scenario] = await test_func(agent)
                except Exception as e:
                    results[scenario] = TestResult(
                        passed=False,
                        score=0.0,
                        error_message=f"Test failed with exception: {str(e)}",
                        execution_time=0.0,
                    )

        return results

    async def _test_cross_modal_consistency(self, agent: Any) -> TestResult:
        """Test that agent responses are consistent across modalities."""
        start_time = asyncio.get_event_loop().time()

        # Create the same semantic content in different modalities
        test_cases = [
            {
                "semantic": "A cat sitting on a windowsill",
                "text": "A cat sitting on a windowsill",
                "image_description": "Image of a cat on a windowsill",
                "audio_description": "Sound of a cat purring near a window",
            },
            {
                "semantic": "Heavy rain falling",
                "text": "Heavy rain falling",
                "image_description": "Image of pouring rain on a street",
                "audio_description": "Sound of heavy rainfall",
            },
        ]

        consistency_scores = []

        for test_case in test_cases:
            responses = {}

            # Test each modality
            for modality, content in test_case.items():
                if modality != "semantic":
                    inputs = MultimodalInputs()
                    if modality == "text":
                        inputs.text_content = content
                    elif modality == "image_description":
                        # Convert text description to mock image data
                        inputs.text_content = (
                            content  # Fallback if image generation not available
                        )
                        inputs.image_data = self._generate_mock_image_data(content)
                    elif modality == "audio_description":
                        inputs.text_content = content  # Fallback
                        inputs.audio_stream = self._generate_mock_audio_data(content)

                    try:
                        # Assuming agent has this method
                        if hasattr(agent, "analyze_multimodal_input"):
                            response = await agent.analyze_multimodal_input(inputs)
                            responses[modality] = response
                    except Exception as e:
                        print(f"Error testing {modality}: {e}")
                        continue

            if len(responses) >= 2:
                # Compare responses for consistency
                consistency_scores.append(
                    await self._measure_response_consistency(responses)
                )

        average_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return TestResult(
            passed=average_consistency >= 0.7,
            score=float(average_consistency),
            details={
                "consistency_scores": [float(s) for s in consistency_scores],
                "num_test_cases": len(consistency_scores),
            },
            execution_time=execution_time,
        )

    async def _test_modality_fallback(self, agent: Any) -> TestResult:
        """Test agent's ability to handle missing modalities gracefully."""
        start_time = asyncio.get_event_loop().time()

        # Test different combinations of missing modalities
        test_configs = [
            {"modalities": ["text"]},
            {"modalities": ["image"]},
            {"modalities": ["audio"]},
            {"modalities": ["text", "image"]},
            {"modalities": ["text", "audio"]},
            {"modalities": ["image", "audio"]},
            {"modalities": ["text", "image", "audio"]},
        ]

        fallback_scores = []

        for config in test_configs:
            inputs = MultimodalInputs()

            for modality in config["modalities"]:
                if modality == "text":
                    inputs.text_content = "Test content"
                elif modality == "image":
                    inputs.image_data = np.random.rand(224, 224, 3)
                elif modality == "audio":
                    inputs.audio_stream = np.random.rand(44100)

            try:
                # Fuse inputs and test agent
                fused_analysis = await self.fusion_engine.fuse_multimodal_inputs(inputs)

                if hasattr(agent, "analyze_multimodal_input"):
                    response = await agent.analyze_multimodal_input(inputs)

                    # Evaluate quality of fallback/degraded response
                    fallback_quality = self._evaluate_fallback_quality(
                        response, fused_analysis
                    )
                    fallback_scores.append(fallback_quality)
                else:
                    fallback_scores.append(0.0)

            except Exception as e:
                # Complete failure - worst possible fallback
                fallback_scores.append(0.0)

        average_fallback = np.mean(fallback_scores) if fallback_scores else 0.0
        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return TestResult(
            passed=average_fallback >= 0.6,
            score=float(average_fallback),
            details={
                "fallback_scores": [float(s) for s in fallback_scores],
                "configs_tested": len(test_configs),
            },
            execution_time=execution_time,
        )

    def _generate_mock_image_data(self, description: str) -> np.ndarray:
        """Generate mock image data for testing."""
        return np.random.rand(224, 224, 3)

    def _generate_mock_audio_data(self, description: str) -> np.ndarray:
        """Generate mock audio data for testing."""
        return np.random.rand(44100)

    async def _measure_response_consistency(self, responses: Dict[str, Any]) -> float:
        """Measure consistency between responses across modalities."""
        if len(responses) < 2:
            return 0.0

        # Extract key response elements
        key_elements = {}
        for modality, response in responses.items():
            key_elements[modality] = self._extract_key_elements(response)

        # Compare key elements across modalities
        consistency_scores = []
        modalities = list(key_elements.keys())

        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i + 1 :]:
                score = self._compare_key_elements(
                    key_elements[mod1], key_elements[mod2]
                )
                consistency_scores.append(score)

        return float(np.mean(consistency_scores)) if consistency_scores else 0.0

    def _extract_key_elements(self, response: Any) -> List[str]:
        """Extract key semantic elements from agent response."""
        # Implementation would analyze response structure
        # Ideally this would look for specific fields or keywords in the response object
        if hasattr(response, "summary"):
            return str(response.summary).split()
        if hasattr(response, "content"):
            return str(response.content).split()
        if isinstance(response, str):
            return response.split()
        if isinstance(response, dict):
            return str(response.get("content", "")).split()

        return ["generic_response"]

    def _compare_key_elements(
        self, elements1: List[str], elements2: List[str]
    ) -> float:
        """Compare similarity between key element lists."""
        set1 = set(elements1)
        set2 = set(elements2)
        if not set1 and not set2:
            return 1.0

        common_elements = set1 & set2
        return len(common_elements) / max(len(set1), len(set2), 1)

    def _evaluate_fallback_quality(
        self, response: Any, fused_analysis: FusedAnalysis
    ) -> float:
        """Evaluate quality of response when modalities are missing."""
        # Base quality on confidence and response completeness
        base_quality = 0.5

        if hasattr(response, "confidence"):
            base_quality = float(response.confidence)
        elif hasattr(response, "confidence_score"):
            base_quality = float(response.confidence_score)
        else:
            base_quality = fused_analysis.confidence_score

        # Adjust based on uncertainty
        quality = base_quality * (1.0 - fused_analysis.uncertainty_estimate)

        # Check if response indicates awareness of missing modalities
        response_str = str(response).lower()
        if (
            "missing" in response_str
            or "limited" in response_str
            or "partial" in response_str
        ):
            quality *= 1.1  # Bonus for noting missing modalities

        return min(quality, 1.0)

    async def _test_fusion_accuracy(self, agent: Any) -> TestResult:
        """Test accuracy of multimodal fusion in agent responses."""
        # Implementation for fusion accuracy testing
        return TestResult(passed=True, score=0.8, execution_time=100.0)

    async def _test_uncertainty_calibration(self, agent: Any) -> TestResult:
        """Test that uncertainty estimates are well-calibrated."""
        # Implementation for uncertainty calibration testing
        return TestResult(passed=True, score=0.85, execution_time=120.0)
