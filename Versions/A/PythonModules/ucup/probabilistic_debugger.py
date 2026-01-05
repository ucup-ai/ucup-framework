"""
Probabilistic Debugging for UCUP Framework.

This module provides specialized debugging tools for probabilistic AI agents,
focusing on uncertainty quantification, confidence analysis, decision probability
distributions, and statistical debugging of agent behavior.

Key Features:
- Confidence score analysis and visualization
- Probability distribution debugging
- Uncertainty quantification metrics
- Statistical decision analysis
- Confidence threshold debugging
- Risk assessment and safety checks
"""

import asyncio
import json
import logging
import math
import statistics
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .observability import DecisionNode, DecisionTrace, DecisionTracer
from .validation import (
    UCUPValidationError,
    UCUPValueError,
    create_error_message,
    validate_non_empty_string,
    validate_probability,
    validate_types,
)


@dataclass
class ConfidenceProfile:
    """Statistical profile of confidence scores for an agent."""

    mean_confidence: float = 0.0
    confidence_std: float = 0.0
    confidence_range: Tuple[float, float] = (0.0, 0.0)
    confidence_trend: str = "stable"  # "increasing", "decreasing", "stable", "volatile"
    overconfidence_count: int = 0
    underconfidence_count: int = 0
    optimal_confidence_range: Tuple[float, float] = (0.7, 0.9)
    calibration_score: float = 0.0  # How well confidence matches accuracy


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty quantification metrics."""

    entropy: float = 0.0
    variance: float = 0.0
    coefficient_of_variation: float = 0.0
    uncertainty_stability: float = 0.0  # How stable uncertainty is over time
    decision_pressure: float = 0.0  # How much pressure from close confidence scores
    ambiguity_index: float = 0.0  # How ambiguous the decision space is


@dataclass
class ProbabilisticBreakpoint:
    """Breakpoint triggered by probabilistic conditions."""

    breakpoint_id: str
    condition_type: str  # "confidence_threshold", "uncertainty_spike", "decision_pressure"
    threshold: float
    description: str
    enabled: bool = True
    hit_count: int = 0
    last_hit: Optional[datetime] = None

    def check_confidence_threshold(self, confidence: float) -> bool:
        """Check if confidence threshold is met."""
        if self.condition_type == "confidence_threshold":
            return confidence < self.threshold
        return False

    def check_uncertainty_spike(self, uncertainty_metrics: UncertaintyMetrics) -> bool:
        """Check if uncertainty spike condition is met."""
        if self.condition_type == "uncertainty_spike":
            return uncertainty_metrics.entropy > self.threshold
        return False

    def check_decision_pressure(self, confidence_diff: float) -> bool:
        """Check if decision pressure condition is met."""
        if self.condition_type == "decision_pressure":
            return confidence_diff < self.threshold  # Small difference = high pressure
        return False


@dataclass
class ProbabilisticDebugSession:
    """Debug session focused on probabilistic analysis."""

    session_id: str
    trace: DecisionTrace
    confidence_profiles: Dict[str, ConfidenceProfile] = field(default_factory=dict)
    uncertainty_history: List[UncertaintyMetrics] = field(default_factory=list)
    probabilistic_breakpoints: List[ProbabilisticBreakpoint] = field(
        default_factory=list
    )
    risk_assessments: List[Dict[str, Any]] = field(default_factory=list)


class ProbabilisticDebugger:
    """
    Advanced probabilistic debugging system for UCUP agents.

    Features:
    - Confidence score analysis and statistical profiling
    - Uncertainty quantification and metrics
    - Probabilistic breakpoint system
    - Risk assessment and safety analysis
    - Decision probability distribution analysis
    """

    def __init__(self):
        self.sessions: Dict[str, ProbabilisticDebugSession] = {}
        self.logger = logging.getLogger(__name__)

    @validate_types
    def create_probabilistic_session(self, trace: DecisionTrace) -> str:
        """
        Create a probabilistic debugging session.

        Args:
            trace: DecisionTrace to analyze probabilistically

        Returns:
            Session ID for probabilistic debugging
        """
        try:
            validate_non_empty_string(trace.session_id, "trace.session_id")
        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="ProbabilisticDebugger Create Session",
                    action="Validating trace session ID",
                    details=str(e),
                    suggestion="Provide a valid DecisionTrace with a non-empty session ID",
                )
            ) from e

        session_id = f"prob_debug_{trace.session_id}"

        session = ProbabilisticDebugSession(session_id=session_id, trace=trace)

        # Analyze the trace probabilistically
        self._analyze_confidence_profiles(session)
        self._analyze_uncertainty_evolution(session)
        self._assess_risks(session)

        self.sessions[session_id] = session

        self.logger.info(f"Created probabilistic debug session: {session_id}")
        return session_id

    def _analyze_confidence_profiles(self, session: ProbabilisticDebugSession):
        """Analyze confidence profiles for all decisions in the trace."""
        confidence_scores = []

        for decision in session.trace.decisions:
            if decision.confidence_scores and decision.chosen_action:
                chosen_confidence = decision.confidence_scores.get(
                    decision.chosen_action, 0
                )
                confidence_scores.append(chosen_confidence)

                # Analyze confidence distribution for this decision
                all_confidences = list(decision.confidence_scores.values())
                if len(all_confidences) > 1:
                    profile = ConfidenceProfile(
                        mean_confidence=statistics.mean(all_confidences),
                        confidence_std=statistics.stdev(all_confidences)
                        if len(all_confidences) > 1
                        else 0,
                        confidence_range=(min(all_confidences), max(all_confidences)),
                    )

                    # Analyze over/under confidence
                    optimal_min, optimal_max = profile.optimal_confidence_range
                    for conf in all_confidences:
                        if conf > optimal_max:
                            profile.overconfidence_count += 1
                        elif conf < optimal_min:
                            profile.underconfidence_count += 1

                    # Calculate calibration score (simplified)
                    profile.calibration_score = self._calculate_calibration_score(
                        all_confidences
                    )

                    session.confidence_profiles[decision.decision_id] = profile

        # Analyze overall confidence trend
        if len(confidence_scores) > 3:
            trend = self._analyze_confidence_trend(confidence_scores)
            for profile in session.confidence_profiles.values():
                profile.confidence_trend = trend

    def _analyze_confidence_trend(self, confidence_scores: List[float]) -> str:
        """Analyze the trend of confidence scores over time."""
        if len(confidence_scores) < 3:
            return "insufficient_data"

        # Simple linear trend analysis
        n = len(confidence_scores)
        x = list(range(n))
        y = confidence_scores

        # Calculate slope using simple linear regression
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        # Calculate volatility (standard deviation of differences)
        diffs = [abs(y[i] - y[i - 1]) for i in range(1, len(y))]
        volatility = statistics.stdev(diffs) if len(diffs) > 1 else 0

        # Determine trend
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "volatile" if volatility > 0.1 else "stable"

    def _calculate_calibration_score(self, confidence_scores: List[float]) -> float:
        """Calculate how well confidence scores are calibrated."""
        if not confidence_scores:
            return 0.0

        # Simplified calibration: check if high confidence scores are actually likely to be correct
        # This is a placeholder - real calibration would need ground truth data
        sorted_scores = sorted(confidence_scores, reverse=True)
        if len(sorted_scores) >= 3:
            # Check if top 3 confidence scores are indeed the highest
            top_three_avg = sum(sorted_scores[:3]) / 3
            overall_avg = sum(sorted_scores) / len(sorted_scores)
            return min(1.0, max(0.0, (top_three_avg - overall_avg) / 0.3 + 0.5))
        return 0.5

    def _analyze_uncertainty_evolution(self, session: ProbabilisticDebugSession):
        """Analyze how uncertainty evolves throughout the decision trace."""
        for i, decision in enumerate(session.trace.decisions):
            if decision.confidence_scores:
                confidences = list(decision.confidence_scores.values())

                # Calculate entropy (measure of uncertainty)
                entropy = self._calculate_entropy(confidences)

                # Calculate variance
                variance = (
                    statistics.variance(confidences) if len(confidences) > 1 else 0
                )

                # Coefficient of variation
                mean_conf = statistics.mean(confidences)
                cv = statistics.stdev(confidences) / mean_conf if mean_conf > 0 else 0

                # Decision pressure (how close the top choices are)
                sorted_confidences = sorted(confidences, reverse=True)
                decision_pressure = (
                    sorted_confidences[0] - sorted_confidences[1]
                    if len(sorted_confidences) > 1
                    else 1.0
                )

                # Ambiguity index (how spread out the confidence distribution is)
                ambiguity_index = (
                    entropy / math.log(len(confidences)) if len(confidences) > 1 else 0
                )

                metrics = UncertaintyMetrics(
                    entropy=entropy,
                    variance=variance,
                    coefficient_of_variation=cv,
                    decision_pressure=decision_pressure,
                    ambiguity_index=ambiguity_index,
                )

                session.uncertainty_history.append(metrics)

        # Calculate uncertainty stability
        if len(session.uncertainty_history) > 1:
            entropies = [m.entropy for m in session.uncertainty_history]
            entropy_stability = (
                1.0 - (statistics.stdev(entropies) / statistics.mean(entropies))
                if statistics.mean(entropies) > 0
                else 1.0
            )

            for metrics in session.uncertainty_history:
                metrics.uncertainty_stability = entropy_stability

    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy of a probability distribution."""
        if not probabilities:
            return 0.0

        # Normalize to probabilities if they don't sum to 1
        total = sum(probabilities)
        if total == 0:
            return 0.0

        normalized = [p / total for p in probabilities]

        entropy = 0.0
        for p in normalized:
            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    def _assess_risks(self, session: ProbabilisticDebugSession):
        """Assess risks based on probabilistic analysis."""
        risks = []

        # Check for consistently low confidence
        avg_confidence = (
            statistics.mean(
                [
                    profile.mean_confidence
                    for profile in session.confidence_profiles.values()
                ]
            )
            if session.confidence_profiles
            else 0
        )

        if avg_confidence < 0.5:
            risks.append(
                {
                    "risk_type": "low_overall_confidence",
                    "severity": "high",
                    "description": f"Agent shows consistently low confidence (avg: {avg_confidence:.2f})",
                    "recommendation": "Review agent training data and decision-making logic",
                }
            )

        # Check for high uncertainty volatility
        if session.uncertainty_history:
            entropies = [m.entropy for m in session.uncertainty_history]
            entropy_std = statistics.stdev(entropies) if len(entropies) > 1 else 0

            if entropy_std > 0.5:
                risks.append(
                    {
                        "risk_type": "high_uncertainty_volatility",
                        "severity": "medium",
                        "description": f"Uncertainty fluctuates significantly (std: {entropy_std:.2f})",
                        "recommendation": "Consider stabilizing decision-making criteria",
                    }
                )

        # Check for decision pressure issues
        high_pressure_decisions = sum(
            1
            for m in session.uncertainty_history
            if m.decision_pressure < 0.1  # Very close confidence scores
        )

        if high_pressure_decisions > len(session.uncertainty_history) * 0.3:
            risks.append(
                {
                    "risk_type": "frequent_close_decisions",
                    "severity": "medium",
                    "description": f"{high_pressure_decisions} decisions had very close confidence scores",
                    "recommendation": "Consider adding more distinguishing features to decision criteria",
                }
            )

        session.risk_assessments = risks

    @validate_types
    def add_probabilistic_breakpoint(
        self, session_id: str, condition_type: str, threshold: float, description: str
    ) -> str:
        """
        Add a probabilistic breakpoint.

        Args:
            session_id: Debug session identifier
            condition_type: Type of condition ("confidence_threshold", "uncertainty_spike", "decision_pressure")
            threshold: Threshold value for the condition
            description: Human-readable description

        Returns:
            Breakpoint ID
        """
        try:
            validate_non_empty_string(session_id, "session_id")
            validate_non_empty_string(condition_type, "condition_type")
            validate_non_empty_string(description, "description")

            if condition_type not in [
                "confidence_threshold",
                "uncertainty_spike",
                "decision_pressure",
            ]:
                raise UCUPValueError(f"Invalid condition type: {condition_type}")

        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="ProbabilisticDebugger Add Breakpoint",
                    action="Validating breakpoint parameters",
                    details=str(e),
                    suggestion="Provide valid session ID, condition type, and parameters",
                )
            ) from e

        if session_id not in self.sessions:
            raise UCUPValueError(f"Debug session '{session_id}' not found")

        breakpoint_id = (
            f"prob_bp_{len(self.sessions[session_id].probabilistic_breakpoints)}"
        )
        breakpoint = ProbabilisticBreakpoint(
            breakpoint_id=breakpoint_id,
            condition_type=condition_type,
            threshold=threshold,
            description=description,
        )

        self.sessions[session_id].probabilistic_breakpoints.append(breakpoint)

        self.logger.info(
            f"Added probabilistic breakpoint '{description}' to session {session_id}"
        )
        return breakpoint_id

    @validate_types
    def analyze_decision_probabilities(
        self, session_id: str, decision_step: int
    ) -> Dict[str, Any]:
        """
        Analyze probability distributions for a specific decision.

        Args:
            session_id: Debug session identifier
            decision_step: Step to analyze

        Returns:
            Detailed probability analysis
        """
        try:
            validate_non_empty_string(session_id, "session_id")
            validate_positive_number(decision_step, "decision_step")

            if not isinstance(decision_step, int):
                raise UCUPTypeError("decision_step must be an integer")

        except (UCUPValueError, UCUPTypeError) as e:
            return {
                "error": create_error_message(
                    context="ProbabilisticDebugger Decision Analysis",
                    action="Validating analysis parameters",
                    details=str(e),
                    suggestion="Provide valid session ID and decision step",
                )
            }

        if session_id not in self.sessions:
            return {"error": f"Debug session '{session_id}' not found"}

        session = self.sessions[session_id]

        if decision_step >= len(session.trace.decisions):
            return {"error": f"Decision step {decision_step} is out of range"}

        decision = session.trace.decisions[decision_step]
        profile = session.confidence_profiles.get(decision.decision_id)
        uncertainty = (
            session.uncertainty_history[decision_step]
            if decision_step < len(session.uncertainty_history)
            else None
        )

        # Analyze probability distribution
        confidences = list(decision.confidence_scores.values())
        actions = list(decision.confidence_scores.keys())

        # Statistical analysis
        stats = {}
        if confidences:
            stats = {
                "mean": statistics.mean(confidences),
                "median": statistics.median(confidences),
                "std_dev": statistics.stdev(confidences) if len(confidences) > 1 else 0,
                "min": min(confidences),
                "max": max(confidences),
                "range": max(confidences) - min(confidences),
            }

        # Decision quality assessment
        quality_assessment = self._assess_decision_quality(
            confidences, decision.chosen_action, decision.confidence_scores
        )

        return {
            "decision_step": decision_step,
            "chosen_action": decision.chosen_action,
            "confidence_distribution": decision.confidence_scores,
            "statistical_summary": stats,
            "uncertainty_metrics": uncertainty.__dict__ if uncertainty else None,
            "confidence_profile": profile.__dict__ if profile else None,
            "quality_assessment": quality_assessment,
            "risk_indicators": self._identify_risk_indicators(
                confidences, decision.chosen_action, decision.confidence_scores
            ),
        }

    def _assess_decision_quality(
        self,
        confidences: List[float],
        chosen_action: str,
        confidence_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assess the quality of a decision based on confidence distribution."""
        if not confidences or not chosen_action:
            return {"quality": "insufficient_data"}

        chosen_confidence = confidence_scores.get(chosen_action, 0)
        sorted_confidences = sorted(confidences, reverse=True)

        # Decision margin (difference between chosen and next best)
        margin = (
            chosen_confidence - sorted_confidences[1]
            if len(sorted_confidences) > 1
            else 1.0
        )

        # Confidence percentile
        percentile = sum(1 for c in confidences if c <= chosen_confidence) / len(
            confidences
        )

        # Quality assessment
        if chosen_confidence >= 0.8 and margin >= 0.2:
            quality = "excellent"
            confidence_level = "high"
        elif chosen_confidence >= 0.6 and margin >= 0.1:
            quality = "good"
            confidence_level = "medium"
        elif chosen_confidence >= 0.4:
            quality = "acceptable"
            confidence_level = "low"
        else:
            quality = "concerning"
            confidence_level = "very_low"

        return {
            "quality": quality,
            "confidence_level": confidence_level,
            "decision_margin": margin,
            "confidence_percentile": percentile,
            "recommendation": self._generate_quality_recommendation(
                quality, margin, chosen_confidence
            ),
        }

    def _generate_quality_recommendation(
        self, quality: str, margin: float, confidence: float
    ) -> str:
        """Generate recommendation based on decision quality."""
        if quality == "excellent":
            return "Decision appears well-founded with strong confidence and clear separation."
        elif quality == "good":
            return (
                "Decision is reasonable but could benefit from additional validation."
            )
        elif quality == "acceptable":
            return (
                "Decision is marginal - consider fallback options or additional data."
            )
        else:
            return "Decision quality is concerning - review decision criteria and consider alternative approaches."

    def _identify_risk_indicators(
        self,
        confidences: List[float],
        chosen_action: str,
        confidence_scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Identify potential risk indicators in the decision."""
        risks = []

        chosen_confidence = confidence_scores.get(chosen_action, 0)

        # Very low confidence
        if chosen_confidence < 0.3:
            risks.append(
                {
                    "type": "very_low_confidence",
                    "severity": "high",
                    "description": f"Chosen action has very low confidence ({chosen_confidence:.2f})",
                }
            )

        # Very close alternatives
        sorted_scores = sorted(
            confidence_scores.items(), key=lambda x: x[1], reverse=True
        )
        if len(sorted_scores) > 1:
            margin = sorted_scores[0][1] - sorted_scores[1][1]
            if margin < 0.05:  # Very close decision
                risks.append(
                    {
                        "type": "very_close_alternatives",
                        "severity": "medium",
                        "description": f"Decision margin is very small ({margin:.3f}) - alternatives are nearly equally likely",
                    }
                )

        # High uncertainty (high entropy)
        entropy = self._calculate_entropy(confidences)
        if entropy > 1.0:  # High uncertainty threshold
            risks.append(
                {
                    "type": "high_uncertainty",
                    "severity": "medium",
                    "description": f"Decision space shows high uncertainty (entropy: {entropy:.2f})",
                }
            )

        return risks

    @validate_types
    def generate_uncertainty_report(self, session_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive uncertainty analysis report.

        Args:
            session_id: Debug session identifier

        Returns:
            Complete uncertainty analysis report
        """
        try:
            validate_non_empty_string(session_id, "session_id")
        except UCUPValueError as e:
            return {
                "error": create_error_message(
                    context="ProbabilisticDebugger Uncertainty Report",
                    action="Validating session ID",
                    details=str(e),
                    suggestion="Provide a valid debug session ID",
                )
            }

        if session_id not in self.sessions:
            return {"error": f"Debug session '{session_id}' not found"}

        session = self.sessions[session_id]

        # Overall uncertainty statistics
        if session.uncertainty_history:
            entropies = [m.entropy for m in session.uncertainty_history]
            variances = [m.variance for m in session.uncertainty_history]
            pressures = [m.decision_pressure for m in session.uncertainty_history]

            overall_stats = {
                "average_entropy": statistics.mean(entropies),
                "entropy_range": (min(entropies), max(entropies)),
                "average_variance": statistics.mean(variances),
                "average_decision_pressure": statistics.mean(pressures),
                "high_pressure_decisions": sum(1 for p in pressures if p < 0.1),
            }
        else:
            overall_stats = {}

        # Confidence analysis
        confidence_stats = {}
        if session.confidence_profiles:
            all_profiles = list(session.confidence_profiles.values())
            confidence_stats = {
                "average_mean_confidence": statistics.mean(
                    [p.mean_confidence for p in all_profiles]
                ),
                "overconfidence_incidents": sum(
                    p.overconfidence_count for p in all_profiles
                ),
                "underconfidence_incidents": sum(
                    p.underconfidence_count for p in all_profiles
                ),
                "average_calibration_score": statistics.mean(
                    [p.calibration_score for p in all_profiles]
                ),
            }

        return {
            "session_id": session_id,
            "total_decisions": len(session.trace.decisions),
            "overall_uncertainty_stats": overall_stats,
            "confidence_analysis": confidence_stats,
            "risk_assessments": session.risk_assessments,
            "probabilistic_breakpoints": [
                {
                    "id": bp.breakpoint_id,
                    "type": bp.condition_type,
                    "threshold": bp.threshold,
                    "description": bp.description,
                    "hits": bp.hit_count,
                }
                for bp in session.probabilistic_breakpoints
            ],
            "recommendations": self._generate_uncertainty_recommendations(session),
        }

    def _generate_uncertainty_recommendations(
        self, session: ProbabilisticDebugSession
    ) -> List[str]:
        """Generate recommendations based on uncertainty analysis."""
        recommendations = []

        # Check overall confidence
        if session.confidence_profiles:
            avg_confidence = statistics.mean(
                [p.mean_confidence for p in session.confidence_profiles.values()]
            )
            if avg_confidence < 0.6:
                recommendations.append(
                    "Consider improving agent training data to increase overall confidence levels"
                )

        # Check decision pressure
        if session.uncertainty_history:
            avg_pressure = statistics.mean(
                [m.decision_pressure for m in session.uncertainty_history]
            )
            if avg_pressure < 0.2:
                recommendations.append(
                    "Many decisions are made under high pressure - consider adding more distinguishing features"
                )

        # Check risk assessments
        if session.risk_assessments:
            high_risks = [
                r for r in session.risk_assessments if r.get("severity") == "high"
            ]
            if high_risks:
                recommendations.append(
                    f"Address {len(high_risks)} high-severity risk(s) identified in the analysis"
                )

        if not recommendations:
            recommendations.append(
                "Agent uncertainty characteristics appear within normal ranges"
            )

        return recommendations
