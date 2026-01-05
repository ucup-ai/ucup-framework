"""
Uncertainty Visualization Tools for UCUP Framework.

This module provides advanced visualization capabilities for uncertainty analysis,
confidence distributions, decision trees, and probabilistic debugging. It creates
interactive charts, graphs, and dashboards to help developers understand and
debug probabilistic agent behavior.

Key Features:
- Interactive confidence score visualizations
- Uncertainty distribution charts
- Decision tree visualization with confidence overlays
- Risk assessment dashboards
- Temporal uncertainty evolution graphs
- Comparative analysis between different agent runs
"""

import asyncio
import json
import logging
import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

    # Create dummy classes for type hints
    class go:
        @staticmethod
        def Figure():
            pass

        class graph_objects:
            pass

    class px:
        @staticmethod
        def scatter():
            pass

        @staticmethod
        def histogram():
            pass

        @staticmethod
        def box():
            pass

    pd = None

from .observability import DecisionNode, DecisionTrace, DecisionTracer
from .probabilistic_debugger import ConfidenceProfile, UncertaintyMetrics
from .validation import (
    UCUPValidationError,
    UCUPValueError,
    create_error_message,
    validate_non_empty_string,
    validate_types,
)


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""

    width: int = 800
    height: int = 600
    theme: str = "plotly_white"  # "plotly", "plotly_white", "plotly_dark"
    color_scheme: str = "viridis"  # Color scheme for plots
    show_grid: bool = True
    interactive: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["html", "png", "svg"])


@dataclass
class UncertaintyVisualization:
    """Container for uncertainty visualization data."""

    plot_data: Dict[str, Any]
    metadata: Dict[str, Any]
    recommendations: List[str]
    risk_indicators: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)


class UncertaintyVisualizer:
    """
    Advanced visualization system for uncertainty analysis and probabilistic debugging.

    Features:
    - Interactive confidence score distributions
    - Uncertainty evolution over time
    - Decision tree visualizations with confidence overlays
    - Risk assessment dashboards
    - Comparative analysis between agent runs
    - Statistical distribution plots
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)

        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available - visualizations will be limited")

    @validate_types
    def create_confidence_distribution_plot(
        self,
        confidence_scores: Dict[str, float],
        chosen_action: Optional[str] = None,
        title: str = "Confidence Score Distribution",
    ) -> UncertaintyVisualization:
        """
        Create a confidence distribution visualization.

        Args:
            confidence_scores: Dictionary mapping actions to confidence scores
            chosen_action: The action that was actually chosen
            title: Plot title

        Returns:
            Visualization data container
        """
        try:
            validate_non_empty_string(title, "title")
        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="UncertaintyVisualizer Confidence Distribution",
                    action="Validating title",
                    details=str(e),
                    suggestion="Provide a non-empty title for the visualization",
                )
            ) from e

        if not confidence_scores:
            return UncertaintyVisualization(
                plot_data={"error": "No confidence scores provided"},
                metadata={"type": "confidence_distribution", "status": "empty"},
                recommendations=["No confidence data available for visualization"],
                risk_indicators=[],
            )

        if not PLOTLY_AVAILABLE:
            return self._create_text_based_visualization(
                confidence_scores, chosen_action, title
            )

        # Create the plot
        actions = list(confidence_scores.keys())
        scores = list(confidence_scores.values())

        # Color coding: chosen action in different color
        colors = ["lightblue"] * len(actions)
        if chosen_action and chosen_action in actions:
            idx = actions.index(chosen_action)
            colors[idx] = "darkblue"

        fig = go.Figure()

        # Add bars
        fig.add_trace(
            go.Bar(x=actions, y=scores, marker_color=colors, name="Confidence Scores")
        )

        # Add threshold line
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="Decision Threshold (0.5)",
            annotation_position="bottom right",
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Actions",
            yaxis_title="Confidence Score",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            showlegend=False,
        )

        # Rotate x-axis labels if too many actions
        if len(actions) > 5:
            fig.update_xaxes(tickangle=45)

        # Generate recommendations
        recommendations = self._analyze_confidence_distribution(
            confidence_scores, chosen_action
        )

        # Identify risk indicators
        risk_indicators = self._identify_visualization_risks(
            confidence_scores, chosen_action
        )

        return UncertaintyVisualization(
            plot_data={
                "figure": fig,
                "confidence_scores": confidence_scores,
                "chosen_action": chosen_action,
                "max_confidence": max(scores) if scores else 0,
                "min_confidence": min(scores) if scores else 0,
                "confidence_range": max(scores) - min(scores) if scores else 0,
            },
            metadata={
                "type": "confidence_distribution",
                "num_actions": len(actions),
                "has_chosen_action": chosen_action is not None,
                "plotly_available": True,
            },
            recommendations=recommendations,
            risk_indicators=risk_indicators,
        )

    def _analyze_confidence_distribution(
        self, confidence_scores: Dict[str, float], chosen_action: Optional[str]
    ) -> List[str]:
        """Analyze confidence distribution and provide recommendations."""
        recommendations = []
        scores = list(confidence_scores.values())

        if not scores:
            return ["No confidence scores available for analysis"]

        max_score = max(scores)
        min_score = min(scores)

        # Check for very low confidence across all options
        if max_score < 0.4:
            recommendations.append(
                "All confidence scores are very low - consider fallback strategies"
            )

        # Check for very close confidence scores
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.1:
            recommendations.append(
                "Multiple actions have very similar confidence scores"
            )

        # Check chosen action confidence
        if chosen_action:
            chosen_confidence = confidence_scores.get(chosen_action, 0)
            if chosen_confidence < 0.5:
                recommendations.append(
                    f"Chosen action '{chosen_action}' has low confidence ({chosen_confidence:.2f})"
                )

        # Check for overconfidence
        if max_score > 0.9:
            recommendations.append(
                "Some confidence scores are very high - verify calibration"
            )

        return (
            recommendations
            if recommendations
            else ["Confidence distribution appears reasonable"]
        )

    def _identify_visualization_risks(
        self, confidence_scores: Dict[str, float], chosen_action: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Identify risk indicators from confidence distribution."""
        risks = []

        if chosen_action:
            chosen_confidence = confidence_scores.get(chosen_action, 0)
            if chosen_confidence < 0.3:
                risks.append(
                    {
                        "type": "low_chosen_confidence",
                        "severity": "high",
                        "description": f"Chosen action confidence is critically low ({chosen_confidence:.2f})",
                        "action_required": "Consider alternative decision or additional data",
                    }
                )

        # Check for extremely polarized confidence
        scores = list(confidence_scores.values())
        if scores:
            confidence_range = max(scores) - min(scores)
            if confidence_range > 0.8:
                risks.append(
                    {
                        "type": "high_confidence_variance",
                        "severity": "medium",
                        "description": f"Large confidence variance ({confidence_range:.2f}) indicates uncertain decision space",
                        "action_required": "Review decision criteria and feature importance",
                    }
                )

        return risks

    def _create_text_based_visualization(
        self,
        confidence_scores: Dict[str, float],
        chosen_action: Optional[str],
        title: str,
    ) -> UncertaintyVisualization:
        """Create text-based visualization when plotly is not available."""
        text_output = f"""
{title}
{'=' * len(title)}

Confidence Scores:
"""

        for action, score in confidence_scores.items():
            marker = " <-- CHOSEN" if action == chosen_action else ""
            text_output += f"  {action}: {score:.2f}{marker}"
            text_output += f"\nMax Confidence: {max(confidence_scores.values()):.2f}"
        text_output += f"\nMin Confidence: {min(confidence_scores.values()):.2f}"
        text_output += f"\nConfidence Range: {max(confidence_scores.values()) - min(confidence_scores.values()):.2f}"

        return UncertaintyVisualization(
            plot_data={"text_visualization": text_output},
            metadata={
                "type": "confidence_distribution",
                "format": "text",
                "plotly_available": False,
            },
            recommendations=self._analyze_confidence_distribution(
                confidence_scores, chosen_action
            ),
            risk_indicators=self._identify_visualization_risks(
                confidence_scores, chosen_action
            ),
        )

    @validate_types
    def create_uncertainty_evolution_plot(
        self,
        uncertainty_history: List[UncertaintyMetrics],
        title: str = "Uncertainty Evolution Over Time",
    ) -> UncertaintyVisualization:
        """
        Create uncertainty evolution visualization over decision steps.

        Args:
            uncertainty_history: List of uncertainty metrics over time
            title: Plot title

        Returns:
            Visualization data container
        """
        try:
            validate_non_empty_string(title, "title")
        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="UncertaintyVisualizer Evolution Plot",
                    action="Validating title",
                    details=str(e),
                    suggestion="Provide a non-empty title for the visualization",
                )
            ) from e

        if not uncertainty_history:
            return UncertaintyVisualization(
                plot_data={"error": "No uncertainty history provided"},
                metadata={"type": "uncertainty_evolution", "status": "empty"},
                recommendations=["No uncertainty data available for visualization"],
                risk_indicators=[],
            )

        if not PLOTLY_AVAILABLE:
            return self._create_text_uncertainty_evolution(uncertainty_history, title)

        # Create multi-line plot
        steps = list(range(len(uncertainty_history)))

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Entropy (Uncertainty Measure)",
                "Decision Pressure",
                "Variance",
                "Ambiguity Index",
            ),
            shared_xaxes=True,
        )

        # Entropy plot
        entropies = [m.entropy for m in uncertainty_history]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=entropies,
                mode="lines+markers",
                name="Entropy",
                line=dict(color="red"),
            ),
            row=1,
            col=1,
        )

        # Decision pressure plot
        pressures = [m.decision_pressure for m in uncertainty_history]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=pressures,
                mode="lines+markers",
                name="Decision Pressure",
                line=dict(color="blue"),
            ),
            row=1,
            col=2,
        )

        # Variance plot
        variances = [m.variance for m in uncertainty_history]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=variances,
                mode="lines+markers",
                name="Variance",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )

        # Ambiguity index plot
        ambiguities = [m.ambiguity_index for m in uncertainty_history]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=ambiguities,
                mode="lines+markers",
                name="Ambiguity Index",
                line=dict(color="orange"),
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Decision Step", row=2, col=1)
        fig.update_xaxes(title_text="Decision Step", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=2)

        # Analyze trends
        recommendations = self._analyze_uncertainty_trends(uncertainty_history)

        return UncertaintyVisualization(
            plot_data={
                "figure": fig,
                "uncertainty_history": [m.__dict__ for m in uncertainty_history],
                "total_steps": len(uncertainty_history),
                "avg_entropy": sum(entropies) / len(entropies) if entropies else 0,
                "avg_pressure": sum(pressures) / len(pressures) if pressures else 0,
            },
            metadata={
                "type": "uncertainty_evolution",
                "num_steps": len(uncertainty_history),
                "plotly_available": True,
            },
            recommendations=recommendations,
            risk_indicators=self._identify_uncertainty_risks(uncertainty_history),
        )

    def _analyze_uncertainty_trends(
        self, uncertainty_history: List[UncertaintyMetrics]
    ) -> List[str]:
        """Analyze uncertainty trends and provide insights."""
        if len(uncertainty_history) < 2:
            return ["Insufficient data for trend analysis"]

        recommendations = []

        # Check entropy trend
        entropies = [m.entropy for m in uncertainty_history]
        entropy_trend = self._calculate_trend(entropies)

        if entropy_trend > 0.1:
            recommendations.append(
                "Uncertainty is increasing over time - agent may be becoming less confident"
            )
        elif entropy_trend < -0.1:
            recommendations.append(
                "Uncertainty is decreasing - agent confidence is improving"
            )

        # Check for high entropy periods
        high_entropy_steps = sum(1 for m in uncertainty_history if m.entropy > 1.0)
        if high_entropy_steps > len(uncertainty_history) * 0.3:
            recommendations.append(
                f"{high_entropy_steps} steps had high uncertainty - review decision criteria"
            )

        # Check decision pressure
        low_pressure_steps = sum(
            1 for m in uncertainty_history if m.decision_pressure < 0.1
        )
        if low_pressure_steps > len(uncertainty_history) * 0.2:
            recommendations.append(
                f"{low_pressure_steps} decisions were made under high pressure"
            )

        return (
            recommendations
            if recommendations
            else ["Uncertainty evolution appears within normal ranges"]
        )

    def _identify_uncertainty_risks(
        self, uncertainty_history: List[UncertaintyMetrics]
    ) -> List[Dict[str, Any]]:
        """Identify risks from uncertainty evolution."""
        risks = []

        if not uncertainty_history:
            return risks

        # Check for consistently high entropy
        avg_entropy = sum(m.entropy for m in uncertainty_history) / len(
            uncertainty_history
        )
        if avg_entropy > 1.2:
            risks.append(
                {
                    "type": "consistently_high_uncertainty",
                    "severity": "high",
                    "description": f"Average entropy ({avg_entropy:.2f}) indicates consistently uncertain decisions",
                    "action_required": "Review agent training and decision-making process",
                }
            )

        # Check for entropy spikes
        entropies = [m.entropy for m in uncertainty_history]
        max_entropy = max(entropies)
        if max_entropy > 2.0:
            step = entropies.index(max_entropy)
            risks.append(
                {
                    "type": "entropy_spike",
                    "severity": "medium",
                    "description": f"Extreme uncertainty spike at step {step} (entropy: {max_entropy:.2f})",
                    "action_required": "Investigate the decision context at this step",
                }
            )

        return risks

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))
        slope = sum(
            (xi - sum(x) / n) * (yi - sum(values) / n) for xi, yi in zip(x, values)
        ) / sum((xi - sum(x) / n) ** 2 for xi in x)
        return slope

    def _create_text_uncertainty_evolution(
        self, uncertainty_history: List[UncertaintyMetrics], title: str
    ) -> UncertaintyVisualization:
        """Create text-based uncertainty evolution when plotly is not available."""
        text_output = f"""
{title}
{'=' * len(title)}

Uncertainty Evolution Summary:
"""

        for i, metrics in enumerate(uncertainty_history):
            text_output += f"""
Step {i}:
  Entropy: {metrics.entropy:.3f}
  Decision Pressure: {metrics.decision_pressure:.3f}
  Variance: {metrics.variance:.3f}
  Ambiguity Index: {metrics.ambiguity_index:.3f}
"""

        text_output += f"\nTotal Steps: {len(uncertainty_history)}"
        if uncertainty_history:
            avg_entropy = sum(m.entropy for m in uncertainty_history) / len(
                uncertainty_history
            )
            text_output += f"\nAverage Entropy: {avg_entropy:.3f}"

        return UncertaintyVisualization(
            plot_data={"text_visualization": text_output},
            metadata={
                "type": "uncertainty_evolution",
                "format": "text",
                "plotly_available": False,
            },
            recommendations=self._analyze_uncertainty_trends(uncertainty_history),
            risk_indicators=self._identify_uncertainty_risks(uncertainty_history),
        )

    @validate_types
    def create_decision_tree_visualization(
        self,
        trace: DecisionTrace,
        max_depth: int = 10,
        title: str = "Decision Tree with Confidence",
    ) -> UncertaintyVisualization:
        """
        Create an interactive decision tree visualization.

        Args:
            trace: DecisionTrace to visualize
            max_depth: Maximum tree depth to display
            title: Visualization title

        Returns:
            Decision tree visualization
        """
        try:
            validate_non_empty_string(title, "title")
            validate_positive_number(max_depth, "max_depth")

            if not isinstance(max_depth, int):
                raise UCUPTypeError("max_depth must be an integer")

        except (UCUPValueError, UCUPTypeError) as e:
            return UncertaintyVisualization(
                plot_data={
                    "error": create_error_message(
                        context="UncertaintyVisualizer Decision Tree",
                        action="Validating parameters",
                        details=str(e),
                        suggestion="Provide valid title and max_depth",
                    )
                },
                metadata={"type": "decision_tree", "status": "error"},
                recommendations=["Invalid parameters for decision tree visualization"],
                risk_indicators=[],
            )

        if not trace.decisions:
            return UncertaintyVisualization(
                plot_data={"error": "No decisions in trace"},
                metadata={"type": "decision_tree", "status": "empty"},
                recommendations=["No decision data available for tree visualization"],
                risk_indicators=[],
            )

        if not PLOTLY_AVAILABLE:
            return self._create_text_decision_tree(trace, max_depth, title)

        # Create tree structure
        nodes = []
        edges = []

        # Add root node (initial state)
        nodes.append(
            {"id": "root", "label": "Start", "level": 0, "type": "root", "size": 20}
        )

        # Build decision nodes
        for i, decision in enumerate(trace.decisions[:max_depth]):
            node_id = f"decision_{i}"
            confidence = decision.confidence_scores.get(decision.chosen_action or "", 0)

            # Color based on confidence
            if confidence >= 0.8:
                color = "green"
            elif confidence >= 0.6:
                color = "yellow"
            elif confidence >= 0.4:
                color = "orange"
            else:
                color = "red"

            nodes.append(
                {
                    "id": node_id,
                    "label": f"D{i}: {decision.chosen_action or 'No Action'}",
                    "level": i + 1,
                    "confidence": confidence,
                    "color": color,
                    "type": "decision",
                    "size": 15 + confidence * 10,  # Size based on confidence
                }
            )

            # Connect to previous node
            prev_node = "root" if i == 0 else f"decision_{i-1}"
            edges.append(
                {"source": prev_node, "target": node_id, "confidence": confidence}
            )

        # Create network visualization
        fig = go.Figure()

        # Add edges
        for edge in edges:
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],  # Placeholder - would need actual positioning
                    y=[0, 1],
                    mode="lines",
                    line=dict(width=2, color="gray"),
                    showlegend=False,
                )
            )

        # Add nodes as scatter plot
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_labels = []

        for node in nodes:
            # Simple positioning logic (could be improved)
            level = node["level"]
            node_x.append(level * 2)
            node_y.append(hash(node["id"]) % 10)  # Simple pseudo-random y position
            node_colors.append(node.get("color", "blue"))
            node_sizes.append(node.get("size", 15))
            node_labels.append(node["label"])

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(size=node_sizes, color=node_colors, sizemode="diameter"),
                text=node_labels,
                textposition="top center",
                showlegend=False,
            )
        )

        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        return UncertaintyVisualization(
            plot_data={
                "figure": fig,
                "nodes": nodes,
                "edges": edges,
                "total_decisions": len(trace.decisions),
                "displayed_depth": min(max_depth, len(trace.decisions)),
            },
            metadata={
                "type": "decision_tree",
                "max_depth": max_depth,
                "actual_depth": len(trace.decisions),
                "plotly_available": True,
            },
            recommendations=self._analyze_decision_tree(trace),
            risk_indicators=self._identify_tree_risks(trace),
        )

    def _analyze_decision_tree(self, trace: DecisionTrace) -> List[str]:
        """Analyze decision tree structure and provide insights."""
        recommendations = []

        if not trace.decisions:
            return ["No decision data available for analysis"]

        # Check decision quality trend
        confidences = []
        for decision in trace.decisions:
            if decision.chosen_action and decision.confidence_scores:
                confidence = decision.confidence_scores.get(decision.chosen_action, 0)
                confidences.append(confidence)

        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence < 0.5:
                recommendations.append(
                    "Overall decision confidence is low - consider improving agent certainty"
                )

            # Check for declining confidence
            if len(confidences) > 3:
                first_half = confidences[: len(confidences) // 2]
                second_half = confidences[len(confidences) // 2 :]
                if second_half and first_half:
                    trend = (sum(second_half) / len(second_half)) / (
                        sum(first_half) / len(first_half)
                    )
                    if trend < 0.9:
                        recommendations.append(
                            "Decision confidence appears to be declining over time"
                        )

        # Check for repeated actions
        actions = [d.chosen_action for d in trace.decisions if d.chosen_action]
        if len(actions) != len(set(actions)):
            recommendations.append(
                "Some actions are being repeated - check for decision loops"
            )

        return (
            recommendations
            if recommendations
            else ["Decision tree structure appears reasonable"]
        )

    def _identify_tree_risks(self, trace: DecisionTrace) -> List[Dict[str, Any]]:
        """Identify risks from decision tree analysis."""
        risks = []

        # Check for very low confidence decisions
        low_confidence_count = 0
        for decision in trace.decisions:
            if decision.chosen_action:
                confidence = decision.confidence_scores.get(decision.chosen_action, 0)
                if confidence < 0.3:
                    low_confidence_count += 1

        if low_confidence_count > len(trace.decisions) * 0.3:
            risks.append(
                {
                    "type": "frequent_low_confidence",
                    "severity": "high",
                    "description": f"{low_confidence_count} decisions had critically low confidence",
                    "action_required": "Review agent decision-making criteria and training data",
                }
            )

        return risks

    def _create_text_decision_tree(
        self, trace: DecisionTrace, max_depth: int, title: str
    ) -> UncertaintyVisualization:
        """Create text-based decision tree when plotly is not available."""
        text_output = f"""
{title}
{'=' * len(title)}

Decision Tree Summary:
"""

        for i, decision in enumerate(trace.decisions[:max_depth]):
            confidence = decision.confidence_scores.get(decision.chosen_action or "", 0)
            text_output += f"""
Decision {i}:
  Action: {decision.chosen_action or 'None'}
  Confidence: {confidence:.3f}
  Available Options: {len(decision.available_actions)}
"""

        text_output += f"\nTotal Decisions: {len(trace.decisions)}"
        text_output += f"\nDisplayed Depth: {min(max_depth, len(trace.decisions))}"

        return UncertaintyVisualization(
            plot_data={"text_visualization": text_output},
            metadata={
                "type": "decision_tree",
                "format": "text",
                "plotly_available": False,
            },
            recommendations=self._analyze_decision_tree(trace),
            risk_indicators=self._identify_tree_risks(trace),
        )

    @validate_types
    def create_comparative_analysis(
        self,
        traces: List[Tuple[str, DecisionTrace]],
        title: str = "Comparative Agent Analysis",
    ) -> UncertaintyVisualization:
        """
        Create comparative analysis between multiple agent runs.

        Args:
            traces: List of (name, trace) tuples for comparison
            title: Visualization title

        Returns:
            Comparative analysis visualization
        """
        try:
            validate_non_empty_string(title, "title")
            if not traces:
                raise UCUPValueError("At least one trace must be provided")

            for name, trace in traces:
                validate_non_empty_string(name, "trace_name")
                if not hasattr(trace, "decisions"):
                    raise UCUPValueError(f"Invalid trace provided for {name}")

        except UCUPValueError as e:
            return UncertaintyVisualization(
                plot_data={
                    "error": create_error_message(
                        context="UncertaintyVisualizer Comparative Analysis",
                        action="Validating traces",
                        details=str(e),
                        suggestion="Provide valid trace names and DecisionTrace objects",
                    )
                },
                metadata={"type": "comparative_analysis", "status": "error"},
                recommendations=["Invalid trace data for comparative analysis"],
                risk_indicators=[],
            )

        if not PLOTLY_AVAILABLE:
            return self._create_text_comparative_analysis(traces, title)

        # Create comparison data
        comparison_data = []
        for name, trace in traces:
            if trace.decisions:
                avg_confidence = (
                    sum(
                        d.confidence_scores.get(d.chosen_action, 0)
                        for d in trace.decisions
                        if d.chosen_action
                    )
                    / len([d for d in trace.decisions if d.chosen_action])
                    if trace.decisions
                    else 0
                )

                comparison_data.append(
                    {
                        "name": name,
                        "decisions": len(trace.decisions),
                        "successful_decisions": trace.successful_decisions,
                        "failed_decisions": trace.failed_decisions,
                        "avg_confidence": avg_confidence,
                        "total_tokens": trace.total_tokens,
                    }
                )

        # Create comparison plot
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Average Confidence by Agent",
                "Decision Success Rate",
                "Total Tokens Used",
                "Decisions Made",
            ),
        )

        names = [d["name"] for d in comparison_data]
        confidences = [d["avg_confidence"] for d in comparison_data]
        success_rates = [
            d["successful_decisions"] / max(d["decisions"], 1) for d in comparison_data
        ]
        tokens = [d["total_tokens"] for d in comparison_data]
        decision_counts = [d["decisions"] for d in comparison_data]

        # Confidence comparison
        fig.add_trace(
            go.Bar(x=names, y=confidences, name="Avg Confidence"), row=1, col=1
        )

        # Success rate comparison
        fig.add_trace(
            go.Bar(x=names, y=success_rates, name="Success Rate"), row=1, col=2
        )

        # Token usage comparison
        fig.add_trace(go.Bar(x=names, y=tokens, name="Total Tokens"), row=2, col=1)

        # Decision count comparison
        fig.add_trace(
            go.Bar(x=names, y=decision_counts, name="Decisions"), row=2, col=2
        )

        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            showlegend=False,
        )

        return UncertaintyVisualization(
            plot_data={
                "figure": fig,
                "comparison_data": comparison_data,
                "num_agents": len(traces),
                "best_performer": max(
                    comparison_data, key=lambda x: x["avg_confidence"]
                )["name"]
                if comparison_data
                else None,
            },
            metadata={
                "type": "comparative_analysis",
                "num_traces": len(traces),
                "plotly_available": True,
            },
            recommendations=self._generate_comparison_recommendations(comparison_data),
            risk_indicators=self._identify_comparison_risks(comparison_data),
        )

    def _generate_comparison_recommendations(
        self, comparison_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations from comparative analysis."""
        if len(comparison_data) < 2:
            return ["Need at least two agents for meaningful comparison"]

        recommendations = []

        # Find best and worst performers
        best_agent = max(comparison_data, key=lambda x: x["avg_confidence"])
        worst_agent = min(comparison_data, key=lambda x: x["avg_confidence"])

        confidence_diff = best_agent["avg_confidence"] - worst_agent["avg_confidence"]

        if confidence_diff > 0.2:
            recommendations.append(
                f"{best_agent['name']} significantly outperforms {worst_agent['name']} in confidence"
            )
            recommendations.append(
                f"Study {best_agent['name']}'s decision patterns for improvement ideas"
            )

        # Check for efficiency differences
        if comparison_data:
            avg_tokens = sum(d["total_tokens"] for d in comparison_data) / len(
                comparison_data
            )
            high_token_agents = [
                d for d in comparison_data if d["total_tokens"] > avg_tokens * 1.5
            ]

            if high_token_agents:
                recommendations.append(
                    f"{' and '.join(d['name'] for d in high_token_agents)} use significantly more tokens - consider optimization"
                )

        return (
            recommendations
            if recommendations
            else ["Agent performances are comparable"]
        )

    def _identify_comparison_risks(
        self, comparison_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify risks from comparative analysis."""
        risks = []

        # Check for very low confidence agents
        low_confidence_agents = [
            d for d in comparison_data if d["avg_confidence"] < 0.4
        ]
        if low_confidence_agents:
            risks.append(
                {
                    "type": "low_confidence_agents",
                    "severity": "high",
                    "description": f"{' and '.join(d['name'] for d in low_confidence_agents)} have critically low confidence",
                    "action_required": "Review training data and decision criteria",
                }
            )

        # Check for high failure rates
        high_failure_agents = [
            d
            for d in comparison_data
            if d["failed_decisions"] > d["successful_decisions"]
        ]
        if high_failure_agents:
            risks.append(
                {
                    "type": "high_failure_rate",
                    "severity": "medium",
                    "description": f"{' and '.join(d['name'] for d in high_failure_agents)} have more failed than successful decisions",
                    "action_required": "Investigate decision-making logic and error handling",
                }
            )

        return risks

    def _create_text_comparative_analysis(
        self, traces: List[Tuple[str, DecisionTrace]], title: str
    ) -> UncertaintyVisualization:
        """Create text-based comparative analysis when plotly is not available."""
        text_output = f"""
{title}
{'=' * len(title)}

Comparative Analysis Summary:
"""

        for name, trace in traces:
            if trace.decisions:
                avg_confidence = sum(
                    d.confidence_scores.get(d.chosen_action, 0)
                    for d in trace.decisions
                    if d.chosen_action
                ) / len([d for d in trace.decisions if d.chosen_action])

                text_output += f"""
{name}:
  Decisions: {len(trace.decisions)}
  Successful: {trace.successful_decisions}
  Failed: {trace.failed_decisions}
  Avg Confidence: {avg_confidence:.3f}
  Total Tokens: {trace.total_tokens}
"""

        text_output += f"\nTotal Agents Compared: {len(traces)}"

        return UncertaintyVisualization(
            plot_data={"text_visualization": text_output},
            metadata={
                "type": "comparative_analysis",
                "format": "text",
                "plotly_available": False,
            },
            recommendations=self._generate_comparison_recommendations(
                []
            ),  # Would need to calculate comparison_data
            risk_indicators=[],
        )
