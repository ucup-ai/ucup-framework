"""
Observability tools for UCUP Framework.

This module provides comprehensive tools for tracking, visualizing, and
analyzing agent decision-making processes. When an agent fails, you need
to understand not just what happened, but WHY it made each decision.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import plotly.graph_objects as go

# Import validation utilities (lazy imports to avoid circular dependencies)
from .validation import (
    TypeValidator,
    UCUPValidationError,
    UCUPValueError,
    create_error_message,
    sanitize_inputs,
    validate_non_empty_string,
    validate_positive_number,
    validate_probability,
    validate_types,
)


@dataclass
class DecisionNode:
    """A single decision point in the reasoning tree."""

    decision_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    available_actions: List[Dict[str, Any]] = field(default_factory=list)
    chosen_action: Optional[str] = None
    rejection_reasons: Dict[str, str] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    token_usage: Optional[int] = None
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate DecisionNode fields with meaningful error messages."""
        try:
            validate_non_empty_string(self.decision_id, "decision_id")
        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="DecisionNode Initialization",
                    action="Validating decision ID",
                    details=str(e),
                    suggestion="Provide a non-empty unique identifier for the decision",
                ),
                context={"decision_id": self.decision_id},
            ) from e

        # Validate confidence scores
        if self.confidence_scores:
            for action_name, confidence in self.confidence_scores.items():
                try:
                    validate_probability(
                        confidence, f"confidence_scores['{action_name}']"
                    )
                except UCUPValueError as e:
                    raise UCUPValueError(
                        create_error_message(
                            context="DecisionNode Initialization",
                            action=f"Validating confidence for action '{action_name}'",
                            details=str(e),
                            suggestion="All confidence scores must be between 0.0 and 1.0",
                        ),
                        context={"action": action_name, "confidence": confidence},
                    ) from e

        # Validate token usage
        if self.token_usage is not None:
            try:
                validate_positive_number(self.token_usage, "token_usage")
                if not isinstance(self.token_usage, int):
                    raise UCUPTypeError("token_usage must be an integer")
            except (UCUPValueError, UCUPTypeError) as e:
                raise UCUPValueError(
                    create_error_message(
                        context="DecisionNode Initialization",
                        action="Validating token usage",
                        details=str(e),
                        suggestion="Token usage must be a non-negative integer",
                    ),
                    context={"token_usage": self.token_usage},
                ) from e


@dataclass
class DecisionTrace:
    """Complete trace of a decision-making session."""

    session_id: str
    decisions: List[DecisionNode] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_tokens: int = 0
    successful_decisions: int = 0
    failed_decisions: int = 0

    def add_decision(self, decision: DecisionNode):
        """Add a decision node to the trace."""
        self.decisions.append(decision)
        if decision.chosen_action:
            self.successful_decisions += 1
        else:
            self.failed_decisions += 1

        if decision.token_usage:
            self.total_tokens += decision.token_usage

    def get_decision_sequence(self) -> List[str]:
        """Get the sequence of chosen actions."""
        return [d.chosen_action for d in self.decisions if d.chosen_action]


class DecisionTracer:
    """
    Traces agent decisions with detailed context for debugging and analysis.

    Captures:
    - Complete reasoning context
    - Rejected alternatives with explanations
    - Confidence levels for each option
    - Token usage and performance metrics
    """

    def __init__(self, enable_detailed_tracing: bool = True):
        self.enable_detailed_tracing = enable_detailed_tracing
        self.traces: Dict[str, DecisionTrace] = {}
        self.logger = logging.getLogger(__name__)

    def start_session(self, session_id: str) -> str:
        """Start a new tracing session."""
        try:
            validate_non_empty_string(session_id, "session_id")
        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="DecisionTracer Session Start",
                    action="Validating session identifier",
                    details=str(e),
                    suggestion="Provide a non-empty unique session identifier",
                ),
                context={"session_id": session_id},
            ) from e

        if session_id in self.traces:
            raise UCUPValueError(
                create_error_message(
                    context="DecisionTracer Session Start",
                    action="Checking session uniqueness",
                    details=f"Session '{session_id}' already exists",
                    suggestion="Use a different session ID or end the existing session first",
                ),
                context={"existing_sessions": list(self.traces.keys())},
            )

        self.traces[session_id] = DecisionTrace(session_id=session_id)
        return session_id

    def record_decision(
        self,
        session_id: str,
        available_actions: List[Dict[str, Any]],
        chosen_action: str,
        confidence_scores: Dict[str, float],
        context_snapshot: Dict[str, Any],
        rejection_reasons: Optional[Dict[str, str]] = None,
        token_usage: Optional[int] = None,
        reasoning_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> DecisionNode:
        """Record a decision with full context."""
        try:
            validate_non_empty_string(session_id, "session_id")
            validate_non_empty_string(chosen_action, "chosen_action")

            # Validate that chosen_action is in available_actions
            available_action_names = [
                action.get("action", "") for action in available_actions
            ]
            if chosen_action not in available_action_names:
                raise UCUPValueError(
                    f"Chosen action '{chosen_action}' not found in available actions"
                )

            # Validate confidence scores for all actions
            for action_name in available_action_names:
                if action_name not in confidence_scores:
                    raise UCUPValueError(
                        f"Missing confidence score for action '{action_name}'"
                    )
                try:
                    validate_probability(
                        confidence_scores[action_name],
                        f"confidence_scores['{action_name}']",
                    )
                except UCUPValueError as e:
                    raise UCUPValueError(
                        f"Invalid confidence for action '{action_name}': {e}"
                    )

            # Validate token usage
            if token_usage is not None:
                validate_positive_number(token_usage, "token_usage")
                if not isinstance(token_usage, int):
                    raise UCUPTypeError("token_usage must be an integer")

        except (UCUPValueError, UCUPTypeError) as e:
            raise UCUPValueError(
                create_error_message(
                    context="DecisionTracer Record Decision",
                    action="Validating decision parameters",
                    details=str(e),
                    suggestion="Ensure all required parameters are provided with correct types and values",
                ),
                context={
                    "session_id": session_id,
                    "chosen_action": chosen_action,
                    "available_actions_count": len(available_actions),
                    "confidence_scores_keys": list(confidence_scores.keys()),
                },
            ) from e

        if session_id not in self.traces:
            raise UCUPValueError(
                create_error_message(
                    context="DecisionTracer Record Decision",
                    action="Checking session existence",
                    details=f"Session '{session_id}' not found",
                    suggestion="Start a session before recording decisions",
                ),
                context={"available_sessions": list(self.traces.keys())},
            )

        decision = DecisionNode(
            decision_id=f"{session_id}_{len(self.traces[session_id].decisions)}",
            available_actions=available_actions,
            chosen_action=chosen_action,
            rejection_reasons=rejection_reasons or {},
            confidence_scores=confidence_scores,
            context_snapshot=context_snapshot,
            token_usage=token_usage,
            reasoning_steps=reasoning_steps or [],
        )

        self.traces[session_id].add_decision(decision)
        return decision

    def end_session(self, session_id: str) -> DecisionTrace:
        """End a tracing session and return the complete trace."""
        if session_id not in self.traces:
            raise ValueError(f"Session {session_id} not found")

        trace = self.traces[session_id]
        trace.end_time = datetime.now()

        # Persist trace if needed (in real implementation, save to database)
        self.logger.info(
            f"Completed tracing session {session_id}: "
            f"{trace.successful_decisions} successful, "
            f"{trace.failed_decisions} failed decisions"
        )

        return trace

    def get_session_trace(self, session_id: str) -> DecisionTrace:
        """Get the trace for a specific session."""
        return self.traces.get(session_id)

    def get_recent_sessions(self, limit: int = 10) -> List[DecisionTrace]:
        """Get the most recent tracing sessions."""
        sessions = list(self.traces.values())
        sessions.sort(key=lambda x: x.start_time, reverse=True)
        return sessions[:limit]


@dataclass
class DecisionVisualization:
    """Visualization data for decision trees."""

    nodes: List[Dict[str, Any]]
    edges: List[Tuple[str, str]]
    highlighted_path: List[str]
    alternative_paths: List[List[str]]
    cost_per_branch: Dict[str, float]


class DecisionExplorer:
    """
    Interactive exploration of decision traces.

    Provides tools to:
    - Replay agent decisions
    - Visualize reasoning trees
    - Perform "what-if" analysis
    """

    def __init__(self, decision_tracer: DecisionTracer):
        self.tracer = decision_tracer

    def visualize_decision_tree(self, session_id: str) -> DecisionVisualization:
        """Create a visualization of the decision tree for a session."""
        trace = self.tracer.get_session_trace(session_id)
        if not trace:
            return DecisionVisualization([], [], [], [], {})

        nodes = []
        edges = []
        node_ids = []
        cost_per_branch = {}

        for i, decision in enumerate(trace.decisions):
            node_id = f"d_{i}"
            node_ids.append(node_id)

            # Create main node
            nodes.append(
                {
                    "id": node_id,
                    "label": f"Decision {i}: {decision.chosen_action or 'No Action'}",
                    "confidence": decision.confidence_scores.get(
                        decision.chosen_action or "", 0
                    ),
                    "level": i,
                    "actions_count": len(decision.available_actions),
                    "timestamp": decision.timestamp.isoformat(),
                }
            )

            # Add edges for all available actions (even rejected ones)
            for action_info in decision.available_actions:
                action_name = action_info.get("action", "unknown")

                if action_name == decision.chosen_action:
                    # Chosen action - solid edge
                    edges.append((node_id, f"{node_id}_{action_name}"))
                    nodes.append(
                        {
                            "id": f"{node_id}_{action_name}",
                            "label": f"Chosen: {action_name}",
                            "confidence": decision.confidence_scores.get(
                                action_name, 0
                            ),
                            "level": i + 1,
                            "type": "chosen",
                        }
                    )
                else:
                    # Rejected action - dashed edge
                    edges.append((node_id, f"{node_id}_{action_name}"))
                    nodes.append(
                        {
                            "id": f"{node_id}_{action_name}",
                            "label": f"Rejected: {action_name}",
                            "reason": decision.rejection_reasons.get(action_name, ""),
                            "confidence": decision.confidence_scores.get(
                                action_name, 0
                            ),
                            "level": i + 1,
                            "type": "rejected",
                        }
                    )

            # Track cost per branch
            decision_cost = decision.token_usage or 0
            cost_per_branch[node_id] = decision_cost

        return DecisionVisualization(
            nodes=nodes,
            edges=edges,
            highlighted_path=node_ids,  # Actual chosen path
            alternative_paths=[],  # Would need more complex logic
            cost_per_branch=cost_per_branch,
        )

    def what_if_analysis(
        self, session_id: str, change_step: int, alternative_choice: str
    ) -> Dict[str, Any]:
        """
        Simulate what would happen if a different choice was made at a specific step.

        This is a conceptual implementation - in practice, this would require
        actually re-running the agent from that decision point.
        """
        try:
            validate_non_empty_string(session_id, "session_id")
            validate_non_empty_string(alternative_choice, "alternative_choice")
            validate_positive_number(change_step, "change_step")

            if not isinstance(change_step, int):
                raise UCUPTypeError("change_step must be an integer")

        except (UCUPValueError, UCUPTypeError) as e:
            return {
                "error": create_error_message(
                    context="DecisionExplorer What-If Analysis",
                    action="Validating analysis parameters",
                    details=str(e),
                    suggestion="Provide valid session ID, step number, and alternative choice",
                )
            }

        trace = self.tracer.get_session_trace(session_id)
        if not trace or change_step >= len(trace.decisions):
            return {
                "error": create_error_message(
                    context="DecisionExplorer What-If Analysis",
                    action="Validating session and step",
                    details=f"Session '{session_id}' not found or step {change_step} is out of range",
                    suggestion="Check that the session exists and step number is within valid range",
                )
            }

        original_decision = trace.decisions[change_step]

        # Check if alternative choice was actually available
        available_actions = [
            action.get("action") for action in original_decision.available_actions
        ]

        if alternative_choice not in available_actions:
            return {
                "error": create_error_message(
                    context="DecisionExplorer What-If Analysis",
                    action="Validating alternative choice",
                    details=f"Alternative choice '{alternative_choice}' was not available at step {change_step}",
                    suggestion=f"Choose from available actions: {available_actions}",
                )
            }

        # Simulate the counterfactual
        simulation = {
            "original_choice": original_decision.chosen_action,
            "alternative_choice": alternative_choice,
            "original_confidence": original_decision.confidence_scores.get(
                original_decision.chosen_action, 0
            ),
            "alternative_confidence": original_decision.confidence_scores.get(
                alternative_choice, 0
            ),
            "rejection_reason_original": original_decision.rejection_reasons.get(
                alternative_choice, "Chosen alternative"
            ),
            "hypothesized_path": [],
            "estimated_outcome_change": "Would need to re-run agent to determine",
        }

        return simulation

    async def replay_session_with_timing(self, session_id: str) -> List[Dict[str, Any]]:
        """Replay a session with original timing and annotations."""
        trace = self.tracer.get_session_trace(session_id)
        if not trace:
            return []

        replay = []
        prev_time = trace.start_time

        for decision in trace.decisions:
            time_diff = (decision.timestamp - prev_time).total_seconds()

            replay.append(
                {
                    "decision_id": decision.decision_id,
                    "chosen_action": decision.chosen_action,
                    "confidence": decision.confidence_scores.get(
                        decision.chosen_action, 0
                    ),
                    "time_since_previous": time_diff,
                    "available_options": len(decision.available_actions),
                    "token_cost": decision.token_usage,
                    "reasoning_snapshot": decision.reasoning_steps,
                }
            )

            prev_time = decision.timestamp

        return replay


@dataclass
class ReasoningMetrics:
    """Metrics for analyzing reasoning quality."""

    contradiction_frequency: float = 0.0
    logical_loops_count: int = 0
    confidence_volatility: float = 0.0
    halluciation_indicators: int = 0
    successful_reasoning_steps: int = 0
    total_reasoning_steps: int = 0


class ReasoningVisualizer:
    """
    Visualizes reasoning processes with various overlays.

    Shows:
    - Thought process diffing between sessions
    - Attention heatmaps
    - Tool usage patterns
    - Temporal decision quality analysis
    """

    def __init__(self):
        self.metrics_cache: Dict[str, ReasoningMetrics] = {}

    def create_thought_process_diff(
        self, successful_trace: DecisionTrace, failed_trace: DecisionTrace
    ) -> go.Figure:
        """Create a diff visualization between successful and failed reasoning."""

        successful_confidences = [
            d.confidence_scores.get(d.chosen_action, 0)
            for d in successful_trace.decisions
            if d.chosen_action
        ]
        failed_confidences = [
            d.confidence_scores.get(d.chosen_action, 0)
            for d in failed_trace.decisions
            if d.chosen_action
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(successful_confidences))),
                y=successful_confidences,
                mode="lines+markers",
                name="Successful Session",
                line=dict(color="green"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(failed_confidences))),
                y=failed_confidences,
                mode="lines+markers",
                name="Failed Session",
                line=dict(color="red"),
            )
        )

        fig.update_layout(
            title="Confidence Comparison: Success vs Failure",
            xaxis_title="Decision Step",
            yaxis_title="Confidence Score",
        )

        return fig

    def create_attention_heatmap(
        self, trace: DecisionTrace, context_elements: List[str]
    ) -> go.Figure:
        """Create attention heatmap showing focus on different context elements."""

        # This would analyze reasoning_steps for mentions of context elements
        # For demonstration, create a mock heatmap
        attention_matrix = []
        for step in range(len(trace.decisions)):
            step_attention = [
                0.1 * (step + 1) * (i + 1) for i in range(len(context_elements))
            ]
            attention_matrix.append(step_attention)

        df = pd.DataFrame(
            attention_matrix,
            columns=context_elements,
            index=[f"Step {i}" for i in range(len(trace.decisions))],
        )

        fig = go.Figure(
            data=go.Heatmap(z=df.values, x=df.columns, y=df.index, colorscale="RdYlGn")
        )
        fig.update_layout(
            title="Attention Heatmap",
            xaxis_title="Context Elements",
            yaxis_title="Decision Step",
        )

        return fig

    def analyze_tool_usage_patterns(
        self, traces: List[DecisionTrace]
    ) -> Dict[str, Any]:
        """Analyze patterns in tool usage across multiple sessions."""

        tool_usage = defaultdict(lambda: defaultdict(int))

        for trace in traces:
            for decision in trace.decisions:
                for action_info in decision.available_actions:
                    tool_name = action_info.get("tool", action_info.get("action"))
                    if tool_name:
                        tool_usage[tool_name]["available"] += 1

                if decision.chosen_action:
                    chosen_tool = None
                    for action_info in decision.available_actions:
                        if action_info.get("action") == decision.chosen_action:
                            chosen_tool = action_info.get(
                                "tool", decision.chosen_action
                            )
                            break

                    if chosen_tool:
                        tool_usage[chosen_tool]["chosen"] += 1

        # Calculate effectiveness ratios
        tool_effectiveness = {}
        for tool, counts in tool_usage.items():
            chosen = counts.get("chosen", 0)
            available = counts.get("available", 1)  # Avoid division by zero
            tool_effectiveness[tool] = chosen / available

        return {
            "raw_usage": dict(tool_usage),
            "effectiveness_ratios": tool_effectiveness,
            "most_effective": max(tool_effectiveness.items(), key=lambda x: x[1])
            if tool_effectiveness
            else None,
            "least_effective": min(tool_effectiveness.items(), key=lambda x: x[1])
            if tool_effectiveness
            else None,
        }

    def create_temporal_analysis(self, trace: DecisionTrace) -> go.Figure:
        """Analyze how decision quality changes over conversation length."""

        steps = [i for i in range(len(trace.decisions))]
        confidences = [
            d.confidence_scores.get(d.chosen_action, 0)
            for d in trace.decisions
            if d.chosen_action
        ]
        token_usage = [d.token_usage or 0 for d in trace.decisions]
        time_diffs = []

        prev_time = trace.start_time
        for decision in trace.decisions:
            time_diffs.append((decision.timestamp - prev_time).total_seconds())
            prev_time = decision.timestamp

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=confidences,
                mode="lines+markers",
                name="Decision Confidence",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=steps,
                y=time_diffs,
                name="Time per Decision",
                opacity=0.6,
                marker_color="orange",
            )
        )

        fig.update_layout(
            title="Temporal Decision Analysis",
            xaxis_title="Decision Step",
            yaxis_title="Confidence / Time (s)",
        )
        fig.update_yaxes(title_text="Confidence")
        fig.add_yaxes(title_text="Time (seconds)", secondary_y=True, showgrid=False)

        return fig


class AlertCondition:
    """Definition of an alert condition for monitoring."""

    def __init__(
        self,
        name: str,
        condition: callable,
        severity: str = "warning",
        message_template: str = "",
    ):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template

    def check(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if the condition is triggered."""
        if self.condition(metrics):
            return {
                "alert_name": self.name,
                "severity": self.severity,
                "message": self.message_template.format(**metrics),
                "timestamp": datetime.now(),
                "metrics": metrics,
            }
        return None


class LiveAgentMonitor:
    """
    Real-time monitoring of agent reasoning with automated alerts.

    Monitors for:
    - High confidence but wrong decisions (Dunning-Kruger detection)
    - Reasoning loops and circular thinking
    - Logical contradictions
    - Escalating uncertainty over time
    """

    def __init__(self):
        self.alert_conditions = self._setup_default_conditions()
        self.active_alerts: List[Dict[str, Any]] = []

    def _setup_default_conditions(self) -> List[AlertCondition]:
        """Set up default monitoring conditions."""

        return [
            AlertCondition(
                name="high_confidence_but_wrong",
                condition=lambda m: m.get("claimed_confidence", 0) > 0.8
                and m.get("actual_success", False) is False,
                severity="error",
                message_template="High confidence ({claimed_confidence}) but decision was wrong",
            ),
            AlertCondition(
                name="reasoning_loops",
                condition=lambda m: m.get("similar_decisions_count", 0) > 5,
                severity="warning",
                message_template="Reasoning loop detected: {similar_decisions_count} similar decisions",
            ),
            AlertCondition(
                name="contradictory_statements",
                condition=lambda m: m.get("contradictions_count", 0) > 3,
                severity="error",
                message_template="Multiple contradictions detected: {contradictions_count}",
            ),
            AlertCondition(
                name="escalating_uncertainty",
                condition=lambda m: m.get("confidence_trend", 1) < 0.9
                and m.get("steps_count", 0) > 3,
                severity="warning",
                message_template="Confidence decreasing over time (trend: {confidence_trend})",
            ),
            AlertCondition(
                name="hallucination_detection",
                condition=lambda m: m.get("fact_check_failures", 0) > 2,
                severity="error",
                message_template="Potential hallucinations: {fact_check_failures} fact check failures",
            ),
        ]

    def add_custom_alert(
        self,
        name: str,
        condition: callable,
        severity: str = "warning",
        message_template: str = "",
    ):
        """Add a custom alert condition."""
        self.alert_conditions.append(
            AlertCondition(name, condition, severity, message_template)
        )

    def check_agent_session(
        self,
        session_id: str,
        trace: DecisionTrace,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check a session trace for alert conditions.
        """

        metrics = self._calculate_metrics(trace, additional_metrics)
        alerts = []

        for condition in self.alert_conditions:
            alert = condition.check(metrics)
            if alert:
                alert["session_id"] = session_id
                alerts.append(alert)

        # Add to active alerts if any found
        self.active_alerts.extend(alerts)

        return alerts

    def _calculate_metrics(
        self, trace: DecisionTrace, additional: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate monitoring metrics from a decision trace."""

        if not trace.decisions:
            return {}

        metrics = {
            "steps_count": len(trace.decisions),
            "successful_decisions": trace.successful_decisions,
            "failed_decisions": trace.failed_decisions,
            "total_tokens": trace.total_tokens,
            "session_duration": (
                trace.end_time or datetime.now() - trace.start_time
            ).total_seconds(),
        }

        # Confidence analysis
        confidences = [
            d.confidence_scores.get(d.chosen_action, 0)
            for d in trace.decisions
            if d.chosen_action
        ]
        if confidences:
            metrics["avg_confidence"] = sum(confidences) / len(confidences)
            metrics["confidence_volatility"] = (
                pd.Series(confidences).std() if len(confidences) > 1 else 0
            )

            # Trend calculation (simple linear trend)
            if len(confidences) > 2:
                recent = confidences[-3:]  # Last 3 decisions
                earlier = confidences[:-3] if len(confidences) > 3 else [confidences[0]]
                metrics["confidence_trend"] = (
                    sum(recent) / len(recent) / (sum(earlier) / len(earlier))
                )
            else:
                metrics["confidence_trend"] = 1.0

        # Add any additional metrics
        if additional:
            metrics.update(additional)

        return metrics

    def get_active_alerts(
        self, severity_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get currently active alerts, optionally filtered by severity."""
        alerts = self.active_alerts

        if severity_filter:
            alerts = [a for a in alerts if a.get("severity") == severity_filter]

        return alerts

    def clear_resolved_alerts(self, resolved_session_ids: List[str]):
        """Clear alerts for resolved sessions."""
        self.active_alerts = [
            alert
            for alert in self.active_alerts
            if alert.get("session_id") not in resolved_session_ids
        ]
