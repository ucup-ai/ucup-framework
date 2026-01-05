"""
Time Travel Debugging for UCUP Framework.

This module provides sophisticated debugging capabilities that allow developers
to step back and forth through agent decision-making processes, analyze
alternative scenarios, and understand the exact reasoning path that led
to specific outcomes.

Key Features:
- Step-by-step execution replay with full state restoration
- Branch point analysis showing alternative decision paths
- Confidence score evolution tracking over time
- Memory and context snapshot management
- Interactive debugging sessions with breakpoints
"""

import asyncio
import copy
import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
class ExecutionState:
    """Complete state snapshot of agent execution at a specific point."""

    timestamp: datetime
    decision_step: int
    agent_state: Dict[str, Any]
    context_snapshot: Dict[str, Any]
    available_actions: List[Dict[str, Any]]
    chosen_action: Optional[str]
    confidence_scores: Dict[str, float]
    reasoning_steps: List[Dict[str, Any]]
    memory_usage: int = 0
    token_count: int = 0
    execution_time: float = 0.0

    def calculate_hash(self) -> str:
        """Calculate a hash of this state for change detection."""
        state_data = {
            "decision_step": self.decision_step,
            "agent_state": str(sorted(self.agent_state.items())),
            "context": str(sorted(self.context_snapshot.items())),
            "actions": str(self.available_actions),
            "chosen": self.chosen_action,
        }
        return hashlib.md5(json.dumps(state_data, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "decision_step": self.decision_step,
            "agent_state": self.agent_state,
            "context_snapshot": self.context_snapshot,
            "available_actions": self.available_actions,
            "chosen_action": self.chosen_action,
            "confidence_scores": self.confidence_scores,
            "reasoning_steps": self.reasoning_steps,
            "memory_usage": self.memory_usage,
            "token_count": self.token_count,
            "execution_time": self.execution_time,
            "state_hash": self.calculate_hash(),
        }


@dataclass
class DebugBreakpoint:
    """Breakpoint definition for time travel debugging."""

    breakpoint_id: str
    condition: Callable[[ExecutionState], bool]
    description: str
    enabled: bool = True
    hit_count: int = 0
    last_hit: Optional[datetime] = None

    def check_hit(self, state: ExecutionState) -> bool:
        """Check if breakpoint condition is met."""
        if not self.enabled:
            return False

        try:
            if self.condition(state):
                self.hit_count += 1
                self.last_hit = datetime.now()
                return True
        except Exception as e:
            logging.warning(f"Breakpoint condition check failed: {e}")

        return False


@dataclass
class TimeTravelSession:
    """Complete time travel debugging session."""

    session_id: str
    trace: DecisionTrace
    execution_states: List[ExecutionState] = field(default_factory=list)
    breakpoints: List[DebugBreakpoint] = field(default_factory=list)
    current_step: int = 0
    is_replaying: bool = False
    bookmarks: Dict[str, int] = field(default_factory=dict)  # Named positions

    def get_current_state(self) -> Optional[ExecutionState]:
        """Get the current execution state."""
        if 0 <= self.current_step < len(self.execution_states):
            return self.execution_states[self.current_step]
        return None

    def can_step_forward(self) -> bool:
        """Check if we can step forward."""
        return self.current_step < len(self.execution_states) - 1

    def can_step_backward(self) -> bool:
        """Check if we can step backward."""
        return self.current_step > 0

    def step_forward(self) -> Optional[ExecutionState]:
        """Step forward one decision."""
        if self.can_step_forward():
            self.current_step += 1
            return self.get_current_state()
        return None

    def step_backward(self) -> Optional[ExecutionState]:
        """Step backward one decision."""
        if self.can_step_backward():
            self.current_step -= 1
            return self.get_current_state()
        return None

    def jump_to_step(self, step: int) -> Optional[ExecutionState]:
        """Jump to a specific decision step."""
        if 0 <= step < len(self.execution_states):
            self.current_step = step
            return self.get_current_state()
        return None

    def add_bookmark(self, name: str, step: int) -> bool:
        """Add a named bookmark at a specific step."""
        if 0 <= step < len(self.execution_states):
            self.bookmarks[name] = step
            return True
        return False

    def goto_bookmark(self, name: str) -> Optional[ExecutionState]:
        """Jump to a named bookmark."""
        if name in self.bookmarks:
            return self.jump_to_step(self.bookmarks[name])
        return None


class TimeTravelDebugger:
    """
    Advanced time travel debugging system for UCUP agents.

    Features:
    - Step-by-step execution replay with full state restoration
    - Breakpoint system with custom conditions
    - Alternative decision path exploration
    - Memory and performance analysis
    - Interactive debugging sessions
    """

    def __init__(self):
        self.sessions: Dict[str, TimeTravelSession] = {}
        self.logger = logging.getLogger(__name__)

    @validate_types
    def create_debug_session(
        self,
        trace: DecisionTrace,
        agent_state_recorder: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> str:
        """
        Create a new time travel debugging session.

        Args:
            trace: DecisionTrace from a completed agent execution
            agent_state_recorder: Function to capture agent state at each step

        Returns:
            Session ID for the debugging session
        """
        try:
            validate_non_empty_string(trace.session_id, "trace.session_id")
        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Create Session",
                    action="Validating trace session ID",
                    details=str(e),
                    suggestion="Provide a valid DecisionTrace with a non-empty session ID",
                )
            ) from e

        session_id = f"debug_{trace.session_id}_{uuid.uuid4().hex[:8]}"

        # Create execution states from the trace
        execution_states = []
        for i, decision in enumerate(trace.decisions):
            state = ExecutionState(
                timestamp=decision.timestamp,
                decision_step=i,
                agent_state=agent_state_recorder() if agent_state_recorder else {},
                context_snapshot=decision.context_snapshot,
                available_actions=decision.available_actions,
                chosen_action=decision.chosen_action,
                confidence_scores=decision.confidence_scores,
                reasoning_steps=decision.reasoning_steps,
                token_count=decision.token_usage or 0,
            )
            execution_states.append(state)

        session = TimeTravelSession(
            session_id=session_id, trace=trace, execution_states=execution_states
        )

        self.sessions[session_id] = session

        self.logger.info(f"Created time travel debug session: {session_id}")
        return session_id

    @validate_types
    def add_breakpoint(
        self,
        session_id: str,
        condition: Callable[[ExecutionState], bool],
        description: str,
        enabled: bool = True,
    ) -> str:
        """
        Add a breakpoint to a debugging session.

        Args:
            session_id: Debug session identifier
            condition: Function that returns True when breakpoint should trigger
            description: Human-readable description of the breakpoint
            enabled: Whether the breakpoint is initially enabled

        Returns:
            Breakpoint ID
        """
        try:
            validate_non_empty_string(session_id, "session_id")
            validate_non_empty_string(description, "description")
        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Add Breakpoint",
                    action="Validating breakpoint parameters",
                    details=str(e),
                    suggestion="Provide valid session ID, condition function, and description",
                )
            ) from e

        if session_id not in self.sessions:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Add Breakpoint",
                    action="Checking session existence",
                    details=f"Debug session '{session_id}' not found",
                    suggestion="Create a debug session before adding breakpoints",
                )
            )

        breakpoint_id = f"bp_{uuid.uuid4().hex[:8]}"
        breakpoint = DebugBreakpoint(
            breakpoint_id=breakpoint_id,
            condition=condition,
            description=description,
            enabled=enabled,
        )

        self.sessions[session_id].breakpoints.append(breakpoint)

        self.logger.info(f"Added breakpoint '{description}' to session {session_id}")
        return breakpoint_id

    @validate_types
    def replay_execution(
        self, session_id: str, start_step: int = 0, speed_multiplier: float = 1.0
    ) -> List[ExecutionState]:
        """
        Replay agent execution with time travel capabilities.

        Args:
            session_id: Debug session identifier
            start_step: Step to start replay from
            speed_multiplier: Speed multiplier for replay (1.0 = real-time)

        Returns:
            List of execution states encountered during replay
        """
        try:
            validate_non_empty_string(session_id, "session_id")
            validate_positive_number(start_step, "start_step")
            validate_positive_number(speed_multiplier, "speed_multiplier")

            if not isinstance(start_step, int):
                raise UCUPTypeError("start_step must be an integer")

        except (UCUPValueError, UCUPTypeError) as e:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Replay Execution",
                    action="Validating replay parameters",
                    details=str(e),
                    suggestion="Provide valid session ID, start step, and speed multiplier",
                )
            ) from e

        if session_id not in self.sessions:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Replay Execution",
                    action="Checking session existence",
                    details=f"Debug session '{session_id}' not found",
                    suggestion="Create a debug session before replaying execution",
                )
            )

        session = self.sessions[session_id]
        session.is_replaying = True

        states = []
        current_step = max(0, min(start_step, len(session.execution_states) - 1))

        while current_step < len(session.execution_states):
            state = session.execution_states[current_step]

            # Check breakpoints
            breakpoint_hit = False
            for bp in session.breakpoints:
                if bp.check_hit(state):
                    self.logger.info(
                        f"Breakpoint hit: {bp.description} at step {current_step}"
                    )
                    breakpoint_hit = True
                    # In a real implementation, this would pause execution
                    # For now, we just log and continue

            states.append(state)

            # Calculate timing for next step
            if current_step < len(session.execution_states) - 1:
                next_state = session.execution_states[current_step + 1]
                time_diff = (next_state.timestamp - state.timestamp).total_seconds()
                adjusted_delay = time_diff / speed_multiplier

                if adjusted_delay > 0:
                    # In real implementation, this would be an async delay
                    pass

            current_step += 1

        session.is_replaying = False
        self.logger.info(
            f"Replay completed for session {session_id}: {len(states)} steps"
        )
        return states

    @validate_types
    def analyze_alternative_paths(
        self, session_id: str, decision_step: int
    ) -> Dict[str, Any]:
        """
        Analyze alternative decision paths from a specific step.

        Args:
            session_id: Debug session identifier
            decision_step: Step to analyze alternatives for

        Returns:
            Analysis of alternative decision paths
        """
        try:
            validate_non_empty_string(session_id, "session_id")
            validate_positive_number(decision_step, "decision_step")

            if not isinstance(decision_step, int):
                raise UCUPTypeError("decision_step must be an integer")

        except (UCUPValueError, UCUPTypeError) as e:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Alternative Paths",
                    action="Validating analysis parameters",
                    details=str(e),
                    suggestion="Provide valid session ID and decision step",
                )
            ) from e

        if session_id not in self.sessions:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Alternative Paths",
                    action="Checking session existence",
                    details=f"Debug session '{session_id}' not found",
                    suggestion="Create a debug session before analyzing alternatives",
                )
            )

        session = self.sessions[session_id]

        if decision_step >= len(session.execution_states):
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Alternative Paths",
                    action="Validating decision step",
                    details=f"Decision step {decision_step} is out of range (max: {len(session.execution_states) - 1})",
                    suggestion="Choose a valid decision step within the session",
                )
            )

        state = session.execution_states[decision_step]

        # Analyze alternative paths
        alternatives = []
        for action_info in state.available_actions:
            action_name = action_info.get("action", "")
            if action_name and action_name != state.chosen_action:
                confidence = state.confidence_scores.get(action_name, 0)
                rejection_reason = None

                # Find rejection reason if available
                if hasattr(session.trace.decisions[decision_step], "rejection_reasons"):
                    rejection_reason = session.trace.decisions[
                        decision_step
                    ].rejection_reasons.get(action_name)

                alternatives.append(
                    {
                        "action": action_name,
                        "confidence": confidence,
                        "rejection_reason": rejection_reason,
                        "would_have_led_to": self._simulate_alternative_path(
                            session, decision_step, action_name
                        ),
                    }
                )

        return {
            "original_choice": state.chosen_action,
            "original_confidence": state.confidence_scores.get(state.chosen_action, 0),
            "decision_step": decision_step,
            "timestamp": state.timestamp.isoformat(),
            "alternatives": alternatives,
            "total_alternatives": len(alternatives),
        }

    def _simulate_alternative_path(
        self, session: TimeTravelSession, start_step: int, alternative_action: str
    ) -> Dict[str, Any]:
        """
        Simulate what might have happened if an alternative action was chosen.
        This is a simplified simulation for debugging purposes.
        """
        # In a real implementation, this would involve complex agent simulation
        # For now, return a placeholder with estimated outcomes

        return {
            "estimated_confidence_impact": "unknown",  # Would need ML model
            "estimated_outcome": "simulation_not_implemented",  # Would need agent re-run
            "risk_assessment": "high",  # Placeholder
            "recommended": False,  # Placeholder
        }

    @validate_types
    def create_performance_profile(self, session_id: str) -> Dict[str, Any]:
        """
        Create a comprehensive performance profile for a debug session.

        Args:
            session_id: Debug session identifier

        Returns:
            Performance profile data
        """
        try:
            validate_non_empty_string(session_id, "session_id")
        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Performance Profile",
                    action="Validating session ID",
                    details=str(e),
                    suggestion="Provide a valid debug session ID",
                )
            ) from e

        if session_id not in self.sessions:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Performance Profile",
                    action="Checking session existence",
                    details=f"Debug session '{session_id}' not found",
                    suggestion="Create a debug session before generating performance profile",
                )
            )

        session = self.sessions[session_id]
        states = session.execution_states

        if not states:
            return {"error": "No execution states available"}

        # Calculate performance metrics
        total_execution_time = sum(state.execution_time for state in states)
        total_tokens = sum(state.token_count for state in states)
        peak_memory = max((state.memory_usage for state in states), default=0)

        decision_times = []
        confidence_scores = []

        for state in states:
            if state.execution_time > 0:
                decision_times.append(state.execution_time)
            if state.chosen_action and state.confidence_scores:
                confidence_scores.append(
                    state.confidence_scores.get(state.chosen_action, 0)
                )

        # Calculate trends
        confidence_trend = "stable"
        if len(confidence_scores) > 3:
            first_half = confidence_scores[: len(confidence_scores) // 2]
            second_half = confidence_scores[len(confidence_scores) // 2 :]
            if second_half and first_half:
                trend_ratio = (
                    sum(second_half)
                    / len(second_half)
                    / (sum(first_half) / len(first_half))
                )
                if trend_ratio > 1.1:
                    confidence_trend = "improving"
                elif trend_ratio < 0.9:
                    confidence_trend = "declining"

        return {
            "session_id": session_id,
            "total_steps": len(states),
            "total_execution_time": total_execution_time,
            "average_decision_time": sum(decision_times) / len(decision_times)
            if decision_times
            else 0,
            "total_tokens": total_tokens,
            "peak_memory_usage": peak_memory,
            "average_confidence": sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0,
            "confidence_trend": confidence_trend,
            "decision_efficiency": len([s for s in states if s.chosen_action])
            / len(states)
            if states
            else 0,
            "breakpoints_hit": sum(bp.hit_count for bp in session.breakpoints),
        }

    @validate_types
    def export_debug_session(self, session_id: str, format: str = "json") -> str:
        """
        Export a debug session for external analysis.

        Args:
            session_id: Debug session identifier
            format: Export format ('json', 'csv', 'html')

        Returns:
            Exported data as string
        """
        try:
            validate_non_empty_string(session_id, "session_id")
            if format not in ["json", "csv", "html"]:
                raise UCUPValueError(f"Unsupported export format: {format}")

        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Export Session",
                    action="Validating export parameters",
                    details=str(e),
                    suggestion="Provide valid session ID and supported format (json, csv, html)",
                )
            ) from e

        if session_id not in self.sessions:
            raise UCUPValueError(
                create_error_message(
                    context="TimeTravelDebugger Export Session",
                    action="Checking session existence",
                    details=f"Debug session '{session_id}' not found",
                    suggestion="Create a debug session before exporting",
                )
            )

        session = self.sessions[session_id]

        if format == "json":
            return json.dumps(
                {
                    "session_id": session.session_id,
                    "trace_session_id": session.trace.session_id,
                    "execution_states": [
                        state.to_dict() for state in session.execution_states
                    ],
                    "breakpoints": [
                        {
                            "id": bp.breakpoint_id,
                            "description": bp.description,
                            "enabled": bp.enabled,
                            "hit_count": bp.hit_count,
                            "last_hit": bp.last_hit.isoformat()
                            if bp.last_hit
                            else None,
                        }
                        for bp in session.breakpoints
                    ],
                    "current_step": session.current_step,
                    "bookmarks": session.bookmarks,
                    "export_timestamp": datetime.now().isoformat(),
                },
                indent=2,
            )

        elif format == "csv":
            # Simplified CSV export of key metrics
            lines = [
                "step,timestamp,chosen_action,confidence,token_count,execution_time"
            ]
            for state in session.execution_states:
                confidence = state.confidence_scores.get(state.chosen_action or "", 0)
                lines.append(
                    f"{state.decision_step},{state.timestamp.isoformat()},{state.chosen_action},{confidence},{state.token_count},{state.execution_time}"
                )
            return "\n".join(lines)

        elif format == "html":
            # Basic HTML report
            html = f"""
            <html>
            <head><title>UCUP Debug Session: {session_id}</title></head>
            <body>
                <h1>UCUP Debug Session Report</h1>
                <p><strong>Session ID:</strong> {session.session_id}</p>
                <p><strong>Steps:</strong> {len(session.execution_states)}</p>
                <p><strong>Breakpoints:</strong> {len(session.breakpoints)}</p>
                <h2>Execution States</h2>
                <table border="1">
                    <tr><th>Step</th><th>Action</th><th>Confidence</th><th>Tokens</th></tr>
            """

            for state in session.execution_states:
                confidence = state.confidence_scores.get(state.chosen_action or "", 0)
                html += f"<tr><td>{state.decision_step}</td><td>{state.chosen_action}</td><td>{confidence:.2f}</td><td>{state.token_count}</td></tr>"

            html += "</table></body></html>"
            return html

        return ""
