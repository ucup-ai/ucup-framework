"""
Probabilistic Agent for UCUP.

Intelligent agent that uses probabilistic reasoning for decision making.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from .engine import ProbabilisticEngine
from ..core.manager import UCUPManager


class ProbabilisticResult:
    """Container for probabilistic reasoning results."""

    def __init__(self, probability: float, confidence_interval: List[float],
                 alternative_paths: List[Dict[str, Any]], uncertainty_measure: float,
                 processing_method: str = "unknown"):
        self.probability = probability
        self.confidence_interval = confidence_interval
        self.alternative_paths = alternative_paths
        self.uncertainty_measure = uncertainty_measure
        self.processing_method = processing_method
        self.timestamp = time.time()

    def __repr__(self) -> str:
        return (".3f")

    def is_confident(self, threshold: float = 0.8) -> bool:
        """Check if the result meets confidence threshold."""
        return self.probability >= threshold

    def get_best_alternative(self) -> Optional[Dict[str, Any]]:
        """Get the highest probability alternative path."""
        if not self.alternative_paths:
            return None
        return max(self.alternative_paths, key=lambda x: x.get("probability", 0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "probability": self.probability,
            "confidence_interval": self.confidence_interval,
            "alternative_paths": self.alternative_paths,
            "uncertainty_measure": self.uncertainty_measure,
            "processing_method": self.processing_method,
            "timestamp": self.timestamp
        }


class ProbabilisticAgent:
    """Intelligent agent using probabilistic reasoning."""

    def __init__(self, manager: UCUPManager, model_path: Optional[str] = None):
        self.manager = manager
        self.engine = manager.get_probabilistic_engine()
        self.model_path = model_path
        self.decision_history: List[Dict[str, Any]] = []
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7

    def make_decision(self, context: Dict[str, Any], options: List[Dict[str, Any]]) -> ProbabilisticResult:
        """
        Make a decision using probabilistic reasoning.

        Args:
            context: Current context information
            options: Available decision options

        Returns:
            ProbabilisticResult with decision recommendation
        """
        # Prepare input data for probabilistic engine
        input_data = self._prepare_input_data(context, options)

        # Get probabilistic prediction
        prediction = self.engine.predict(input_data)

        # Create result object
        result = ProbabilisticResult(
            probability=prediction.get("probability", 0.5),
            confidence_interval=prediction.get("confidence_interval", [0.4, 0.6]),
            alternative_paths=prediction.get("alternative_paths", []),
            uncertainty_measure=prediction.get("uncertainty_measure", 0.2),
            processing_method=prediction.get("processing_method", "probabilistic_agent")
        )

        # Store decision in history
        self._record_decision(context, options, result)

        return result

    def evaluate_scenario(self, scenario: Dict[str, Any]) -> ProbabilisticResult:
        """
        Evaluate a specific scenario probabilistically.

        Args:
            scenario: Scenario description with parameters

        Returns:
            ProbabilisticResult for the scenario
        """
        # Extract scenario parameters
        context = scenario.get("context", {})
        parameters = scenario.get("parameters", {})

        # Combine context and parameters
        input_data = {**context, **parameters}

        # Make probabilistic prediction
        prediction = self.engine.predict(input_data)

        return ProbabilisticResult(
            probability=prediction.get("probability", 0.5),
            confidence_interval=prediction.get("confidence_interval", [0.4, 0.6]),
            alternative_paths=prediction.get("alternative_paths", []),
            uncertainty_measure=prediction.get("uncertainty_measure", 0.2),
            processing_method="scenario_evaluation"
        )

    def optimize_decision(self, options: List[Dict[str, Any]],
                         constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize decision making with constraints.

        Args:
            options: Decision options to evaluate
            constraints: Optional constraints on the decision

        Returns:
            Optimized decision recommendation
        """
        return self.engine.optimize_decision(options, constraints)

    def learn_from_feedback(self, decision_id: str, actual_outcome: float,
                          feedback: Optional[Dict[str, Any]] = None):
        """
        Learn from decision outcomes to improve future predictions.

        Args:
            decision_id: ID of the decision to learn from
            actual_outcome: Actual outcome (0.0 to 1.0)
            feedback: Additional feedback information
        """
        # Find the decision in history
        decision = None
        for d in self.decision_history:
            if d.get("id") == decision_id:
                decision = d
                break

        if not decision:
            print(f"Warning: Decision {decision_id} not found in history")
            return

        # Calculate prediction error
        predicted_probability = decision["result"].probability
        prediction_error = abs(predicted_probability - actual_outcome)

        # Update learning parameters (simple adaptive learning)
        self.confidence_threshold = max(0.5, self.confidence_threshold - self.learning_rate * prediction_error)

        # Store feedback for future reference
        decision["feedback"] = {
            "actual_outcome": actual_outcome,
            "prediction_error": prediction_error,
            "feedback_data": feedback or {},
            "learning_timestamp": time.time()
        }

    def get_decision_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get decision history."""
        history = self.decision_history
        if limit:
            history = history[-limit:]
        return history

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze agent performance over time."""
        if not self.decision_history:
            return {"error": "No decision history available"}

        total_decisions = len(self.decision_history)
        confident_decisions = sum(1 for d in self.decision_history
                                if d["result"].is_confident(self.confidence_threshold))

        feedback_count = sum(1 for d in self.decision_history if "feedback" in d)
        if feedback_count > 0:
            avg_prediction_error = sum(d["feedback"]["prediction_error"]
                                     for d in self.decision_history if "feedback" in d) / feedback_count
        else:
            avg_prediction_error = None

        return {
            "total_decisions": total_decisions,
            "confident_decisions": confident_decisions,
            "confidence_ratio": confident_decisions / total_decisions if total_decisions > 0 else 0,
            "decisions_with_feedback": feedback_count,
            "average_prediction_error": avg_prediction_error,
            "current_confidence_threshold": self.confidence_threshold
        }

    def _prepare_input_data(self, context: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare input data for probabilistic engine."""
        # Extract relevant features from context
        input_data = {}

        # Add context features
        for key, value in context.items():
            if isinstance(value, (int, float)):
                input_data[f"context_{key}"] = float(value)
            elif isinstance(value, str):
                # Convert strings to numeric features
                input_data[f"context_{key}_hash"] = hash(value) % 1000
            elif isinstance(value, bool):
                input_data[f"context_{key}"] = 1.0 if value else 0.0

        # Add option count as a feature
        input_data["option_count"] = len(options)

        # Add time-based features
        current_time = time.time()
        input_data["time_of_day"] = (current_time % 86400) / 3600  # Hour of day
        input_data["day_of_week"] = ((current_time // 86400) + 4) % 7  # Day of week

        return input_data

    def _record_decision(self, context: Dict[str, Any], options: List[Dict[str, Any]],
                        result: ProbabilisticResult):
        """Record decision in history."""
        decision_record = {
            "id": f"decision_{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "context": context.copy(),
            "options_count": len(options),
            "result": result,
            "confidence_threshold": self.confidence_threshold
        }

        self.decision_history.append(decision_record)

        # Limit history size to prevent memory issues
        max_history = self.manager.config.get("probabilistic.max_history_size", 1000)
        if len(self.decision_history) > max_history:
            self.decision_history = self.decision_history[-max_history:]

    def reset(self):
        """Reset agent state."""
        self.decision_history.clear()
        self.confidence_threshold = 0.7

    def __repr__(self) -> str:
        return f"ProbabilisticAgent(decisions={len(self.decision_history)}, confidence_threshold={self.confidence_threshold:.2f})"
