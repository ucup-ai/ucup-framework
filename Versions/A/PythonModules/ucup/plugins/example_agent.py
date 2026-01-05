"""
Example Agent Plugin for UCUP Framework

This plugin demonstrates how to create custom agents that extend UCUP's capabilities.
"""

from typing import Any, Dict, Set

from ..plugins import AgentPlugin, PluginMetadata
from ..probabilistic import ProbabilisticAgent, ProbabilisticResult


class CustomerServiceAgentPlugin(AgentPlugin):
    """
    Example customer service agent plugin.

    This plugin creates agents specialized in customer service interactions
    with domain-specific reasoning and response strategies.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="customer_service_agent",
            version="1.0.0",
            description="Specialized customer service agent with sentiment analysis",
            author="UCUP Team",
            dependencies=["nltk"],  # Would need sentiment analysis library
            config_schema={
                "type": "object",
                "properties": {
                    "sentiment_threshold": {"type": "number", "default": 0.5},
                    "escalation_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "response_templates": {"type": "object"},
                },
            },
        )

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the plugin."""
        self.config = config or {}
        self.sentiment_threshold = self.config.get("sentiment_threshold", 0.5)
        self.escalation_keywords = set(
            self.config.get("escalation_keywords", ["urgent", "escalate", "manager"])
        )
        self.response_templates = self.config.get("response_templates", {})
        return True

    def shutdown(self) -> bool:
        """Clean up plugin resources."""
        return True

    def create_agent(self, config: Dict[str, Any]) -> ProbabilisticAgent:
        """Create and return a configured customer service agent."""
        return CustomerServiceAgent(
            sentiment_threshold=self.sentiment_threshold,
            escalation_keywords=self.escalation_keywords,
            response_templates=self.response_templates,
            **config,
        )

    def get_supported_capabilities(self) -> Set[str]:
        """Return supported capabilities."""
        return {"customer_service", "sentiment_analysis", "escalation_detection"}

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_keys = ["llm"]
        return all(key in config for key in required_keys)


class CustomerServiceAgent(ProbabilisticAgent):
    """
    Customer service agent with specialized reasoning for customer interactions.
    """

    def __init__(
        self,
        sentiment_threshold: float = 0.5,
        escalation_keywords: Set[str] = None,
        response_templates: Dict[str, str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sentiment_threshold = sentiment_threshold
        self.escalation_keywords = escalation_keywords or set()
        self.response_templates = response_templates or {}

    async def execute(self, task: str, **kwargs) -> ProbabilisticResult:
        """Execute customer service task with specialized reasoning."""

        # Analyze sentiment and urgency
        sentiment_score = self._analyze_sentiment(task)
        urgency_level = self._assess_urgency(task)

        # Choose reasoning strategy based on context
        if urgency_level > 0.8:
            reasoning_strategy = "chain_of_thought"  # Quick, systematic response
            confidence_adjustment = -0.1  # More conservative for urgent issues
        elif sentiment_score < self.sentiment_threshold:
            reasoning_strategy = (
                "step_back_questioning"  # Careful for negative sentiment
            )
            confidence_adjustment = -0.2
        else:
            reasoning_strategy = "chain_of_thought"
            confidence_adjustment = 0.0

        # Generate response
        result, base_confidence = await self._generate_with_confidence(
            task, reasoning_strategy
        )

        # Apply confidence adjustments
        adjusted_confidence = min(
            1.0, max(0.0, base_confidence + confidence_adjustment)
        )

        # Check for low confidence workflow
        if adjusted_confidence < self.min_confidence_threshold:
            return await self.low_confidence_workflow(result, adjusted_confidence, task)

        # Generate alternatives
        alternatives = self.get_alternative_interpretations(task)

        # Create metadata
        metadata = {
            "sentiment_score": sentiment_score,
            "urgency_level": urgency_level,
            "reasoning_strategy": reasoning_strategy,
            "escalation_recommended": self._should_escalate(task),
            "response_category": self._categorize_response(result),
        }

        return ProbabilisticResult(
            value=self._format_customer_response(result, task),
            confidence=adjusted_confidence,
            alternatives=alternatives,
            metadata=metadata,
        )

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (would use NLP library in practice)."""
        negative_words = [
            "unhappy",
            "angry",
            "frustrated",
            "problem",
            "issue",
            "complaint",
        ]
        positive_words = [
            "happy",
            "satisfied",
            "great",
            "excellent",
            "thank",
            "appreciate",
        ]

        text_lower = text.lower()
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return 0.5

        # Normalize sentiment score (0.0 = very negative, 1.0 = very positive)
        sentiment_score = (positive_count - negative_count) / max(1, total_words / 10)
        return min(1.0, max(0.0, 0.5 + sentiment_score))

    def _assess_urgency(self, text: str) -> float:
        """Assess urgency level from text."""
        urgent_keywords = ["urgent", "asap", "immediately", "emergency", "critical"]
        text_lower = text.lower()

        urgency_score = sum(1 for keyword in urgent_keywords if keyword in text_lower)
        return min(1.0, urgency_score / 2.0)  # Normalize

    def _should_escalate(self, text: str) -> bool:
        """Determine if issue should be escalated."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.escalation_keywords)

    def _categorize_response(self, response: str) -> str:
        """Categorize the response type."""
        response_lower = response.lower()

        if any(word in response_lower for word in ["refund", "credit", "compensation"]):
            return "compensation"
        elif any(
            word in response_lower for word in ["escalate", "manager", "supervisor"]
        ):
            return "escalation"
        elif any(word in response_lower for word in ["sorry", "apology", "regret"]):
            return "apology"
        else:
            return "information"

    def _format_customer_response(self, response: str, task: str) -> str:
        """Format response with customer service appropriate language."""
        confidence_indicator = (
            "🤔" if getattr(self.state, "task_completion_confidence", 1.0) < 0.7 else "✅"
        )

        formatted = f"{confidence_indicator} {response}"

        # Add empathy for negative sentiment tasks
        if self._analyze_sentiment(task) < self.sentiment_threshold:
            formatted = "I understand your concern. " + formatted

        return formatted
