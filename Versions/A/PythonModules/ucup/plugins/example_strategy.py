"""
Example Strategy Plugin for UCUP Framework

This plugin demonstrates how to create custom reasoning strategies and coordination patterns.
"""

from typing import Any, Dict

from ..plugins import PluginMetadata, StrategyPlugin


class CreativeBrainstormingStrategyPlugin(StrategyPlugin):
    """
    Creative brainstorming strategy plugin.

    This strategy uses divergent thinking techniques to generate multiple
    creative solutions before converging on the best approach.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="creative_brainstorming",
            version="1.0.0",
            description="Divergent thinking strategy for creative problem solving",
            author="UCUP Team",
            config_schema={
                "type": "object",
                "properties": {
                    "divergence_rounds": {"type": "integer", "default": 3},
                    "convergence_criteria": {
                        "type": "string",
                        "enum": ["vote", "score", "consensus"],
                    },
                    "creativity_techniques": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        )

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the plugin."""
        self.config = config or {}
        self.divergence_rounds = self.config.get("divergence_rounds", 3)
        self.convergence_criteria = self.config.get("convergence_criteria", "vote")
        self.creativity_techniques = self.config.get(
            "creativity_techniques",
            ["random_association", "question_storming", "analogy_mapping"],
        )
        return True

    def shutdown(self) -> bool:
        """Clean up plugin resources."""
        return True

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "creative_brainstorming"

    def get_strategy_type(self) -> str:
        """Return strategy type."""
        return "reasoning"

    def execute_strategy(self, context: Dict[str, Any]) -> Any:
        """Execute creative brainstorming strategy."""
        problem = context.get("problem", "")
        constraints = context.get("constraints", [])

        # Divergence phase: generate multiple creative solutions
        ideas = []
        for round_num in range(self.divergence_rounds):
            round_ideas = self._divergence_phase(problem, constraints, round_num)
            ideas.extend(round_ideas)

        # Convergence phase: select best ideas
        selected_ideas = self._convergence_phase(ideas, context)

        # Synthesis phase: combine selected ideas
        final_solution = self._synthesis_phase(selected_ideas, problem)

        return {
            "solution": final_solution,
            "all_ideas_generated": len(ideas),
            "selected_ideas": len(selected_ideas),
            "creativity_techniques_used": self.creativity_techniques,
            "confidence": self._calculate_solution_confidence(
                selected_ideas, final_solution
            ),
        }

    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this strategy is applicable."""
        # Applicable for creative tasks, innovation problems, design challenges
        task_type = context.get("task_type", "").lower()
        applicable_types = [
            "creative",
            "design",
            "innovation",
            "brainstorm",
            "ideation",
        ]

        return any(app_type in task_type for app_type in applicable_types)

    def _divergence_phase(
        self, problem: str, constraints: list, round_num: int
    ) -> list:
        """Generate diverse ideas using different creativity techniques."""
        ideas = []

        for technique in self.creativity_techniques:
            if technique == "random_association":
                ideas.extend(self._random_association_technique(problem, 5))
            elif technique == "question_storming":
                ideas.extend(self._question_storming_technique(problem, 5))
            elif technique == "analogy_mapping":
                ideas.extend(self._analogy_mapping_technique(problem, 3))

        return ideas

    def _random_association_technique(self, problem: str, num_ideas: int) -> list:
        """Generate ideas through random word association."""
        # In practice, this would use a word database or LLM
        random_words = [
            "cloud",
            "river",
            "mountain",
            "lightning",
            "ocean",
            "forest",
            "star",
            "crystal",
        ]

        ideas = []
        for i in range(num_ideas):
            random_word = random_words[i % len(random_words)]
            idea = f"Combine {problem} with {random_word} concept"
            ideas.append(
                {
                    "idea": idea,
                    "technique": "random_association",
                    "creativity_score": 0.7 + (i * 0.1),  # Mock creativity score
                    "feasibility_score": 0.5 + (i * 0.05),  # Mock feasibility score
                }
            )

        return ideas

    def _question_storming_technique(self, problem: str, num_ideas: int) -> list:
        """Generate ideas by asking provocative questions."""
        question_templates = [
            "What if {} were impossible?",
            "What would {} look like in reverse?",
            "How would a child approach {}?",
            "What if {} had superpowers?",
            "How can {} be made fun?",
        ]

        ideas = []
        for i in range(min(num_ideas, len(question_templates))):
            question = question_templates[i].format(problem)
            idea = f"Answer: {question} - This suggests..."
            ideas.append(
                {
                    "idea": idea,
                    "technique": "question_storming",
                    "creativity_score": 0.8 + (i * 0.05),
                    "feasibility_score": 0.4 + (i * 0.08),
                }
            )

        return ideas

    def _analogy_mapping_technique(self, problem: str, num_ideas: int) -> list:
        """Generate ideas by mapping analogies from different domains."""
        analogies = [
            ("bird", "flying", "How can we make {} soar?"),
            ("river", "flowing", "How can {} flow naturally?"),
            ("tree", "growing", "How can {} grow organically?"),
            ("spider", "weaving", "How can {} create intricate connections?"),
        ]

        ideas = []
        for i in range(min(num_ideas, len(analogies))):
            source, characteristic, template = analogies[i]
            idea = (
                template.format(problem) + f" (Inspired by {source}'s {characteristic})"
            )
            ideas.append(
                {
                    "idea": idea,
                    "technique": "analogy_mapping",
                    "creativity_score": 0.9 + (i * 0.02),
                    "feasibility_score": 0.6 + (i * 0.03),
                }
            )

        return ideas

    def _convergence_phase(self, ideas: list, context: Dict[str, Any]) -> list:
        """Select the most promising ideas."""
        if self.convergence_criteria == "vote":
            # Simple voting: select top N by average score
            scored_ideas = [
                (idea, (idea["creativity_score"] + idea["feasibility_score"]) / 2)
                for idea in ideas
            ]
            scored_ideas.sort(key=lambda x: x[1], reverse=True)
            return [idea for idea, score in scored_ideas[:5]]  # Top 5

        elif self.convergence_criteria == "score":
            # Score-based selection with thresholds
            return [
                idea
                for idea in ideas
                if idea["creativity_score"] > 0.7 and idea["feasibility_score"] > 0.5
            ]

        else:  # consensus
            # Mock consensus: select ideas that appear multiple times or are highly rated
            return ideas[:3]  # Just take first 3

    def _synthesis_phase(self, selected_ideas: list, problem: str) -> str:
        """Synthesize selected ideas into a final solution."""
        if not selected_ideas:
            return f"No creative solution found for: {problem}"

        # Combine the best elements from selected ideas
        synthesis_parts = []
        for idea in selected_ideas[:3]:  # Use top 3
            synthesis_parts.append(idea["idea"].split("-")[0].strip())

        combined_solution = "; ".join(synthesis_parts)
        return f"Creative solution combining: {combined_solution}"

    def _calculate_solution_confidence(
        self, selected_ideas: list, final_solution: str
    ) -> float:
        """Calculate confidence in the final solution."""
        if not selected_ideas:
            return 0.0

        # Average creativity and feasibility scores
        avg_creativity = sum(idea["creativity_score"] for idea in selected_ideas) / len(
            selected_ideas
        )
        avg_feasibility = sum(
            idea["feasibility_score"] for idea in selected_ideas
        ) / len(selected_ideas)

        # Diversity bonus: more diverse techniques = higher confidence
        techniques_used = set(idea["technique"] for idea in selected_ideas)
        diversity_bonus = min(0.2, len(techniques_used) * 0.05)

        confidence = avg_creativity * 0.6 + avg_feasibility * 0.3 + diversity_bonus
        return min(1.0, confidence)
