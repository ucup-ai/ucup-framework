"""
TOON (Token-Oriented Object Notation) Library for UCUP

TOON is a compact, human-readable encoding of JSON that minimizes LLM tokens
while maintaining readability and structure. This library provides comprehensive
TOON integration for UCUP to help users reduce token usage and optimize costs.

Key Features:
- TOON serialization/deserialization with JSON compatibility
- Automatic token optimization recommendations
- Schema-aware formatting for structured data
- Cost savings calculations and reporting
- Integration with UCUP observability and debugging
- Educational tools and best practices
"""

from .toon_formatter import (
    TokenMetrics,
    ToonConversionResult,
    ToonFormatter,
    TOONOptimizer,
    ToonSchema,
)

__all__ = [
    "ToonFormatter",
    "ToonSchema",
    "ToonConversionResult",
    "TokenMetrics",
    "TOONOptimizer",
]
