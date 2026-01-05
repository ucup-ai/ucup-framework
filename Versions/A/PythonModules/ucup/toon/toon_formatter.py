"""
TOON (Token-Oriented Object Notation) Integration for UCUP Framework.

TOON is a compact, human-readable encoding of JSON that minimizes LLM tokens.
This module provides comprehensive TOON integration for UCUP to help users
reduce token usage and optimize payloads.

Copyright (c) 2025 UCUP Framework Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Key Features:
- TOON serialization/deserialization with JSON compatibility
- Automatic token optimization recommendations
- Schema-aware formatting for structured data
- Benchmarking against JSON for token savings
- Educational tools and best practices
- Integration with UCUP observability and debugging
"""

import json
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .validation import (
    UCUPValidationError,
    UCUPValueError,
    create_error_message,
    validate_non_empty_string,
    validate_types,
)


@dataclass
class TokenMetrics:
    """Metrics for comparing JSON vs TOON token usage."""

    json_tokens: int = 0
    toon_tokens: int = 0
    savings_percentage: float = field(init=False)
    compression_ratio: float = field(init=False)
    estimated_cost_savings: float = field(init=False)  # In USD for common LLM rates

    def __post_init__(self):
        """Calculate metrics after initialization."""
        self.calculate_savings()

    def calculate_savings(self):
        """Calculate token savings metrics."""
        if self.json_tokens > 0:
            self.savings_percentage = (
                (self.json_tokens - self.toon_tokens) / self.json_tokens
            ) * 100
            self.compression_ratio = (
                self.json_tokens / self.toon_tokens if self.toon_tokens > 0 else 1.0
            )
            # Estimate cost savings (rough approximation: $0.002 per 1K tokens)
            self.estimated_cost_savings = (
                self.json_tokens - self.toon_tokens
            ) * 0.000002
        else:
            self.savings_percentage = 0.0
            self.compression_ratio = 1.0
            self.estimated_cost_savings = 0.0


@dataclass
class ToonSchema:
    """Schema definition for TOON formatting."""

    name: str
    description: str
    fields: List[str]
    array_fields: List[str] = field(default_factory=list)
    nested_objects: Dict[str, "ToonSchema"] = field(default_factory=dict)
    optimizations: List[str] = field(default_factory=list)


@dataclass
class ToonConversionResult:
    """Result of JSON to TOON conversion."""

    original_json: str
    toon_output: str
    json_output: str  # Formatted JSON for comparison
    schema_used: Optional[ToonSchema]
    metrics: TokenMetrics
    recommendations: List[str]
    conversion_time: float
    format_choice: str = "auto"  # "toon", "json", or "auto"


class ToonFormatter:
    """
    Main TOON formatter for UCUP with token optimization.

    Features:
    - JSON to TOON conversion with automatic optimization
    - Schema-aware formatting for structured data
    - Token usage analysis and benchmarking
    - Educational recommendations and best practices
    - Integration with UCUP debugging and observability
    """

    def __init__(self):
        self.schemas: Dict[str, ToonSchema] = {}
        self.conversion_history: List[ToonConversionResult] = []
        self.logger = None  # Will be set by UCUP framework

        # Initialize with common UCUP schemas
        self._initialize_ucup_schemas()

    def _initialize_ucup_schemas(self):
        """Initialize common schemas for UCUP data structures."""

        # Decision trace schema
        decision_schema = ToonSchema(
            name="decision_trace",
            description="UCUP decision trace format",
            fields=["session_id", "decisions", "total_tokens", "successful_decisions"],
            array_fields=["decisions"],
            optimizations=["compact_arrays", "omit_nulls", "short_field_names"],
        )

        # Probabilistic analysis schema
        prob_schema = ToonSchema(
            name="probabilistic_analysis",
            description="Confidence scores and uncertainty metrics",
            fields=[
                "confidence_scores",
                "entropy",
                "decision_pressure",
                "risk_indicators",
            ],
            array_fields=["confidence_scores", "risk_indicators"],
            optimizations=["numeric_precision", "omit_defaults"],
        )

        # Agent state schema
        agent_schema = ToonSchema(
            name="agent_state",
            description="Agent execution state snapshot",
            fields=["agent_id", "current_task", "memory_usage", "confidence_level"],
            optimizations=["compact_nested", "omit_empty_arrays"],
        )

        self.schemas.update(
            {
                "decision_trace": decision_schema,
                "probabilistic_analysis": prob_schema,
                "agent_state": agent_schema,
            }
        )

    @validate_types
    def format_with_choice(
        self,
        data: Union[str, Dict, List],
        preferred_format: str = "auto",
        schema_name: Optional[str] = None,
        show_comparison: bool = True,
    ) -> ToonConversionResult:
        """
        Format data with user choice between TOON and JSON.

        Args:
            data: Data to format (JSON string, dict, or list)
            preferred_format: "toon", "json", or "auto" (auto chooses best based on savings)
            schema_name: Optional schema for TOON optimization
            show_comparison: Whether to include format comparison in result

        Returns:
            ToonConversionResult with chosen format and comparison data
        """
        import time

        start_time = time.time()

        try:
            # Parse input data
            if isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                    original_json = data
                except json.JSONDecodeError as e:
                    raise UCUPValueError(f"Invalid JSON input: {e}")
            else:
                parsed_data = data
                original_json = json.dumps(parsed_data, indent=2)

            # Create formatted JSON version
            json_output = json.dumps(parsed_data, indent=2)

            # Get schema and convert to TOON
            schema = self.schemas.get(schema_name) if schema_name else None
            toon_output = self._convert_to_toon(parsed_data, schema, optimize=True)

            # Calculate metrics
            json_tokens = self._estimate_tokens(json_output)
            toon_tokens = self._estimate_tokens(toon_output)

            metrics = TokenMetrics(json_tokens=json_tokens, toon_tokens=toon_tokens)
            metrics.calculate_savings()

            # Validate and determine best format based on preference and savings
            valid_formats = ["auto", "toon", "json"]
            if preferred_format not in valid_formats:
                raise UCUPValidationError(
                    create_error_message(
                        context="TOON Formatter Format Choice",
                        action="Validating format preference",
                        details=f"Invalid format: {preferred_format}",
                        suggestion=f"Use one of: {', '.join(valid_formats)}",
                    )
                )

            if preferred_format == "auto":
                # Auto-select based on token savings
                chosen_format = "toon" if metrics.savings_percentage > 15 else "json"
                reasoning = (
                    f"Auto-selected {chosen_format} (saves {metrics.savings_percentage:.1f}% tokens)"
                    if chosen_format == "toon"
                    else f"Auto-selected JSON (only {metrics.savings_percentage:.1f}% savings)"
                )
            else:
                chosen_format = preferred_format
                reasoning = f"User selected {preferred_format} format"

            # Generate recommendations
            recommendations = self._generate_format_recommendations(
                parsed_data, metrics, schema, chosen_format, preferred_format
            )

            result = ToonConversionResult(
                original_json=original_json,
                toon_output=toon_output,
                json_output=json_output,
                schema_used=schema,
                metrics=metrics,
                recommendations=[reasoning] + recommendations,
                conversion_time=time.time() - start_time,
                format_choice=chosen_format,
            )

            self.conversion_history.append(result)
            return result

        except Exception as e:
            raise UCUPValueError(
                create_error_message(
                    context="TOON Formatter Format Choice",
                    action="Formatting data with user choice",
                    details=str(e),
                    suggestion="Ensure valid data format and format preference",
                )
            ) from e

    @validate_types
    def json_to_toon(
        self,
        json_data: Union[str, Dict, List],
        schema_name: Optional[str] = None,
        optimize: bool = True,
    ) -> ToonConversionResult:
        """
        Convert JSON data to TOON format with optimization.

        Args:
            json_data: JSON string, dict, or list to convert
            schema_name: Optional schema to use for optimization
            optimize: Whether to apply token optimizations

        Returns:
            ToonConversionResult with conversion details and metrics
        """
        import time

        start_time = time.time()

        try:
            # Parse input to dict/list if it's a string
            if isinstance(json_data, str):
                try:
                    parsed_data = json.loads(json_data)
                    original_json = json_data
                except json.JSONDecodeError as e:
                    raise UCUPValueError(f"Invalid JSON input: {e}")
            else:
                parsed_data = json_data
                original_json = json.dumps(parsed_data, indent=2)

            # Get schema if specified
            schema = self.schemas.get(schema_name) if schema_name else None

            # Convert to TOON
            toon_output = self._convert_to_toon(parsed_data, schema, optimize)

            # Create formatted JSON for comparison
            json_output = json.dumps(parsed_data, indent=2)

            # Calculate metrics
            json_tokens = self._estimate_tokens(original_json)
            toon_tokens = self._estimate_tokens(toon_output)

            metrics = TokenMetrics(json_tokens=json_tokens, toon_tokens=toon_tokens)
            metrics.calculate_savings()

            # Generate recommendations
            recommendations = self._generate_toon_recommendations(
                parsed_data, metrics, schema
            )

            result = ToonConversionResult(
                original_json=original_json,
                toon_output=toon_output,
                json_output=json_output,
                schema_used=schema,
                metrics=metrics,
                recommendations=recommendations,
                conversion_time=time.time() - start_time,
                format_choice="toon",
            )

            self.conversion_history.append(result)
            return result

        except Exception as e:
            raise UCUPValueError(
                create_error_message(
                    context="TOON Formatter JSON to TOON",
                    action="Converting data to TOON format",
                    details=str(e),
                    suggestion="Ensure input is valid JSON and schema exists if specified",
                )
            ) from e

    def _convert_to_toon(
        self,
        data: Union[Dict, List],
        schema: Optional[ToonSchema] = None,
        optimize: bool = True,
    ) -> str:
        """Convert Python data structure to TOON format."""
        if isinstance(data, list):
            return self._convert_array_to_toon(data, schema, optimize)
        elif isinstance(data, dict):
            return self._convert_object_to_toon(data, schema, optimize)
        else:
            return str(data)

    def _convert_object_to_toon(
        self, obj: Dict, schema: Optional[ToonSchema] = None, optimize: bool = True
    ) -> str:
        """Convert dictionary to TOON object format."""
        lines = []

        for key, value in obj.items():
            if value is None and optimize and self._should_omit_nulls(schema):
                continue

            if isinstance(value, (dict, list)):
                # Nested object or array
                nested_toon = self._convert_to_toon(value, schema, optimize)
                lines.append(f"{key}:")
                # Indent nested content
                indented = "\n".join(
                    "  " + line for line in nested_toon.split("\n") if line.strip()
                )
                lines.append(indented)
            else:
                # Simple value
                formatted_value = self._format_value(value, optimize)
                lines.append(f"{key}: {formatted_value}")

        return "\n".join(lines)

    def _convert_array_to_toon(
        self, arr: List, schema: Optional[ToonSchema] = None, optimize: bool = True
    ) -> str:
        """Convert array to TOON format - uses tabular format for uniform objects."""
        if not arr:
            return "[]"

        # Check if array contains uniform objects (same keys)
        if self._is_uniform_object_array(arr) and optimize:
            return self._convert_uniform_array_to_table(arr, schema)
        else:
            # Use YAML-style list
            lines = []
            for item in arr:
                if isinstance(item, (dict, list)):
                    item_toon = self._convert_to_toon(item, schema, optimize)
                    indented = "\n".join(
                        "  " + line for line in item_toon.split("\n") if line.strip()
                    )
                    lines.append(f"- {indented}")
                else:
                    lines.append(f"- {self._format_value(item, optimize)}")
            return "\n".join(lines)

    def _convert_uniform_array_to_table(
        self, arr: List[Dict], schema: Optional[ToonSchema] = None
    ) -> str:
        """Convert uniform object array to TOON table format (most token-efficient)."""
        if not arr or not isinstance(arr[0], dict):
            return self._convert_array_to_toon(arr, schema, False)

        # Get all keys from first object (assuming uniformity)
        keys = list(arr[0].keys())

        # Create header row
        header = "| " + " | ".join(keys) + " |"
        separator = "|" + "|".join("-" * (len(key) + 2) for key in keys) + "|"

        # Create data rows
        rows = []
        for item in arr:
            values = []
            for key in keys:
                value = item.get(key, "")
                formatted = self._format_table_value(value)
                values.append(formatted)
            rows.append("| " + " | ".join(values) + " |")

        # Combine all parts
        return "\n".join([header, separator] + rows)

    def _is_uniform_object_array(self, arr: List) -> bool:
        """Check if array contains objects with the same structure."""
        if not arr or len(arr) < 2:
            return False

        if not all(isinstance(item, dict) for item in arr):
            return False

        # Check if all objects have the same keys
        first_keys = set(arr[0].keys())
        return all(set(item.keys()) == first_keys for item in arr)

    def _format_value(self, value: Any, optimize: bool = True) -> str:
        """Format a value for TOON output."""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            if optimize and isinstance(value, float):
                # Limit decimal precision for token savings
                return f"{value:.3f}".rstrip("0").rstrip(".")
            return str(value)
        elif isinstance(value, str):
            # Quote strings that need it
            if " " in value or "\n" in value or value in ["true", "false", "null"]:
                return f'"{value}"'
            return value
        elif value is None:
            return "null"
        else:
            return str(value)

    def _format_table_value(self, value: Any) -> str:
        """Format value for table cell (compact)."""
        if value is None:
            return ""
        elif isinstance(value, bool):
            return "✓" if value else "✗"
        elif isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        else:
            return str(value)[:20]  # Limit length for table cells

    def _should_omit_nulls(self, schema: Optional[ToonSchema]) -> bool:
        """Check if null values should be omitted based on schema."""
        if schema and "omit_nulls" in schema.optimizations:
            return True
        return False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token
        # In practice, you'd use tiktoken or similar for accurate counts
        return len(text) // 4

    def _generate_toon_recommendations(
        self,
        data: Union[Dict, List],
        metrics: TokenMetrics,
        schema: Optional[ToonSchema],
    ) -> List[str]:
        """Generate recommendations for TOON usage."""
        recommendations = []

        if metrics.savings_percentage > 10:
            recommendations.append(
                f"TOON saves {metrics.savings_percentage:.1f}% tokens vs JSON"
            )

        # Check data structure suitability
        if self._has_uniform_arrays(data):
            recommendations.append(
                "Data contains uniform arrays - ideal for TOON table format"
            )
        elif self._has_deep_nesting(data):
            recommendations.append(
                "Deeply nested data - consider TOON's structured format"
            )

        # Schema recommendations
        if not schema and self._could_benefit_from_schema(data):
            recommendations.append(
                "Consider defining a schema for this data type to optimize further"
            )

        return recommendations

    def _generate_format_recommendations(
        self,
        data: Union[Dict, List],
        metrics: TokenMetrics,
        schema: Optional[ToonSchema],
        chosen_format: str,
        preferred_format: str,
    ) -> List[str]:
        """Generate recommendations for format choice."""
        recommendations = []

        # Format-specific recommendations
        if chosen_format == "toon":
            recommendations.extend(
                self._generate_toon_recommendations(data, metrics, schema)
            )
            if preferred_format == "auto" and metrics.savings_percentage < 20:
                recommendations.append(
                    "TOON selected automatically - savings may be marginal for this data"
                )
        elif chosen_format == "json":
            if metrics.savings_percentage > 30:
                recommendations.append(
                    f"Consider TOON format to save {metrics.savings_percentage:.1f}% on tokens"
                )
            else:
                recommendations.append(
                    "JSON format selected - suitable for this data structure"
                )

        # General recommendations
        if self._has_uniform_arrays(data) and chosen_format == "json":
            recommendations.append(
                "Data has uniform arrays - TOON table format would be more efficient"
            )

        if self._count_objects(data) > 20 and chosen_format == "json":
            recommendations.append(
                "Large data structure - TOON could significantly reduce token usage"
            )

        return recommendations

    def _has_uniform_arrays(self, data: Union[Dict, List]) -> bool:
        """Check if data contains uniform object arrays."""

        def check_item(item):
            if isinstance(item, list):
                return self._is_uniform_object_array(item)
            elif isinstance(item, dict):
                return any(check_item(v) for v in item.values())
            return False

        return check_item(data)

    def _has_deep_nesting(self, data: Union[Dict, List], depth: int = 0) -> bool:
        """Check if data has deep nesting."""
        if depth > 3:
            return True

        if isinstance(data, dict):
            return any(self._has_deep_nesting(v, depth + 1) for v in data.values())
        elif isinstance(data, list):
            return any(self._has_deep_nesting(item, depth + 1) for item in data)

        return False

    def _could_benefit_from_schema(self, data: Union[Dict, List]) -> bool:
        """Check if data could benefit from a custom schema."""
        # Simple heuristic: has repeated structures
        return self._has_uniform_arrays(data) or self._count_objects(data) > 5

    def _count_objects(self, data: Union[Dict, List]) -> int:
        """Count total number of objects in data structure."""
        count = 0

        if isinstance(data, dict):
            count += 1
            for v in data.values():
                count += self._count_objects(v)
        elif isinstance(data, list):
            for item in data:
                count += self._count_objects(item)

        return count

    @validate_types
    def create_custom_schema(
        self, schema_name: str, sample_data: Union[Dict, List], description: str = ""
    ) -> ToonSchema:
        """
        Create a custom TOON schema based on sample data.

        Args:
            schema_name: Name for the new schema
            sample_data: Sample data to analyze for schema creation
            description: Optional description

        Returns:
            New ToonSchema optimized for the data
        """
        try:
            validate_non_empty_string(schema_name, "schema_name")

            # Analyze data structure
            fields = []
            array_fields = []
            nested_objects = {}

            def analyze_structure(data, path=""):
                if isinstance(data, dict):
                    for key, value in data.items():
                        full_path = f"{path}.{key}" if path else key
                        fields.append(full_path)

                        if isinstance(value, (dict, list)):
                            analyze_structure(value, full_path)
                        elif (
                            isinstance(value, list)
                            and value
                            and isinstance(value[0], dict)
                        ):
                            array_fields.append(full_path)

                elif isinstance(data, list) and data and isinstance(data[0], dict):
                    # Analyze first item for structure
                    analyze_structure(data[0], path)

            analyze_structure(sample_data)

            # Determine optimizations based on data analysis
            optimizations = []
            if self._has_uniform_arrays(sample_data):
                optimizations.append("compact_arrays")
            if self._count_objects(sample_data) > 10:
                optimizations.append("omit_nulls")
                optimizations.append("short_field_names")

            schema = ToonSchema(
                name=schema_name,
                description=description or f"Custom schema for {schema_name}",
                fields=fields,
                array_fields=array_fields,
                nested_objects=nested_objects,
                optimizations=optimizations,
            )

            self.schemas[schema_name] = schema
            return schema

        except UCUPValueError as e:
            raise UCUPValueError(
                create_error_message(
                    context="TOON Formatter Create Schema",
                    action="Creating custom TOON schema",
                    details=str(e),
                    suggestion="Provide valid schema name and sample data",
                )
            ) from e

    def get_token_savings_report(self, limit: int = 10) -> Dict[str, Any]:
        """Generate a report of token savings from recent conversions."""
        recent_conversions = self.conversion_history[-limit:]

        if not recent_conversions:
            return {"message": "No conversions to report"}

        total_savings = sum(c.metrics.savings_percentage for c in recent_conversions)
        avg_savings = total_savings / len(recent_conversions)

        best_conversion = max(
            recent_conversions, key=lambda c: c.metrics.savings_percentage
        )

        return {
            "total_conversions": len(recent_conversions),
            "average_savings_percentage": avg_savings,
            "total_token_savings": sum(
                c.metrics.json_tokens - c.metrics.toon_tokens
                for c in recent_conversions
            ),
            "best_conversion": {
                "savings_percentage": best_conversion.metrics.savings_percentage,
                "schema_used": best_conversion.schema_used.name
                if best_conversion.schema_used
                else None,
            },
            "recommendations": [
                "Consider using TOON for all LLM prompts to reduce costs",
                "Define schemas for frequently used data structures",
                "Monitor token usage with built-in metrics",
            ],
        }

    def toon_to_json(self, toon_string: str) -> Union[Dict, List]:
        """
        Convert TOON back to JSON (for completeness).

        Note: This is a basic implementation. Full TOON parsing would require
        a complete parser. For now, this handles simple cases.
        """
        try:
            # For now, assume it's YAML-compatible and use basic parsing
            # In practice, you'd want a full TOON parser
            import yaml

            try:
                return yaml.safe_load(toon_string)
            except ImportError:
                # Fallback: try to parse as basic structure
                return self._simple_toon_parser(toon_string)
        except Exception as e:
            raise UCUPValueError(f"Failed to parse TOON: {e}")

    def _simple_toon_parser(self, toon: str) -> Union[Dict, List]:
        """Simple TOON parser for basic cases."""
        # This is a placeholder - real TOON parsing is complex
        # For production use, integrate with official TOON parser
        lines = toon.strip().split("\n")
        result = {}

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if value:  # Simple value
                    result[key] = self._parse_toon_value(value)
                else:  # Nested object
                    i += 1
                    nested = {}
                    while i < len(lines) and lines[i].startswith("  "):
                        nested_line = lines[i].strip()
                        if ":" in nested_line:
                            n_key, n_value = nested_line.split(":", 1)
                            nested[n_key.strip()] = self._parse_toon_value(
                                n_value.strip()
                            )
                        i += 1
                    result[key] = nested
                    continue
            i += 1

        return result

    def _parse_toon_value(self, value: str) -> Any:
        """Parse a TOON value to Python type."""
        value = value.strip()
        if value == "null":
            return None
        elif value == "true":
            return True
        elif value == "false":
            return False
        elif value.isdigit():
            return int(value)
        elif re.match(r"^\d+\.\d+$", value):
            return float(value)
        elif value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        else:
            return value


@dataclass
class TOONOptimizer:
    """Optimizes UCUP results using TOON format for token efficiency."""

    formatter: ToonFormatter
    enable_optimization: bool = True
    target_token_reduction: float = 0.15  # 15% minimum reduction

    def __init__(self, formatter: Optional[ToonFormatter] = None):
        """Initialize TOON optimizer."""
        self.formatter = formatter or ToonFormatter()

    def optimize(self, probabilistic_result: Any) -> Any:
        """
        Optimize a probabilistic result using TOON format.

        Args:
            probabilistic_result: UCUP result to optimize

        Returns:
            Optimized result with TOON formatting where beneficial
        """
        if not self.enable_optimization:
            return probabilistic_result

        try:
            # Extract data that can be optimized
            optimizable_data = self._extract_optimizable_data(probabilistic_result)

            if not optimizable_data:
                return probabilistic_result

            # Convert to TOON and check savings
            toon_result = self.formatter.json_to_toon(optimizable_data)

            # Only optimize if we get significant savings
            if toon_result.metrics.savings_percentage >= (
                self.target_token_reduction * 100
            ):
                return self._create_optimized_result(probabilistic_result, toon_result)
            else:
                return probabilistic_result

        except Exception:
            # If optimization fails, return original result
            return probabilistic_result

    def _extract_optimizable_data(self, result: Any) -> Optional[Dict]:
        """Extract data that can be optimized from UCUP result."""
        try:
            if hasattr(result, "metadata") and result.metadata:
                # Look for arrays or structured data in metadata
                metadata = result.metadata
                optimizable = {}

                # Extract alternatives if present
                if hasattr(result, "alternatives") and result.alternatives:
                    alternatives_data = []
                    for alt in result.alternatives:
                        if hasattr(alt, "__dict__"):
                            alternatives_data.append(alt.__dict__)
                        else:
                            alternatives_data.append(alt)
                    optimizable["alternatives"] = alternatives_data

                # Extract metadata arrays
                for key, value in metadata.items():
                    if isinstance(value, list) and len(value) > 2:
                        optimizable[key] = value
                    elif isinstance(value, dict) and len(value) > 3:
                        optimizable[key] = value

                return optimizable if optimizable else None

        except Exception:
            return None

    def _create_optimized_result(
        self, original_result: Any, toon_result: "ToonConversionResult"
    ) -> Any:
        """Create optimized result with TOON formatting."""
        try:
            # Create a copy of the original result
            import copy

            optimized_result = copy.deepcopy(original_result)

            # Add TOON-optimized data
            if not hasattr(optimized_result, "metadata"):
                optimized_result.metadata = {}

            optimized_result.metadata.update(
                {
                    "toon_optimized": True,
                    "token_savings": toon_result.metrics.savings_percentage,
                    "toon_data": toon_result.toon_output,
                }
            )

            return optimized_result

        except Exception:
            return original_result
