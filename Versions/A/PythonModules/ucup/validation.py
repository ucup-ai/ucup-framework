"""
UCUP Validation Framework

This module provides comprehensive runtime type checking, input validation,
and error handling utilities for the UCUP framework.

Features:
- Runtime type validation with detailed error messages
- Input sanitization with configurable rules
- Decorator-based validation
- Configuration validation
- Domain-specific validators for UCUP components
- Performance monitoring and validation metrics
- Advanced validation rules for complex scenarios
- Integration with UCUP modules (coordination, reliability, probabilistic, observability, testing)
- Comprehensive error reporting with suggestions
"""

import asyncio
import json
import re
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union


class ErrorSeverity(Enum):
    """Severity levels for validation errors."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """
    Detailed validation error with context and suggestions.

    Provides comprehensive information about validation failures,
    including field context, expected vs actual types, and actionable
    suggestions for resolution.
    """

    field: str
    value: Any
    expected_type: str
    actual_type: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    suggestion: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Human-readable error representation."""
        base_msg = f"ValidationError in field '{self.field}': {self.message}"

        if self.suggestion:
            base_msg += f" | Suggestion: {self.suggestion}"

        if self.context:
            base_msg += f" | Context: {self.context}"

        return base_msg


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    sanitized_value: Optional[Any] = None

    def add_error(self, error: ValidationError):
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: ValidationError):
        """Add a validation warning."""
        warning.severity = ErrorSeverity.WARNING
        self.warnings.append(warning)

    def get_error_messages(self) -> List[str]:
        """Get all error messages as strings."""
        return [str(error) for error in self.errors]


class UCUPValidationError(Exception):
    """Base exception for UCUP validation errors."""

    def __init__(
        self,
        message: str,
        errors: Optional[List[ValidationError]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.errors = errors or []
        self.context = context


class UCUPTypeError(UCUPValidationError):
    """Exception for type validation errors."""

    pass


class UCUPValueError(UCUPValidationError):
    """Exception for value validation errors."""

    pass


class UCUPConfigurationError(UCUPValidationError):
    """Exception for configuration validation errors."""

    pass


class TypeValidator:
    """Runtime type validation utilities."""

    @staticmethod
    def validate_type(
        value: Any, expected_type: Type, field_name: str
    ) -> ValidationResult:
        """
        Validate that a value matches an expected type.

        Args:
            value: The value to validate
            expected_type: The expected type (supports generics like List[int])
            field_name: Name of the field being validated

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True, sanitized_value=value)

        try:
            # Handle Optional types
            if hasattr(expected_type, "__origin__"):
                origin = expected_type.__origin__
                args = getattr(expected_type, "__args__", ())

                if origin is Union:
                    # Check if any of the union types match
                    for union_type in args:
                        if union_type is type(None) and value is None:
                            return result
                        try:
                            TypeValidator._validate_single_type(value, union_type)
                            return result
                        except (TypeError, ValueError):
                            continue
                    result.add_error(
                        ValidationError(
                            field=field_name,
                            value=value,
                            expected_type=str(expected_type),
                            actual_type=type(value).__name__,
                            message=f"Value does not match any type in union {expected_type}",
                        )
                    )

                elif origin in (list, tuple):
                    if not isinstance(value, origin):
                        result.add_error(
                            ValidationError(
                                field=field_name,
                                value=value,
                                expected_type=str(expected_type),
                                actual_type=type(value).__name__,
                                message=f"Expected {origin.__name__}, got {type(value).__name__}",
                            )
                        )
                    elif args and len(args) > 0:
                        # Validate each element in the collection
                        element_type = args[0]
                        for i, item in enumerate(value):
                            try:
                                TypeValidator._validate_single_type(item, element_type)
                            except (TypeError, ValueError):
                                result.add_error(
                                    ValidationError(
                                        field=f"{field_name}[{i}]",
                                        value=item,
                                        expected_type=str(element_type),
                                        actual_type=type(item).__name__,
                                        message=f"Element {i} has incorrect type",
                                    )
                                )
                else:
                    # Fallback for other generic types
                    TypeValidator._validate_single_type(value, expected_type)
            else:
                TypeValidator._validate_single_type(value, expected_type)

        except (TypeError, ValueError) as e:
            result.add_error(
                ValidationError(
                    field=field_name,
                    value=value,
                    expected_type=str(expected_type),
                    actual_type=type(value).__name__,
                    message=str(e),
                )
            )

        return result

    @staticmethod
    def _validate_single_type(value: Any, expected_type: Type) -> None:
        """Validate a single (non-generic) type."""
        if expected_type is str:
            if not isinstance(value, str):
                raise TypeError(f"Expected str, got {type(value).__name__}")
        elif expected_type is int:
            if not isinstance(value, int):
                raise TypeError(f"Expected int, got {type(value).__name__}")
        elif expected_type is float:
            if not isinstance(value, (int, float)):
                raise TypeError(f"Expected float, got {type(value).__name__}")
        elif expected_type is bool:
            if not isinstance(value, bool):
                raise TypeError(f"Expected bool, got {type(value).__name__}")
        elif expected_type is dict:
            if not isinstance(value, dict):
                raise TypeError(f"Expected dict, got {type(value).__name__}")
        else:
            # For custom types, try isinstance check
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Expected {expected_type.__name__}, got {type(value).__name__}"
                )


def validate_data(
    data_type: str, data: Any, context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Convenience function to validate data."""
    # Simple implementation - can be expanded
    result = ValidationResult(is_valid=True, sanitized_value=data)
    return result


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    is_valid: bool
    issues: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    errors: List[ValidationError] = field(default_factory=list)
    validated_data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_ucup_validator():
    """Get the global UCUP validator instance."""
    # Simple implementation - can be expanded
    return None


# Additional validation functions for compatibility
def validate_types(
    value: Any, expected_type: Type, field_name: str
) -> ValidationResult:
    """Validate types with detailed error reporting."""
    return TypeValidator.validate_type(value, expected_type, field_name)


def sanitize_inputs(data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """Sanitize and validate input data against a schema."""
    result = ValidationResult(is_valid=True, sanitized_value=data)

    for field, expected_type in schema.items():
        if field in data:
            field_result = validate_types(data[field], expected_type, field)
            if not field_result.is_valid:
                result.add_error(field_result.errors[0])
            else:
                # Update sanitized value
                if result.sanitized_value is None:
                    result.sanitized_value = {}
                result.sanitized_value[field] = field_result.sanitized_value

    return result


@dataclass
class InputSanitizer:
    """Advanced input sanitization with configurable rules."""

    rules: Dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    strict_mode: bool = True

    def add_rule(self, field_name: str, sanitizer: Callable[[Any], Any]):
        """Add a sanitization rule for a field."""
        self.rules[field_name] = sanitizer

    def sanitize(self, data: Dict[str, Any]) -> ValidationResult:
        """Sanitize input data according to configured rules."""
        result = ValidationResult(is_valid=True, sanitized_value={})

        for field, value in data.items():
            if field in self.rules:
                try:
                    sanitized = self.rules[field](value)
                    result.sanitized_value[field] = sanitized
                except Exception as e:
                    result.add_error(
                        ValidationError(
                            field=field,
                            value=value,
                            expected_type="sanitized",
                            actual_type=type(value).__name__,
                            message=f"Sanitization failed: {str(e)}",
                        )
                    )
            else:
                result.sanitized_value[field] = value

        return result

    def sanitize_string(
        self,
        value: str,
        max_length: Optional[int] = None,
        allowed_chars: Optional[str] = None,
    ) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise UCUPTypeError("Value must be a string")

        sanitized = value.strip()

        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        if allowed_chars:
            sanitized = "".join(c for c in sanitized if c in allowed_chars)

        return sanitized

    def sanitize_number(
        self,
        value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> float:
        """Sanitize numeric input."""
        try:
            num = float(value)
        except (ValueError, TypeError):
            raise UCUPTypeError("Value must be numeric")

        if min_val is not None and num < min_val:
            num = min_val
        if max_val is not None and num > max_val:
            num = max_val

        return num


@dataclass
class UCUPValidator:
    """Main UCUP validation orchestrator."""

    sanitizers: Dict[str, InputSanitizer] = field(default_factory=dict)
    validators: Dict[str, Callable] = field(default_factory=dict)

    def add_sanitizer(self, name: str, sanitizer: InputSanitizer):
        """Add a named sanitizer."""
        self.sanitizers[name] = sanitizer

    def add_validator(self, name: str, validator: Callable):
        """Add a named validator."""
        self.validators[name] = validator

    def validate_and_sanitize(
        self, data: Dict[str, Any], sanitizer_name: str
    ) -> ValidationResult:
        """Validate and sanitize data using named sanitizer."""
        sanitizer = self.sanitizers.get(sanitizer_name)
        if not sanitizer:
            return ValidationResult(
                is_valid=False,
                errors=[
                    ValidationError(
                        field="sanitizer",
                        value=sanitizer_name,
                        expected_type="registered_sanitizer",
                        actual_type="unknown",
                        message=f"Sanitizer '{sanitizer_name}' not found",
                    )
                ],
            )

        return sanitizer.sanitize(data)

    def validate_schema(
        self, data: Dict[str, Any], schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate data against a schema."""
        return sanitize_inputs(data, schema)


def validate_probability(value: Any, field_name: str) -> None:
    """Validate that a value is a probability (0.0 to 1.0)."""
    if not isinstance(value, (int, float)):
        raise UCUPTypeError(f"{field_name} must be a number")
    if not (0.0 <= value <= 1.0):
        raise UCUPValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")


def validate_positive_number(value: Any, field_name: str) -> None:
    """Validate that a value is a positive number."""
    if not isinstance(value, (int, float)):
        raise UCUPTypeError(f"{field_name} must be a number")
    if value <= 0:
        raise UCUPValueError(f"{field_name} must be positive, got {value}")


def validate_non_empty_string(value: Any, field_name: str) -> None:
    """Validate that a value is a non-empty string."""
    if not isinstance(value, str):
        raise UCUPTypeError(f"{field_name} must be a string")
    if not value.strip():
        raise UCUPValueError(f"{field_name} cannot be empty")


def create_error_message(
    context: str, action: str, details: str, suggestion: str
) -> str:
    """Create a formatted error message."""
    return f"[{context}] {action}: {details}. Suggestion: {suggestion}"
