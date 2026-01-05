#!/usr/bin/env python3
"""
TOON Validation Utilities

Simple validation utilities for the TOON formatter.
"""

from typing import Any


class UCUPValidationError(Exception):
    """Base validation error for UCUP."""

    pass


class UCUPValueError(UCUPValidationError):
    """Value validation error."""

    pass


def validate_types(func):
    """Decorator to validate function argument types (placeholder)."""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def validate_non_empty_string(value: str, field_name: str):
    """Validate that a string is not empty."""
    if not isinstance(value, str) or not value.strip():
        raise UCUPValueError(f"{field_name} must be a non-empty string")


def create_error_message(
    context: str, action: str, details: str, suggestion: str = ""
) -> str:
    """Create a formatted error message."""
    message = f"[{context}] {action}: {details}"
    if suggestion:
        message += f" | Suggestion: {suggestion}"
    return message
