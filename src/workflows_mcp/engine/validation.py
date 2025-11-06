"""Validation utilities for workflow execution.

This module provides validation functions for workflow components,
particularly for ADR-009 for_each iteration keys.
"""

from typing import Any


class IterationKeyValidationError(ValueError):
    """Raised when iteration key validation fails."""

    pass


def validate_iteration_keys(iterations: dict[str, Any]) -> None:
    """
    Validate iteration keys for for_each blocks (ADR-009).

    Security and stability requirements:
    - Length < 255 characters
    - No control characters (ord(c) < 32 or ord(c) == 127)
    - No `__` prefix (reserved for internal use)
    - Otherwise permissive (bracket notation handles special chars)

    Args:
        iterations: Dictionary of iteration key -> value pairs

    Raises:
        IterationKeyValidationError: If any key violates validation rules
            with detailed error message indicating which key and which rule

    Examples:
        >>> validate_iteration_keys({"api": {...}, "worker": {...}})  # OK
        >>> validate_iteration_keys({"0": ..., "1": ...})  # OK (numeric strings)
        >>> validate_iteration_keys({"file-1.txt": ..., "file-2.txt": ...})  # OK
        >>> validate_iteration_keys({"__internal": ...})  # Raises (reserved prefix)
        >>> validate_iteration_keys({"x" * 300: ...})  # Raises (too long)
    """
    for key in iterations.keys():
        # Rule 1: Length < 255 characters
        if len(key) >= 255:
            raise IterationKeyValidationError(
                f"Iteration key '{key[:50]}...' exceeds maximum length of 254 characters "
                f"(length: {len(key)})"
            )

        # Rule 2: No control characters (ord(c) < 32 or ord(c) == 127)
        for i, char in enumerate(key):
            if ord(char) < 32 or ord(char) == 127:
                raise IterationKeyValidationError(
                    f"Iteration key '{key[:50]}...' contains control character at position {i} "
                    f"(ord: {ord(char)}). Control characters are not allowed for security and "
                    f"stability reasons."
                )

        # Rule 3: No `__` prefix (reserved for internal use)
        if key.startswith("__"):
            raise IterationKeyValidationError(
                f"Iteration key '{key}' starts with '__' which is reserved for internal use. "
                f"Please use a different key name."
            )


__all__ = ["validate_iteration_keys", "IterationKeyValidationError"]
