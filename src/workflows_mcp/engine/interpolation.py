"""
Utilities for handling interpolatable fields in workflow block inputs.

This module provides a reusable pattern for fields that need strict validation
(like enums or literals) but also support variable interpolation at load time.

Pattern:
1. Load time: Accept Union[StrictType, str], validate interpolation syntax
2. Execution time: Resolve variables, validate against strict type

Example:
    ```python
    from pydantic import Field, field_validator
    from typing import Union

    class MyInputs(BlockInput):
        provider: Union[LLMProvider, str] = Field(...)

        _validate_provider = field_validator('provider', mode='before')(
            interpolatable_enum_validator(LLMProvider)
        )

    class MyExecutor(BlockExecutor):
        async def execute(self, inputs: BlockInput, context: Execution) -> MyOutput:
            # Resolve and validate after variable interpolation
            provider = resolve_interpolatable_enum(
                inputs.provider, LLMProvider, 'provider'
            )
            # Now provider is guaranteed to be LLMProvider enum
    ```
"""

import re
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar, overload

T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)

# Pattern to detect {{...}} interpolation
INTERPOLATION_PATTERN = re.compile(r"\{\{[^}]+\}\}")


def has_interpolation(value: Any) -> bool:
    """
    Check if a value contains variable interpolation syntax.

    Args:
        value: Value to check

    Returns:
        True if value is a string containing {{...}} pattern

    Examples:
        >>> has_interpolation("{{inputs.provider}}")
        True
        >>> has_interpolation("openai")
        False
        >>> has_interpolation("prefix-{{var}}-suffix")
        True
    """
    return isinstance(value, str) and bool(INTERPOLATION_PATTERN.search(value))


def interpolatable_enum_validator[E: Enum](
    enum_class: type[E],
) -> Callable[[type[Any], E | str], E | str]:
    """
    Create a Pydantic field validator for enum fields that support interpolation.

    This validator allows three types of values:
    1. Enum instances (already validated) - pass through
    2. Interpolation strings like "{{inputs.provider}}" - pass through for later resolution
    3. Valid enum value strings - convert to enum instance

    Invalid values are rejected with helpful error messages.

    Args:
        enum_class: The enum class to validate against

    Returns:
        A classmethod validator function for use with Pydantic's field_validator

    Usage:
        ```python
        class MyInputs(BlockInput):
            provider: LLMProvider | str = Field(
                description="Provider name or interpolation"
            )

            # Apply validator (Pydantic v2 pattern)
            _validate_provider = field_validator('provider', mode='before')(
                interpolatable_enum_validator(LLMProvider)
            )
        ```

    Raises:
        ValueError: If value is not an enum, valid enum string, or interpolation
    """

    def validator(cls: type[Any], v: E | str) -> E | str:
        # Already the enum type - pass through
        if isinstance(v, enum_class):
            return v

        # Must be a string from here
        if not isinstance(v, str):
            raise ValueError(
                f"must be a {enum_class.__name__} instance or string, got {type(v).__name__}"
            )

        # Interpolation string - pass through for later resolution
        if has_interpolation(v):
            return v

        # Try to convert to enum (handles both .value and .name)
        try:
            return enum_class(v)
        except ValueError:
            # Build helpful error message
            valid_values = ", ".join(repr(e.value) for e in enum_class)
            raise ValueError(
                f"must be one of [{valid_values}] or an interpolation string "
                f"like '{{{{inputs.field}}}}'. Got: {v!r}"
            )

    # Return the function - field_validator handles classmethod wrapping
    return validator


def interpolatable_literal_validator(
    *allowed_values: Any,
) -> Callable[[type[Any], Any], Any]:
    """
    Create a Pydantic field validator for literal fields that support interpolation.

    Similar to interpolatable_enum_validator but for Literal types.

    Args:
        *allowed_values: Valid literal values

    Returns:
        A classmethod validator function for use with Pydantic's field_validator

    Usage:
        ```python
        class MyInputs(BlockInput):
            mode: str = Field(...)

            _validate_mode = field_validator('mode', mode='before')(
                interpolatable_literal_validator("fast", "slow")
            )
        ```
    """

    def validator(cls: type[Any], v: Any) -> Any:
        # Interpolation string - pass through
        if isinstance(v, str) and has_interpolation(v):
            return v

        # Check if value is in allowed literals
        if v in allowed_values:
            return v

        # Invalid value
        valid_repr = ", ".join(repr(val) for val in allowed_values)
        raise ValueError(f"must be one of [{valid_repr}] or an interpolation string. Got: {v!r}")

    # Return the function - field_validator handles classmethod wrapping
    return validator


def resolve_interpolatable_enum(
    value: EnumT | str, enum_class: type[EnumT], field_name: str
) -> EnumT:
    """
    Resolve an interpolatable enum field to its strict enum type after variable resolution.

    Call this in the executor's execute() method after variables have been resolved
    to ensure the value is a valid enum instance.

    Args:
        value: The field value (either enum instance or resolved string)
        enum_class: The enum class to validate against
        field_name: Field name for error messages

    Returns:
        The enum instance

    Raises:
        ValueError: If the resolved value is not a valid enum member

    Usage:
        ```python
        class MyExecutor(BlockExecutor):
            async def execute(self, inputs: BlockInput, context: Execution):
                # After variable resolution, validate and coerce
                provider = resolve_interpolatable_enum(
                    inputs.provider, LLMProvider, 'provider'
                )
                # Now provider is guaranteed to be LLMProvider enum
                # (mypy will recognize this via type narrowing)
        ```
    """
    # Already the enum type
    if isinstance(value, enum_class):
        return value

    # Must be a resolved string at this point
    if isinstance(value, str):
        # Should NOT contain interpolation anymore (variables resolved)
        if has_interpolation(value):
            raise ValueError(
                f"{field_name}: Variable interpolation was not resolved. Still contains: {value!r}"
            )

        # Try to convert to enum
        try:
            return enum_class(value)
        except ValueError:
            valid_values = ", ".join(repr(e.value) for e in enum_class)
            raise ValueError(f"Invalid {field_name}: {value!r}. Must be one of: {valid_values}")

    # Unexpected type
    raise ValueError(
        f"{field_name} must be {enum_class.__name__} or string, got {type(value).__name__}"
    )


def resolve_interpolatable_literal(
    value: Any, allowed_values: tuple[Any, ...], field_name: str
) -> Any:
    """
    Resolve an interpolatable literal field after variable resolution.

    Similar to resolve_interpolatable_enum but for Literal types.

    Args:
        value: The field value (resolved)
        allowed_values: Tuple of allowed literal values
        field_name: Field name for error messages

    Returns:
        The validated value

    Raises:
        ValueError: If the resolved value is not in allowed_values
    """
    # Check for unresolved interpolation
    if isinstance(value, str) and has_interpolation(value):
        raise ValueError(
            f"{field_name}: Variable interpolation was not resolved. Still contains: {value!r}"
        )

    # Check if value is allowed
    if value in allowed_values:
        return value

    # Invalid value
    valid_repr = ", ".join(repr(val) for val in allowed_values)
    raise ValueError(f"Invalid {field_name}: {value!r}. Must be one of: {valid_repr}")


def interpolatable_numeric_validator(
    numeric_type: type[int] | type[float],
    *,
    ge: int | float | None = None,
    le: int | float | None = None,
    gt: int | float | None = None,
    lt: int | float | None = None,
) -> Callable[[type[Any], int | float | str | None], int | float | str | None]:
    """
    Create a Pydantic field validator for numeric fields that support interpolation.

    This validator allows:
    1. None for optional fields - pass through without validation
    2. Numeric values (already validated) - pass through with constraint checks
    3. Interpolation strings like "{{inputs.timeout}}" - pass through for later resolution
    4. Numeric strings - convert to numeric type with constraint checks

    Args:
        numeric_type: The numeric type (int or float)
        ge: Greater than or equal to (>=)
        le: Less than or equal to (<=)
        gt: Greater than (>)
        lt: Less than (<)

    Returns:
        A classmethod validator function for use with Pydantic's field_validator

    Usage:
        ```python
        class MyInputs(BlockInput):
            timeout: int | str = Field(description="Timeout in seconds")

            _validate_timeout = field_validator('timeout', mode='before')(
                interpolatable_numeric_validator(int, ge=1, le=3600)
            )
        ```

    Raises:
        ValueError: If value violates constraints or is invalid type
    """

    def validator(cls: type[Any], v: int | float | str | None) -> int | float | str | None:
        # None for optional fields - pass through
        if v is None:
            return None

        # Interpolation string - pass through for later resolution
        if isinstance(v, str) and has_interpolation(v):
            return v

        # Try to convert to numeric type
        try:
            if isinstance(v, str):
                # Convert string to numeric
                num_value = numeric_type(v)
            elif isinstance(v, (int, float)):
                # Already numeric, coerce to target type
                num_value = numeric_type(v)
            else:
                raise ValueError(
                    f"must be {numeric_type.__name__}, string, or interpolation, "
                    f"got {type(v).__name__}"
                )

            # Validate constraints
            if ge is not None and num_value < ge:
                raise ValueError(f"must be >= {ge}, got {num_value}")
            if le is not None and num_value > le:
                raise ValueError(f"must be <= {le}, got {num_value}")
            if gt is not None and num_value <= gt:
                raise ValueError(f"must be > {gt}, got {num_value}")
            if lt is not None and num_value >= lt:
                raise ValueError(f"must be < {lt}, got {num_value}")

            return num_value

        except (ValueError, TypeError) as e:
            # Build helpful error message
            constraints = []
            if ge is not None:
                constraints.append(f">= {ge}")
            if le is not None:
                constraints.append(f"<= {le}")
            if gt is not None:
                constraints.append(f"> {gt}")
            if lt is not None:
                constraints.append(f"< {lt}")

            constraint_str = f" ({', '.join(constraints)})" if constraints else ""
            raise ValueError(
                f"must be a valid {numeric_type.__name__}{constraint_str} or "
                f"an interpolation string like '{{{{inputs.field}}}}'. Got: {v!r}. Error: {e}"
            )

    return validator


@overload
def resolve_interpolatable_numeric(
    value: int | str,
    numeric_type: type[int],
    field_name: str,
    *,
    ge: int | float | None = None,
    le: int | float | None = None,
    gt: int | float | None = None,
    lt: int | float | None = None,
) -> int: ...


@overload
def resolve_interpolatable_numeric(
    value: float | str,
    numeric_type: type[float],
    field_name: str,
    *,
    ge: int | float | None = None,
    le: int | float | None = None,
    gt: int | float | None = None,
    lt: int | float | None = None,
) -> float: ...


def resolve_interpolatable_numeric(
    value: int | float | str,
    numeric_type: type[int] | type[float],
    field_name: str,
    *,
    ge: int | float | None = None,
    le: int | float | None = None,
    gt: int | float | None = None,
    lt: int | float | None = None,
) -> int | float:
    """
    Resolve an interpolatable numeric field after variable resolution.

    Call this in the executor's execute() method after variables have been resolved
    to ensure the value is a valid numeric type with constraints enforced.

    Args:
        value: The field value (numeric or resolved string)
        numeric_type: The numeric type (int or float)
        field_name: Field name for error messages
        ge: Greater than or equal to (>=)
        le: Less than or equal to (<=)
        gt: Greater than (>)
        lt: Less than (<)

    Returns:
        The validated numeric value

    Raises:
        ValueError: If the resolved value violates constraints

    Usage:
        ```python
        timeout = resolve_interpolatable_numeric(
            inputs.timeout, int, 'timeout', ge=1, le=3600
        )
        ```
    """
    # Convert to numeric type (handle each case and validate inline)
    if isinstance(value, str):
        # Check for unresolved interpolation first
        if has_interpolation(value):
            raise ValueError(
                f"{field_name}: Variable interpolation was not resolved. Still contains: {value!r}"
            )

        # Convert string to numeric
        try:
            num_value = numeric_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid {field_name}: {value!r} is not a valid {numeric_type.__name__}. "
                f"Error: {e}"
            )
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        # Numeric value - coerce to target type if needed
        num_value = numeric_type(value)
    else:
        raise ValueError(
            f"{field_name} must be {numeric_type.__name__} or string, got {type(value).__name__}"
        )

    # Validate constraints
    constraints_violated = []
    if ge is not None and num_value < ge:
        constraints_violated.append(f">= {ge}")
    if le is not None and num_value > le:
        constraints_violated.append(f"<= {le}")
    if gt is not None and num_value <= gt:
        constraints_violated.append(f"> {gt}")
    if lt is not None and num_value >= lt:
        constraints_violated.append(f"< {lt}")

    if constraints_violated:
        raise ValueError(
            f"Invalid {field_name}: {num_value} violates constraints "
            f"({', '.join(constraints_violated)})"
        )

    return num_value


def interpolatable_boolean_validator() -> Callable[[type[Any], bool | str], bool | str]:
    """
    Create a Pydantic field validator for boolean fields that support interpolation.

    This validator allows:
    1. Boolean values (True/False) - pass through
    2. Interpolation strings like "{{inputs.enable_ssl}}" - pass through for later resolution
    3. Boolean-like strings ("true", "false", "1", "0", "yes", "no") - convert to bool

    Returns:
        A classmethod validator function for use with Pydantic's field_validator

    Usage:
        ```python
        class MyInputs(BlockInput):
            verify_ssl: bool | str = Field(description="Verify SSL certificates")

            _validate_verify_ssl = field_validator('verify_ssl', mode='before')(
                interpolatable_boolean_validator()
            )
        ```

    Raises:
        ValueError: If value is not a valid boolean representation
    """

    def validator(cls: type[Any], v: bool | str) -> bool | str:
        # Already a boolean - pass through
        if isinstance(v, bool):
            return v

        # Must be a string from here
        if not isinstance(v, str):
            raise ValueError(f"must be a boolean or string, got {type(v).__name__}")

        # Interpolation string - pass through for later resolution
        if has_interpolation(v):
            return v

        # Convert string to boolean
        v_lower = v.lower().strip()
        if v_lower in ("true", "1", "yes", "on", "y"):
            return True
        elif v_lower in ("false", "0", "no", "off", "n", ""):
            return False
        else:
            raise ValueError(
                f"must be a boolean (true/false) or an interpolation string. Got: {v!r}"
            )

    return validator


def resolve_interpolatable_boolean(value: bool | str, field_name: str) -> bool:
    """
    Resolve an interpolatable boolean field after variable resolution.

    Call this in the executor's execute() method after variables have been resolved
    to ensure the value is a valid boolean.

    Args:
        value: The field value (boolean or resolved string)
        field_name: Field name for error messages

    Returns:
        The validated boolean value

    Raises:
        ValueError: If the resolved value is not a valid boolean

    Usage:
        ```python
        verify_ssl = resolve_interpolatable_boolean(inputs.verify_ssl, 'verify_ssl')
        ```
    """
    # Already a boolean
    if isinstance(value, bool):
        return value

    # Must be a string from here
    if isinstance(value, str):
        # Check for unresolved interpolation
        if has_interpolation(value):
            raise ValueError(
                f"{field_name}: Variable interpolation was not resolved. Still contains: {value!r}"
            )

        # Convert string to boolean
        v_lower = value.lower().strip()
        if v_lower in ("true", "1", "yes", "on", "y"):
            return True
        elif v_lower in ("false", "0", "no", "off", "n", ""):
            return False
        else:
            raise ValueError(
                f"Invalid {field_name}: {value!r} is not a valid boolean. "
                f"Use 'true'/'false', '1'/'0', 'yes'/'no', etc."
            )

    raise ValueError(f"{field_name} must be boolean or string, got {type(value).__name__}")


# Type alias for documentation
InterpolatableEnum = Enum | str
InterpolatableLiteral = Any | str
InterpolatableNumeric = int | float | str
InterpolatableBoolean = bool | str
