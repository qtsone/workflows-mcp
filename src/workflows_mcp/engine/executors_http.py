"""HTTP/REST API call executor for ADR-006 - HttpCall.

Architecture (ADR-006):
- Execute returns output directly (no Result wrapper)
- Raises exceptions for failures
- Uses Execution context (not dict)
- Orchestrator creates Metadata based on success/exceptions

Features:
- Generic HTTP/REST API caller
- Support for all HTTP methods (GET, POST, PUT, DELETE, PATCH, etc.)
- JSON and text body support
- Custom headers with environment variable substitution
- Configurable timeout and SSL verification
- Response parsing (text and JSON)
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
from pydantic import Field, field_validator, model_validator

if TYPE_CHECKING:
    from typing import Self

from .block import BlockInput, BlockOutput
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)
from .interpolation import (
    interpolatable_boolean_validator,
    interpolatable_numeric_validator,
    resolve_interpolatable_boolean,
    resolve_interpolatable_numeric,
)

# ============================================================================
# HttpCall Executor
# ============================================================================


class HttpCallInput(BlockInput):
    """Input model for HttpCall executor.

    Field names match httpx.AsyncClient.request() parameters for consistency
    and to avoid type confusion during parameter passing.
    """

    url: str = Field(description="Request URL (supports ${ENV_VAR} substitution)")
    method: str = Field(
        default="POST",
        description="HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers (supports ${ENV_VAR} substitution in values)",
    )
    json: dict[str, Any] | None = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "JSON request body (mutually exclusive with content). Matches httpx parameter name."
        ),
    )
    # Note: Field name 'json' intentionally shadows Pydantic's deprecated .json() method
    # to match httpx.AsyncClient.request() API for type-safe parameter passthrough
    content: str | bytes | None = Field(
        default=None,
        description=(
            "Text or binary request body (mutually exclusive with json). "
            "Matches httpx parameter name."
        ),
    )
    timeout: int | str = Field(
        default=30,
        description="Request timeout in seconds (or interpolation string)",
    )
    follow_redirects: bool | str = Field(
        default=True, description="Whether to follow HTTP redirects (or interpolation string)"
    )
    verify_ssl: bool | str = Field(
        default=True, description="Whether to verify SSL certificates (or interpolation string)"
    )

    # Validators for numeric and boolean fields with interpolation support
    _validate_timeout = field_validator("timeout", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=1800)
    )
    _validate_follow_redirects = field_validator("follow_redirects", mode="before")(
        interpolatable_boolean_validator()
    )
    _validate_verify_ssl = field_validator("verify_ssl", mode="before")(
        interpolatable_boolean_validator()
    )

    @model_validator(mode="after")
    def validate_body_exclusive(self) -> Self:
        """Validate that json and content are mutually exclusive."""
        if self.json is not None and self.content is not None:
            raise ValueError("Cannot specify both 'json' and 'content' parameters")
        return self


class HttpCallOutput(BlockOutput):
    """Output model for HttpCall executor.

    All fields have defaults to support graceful degradation when HTTP calls fail.
    A default-constructed instance represents a failed/crashed HTTP call.
    """

    status_code: int = Field(
        default=0,
        description="HTTP response status code (0 if request failed before receiving response)",
    )
    response_body: str = Field(
        default="",
        description="Response body as text (empty string if request failed)",
    )
    response_json: dict[str, Any] | None = Field(
        default=None,
        description="Parsed JSON response (None if not valid JSON or request failed)",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Response headers (empty dict if request failed)",
    )
    success: bool = Field(
        default=False,
        description="True if status code is 2xx, False otherwise or if request failed",
    )


class HttpCallExecutor(BlockExecutor):
    """
    HTTP/REST API call executor.

    Architecture (ADR-006):
    - Returns HttpCallOutput directly
    - Raises exceptions for network failures, timeouts, SSL errors
    - Uses Execution context

    Features:
    - Support all HTTP methods (GET, POST, PUT, DELETE, PATCH, etc.)
    - JSON and text request bodies
    - Environment variable substitution in URL and headers: ${ENV_VAR}
    - Configurable timeout and SSL verification
    - Response parsing (text and JSON)
    - Proper error handling for network issues

    Security:
    - Environment variable substitution uses whitelist approach
    - No shell expansion or command injection risk
    - SSL verification enabled by default
    - Response size limited by httpx defaults
    """

    type_name: ClassVar[str] = "HttpCall"
    input_type: ClassVar[type[BlockInput]] = HttpCallInput
    output_type: ClassVar[type[BlockOutput]] = HttpCallOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_network=True)

    async def execute(  # type: ignore[override]
        self, inputs: HttpCallInput, context: Execution
    ) -> HttpCallOutput:
        """Execute HTTP request.

        Returns:
            HttpCallOutput with status code, response body, headers

        Raises:
            ValueError: Invalid inputs (validated by Pydantic)
            httpx.TimeoutException: Request timeout
            httpx.NetworkError: Network connectivity issues
            httpx.HTTPStatusError: HTTP error responses (can be caught)
            Exception: Other HTTP client errors
        """
        # Resolve interpolatable fields to their actual types
        timeout = resolve_interpolatable_numeric(inputs.timeout, int, "timeout", ge=1, le=1800)
        follow_redirects = resolve_interpolatable_boolean(
            inputs.follow_redirects, "follow_redirects"
        )
        verify_ssl = resolve_interpolatable_boolean(inputs.verify_ssl, "verify_ssl")

        # Substitute environment variables in URL and headers
        url = self._substitute_env_vars(inputs.url)
        headers = {key: self._substitute_env_vars(value) for key, value in inputs.headers.items()}

        # Set Content-Type for JSON if not already specified
        if inputs.json is not None:
            if "Content-Type" not in headers and "content-type" not in headers:
                headers["Content-Type"] = "application/json"

        # Create HTTP client with configuration
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=follow_redirects,
            verify=verify_ssl,
        ) as client:
            # Make HTTP request (parameters pass through directly to httpx)
            try:
                response = await client.request(
                    method=inputs.method.upper(),
                    url=url,
                    headers=headers,
                    json=inputs.json,  # Direct passthrough - type-safe!
                    content=inputs.content,  # Direct passthrough - type-safe!
                )
            except httpx.TimeoutException as e:
                raise httpx.TimeoutException(f"Request timeout after {timeout}s: {url}") from e
            except httpx.NetworkError as e:
                raise httpx.NetworkError(f"Network error for {url}: {e}") from e

            # Parse response
            response_body = response.text
            response_json: dict[str, Any] | None = None

            # Try to parse as JSON if Content-Type indicates JSON
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type.lower():
                try:
                    response_json = response.json()
                except json.JSONDecodeError:
                    # Invalid JSON in response, leave as None
                    pass

            # Determine success (2xx status codes)
            success = 200 <= response.status_code < 300

            return HttpCallOutput(
                status_code=response.status_code,
                response_body=response_body,
                response_json=response_json,
                headers=dict(response.headers),
                success=success,
            )

    @staticmethod
    def _substitute_env_vars(text: str) -> str:
        """Substitute environment variables in text using ${ENV_VAR} syntax.

        Args:
            text: Text potentially containing ${ENV_VAR} references

        Returns:
            Text with environment variables substituted

        Examples:
            >>> os.environ['API_KEY'] = 'secret123'
            >>> _substitute_env_vars('Bearer ${API_KEY}')
            'Bearer secret123'
            >>> _substitute_env_vars('No vars here')
            'No vars here'
        """
        import re

        def replace_env_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            # Get env var value, empty string if not found (fail silently)
            return os.environ.get(var_name, "")

        # Pattern: ${VAR_NAME} where VAR_NAME is alphanumeric + underscore
        pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"
        return re.sub(pattern, replace_env_var, text)
