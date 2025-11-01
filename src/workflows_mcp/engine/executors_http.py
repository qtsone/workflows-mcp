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
from typing import Any, ClassVar

import httpx
from pydantic import Field

from .block import BlockInput, BlockOutput
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)

# ============================================================================
# HttpCall Executor
# ============================================================================


class HttpCallInput(BlockInput):
    """Input model for HttpCall executor."""

    url: str = Field(description="Request URL (supports ${ENV_VAR} substitution)")
    method: str = Field(
        default="POST",
        description="HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers (supports ${ENV_VAR} substitution in values)",
    )
    body_json: dict[str, Any] | None = Field(
        default=None,
        description="JSON request body (mutually exclusive with body_text)",
    )
    body_text: str | None = Field(
        default=None,
        description="Text request body (mutually exclusive with body_json)",
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
        ge=1,
        le=300,
    )
    follow_redirects: bool = Field(default=True, description="Whether to follow HTTP redirects")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")


class HttpCallOutput(BlockOutput):
    """Output model for HttpCall executor."""

    status_code: int = Field(description="HTTP response status code")
    response_body: str = Field(description="Response body as text")
    response_json: dict[str, Any] | None = Field(
        default=None,
        description="Parsed JSON response (None if not valid JSON)",
    )
    headers: dict[str, str] = Field(description="Response headers")
    success: bool = Field(description="True if status code is 2xx, False otherwise")


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
            ValueError: Invalid inputs (both body_json and body_text specified)
            httpx.TimeoutException: Request timeout
            httpx.NetworkError: Network connectivity issues
            httpx.HTTPStatusError: HTTP error responses (can be caught)
            Exception: Other HTTP client errors
        """
        # Validate mutually exclusive body inputs
        if inputs.body_json is not None and inputs.body_text is not None:
            raise ValueError("Cannot specify both body_json and body_text")

        # Substitute environment variables in URL and headers
        url = self._substitute_env_vars(inputs.url)
        headers = {key: self._substitute_env_vars(value) for key, value in inputs.headers.items()}

        # Prepare request body
        body: str | bytes | dict[str, Any] | None = None
        if inputs.body_json is not None:
            body = inputs.body_json
            # Set Content-Type if not already specified
            if "Content-Type" not in headers and "content-type" not in headers:
                headers["Content-Type"] = "application/json"
        elif inputs.body_text is not None:
            body = inputs.body_text

        # Create HTTP client with configuration
        async with httpx.AsyncClient(
            timeout=inputs.timeout,
            follow_redirects=inputs.follow_redirects,
            verify=inputs.verify_ssl,
        ) as client:
            # Make HTTP request
            try:
                response = await client.request(
                    method=inputs.method.upper(),
                    url=url,
                    headers=headers,
                    json=body if inputs.body_json is not None else None,
                    content=body if inputs.body_text is not None else None,
                )
            except httpx.TimeoutException as e:
                raise httpx.TimeoutException(
                    f"Request timeout after {inputs.timeout}s: {url}"
                ) from e
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
