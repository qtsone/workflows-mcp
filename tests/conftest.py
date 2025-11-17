"""Shared test configuration for workflows-mcp tests.

Configures test environment including:
- Test secrets for secrets management tests
- HTTP mock server (replaces httpbin.org for reliable tests)
- Common fixtures and test utilities
- Environment setup and teardown
"""

import json
from collections.abc import Iterator
from typing import Any

import pytest
from pytest_httpserver import HTTPServer
from test_secrets import setup_test_secrets as _setup_secrets
from test_secrets import teardown_test_secrets as _teardown_secrets
from werkzeug.wrappers import Request, Response


@pytest.fixture(scope="session", autouse=True)
def setup_test_secrets() -> Iterator[None]:
    """Configure test secrets for all tests.

    These secrets are used by workflows in tests/workflows/core/secrets/
    to validate secrets management functionality (ADR-008).

    The secrets are set as environment variables with the WORKFLOW_SECRET_ prefix,
    matching the production secrets configuration pattern.

    Secret values defined in test_secrets.py (single source of truth).
    """
    _setup_secrets()
    yield
    _teardown_secrets()


@pytest.fixture
def httpbin_mock(httpserver: HTTPServer) -> HTTPServer:
    """
    Local HTTP mock server that replaces httpbin.org for reliable testing.

    This fixture provides a local HTTP server that mimics httpbin.org behavior,
    eliminating flaky tests from external service downtime and network issues.

    The server dynamically handles requests (not hardcoded responses):
    - GET /get: Echoes back request args, headers, URL
    - POST /post: Echoes back JSON body and headers
    - GET /headers: Echoes back request headers

    Usage in tests:
        # Pass mock URL to workflow as input
        base_url = httpbin_mock.url_for("/").rstrip("/")
        result = await execute_workflow("http-basic-get", {"base_url": base_url})

    Usage in workflows:
        inputs:
          base_url:
            type: str
            default: "https://httpbin.org"  # No trailing slash

        blocks:
          - id: get_request
            inputs:
              url: "{{inputs.base_url}}/get"  # Add slash here
    """

    # GET endpoint - echoes back request information
    def get_handler(request: Request) -> Response:
        """Echo request information like httpbin.org/get."""
        data = {
            "args": dict(request.args),
            "headers": {k: v for k, v in request.headers},
            "origin": request.remote_addr or "127.0.0.1",
            "url": str(request.url),
        }
        return Response(json.dumps(data), content_type="application/json")

    httpserver.expect_request("/get").respond_with_handler(get_handler)

    # POST endpoint - echoes back JSON body and headers
    def post_handler(request: Request) -> Response:
        """Echo JSON body and headers like httpbin.org/post."""
        json_data = None
        if request.is_json:
            try:
                json_data = request.json
            except Exception:
                json_data = None

        data = {
            "json": json_data,
            "data": request.data.decode() if request.data else "",
            "headers": {k: v for k, v in request.headers},
            "origin": request.remote_addr or "127.0.0.1",
        }
        return Response(json.dumps(data), content_type="application/json")

    httpserver.expect_request("/post", method="POST").respond_with_handler(post_handler)

    # /headers endpoint - echoes back request headers (used in secrets tests)
    def headers_handler(request: Request) -> Response:
        """Echo request headers like httpbin.org/headers."""
        data = {"headers": {k: v for k, v in request.headers}}
        return Response(json.dumps(data), content_type="application/json")

    httpserver.expect_request("/headers").respond_with_handler(headers_handler)

    return httpserver


@pytest.fixture
def llm_mock(httpserver: HTTPServer) -> HTTPServer:
    """
    Local LLM mock server that mimics OpenAI-compatible chat completion API.

    This fixture provides a local HTTP server that responds to chat completion requests,
    enabling LLM tests without external API dependencies or credentials.

    The server handles POST requests to /v1/chat/completions and returns OpenAI-compatible
    JSON responses with configurable content based on the request prompt.

    Usage in tests:
        # Pass mock URL to workflow as input
        api_url = llm_mock.url_for("/").rstrip("/")
        result = await execute_workflow("executor-llm-operations-test", {"api_url": api_url})

    Usage in workflows:
        inputs:
          api_url:
            type: str
            default: ""  # Required for tests

        blocks:
          - id: llm_call
            type: LLMCall
            inputs:
              provider: openai
              model: gpt-4o-mini
              api_url: "{{inputs.api_url}}"
              prompt: "What is the capital of France?"
    """

    def chat_completion_handler(request: Request) -> Response:
        """Return OpenAI-compatible chat completion response."""
        # Parse request body - handle potential None from request.json
        if request.is_json and request.json is not None:
            request_data = dict(request.json)
        else:
            request_data = {}

        # Extract prompt from messages (simple extraction for test purposes)
        messages = request_data.get("messages", [])
        user_message = next((msg["content"] for msg in messages if msg.get("role") == "user"), "")

        # If prompt contains specific keywords, customize response
        if "capital" in user_message.lower() and "france" in user_message.lower():
            response_text = "Paris"
        elif "test" in user_message.lower():
            response_text = "This is a test response from the mock LLM server."
        else:
            response_text = f"Mock response to: {user_message[:50]}"

        # Return OpenAI-compatible response
        response_data = {
            "id": "chatcmpl-mock123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": request_data.get("model", "gpt-5-mini"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        return Response(json.dumps(response_data), content_type="application/json")

    # OpenAI SDK uses /v1/chat/completions or /chat/completions depending on base_url
    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_completion_handler
    )
    httpserver.expect_request("/chat/completions", method="POST").respond_with_handler(
        chat_completion_handler
    )

    return httpserver


@pytest.fixture
def workflow_inputs(workflow_name: str, request: pytest.FixtureRequest) -> dict[str, Any]:
    """
    Provide workflow-specific inputs for tests.

    Automatically injects base_url for HTTP-related workflows to point to the
    local mock server instead of external httpbin.org.

    Uses lazy loading: only creates httpbin_mock fixture when needed (for HTTP
    workflows), avoiding unnecessary server instantiation for non-HTTP tests.

    Args:
        workflow_name: Name of the workflow being tested (from parametrization)
        request: Pytest fixture request for lazy fixture loading

    Returns:
        Dict of workflow inputs (empty for non-HTTP workflows)
    """
    # HTTP workflows that need base_url injected
    http_workflows = {
        "http-basic-get",
        "secrets-http-auth",
        "secrets-multiple-blocks",
        "core-secrets-management-test",
    }

    if workflow_name in http_workflows:
        # Lazy load: only create httpbin_mock when needed
        httpbin_mock = request.getfixturevalue("httpbin_mock")
        # Strip trailing slash from mock server URL
        base_url = httpbin_mock.url_for("/").rstrip("/")
        return {"base_url": base_url}

    # LLM workflows that need api_url injected
    llm_workflows = {
        "executor-llm-operations-test",
    }

    if workflow_name in llm_workflows:
        # Lazy load: only create llm_mock when needed
        llm_mock = request.getfixturevalue("llm_mock")
        # Strip trailing slash from mock server URL
        api_url = llm_mock.url_for("/").rstrip("/")
        return {"api_url": api_url}

    # Secrets redaction workflow needs expected hash of the secret value
    if workflow_name == "secrets-redaction":
        import hashlib

        from test_secrets import TEST_SECRETS

        # Calculate expected hash from actual test secret (single source of truth)
        secret_value = TEST_SECRETS["WORKFLOW_SECRET_REDACTION_TEST"]
        expected_hash = hashlib.sha256(secret_value.encode()).hexdigest()
        return {"expected_hash": expected_hash}

    return {}
