"""Shared test configuration for workflows-mcp tests.

Configures test environment including:
- Test secrets for secrets management tests
- HTTP mock server (replaces httpbin.org for reliable tests)
- Common fixtures and test utilities
- Environment setup and teardown
"""

import json

import pytest
from pytest_httpserver import HTTPServer
from test_secrets import setup_test_secrets as _setup_secrets
from test_secrets import teardown_test_secrets as _teardown_secrets
from werkzeug.wrappers import Response


@pytest.fixture(scope="session", autouse=True)
def setup_test_secrets():
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
def httpbin_mock(httpserver: HTTPServer):
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
    def get_handler(request):
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
    def post_handler(request):
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
    def headers_handler(request):
        """Echo request headers like httpbin.org/headers."""
        data = {"headers": {k: v for k, v in request.headers}}
        return Response(json.dumps(data), content_type="application/json")

    httpserver.expect_request("/headers").respond_with_handler(headers_handler)

    return httpserver


@pytest.fixture
def workflow_inputs(workflow_name: str, request: pytest.FixtureRequest) -> dict[str, str]:
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
    }

    if workflow_name in http_workflows:
        # Lazy load: only create httpbin_mock when needed
        httpbin_mock = request.getfixturevalue("httpbin_mock")
        # Strip trailing slash from mock server URL
        base_url = httpbin_mock.url_for("/").rstrip("/")
        return {"base_url": base_url}

    return {}
