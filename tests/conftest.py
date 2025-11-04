"""Shared test configuration for workflows-mcp tests.

Configures test environment including:
- Test secrets for secrets management tests
- Common fixtures and test utilities
- Environment setup and teardown
"""

import pytest
from test_secrets import setup_test_secrets as _setup_secrets
from test_secrets import teardown_test_secrets as _teardown_secrets


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
