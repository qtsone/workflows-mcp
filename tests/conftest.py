"""Shared test configuration for workflows-mcp tests.

Configures test environment including:
- Test secrets for secrets management tests
- Common fixtures and test utilities
- Environment setup and teardown
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_secrets():
    """Configure test secrets for all tests.

    These secrets are used by workflows in tests/workflows/core/secrets/
    to validate secrets management functionality (ADR-008).

    The secrets are set as environment variables with the WORKFLOW_SECRET_ prefix,
    matching the production secrets configuration pattern.

    Secret values match those documented in tests/workflows/core/secrets/README.md
    """
    test_secrets = {
        # Basic tests
        "WORKFLOW_SECRET_TEST_SECRET": "test-value-123",
        "WORKFLOW_SECRET_API_KEY": "sk-test-key-456",
        "WORKFLOW_SECRET_API_TOKEN": "valid-bearer-token",
        # Database tests
        "WORKFLOW_SECRET_DB_PASSWORD": "db-secure-password",
        "WORKFLOW_SECRET_DB_CONNECTION_STRING": "postgresql://user:pass@localhost/db",
        # Integration tests
        "WORKFLOW_SECRET_MULTI_SECRET": "multi-block-test-value",
        # Audit tests
        "WORKFLOW_SECRET_AUDIT_SECRET_1": "audit-test-1",
        "WORKFLOW_SECRET_AUDIT_SECRET_2": "audit-test-2",
        # Redaction tests
        "WORKFLOW_SECRET_REDACTION_TEST": "secret-to-be-redacted",
    }

    # Set test secrets
    for key, value in test_secrets.items():
        os.environ[key] = value

    yield

    # Cleanup: Remove test secrets after session
    for key in test_secrets:
        os.environ.pop(key, None)
