"""Shared test secrets configuration.

Single source of truth for test secrets used across:
- conftest.py (pytest fixture setup)
- generate_snapshots.py (snapshot generation)
- regenerate_snapshots.sh (legacy shell script - can be removed)

Secret values match tests/workflows/core/secrets/README.md
"""

TEST_SECRETS = {
    # Basic tests
    "WORKFLOW_SECRET_TEST_SECRET": "test-value-123",
    "WORKFLOW_SECRET_API_KEY": "sk-test-key-456",
    "WORKFLOW_SECRET_API_TOKEN": "valid-bearer-token",
    "WORKFLOW_SECRET_HTTP_TOKEN": "bearer-token-xyz",
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


def setup_test_secrets() -> None:
    """Configure test secrets in environment variables."""
    import os

    for key, value in TEST_SECRETS.items():
        os.environ[key] = value


def teardown_test_secrets() -> None:
    """Remove test secrets from environment variables."""
    import os

    for key in TEST_SECRETS:
        os.environ.pop(key, None)
