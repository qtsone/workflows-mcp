"""Secrets management system for workflows-mcp.

This package provides comprehensive secrets management for workflow execution,
including secret providers, automatic redaction, and audit logging.

Core Components:
    - SecretProvider: Abstract base class for secret sources
    - EnvVarSecretProvider: Environment variable-based secrets
    - SecretRedactor: Automatic secret redaction from outputs
    - SecretAuditLog: Compliance and security audit logging
    - Custom exceptions: Structured error handling

Example:
    >>> import os
    >>> from workflows_mcp.engine.secrets import (
    ...     EnvVarSecretProvider,
    ...     SecretRedactor,
    ...     SecretAuditLog,
    ... )
    >>>
    >>> # Set up environment variable secret
    >>> os.environ["WORKFLOW_SECRET_API_KEY"] = "sk-1234567890abcdef"
    >>>
    >>> # Initialize provider and redactor
    >>> provider = EnvVarSecretProvider()
    >>> redactor = SecretRedactor(provider)
    >>> await redactor.initialize()
    >>>
    >>> # Initialize audit log
    >>> audit_log = SecretAuditLog()
    >>>
    >>> # Access secret with audit logging
    >>> secret = await provider.get_secret("api_key")
    >>> await audit_log.log_access(
    ...     workflow_name="my-workflow",
    ...     block_id="fetch_api_key",
    ...     secret_key="api_key",
    ...     success=True
    ... )
    >>>
    >>> # Redact secret from outputs
    >>> output = {"key": "sk-1234567890abcdef", "status": "success"}
    >>> safe_output = redactor.redact(output)
    >>> print(safe_output)
    {'key': '***REDACTED***', 'status': 'success'}
"""

from .audit import SecretAccessEvent, SecretAuditLog
from .exceptions import (
    SecretAccessDeniedError,
    SecretError,
    SecretNotFoundError,
    SecretProviderError,
)
from .provider import (
    AWSSecretsProvider,
    EnvVarSecretProvider,
    SecretProvider,
    VaultSecretProvider,
)
from .redactor import SecretRedactor

__all__ = [
    # Exceptions
    "SecretError",
    "SecretNotFoundError",
    "SecretAccessDeniedError",
    "SecretProviderError",
    # Providers
    "SecretProvider",
    "EnvVarSecretProvider",
    "VaultSecretProvider",
    "AWSSecretsProvider",
    # Redaction
    "SecretRedactor",
    # Audit
    "SecretAccessEvent",
    "SecretAuditLog",
]
