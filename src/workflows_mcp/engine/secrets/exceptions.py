"""Custom exceptions for secrets management system.

This module defines the exception hierarchy for the secrets management system,
providing structured error handling for secret access, provider errors, and
security violations.

Exception Hierarchy:
    SecretError (base)
    ├── SecretNotFoundError (missing secret)
    ├── SecretAccessDeniedError (authorization failure)
    └── SecretProviderError (provider-level failure)

Example:
    >>> try:
    ...     secret = await provider.get_secret("missing_key")
    ... except SecretNotFoundError as e:
    ...     print(f"Secret {e.key} not found. Try: {e.provider_hint}")
"""


class SecretError(Exception):
    """Base exception for all secrets-related errors.

    This is the root exception class for the secrets management system.
    All other secret-related exceptions inherit from this class, allowing
    for broad exception handling when needed.

    Example:
        >>> try:
        ...     await secret_operation()
        ... except SecretError as e:
        ...     logger.error(f"Secret operation failed: {e}")
    """

    pass


class SecretNotFoundError(SecretError):
    """Exception raised when a requested secret is not found.

    This exception indicates that a secret key was requested but could not
    be located in any configured provider. It includes the missing key and
    an optional provider hint for troubleshooting.

    Attributes:
        key: The secret key that was not found
        provider_hint: Optional hint about where to configure the secret

    Example:
        >>> raise SecretNotFoundError(
        ...     key="database_password",
        ...     provider_hint="Set WORKFLOW_SECRET_DATABASE_PASSWORD env var"
        ... )
    """

    def __init__(self, key: str, provider_hint: str | None = None) -> None:
        """Initialize SecretNotFoundError.

        Args:
            key: The secret key that was not found
            provider_hint: Optional hint about where to configure the secret
        """
        self.key = key
        self.provider_hint = provider_hint

        message = f"Secret '{key}' not found"
        if provider_hint:
            message += f". {provider_hint}"

        super().__init__(message)


class SecretAccessDeniedError(SecretError):
    """Exception raised when access to a secret is denied.

    This exception indicates an authorization failure when attempting to
    access a secret. It includes the secret key and the reason for denial.

    Attributes:
        key: The secret key that access was denied for
        reason: Human-readable reason for the denial

    Example:
        >>> raise SecretAccessDeniedError(
        ...     key="admin_token",
        ...     reason="Workflow does not have admin permissions"
        ... )
    """

    def __init__(self, key: str, reason: str) -> None:
        """Initialize SecretAccessDeniedError.

        Args:
            key: The secret key that access was denied for
            reason: Human-readable reason for the denial
        """
        self.key = key
        self.reason = reason

        message = f"Access denied for secret '{key}': {reason}"
        super().__init__(message)


class SecretProviderError(SecretError):
    """Exception raised when a secret provider encounters an error.

    This exception indicates a provider-level failure such as connection
    issues, authentication problems, or configuration errors. It includes
    the provider name and detailed error information.

    Attributes:
        provider_name: Name of the provider that encountered the error
        details: Detailed error information from the provider

    Example:
        >>> raise SecretProviderError(
        ...     provider_name="VaultSecretProvider",
        ...     details="Connection timeout to vault.example.com:8200"
        ... )
    """

    def __init__(self, provider_name: str, details: str) -> None:
        """Initialize SecretProviderError.

        Args:
            provider_name: Name of the provider that encountered the error
            details: Detailed error information from the provider
        """
        self.provider_name = provider_name
        self.details = details

        message = f"Secret provider '{provider_name}' error: {details}"
        super().__init__(message)
