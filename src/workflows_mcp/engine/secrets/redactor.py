"""Secret redaction for workflow outputs and logs.

This module provides automatic redaction of secrets from workflow outputs, logs,
and any other data structures to prevent accidental exposure of sensitive information.

The SecretRedactor uses loaded secrets from a SecretProvider to identify and redact
secret values anywhere they appear in strings, dictionaries, lists, and nested structures.

Features:
    - Recursive redaction of nested data structures
    - Minimum secret length validation (MIN_SECRET_LENGTH = 8)
    - Regex escaping for special characters
    - Type-preserving redaction (maintains dict/list structure)

Example:
    >>> from .provider import EnvVarSecretProvider
    >>> provider = EnvVarSecretProvider()
    >>> redactor = SecretRedactor(provider)
    >>> await redactor.initialize()
    >>>
    >>> data = {"password": "my_secret_password", "user": "admin"}
    >>> redacted = redactor.redact(data)
    >>> print(redacted)
    {'password': '***REDACTED***', 'user': 'admin'}
"""

import re
from typing import Any

from .provider import SecretProvider


class SecretRedactor:
    """Redacts secrets from data structures to prevent exposure.

    The SecretRedactor loads secrets from a provider and creates regex patterns
    to identify and redact secret values anywhere they appear. It handles nested
    data structures recursively while maintaining type safety.

    Security Features:
        - Minimum secret length (8 chars) to avoid false positives
        - Regex escaping to handle special characters
        - Recursive traversal of nested structures
        - Type preservation (dict/list structure maintained)

    Attributes:
        provider: SecretProvider to load secrets from
        secrets: Dictionary mapping secret keys to their values
        redaction_patterns: Compiled regex patterns for redaction
        MIN_SECRET_LENGTH: Minimum length for a value to be considered a secret (8)
        REDACTION_MARKER: String used to replace redacted secrets ("***REDACTED***")

    Example:
        >>> import os
        >>> from .provider import EnvVarSecretProvider
        >>> os.environ["WORKFLOW_SECRET_API_KEY"] = "sk-1234567890abcdef"
        >>> os.environ["WORKFLOW_SECRET_DB_PASSWORD"] = "my_secure_password"
        >>>
        >>> provider = EnvVarSecretProvider()
        >>> redactor = SecretRedactor(provider)
        >>> await redactor.initialize()
        >>>
        >>> # Redact a string
        >>> text = "Using API key: sk-1234567890abcdef"
        >>> print(redactor.redact(text))
        Using API key: ***REDACTED***
        >>>
        >>> # Redact nested structures
        >>> config = {
        ...     "database": {
        ...         "host": "localhost",
        ...         "password": "my_secure_password"
        ...     },
        ...     "api": {
        ...         "key": "sk-1234567890abcdef"
        ...     }
        ... }
        >>> redacted = redactor.redact(config)
        >>> print(redacted)
        {
            'database': {
                'host': 'localhost',
                'password': '***REDACTED***'
            },
            'api': {
                'key': '***REDACTED***'
            }
        }
    """

    # Minimum length for a secret to avoid redacting common short strings
    MIN_SECRET_LENGTH = 8

    # Marker used to replace redacted secrets
    REDACTION_MARKER = "***REDACTED***"

    def __init__(self, provider: SecretProvider) -> None:
        """Initialize the secret redactor.

        Args:
            provider: SecretProvider to load secrets from

        Note:
            You must call initialize() after construction to load secrets
            from the provider.
        """
        self.provider = provider
        self.secrets: dict[str, str] = {}
        self.redaction_patterns: list[re.Pattern[str]] = []

    async def initialize(self) -> None:
        """Load secrets from the provider and compile redaction patterns.

        This method must be called after construction before using the redactor.
        It loads all secrets from the provider and creates regex patterns for
        efficient redaction.

        Only secrets meeting the minimum length requirement (MIN_SECRET_LENGTH)
        are included to avoid false positive redactions of common short strings.

        Example:
            >>> redactor = SecretRedactor(provider)
            >>> await redactor.initialize()
            >>> # Now redactor is ready to use
        """
        # Load all secrets from the provider
        secret_keys = await self.provider.list_secret_keys()

        for key in secret_keys:
            try:
                value = await self.provider.get_secret(key)

                # Only include secrets meeting minimum length requirement
                if len(value) >= self.MIN_SECRET_LENGTH:
                    self.secrets[key] = value
            except Exception:
                # Skip secrets that fail to load
                # (provider may list keys that are inaccessible)
                pass

        # Compile regex patterns for each secret value
        self._compile_redaction_patterns()

    def _compile_redaction_patterns(self) -> None:
        """Compile regex patterns for redacting secret values.

        This method creates compiled regex patterns for each secret value,
        properly escaping special characters to ensure accurate matching.

        The patterns are sorted by length (longest first) to ensure longer
        secrets are matched before potential substrings.
        """
        self.redaction_patterns = []

        # Sort secrets by length (longest first) to match longer secrets first
        sorted_secrets = sorted(self.secrets.values(), key=len, reverse=True)

        for secret_value in sorted_secrets:
            # Escape special regex characters in the secret value
            escaped_value = re.escape(secret_value)

            # Compile pattern for this secret
            pattern = re.compile(escaped_value)
            self.redaction_patterns.append(pattern)

    def redact(self, data: Any) -> Any:  # noqa: ANN401
        """Redact secrets from any data structure.

        Recursively traverses the data structure and redacts any secret values
        found in strings. Maintains the original structure and types.

        Args:
            data: Data to redact (can be str, dict, list, or any other type)

        Returns:
            Redacted data with the same structure and types as the input

        Example:
            >>> # Redact a string
            >>> redacted = redactor.redact("Password: my_secret_password")
            >>> print(redacted)
            Password: ***REDACTED***
            >>>
            >>> # Redact a list
            >>> data = ["safe_value", "my_secret_password", 123]
            >>> redacted = redactor.redact(data)
            >>> print(redacted)
            ['safe_value', '***REDACTED***', 123]
            >>>
            >>> # Redact a nested dict
            >>> data = {
            ...     "config": {
            ...         "key": "my_secret_password",
            ...         "timeout": 30
            ...     }
            ... }
            >>> redacted = redactor.redact(data)
            >>> print(redacted)
            {'config': {'key': '***REDACTED***', 'timeout': 30}}
        """
        # Handle strings: apply redaction patterns
        if isinstance(data, str):
            return self._redact_string(data)

        # Handle dictionaries: redact keys and values recursively
        if isinstance(data, dict):
            return {key: self.redact(value) for key, value in data.items()}

        # Handle lists: redact each element recursively
        if isinstance(data, list):
            return [self.redact(item) for item in data]

        # Handle tuples: redact each element and return as tuple
        if isinstance(data, tuple):
            return tuple(self.redact(item) for item in data)

        # For other types (int, float, bool, None, etc.), return as-is
        return data

    def _redact_string(self, text: str) -> str:
        """Redact secrets from a string.

        Applies all compiled redaction patterns to the string, replacing
        secret values with the redaction marker.

        Args:
            text: String to redact

        Returns:
            String with secrets replaced by REDACTION_MARKER

        Example:
            >>> text = "API key: sk-1234567890abcdef, DB password: my_secure_password"
            >>> redacted = redactor._redact_string(text)
            >>> print(redacted)
            API key: ***REDACTED***, DB password: ***REDACTED***
        """
        redacted_text = text

        # Apply each redaction pattern
        for pattern in self.redaction_patterns:
            redacted_text = pattern.sub(self.REDACTION_MARKER, redacted_text)

        return redacted_text

    def add_secret(self, key: str, value: str) -> None:
        """Manually add a secret for redaction.

        This method allows adding secrets that may not be in the provider,
        such as dynamically generated secrets or secrets from other sources.

        Args:
            key: Secret identifier (for tracking purposes)
            value: Secret value to redact

        Example:
            >>> redactor.add_secret("temp_token", "dynamically_generated_token_12345")
            >>> text = "Token: dynamically_generated_token_12345"
            >>> print(redactor.redact(text))
            Token: ***REDACTED***
        """
        # Only add if meets minimum length requirement
        if len(value) >= self.MIN_SECRET_LENGTH:
            self.secrets[key] = value

            # Recompile patterns to include the new secret
            self._compile_redaction_patterns()

    def get_loaded_secret_keys(self) -> list[str]:
        """Get list of secret keys currently loaded for redaction.

        Returns:
            List of secret keys that are actively being redacted

        Example:
            >>> keys = redactor.get_loaded_secret_keys()
            >>> print(keys)
            ['api_key', 'db_password', 'temp_token']
        """
        return list(self.secrets.keys())
