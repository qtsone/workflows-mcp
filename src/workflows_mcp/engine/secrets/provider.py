"""Secret provider abstraction and implementations.

This module defines the SecretProvider abstraction and concrete implementations
for retrieving secrets from various sources (environment variables, Vault, AWS, etc.).

Providers:
    - SecretProvider: Abstract base class defining the provider interface
    - EnvVarSecretProvider: Retrieves secrets from environment variables (WORKFLOW_SECRET_*)
    - VaultSecretProvider: HashiCorp Vault integration (future implementation)
    - AWSSecretsProvider: AWS Secrets Manager integration (future implementation)

Example:
    >>> provider = EnvVarSecretProvider()
    >>> secret = await provider.get_secret("database_password")
    >>> keys = await provider.list_secret_keys()
"""

import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from .exceptions import SecretNotFoundError


class SecretProvider(ABC):
    """Abstract base class for secret providers.

    Secret providers are responsible for retrieving secrets from external sources
    and listing available secret keys. Implementations must handle provider-specific
    authentication, connection management, and error handling.

    All methods are async to support both local and remote secret sources.

    Example:
        >>> class MyProvider(SecretProvider):
        ...     async def get_secret(self, key: str) -> str:
        ...         return await fetch_from_source(key)
        ...
        ...     async def list_secret_keys(self) -> list[str]:
        ...         return await list_from_source()
    """

    @abstractmethod
    async def get_secret(self, key: str) -> str:
        """Retrieve a secret value by key.

        Args:
            key: The secret key to retrieve (e.g., "database_password")

        Returns:
            The secret value as a string

        Raises:
            SecretNotFoundError: If the secret key does not exist
            SecretProviderError: If the provider encounters an error
        """
        pass

    @abstractmethod
    async def list_secret_keys(self) -> list[str]:
        """List all available secret keys.

        Returns:
            List of secret keys available in this provider

        Raises:
            SecretProviderError: If the provider encounters an error
        """
        pass


class EnvVarSecretProvider(SecretProvider):
    """Secret provider that reads from environment variables.

    This provider retrieves secrets from environment variables prefixed with
    WORKFLOW_SECRET_. The prefix is automatically added when looking up secrets.

    Environment Variable Format:
        WORKFLOW_SECRET_{KEY_UPPER} = secret_value

    Examples:
        WORKFLOW_SECRET_DATABASE_PASSWORD=my_secure_password
        WORKFLOW_SECRET_API_KEY=sk-1234567890abcdef

    Usage:
        >>> os.environ["WORKFLOW_SECRET_DATABASE_PASSWORD"] = "secret123"
        >>> provider = EnvVarSecretProvider()
        >>> password = await provider.get_secret("database_password")
        >>> print(password)
        secret123

        >>> keys = await provider.list_secret_keys()
        >>> print(keys)
        ['database_password']

    Attributes:
        prefix: Environment variable prefix (default: "WORKFLOW_SECRET_")
    """

    def __init__(self, prefix: str = "WORKFLOW_SECRET_") -> None:
        """Initialize the environment variable secret provider.

        Args:
            prefix: Environment variable prefix for secrets
                   (default: "WORKFLOW_SECRET_")
        """
        self.prefix = prefix

    def _get_env_var_name(self, key: str) -> str:
        """Convert secret key to environment variable name.

        Args:
            key: Secret key (e.g., "database_password")

        Returns:
            Environment variable name (e.g., "WORKFLOW_SECRET_DATABASE_PASSWORD")
        """
        return f"{self.prefix}{key.upper()}"

    async def get_secret(self, key: str) -> str:
        """Retrieve a secret from environment variables.

        Args:
            key: The secret key to retrieve (e.g., "database_password")

        Returns:
            The secret value from the corresponding environment variable

        Raises:
            SecretNotFoundError: If the environment variable does not exist
        """
        env_var_name = self._get_env_var_name(key)
        value = os.environ.get(env_var_name)

        if value is None:
            raise SecretNotFoundError(
                key=key,
                provider_hint=f"Set environment variable: {env_var_name}=<secret_value>",
            )

        return value

    async def list_secret_keys(self) -> list[str]:
        """List all available secret keys from environment variables.

        Scans environment variables for those matching the prefix and
        returns the secret keys (with prefix and uppercase conversion removed).

        Returns:
            List of secret keys (e.g., ["database_password", "api_key"])

        Example:
            >>> os.environ["WORKFLOW_SECRET_DB_PASSWORD"] = "secret"
            >>> os.environ["WORKFLOW_SECRET_API_KEY"] = "key"
            >>> provider = EnvVarSecretProvider()
            >>> keys = await provider.list_secret_keys()
            >>> print(sorted(keys))
            ['api_key', 'db_password']
        """
        secret_keys = []

        for env_var_name in os.environ.keys():
            if env_var_name.startswith(self.prefix):
                # Remove prefix and convert to lowercase
                key = env_var_name[len(self.prefix) :].lower()
                secret_keys.append(key)

        return secret_keys


class SQLiteSecretProvider(SecretProvider):
    """Secret provider that reads encrypted admin-managed secrets from SQLite."""

    def __init__(self, db_path: Path, key_path: Path) -> None:
        self.db_path = db_path
        self.key_path = key_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _key_candidates(self, key: str) -> tuple[str, ...]:
        candidates = [key]
        for variant in (key.upper(), key.lower()):
            if variant not in candidates:
                candidates.append(variant)
        return tuple(candidates)

    async def get_secret(self, key: str) -> str:
        from workflows_mcp.metadata.repos.secrets_repo import SQLiteSecretsRepository

        if not self.db_path.exists():
            raise SecretNotFoundError(
                key=key,
                provider_hint=f"Create admin secret in SQLite store: {self.db_path.name}",
            )

        conn = self._connect()
        try:
            repo = SQLiteSecretsRepository(conn=conn, key_path=self.key_path)
            for candidate in self._key_candidates(key):
                value = repo.get_secret_value(candidate)
                if value is not None:
                    return value
        finally:
            conn.close()

        raise SecretNotFoundError(
            key=key,
            provider_hint="Create secret via the admin secrets store",
        )

    async def list_secret_keys(self) -> list[str]:
        from workflows_mcp.metadata.repos.secrets_repo import SQLiteSecretsRepository

        if not self.db_path.exists():
            return []

        conn = self._connect()
        try:
            repo = SQLiteSecretsRepository(conn=conn, key_path=self.key_path)
            return [metadata.name for metadata in repo.list_metadata()]
        finally:
            conn.close()


class CompositeSecretProvider(SecretProvider):
    """Secret provider that checks multiple providers in priority order."""

    def __init__(self, providers: list[SecretProvider]) -> None:
        self.providers = providers

    async def get_secret(self, key: str) -> str:
        last_error: SecretNotFoundError | None = None
        for provider in self.providers:
            try:
                return await provider.get_secret(key)
            except SecretNotFoundError as exc:
                last_error = exc

        if last_error is not None:
            raise SecretNotFoundError(key=key, provider_hint=str(last_error))

        raise SecretNotFoundError(key=key)

    async def list_secret_keys(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for provider in self.providers:
            for key in await provider.list_secret_keys():
                normalized = key.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                keys.append(key)
        return keys


class VaultSecretProvider(SecretProvider):
    """Secret provider for HashiCorp Vault (future implementation).

    This provider will integrate with HashiCorp Vault for enterprise secret
    management with advanced features like dynamic secrets, encryption as a service,
    and comprehensive audit logging.

    Planned Features:
        - Token-based authentication
        - AppRole authentication
        - Kubernetes authentication
        - Dynamic secret generation
        - Secret rotation support
        - KV v2 secrets engine support

    Note:
        This is a placeholder for future implementation. Methods will raise
        NotImplementedError until the Vault integration is complete.

    Example (future):
        >>> provider = VaultSecretProvider(
        ...     url="https://vault.example.com:8200",
        ...     token="s.1234567890abcdef",
        ...     mount_point="workflows"
        ... )
        >>> secret = await provider.get_secret("database/credentials")
    """

    def __init__(
        self,
        url: str,
        token: str | None = None,
        mount_point: str = "secret",
    ) -> None:
        """Initialize the Vault secret provider.

        Args:
            url: Vault server URL (e.g., "https://vault.example.com:8200")
            token: Vault authentication token (optional, can use other auth methods)
            mount_point: Vault secrets engine mount point (default: "secret")
        """
        self.url = url
        self.token = token
        self.mount_point = mount_point

    async def get_secret(self, key: str) -> str:
        """Retrieve a secret from HashiCorp Vault.

        Args:
            key: The secret key/path to retrieve

        Returns:
            The secret value

        Raises:
            NotImplementedError: This provider is not yet implemented
        """
        raise NotImplementedError(
            "VaultSecretProvider is not yet implemented. Use EnvVarSecretProvider for now."
        )

    async def list_secret_keys(self) -> list[str]:
        """List all available secret keys from Vault.

        Returns:
            List of secret keys

        Raises:
            NotImplementedError: This provider is not yet implemented
        """
        raise NotImplementedError(
            "VaultSecretProvider is not yet implemented. Use EnvVarSecretProvider for now."
        )


class AWSSecretsProvider(SecretProvider):
    """Secret provider for AWS Secrets Manager (future implementation).

    This provider will integrate with AWS Secrets Manager for cloud-native secret
    management with automatic rotation, fine-grained access control, and integration
    with AWS services.

    Planned Features:
        - IAM role-based authentication
        - Automatic secret rotation
        - Cross-region replication support
        - Binary and JSON secret support
        - Version management
        - Audit logging via CloudTrail

    Note:
        This is a placeholder for future implementation. Methods will raise
        NotImplementedError until the AWS integration is complete.

    Example (future):
        >>> provider = AWSSecretsProvider(
        ...     region_name="us-east-1",
        ...     role_arn="arn:aws:iam::123456789012:role/WorkflowsRole"
        ... )
        >>> secret = await provider.get_secret("prod/database/password")
    """

    def __init__(
        self,
        region_name: str = "us-east-1",
        role_arn: str | None = None,
    ) -> None:
        """Initialize the AWS Secrets Manager provider.

        Args:
            region_name: AWS region (default: "us-east-1")
            role_arn: Optional IAM role ARN to assume
        """
        self.region_name = region_name
        self.role_arn = role_arn

    async def get_secret(self, key: str) -> str:
        """Retrieve a secret from AWS Secrets Manager.

        Args:
            key: The secret name/ARN to retrieve

        Returns:
            The secret value

        Raises:
            NotImplementedError: This provider is not yet implemented
        """
        raise NotImplementedError(
            "AWSSecretsProvider is not yet implemented. Use EnvVarSecretProvider for now."
        )

    async def list_secret_keys(self) -> list[str]:
        """List all available secret names from AWS Secrets Manager.

        Returns:
            List of secret names

        Raises:
            NotImplementedError: This provider is not yet implemented
        """
        raise NotImplementedError(
            "AWSSecretsProvider is not yet implemented. Use EnvVarSecretProvider for now."
        )
