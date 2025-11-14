"""
Proxy system for namespace enhancement in variable resolution.

Proxies provide attribute shortcuts and special handling for workflow objects:
    - BlockProxy: Shortcuts for block status and outputs
    - SecretProxy: Lazy secret loading with audit logging

Example:
    # Block shortcuts (ADR-007)
    blocks.foo.succeeded  # → blocks.foo.metadata.succeeded
    blocks.foo.exit_code  # → blocks.foo.outputs.exit_code

    # Secret proxy
    secrets.API_KEY  # → Lazy load from provider with audit log
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..secrets import SecretAuditLog, SecretProvider


class ProxyBase:
    """Base class for proxy objects."""

    def __init__(self, data: Any, rules: list[Any] | None = None):
        """
        Initialize proxy with data and optional rules.

        Args:
            data: Underlying data to wrap
            rules: Optional transformation rules
        """
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_rules", rules or [])

    def _apply_rules(self, key: str, value: Any) -> Any:
        """Apply transformation rules to accessed values."""
        rules = object.__getattribute__(self, "_rules")
        for rule in rules:
            if hasattr(rule, "applies_to_value") and rule.applies_to_value(key, value):
                value = rule.transform_value(key, value)
        return value


class BlockProxy(ProxyBase):
    """
    Proxy for block objects providing attribute shortcuts.

    Shortcuts (ADR-007):
    - blocks.foo.succeeded → blocks.foo.metadata.succeeded
    - blocks.foo.failed → blocks.foo.metadata.failed
    - blocks.foo.status → blocks.foo.metadata.status
    - blocks.foo.outcome → blocks.foo.metadata.outcome
    - blocks.foo.exit_code → blocks.foo.outputs.exit_code (output shortcut)

    Priority:
    1. Metadata shortcuts (succeeded, failed, skipped, status, outcome)
    2. Direct attributes (inputs, outputs, metadata)
    3. Output shortcuts (any field in outputs)
    4. Nested block access (for composed workflows)
    """

    # Metadata shortcuts
    METADATA_ATTRIBUTES = {"succeeded", "failed", "skipped", "status", "outcome"}

    def __getattr__(self, name: str) -> Any:
        # Priority 1: Check metadata shortcuts
        if name in self.METADATA_ATTRIBUTES:
            data = object.__getattribute__(self, "_data")
            metadata = data.get("metadata", {})
            return metadata.get(name)

        # Priority 2: Direct attributes
        data = object.__getattribute__(self, "_data")
        if name in data:
            value = data[name]
            # Wrap nested blocks dicts in proxy
            if name == "blocks" and isinstance(value, dict):
                return {k: BlockProxy(v) if isinstance(v, dict) else v for k, v in value.items()}
            return value

        # Priority 3: Output shortcuts
        if "outputs" in data:
            outputs = data["outputs"]
            if isinstance(outputs, dict) and name in outputs:
                return outputs[name]

        # Priority 4: Nested block access (for composed workflows)
        if "blocks" in data:
            blocks = data["blocks"]
            if isinstance(blocks, dict) and name in blocks:
                # Recursively wrap nested blocks
                return BlockProxy(blocks[name])

        # Return None for missing attributes (Jinja2 handles gracefully)
        return None

    def __getitem__(self, key: str) -> Any:
        """Support bracket notation: blocks['foo-bar'] and for_each iteration access"""
        data = object.__getattribute__(self, "_data")

        # Priority 1: Direct attributes (top-level keys)
        if key in data:
            value = data[key]
            # Wrap nested blocks in proxy
            if key == "blocks" and isinstance(value, dict):
                return {k: BlockProxy(v) if isinstance(v, dict) else v for k, v in value.items()}
            return value

        # Priority 2: Nested block access (for for_each iterations or composed workflows)
        if "blocks" in data:
            blocks = data["blocks"]
            if isinstance(blocks, dict) and key in blocks:
                # Recursively wrap nested blocks
                return BlockProxy(blocks[key])

        # Priority 3: Output shortcuts (for array/dict outputs)
        if "outputs" in data:
            outputs = data["outputs"]
            if isinstance(outputs, dict) and key in outputs:
                return outputs[key]

        return None

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator"""
        data = object.__getattribute__(self, "_data")
        return key in data

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute modification"""
        raise AttributeError("BlockProxy is read-only")

    def __delattr__(self, name: str) -> None:
        """Prevent attribute deletion"""
        raise AttributeError("BlockProxy is read-only")


class SecretProxy(ProxyBase):
    """
    Proxy for lazy secret resolution with audit logging.

    Features:
    - Lazy loading (only fetch when accessed)
    - Caching to avoid repeated fetches
    - Audit logging for compliance
    - Async support for secret providers
    """

    def __init__(
        self,
        provider: "SecretProvider",
        audit_log: "SecretAuditLog | None" = None,
    ):
        """
        Initialize secret proxy.

        Args:
            provider: Secret provider for fetching secrets
            audit_log: Optional audit log for tracking access
        """
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_audit_log", audit_log)
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_pending", {})

    async def get(self, key: str) -> Any:
        """
        Async secret resolution with caching.

        Args:
            key: Secret key name

        Returns:
            Secret value

        Raises:
            SecretNotFoundError: If secret not found
            RuntimeError: If circular reference detected
        """
        cache = object.__getattribute__(self, "_cache")
        if key in cache:
            return cache[key]

        pending = object.__getattribute__(self, "_pending")
        if pending.get(key):
            raise RuntimeError(f"Circular secret reference: {key}")

        try:
            pending[key] = True
            provider = object.__getattribute__(self, "_provider")

            # Fetch from provider
            value = await provider.get_secret(key)

            # Audit log
            audit_log = object.__getattribute__(self, "_audit_log")
            if audit_log:
                await audit_log.log_access(
                    workflow_name="",  # TODO: Thread context through resolver
                    block_id="",  # TODO: Thread context through resolver
                    secret_key=key,
                    success=True,
                )

            # Cache for future access
            cache[key] = value
            return value

        except Exception as e:
            audit_log = object.__getattribute__(self, "_audit_log")
            if audit_log:
                await audit_log.log_access(
                    workflow_name="",  # TODO: Thread context through resolver
                    block_id="",  # TODO: Thread context through resolver
                    secret_key=key,
                    success=False,
                    error_message=str(e),
                )
            raise
        finally:
            pending.pop(key, None)

    def __getattr__(self, name: str) -> Any:
        """Synchronous attribute access (raises error for secrets)."""
        raise RuntimeError(
            f"Secret access requires async context. Use await secrets.get('{name}') "
            f"or use resolver.resolve_async()"
        )

    def __getitem__(self, key: str) -> Any:
        """Synchronous bracket access (raises error for secrets)."""
        raise RuntimeError(
            f"Secret access requires async context. Use await secrets.get('{key}') "
            f"or use resolver.resolve_async()"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute modification"""
        raise AttributeError("SecretProxy is read-only")

    def __delattr__(self, name: str) -> None:
        """Prevent attribute deletion"""
        raise AttributeError("SecretProxy is read-only")
