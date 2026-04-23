"""Bootstrap token storage and authentication helpers.

Security model
--------------
- Tokens are **never** stored in plaintext.  Only their SHA-256 digest is
  written to disk.
- The token store file is created with mode ``0600`` so only the owning
  user can read or write it.
- Constant-time comparison (``hmac.compare_digest``) is used during
  validation to prevent timing-oracle attacks.
- The ``WORKFLOWS_BOOTSTRAP_TOKEN`` environment variable must carry at
  least 32 bytes of UTF-8 entropy.  Anything shorter is rejected at
  startup to prevent accidental weak secrets.
- Token material is **never** logged or returned in API responses.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum required entropy: 32 bytes encoded as UTF-8.
_MIN_TOKEN_BYTES: int = 32


class BootstrapTokenError(ValueError):
    """Raised when the bootstrap token is absent or does not meet the minimum
    entropy requirement."""


def _hash_token(token: str) -> str:
    """Return the hex-encoded SHA-256 digest of *token*.

    The raw token is never stored; only this digest is written to disk.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class TokenStore:
    """Atomic, permission-safe store for a single hashed bearer token.

    Parameters
    ----------
    path:
        Absolute path to the JSON file used as the backing store.
        The parent directory is created on first write if it does not
        already exist.
    """

    path: Path

    def write_token(self, token: str) -> None:
        """Hash *token* and persist only the digest to :attr:`path`.

        The write is atomic: the payload is flushed and fsync'd to a
        temporary file in the same directory, then renamed over the
        target path.  File mode ``0600`` is set immediately after the
        rename.

        Parameters
        ----------
        token:
            The raw bearer token value.  It is hashed immediately and
            never written to disk or logs.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"token_hash": _hash_token(token)}
        # Write to a sibling temp file then atomically replace target.
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=self.path.parent, suffix=".tmp"
        ) as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_name = handle.name
        Path(tmp_name).replace(self.path)
        # Restrict read/write to the owner only.
        self.path.chmod(0o600)
        logger.debug("Token store written to %s", self.path)

    def rotate_token(self, token: str) -> None:
        """Replace the current token with a new one.

        The previous token is immediately invalidated.  The write is atomic
        (via :meth:`write_token`) so no window exists where neither token is
        valid.  File mode ``0600`` is enforced after the rename.

        Parameters
        ----------
        token:
            The new raw bearer token value.  It is hashed before storage.
        """
        self.write_token(token)
        logger.debug("Token rotated at %s", self.path)

    def revoke(self) -> None:
        """Delete the token store, immediately invalidating all tokens.

        Calling this method when no store file exists is a no-op.

        Recovery:
            After revocation, calling :func:`ensure_bootstrap_token` with
            ``WORKFLOWS_BOOTSTRAP_TOKEN`` set will issue a new admin token
            without requiring a reinstall.
        """
        if self.path.exists():
            self.path.unlink()
            logger.info("Token store revoked and removed at %s", self.path)
        else:
            logger.debug("revoke() called but no store exists at %s", self.path)

    def validate(self, token: str) -> bool:
        """Return ``True`` iff *token* matches the stored digest.

        Uses :func:`hmac.compare_digest` for constant-time comparison.
        Returns ``False`` (rather than raising) when the store file is
        absent, so callers can treat a missing store as "not yet
        initialised" without extra existence checks.

        Parameters
        ----------
        token:
            The raw bearer token to verify.
        """
        if not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text())
            expected: str = str(payload["token_hash"])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Token store at %s is corrupt: %s", self.path, exc)
            return False
        return hmac.compare_digest(expected, _hash_token(token))


def ensure_bootstrap_token(base_dir: Path) -> TokenStore:
    """Load and validate the bootstrap token from the environment.

    On the very first start (no ``auth.json`` exists yet), the raw token
    is hashed and written to *base_dir*/auth.json.  On subsequent calls
    the store is returned as-is so that an operator-provided token does
    not silently overwrite a previously initialised store.

    Parameters
    ----------
    base_dir:
        Directory under which ``auth.json`` will be written.  Typically
        ``~/.workflows/``.

    Returns
    -------
    TokenStore
        A :class:`TokenStore` pointing at *base_dir*/auth.json.

    Raises
    ------
    BootstrapTokenError
        If the auth store does not yet exist and ``WORKFLOWS_BOOTSTRAP_TOKEN``
        is unset or its UTF-8 encoding is shorter than
        :data:`_MIN_TOKEN_BYTES` bytes.  When the store already exists the
        env var is not required and is ignored.
    """
    store = TokenStore(base_dir / "auth.json")
    if store.path.exists():
        logger.debug("Token store already exists at %s; skipping bootstrap", store.path)
        return store
    # First start: store absent — bootstrap token is mandatory.
    raw = os.getenv("WORKFLOWS_BOOTSTRAP_TOKEN")
    if raw is None or len(raw.encode("utf-8")) < _MIN_TOKEN_BYTES:
        raise BootstrapTokenError(
            "WORKFLOWS_BOOTSTRAP_TOKEN is required on first start and must be "
            f"at least {_MIN_TOKEN_BYTES} bytes of UTF-8 entropy.  "
            "Set the environment variable to a sufficiently long random value "
            "before starting the service."
        )
    store.write_token(raw)
    logger.info("Bootstrap token written to %s", store.path)
    return store


def generate_request_id() -> str:
    """Return a cryptographically random hex string for use as a request ID."""
    return secrets.token_hex(16)
