from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

ALGORITHM = "pbkdf2_sha256"
MIN_ITERATIONS = 600_000
DEFAULT_ITERATIONS = 600_000
MAX_ITERATIONS = 2_000_000
SALT_BYTES = 16
DKLEN = 32


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


def hash_password(password: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    if not (MIN_ITERATIONS <= iterations <= MAX_ITERATIONS):
        msg = f"iterations must be between {MIN_ITERATIONS} and {MAX_ITERATIONS}"
        raise ValueError(msg)

    salt = secrets.token_bytes(SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=DKLEN,
    )
    return f"{ALGORITHM}${iterations}${_b64encode(salt)}${_b64encode(derived)}"


def verify_password(password: str, password_hash: str) -> bool:
    parts = password_hash.split("$")
    if len(parts) != 4:
        return False

    algorithm, iterations_text, salt_text, expected_text = parts
    if algorithm != ALGORITHM:
        return False

    try:
        iterations = int(iterations_text)
        if not (MIN_ITERATIONS <= iterations <= MAX_ITERATIONS):
            return False
        salt = _b64decode(salt_text)
        expected = _b64decode(expected_text)
        if len(salt) != SALT_BYTES:
            return False
        if len(expected) != DKLEN:
            return False
    except (ValueError, TypeError):
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=len(expected),
    )
    return hmac.compare_digest(candidate, expected)
