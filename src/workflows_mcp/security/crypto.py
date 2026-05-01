from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_SIZE_BYTES = 32
_NONCE_SIZE_BYTES = 12


class SecretKeyError(RuntimeError):
    """Raised when the local secrets key is missing or invalid."""


class SecretCryptoError(RuntimeError):
    """Raised when encrypted secret payloads are invalid or cannot decrypt."""


@dataclass(frozen=True)
class EncryptedSecretEnvelope:
    v: int
    alg: str
    nonce_b64: str
    ciphertext_b64: str


def load_secret_key(key_path: Path) -> bytes:
    try:
        key = key_path.read_bytes()
    except FileNotFoundError as exc:
        raise SecretKeyError("Secret key file is missing") from exc
    except OSError as exc:
        raise SecretKeyError("Secret key file cannot be read") from exc

    if len(key) != _KEY_SIZE_BYTES:
        raise SecretKeyError("Secret key length is invalid")
    return key


def encrypt_secret_value(value: str, key: bytes) -> str:
    if len(key) != _KEY_SIZE_BYTES:
        raise SecretKeyError("Secret key length is invalid")

    nonce = os.urandom(_NONCE_SIZE_BYTES)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)

    envelope = EncryptedSecretEnvelope(
        v=1,
        alg="AES-256-GCM",
        nonce_b64=base64.b64encode(nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
    )
    return json.dumps(
        {
            "v": envelope.v,
            "alg": envelope.alg,
            "nonce": envelope.nonce_b64,
            "ciphertext": envelope.ciphertext_b64,
        },
        separators=(",", ":"),
    )


def decrypt_secret_value(envelope_json: str, key: bytes) -> str:
    if len(key) != _KEY_SIZE_BYTES:
        raise SecretKeyError("Secret key length is invalid")

    try:
        raw = json.loads(envelope_json)
        v = int(raw["v"])
        alg = str(raw["alg"])
        nonce = base64.b64decode(str(raw["nonce"]), validate=True)
        ciphertext = base64.b64decode(str(raw["ciphertext"]), validate=True)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SecretCryptoError("Encrypted secret payload is invalid") from exc

    if v != 1 or alg != "AES-256-GCM" or len(nonce) != _NONCE_SIZE_BYTES:
        raise SecretCryptoError("Encrypted secret payload metadata is unsupported")

    try:
        plaintext_bytes = AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception as exc:  # pragma: no cover - type from cryptography backend
        raise SecretCryptoError("Encrypted secret payload failed authentication") from exc

    try:
        return plaintext_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SecretCryptoError("Decrypted secret payload is not valid UTF-8") from exc
