from __future__ import annotations

import secrets
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.secrets_repo import SQLiteSecretsRepository
from workflows_mcp.security.crypto import (
    SecretCryptoError,
    SecretKeyError,
    decrypt_secret_value,
    encrypt_secret_value,
    load_secret_key,
)
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase4-admin-password"


def _write_valid_key(path: Path) -> None:
    path.write_bytes(secrets.token_bytes(32))


def _bootstrap_base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _MCP_BOOTSTRAP_TOKEN)
    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )
    return base_dir


def _client_from_base_dir(base_dir: Path) -> TestClient:
    return TestClient(
        build_app(base_dir=base_dir),
        base_url="http://testserver",
        raise_server_exceptions=False,
    )


def _login_and_csrf(client: TestClient) -> str:
    login_response = client.post("/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD})
    assert login_response.status_code == 200
    csrf_token = login_response.headers.get("X-CSRF-Token") or login_response.json().get(
        "csrf_token"
    )
    assert csrf_token
    return str(csrf_token)


def test_encrypt_decrypt_round_trip_uses_versioned_envelope(tmp_path: Path) -> None:
    key_path = tmp_path / "secrets.key"
    _write_valid_key(key_path)
    key = load_secret_key(key_path)

    envelope = encrypt_secret_value("super-secret-value", key)

    assert "super-secret-value" not in envelope
    assert '"v":1' in envelope
    assert decrypt_secret_value(envelope, key) == "super-secret-value"


def test_load_secret_key_missing_or_invalid_fails_closed(tmp_path: Path) -> None:
    missing_key_path = tmp_path / "missing.key"
    with pytest.raises(SecretKeyError):
        load_secret_key(missing_key_path)

    invalid_key_path = tmp_path / "invalid.key"
    invalid_key_path.write_bytes(b"too-short")

    with pytest.raises(SecretKeyError):
        load_secret_key(invalid_key_path)


def test_repository_stores_only_ciphertext_and_hides_values(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    key_path = tmp_path / "secrets.key"
    _write_valid_key(key_path)

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)
        repo = SQLiteSecretsRepository(conn=conn, key_path=key_path)

        first_meta = repo.upsert_secret("OPENAI_API_KEY", "sk-live-123", key_id="k1")
        second_meta = repo.upsert_secret("OPENAI_API_KEY", "sk-live-123", key_id="k1")

        # Public metadata never includes secret material.
        assert first_meta.name == "OPENAI_API_KEY"
        assert second_meta.name == "OPENAI_API_KEY"
        assert not hasattr(first_meta, "value")
        assert not hasattr(first_meta, "ciphertext")

        row = conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
            ("OPENAI_API_KEY",),
        ).fetchone()
        assert row is not None
        stored_payload = str(row[0])
        assert "sk-live-123" not in stored_payload

        repo.upsert_secret("OPENAI_API_KEY", "sk-live-123", key_id="k1")
        row_2 = conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
            ("OPENAI_API_KEY",),
        ).fetchone()
        assert row_2 is not None
        assert str(row_2[0]) != stored_payload

        listed = repo.list_metadata()
        assert [item.name for item in listed] == ["OPENAI_API_KEY"]
        assert repo.get_secret_value("OPENAI_API_KEY") == "sk-live-123"
    finally:
        conn.close()


def test_repository_secret_access_fails_closed_for_missing_or_corrupt_key(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.db"
    key_path = tmp_path / "secrets.key"
    _write_valid_key(key_path)

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)
        repo = SQLiteSecretsRepository(conn=conn, key_path=key_path)
        repo.upsert_secret("SERVICE_TOKEN", "token-xyz")

        key_path.unlink()
        with pytest.raises(SecretKeyError):
            repo.get_secret_value("SERVICE_TOKEN")

        key_path.write_bytes(b"corrupt")
        with pytest.raises(SecretKeyError):
            repo.get_secret_value("SERVICE_TOKEN")
    finally:
        conn.close()


def test_bootstrap_generated_secrets_key_is_used_for_repository_crypto(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    bootstrap_if_needed(
        config_dir=config_dir,
        host="127.0.0.1",
        port=8765,
        admin_password="AdminPass-123!",
    )

    key_path = config_dir / "secrets.key"
    db_path = config_dir / "server.db"
    assert key_path.exists()

    key = load_secret_key(key_path)
    assert len(key) == 32

    conn = connect_metadata_db(db_path)
    try:
        repo = SQLiteSecretsRepository(conn=conn, key_path=key_path)
        repo.upsert_secret("BOOTSTRAP_SECRET", "bootstrap-value")
        assert repo.get_secret_value("BOOTSTRAP_SECRET") == "bootstrap-value"
    finally:
        conn.close()


def test_repository_delete_secret_returns_status_and_removes_value(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    key_path = tmp_path / "secrets.key"
    _write_valid_key(key_path)

    conn = connect_metadata_db(db_path)
    try:
        migrate_metadata_db(conn)
        repo = SQLiteSecretsRepository(conn=conn, key_path=key_path)

        repo.upsert_secret("DELETE_ME", "value")
        assert repo.delete_secret("DELETE_ME") is True
        assert repo.get_secret_value("DELETE_ME") is None
        assert repo.delete_secret("DELETE_ME") is False
    finally:
        conn.close()


def test_decrypt_secret_value_unsupported_or_corrupt_envelope_fails_closed(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "secrets.key"
    _write_valid_key(key_path)
    key = load_secret_key(key_path)

    with pytest.raises(SecretCryptoError):
        decrypt_secret_value('{"v":2,"alg":"AES-256-GCM","nonce":"AA==","ciphertext":"AA=="}', key)

    with pytest.raises(SecretCryptoError):
        decrypt_secret_value('{"v":1,"alg":"UNKNOWN","nonce":"AA==","ciphertext":"AA=="}', key)

    with pytest.raises(SecretCryptoError):
        decrypt_secret_value("not-json", key)


def test_admin_secrets_list_and_create_require_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))

    unauth_list = client.get("/api/admin/v1/secrets")
    assert unauth_list.status_code == 401

    unauth_create = client.post(
        "/api/admin/v1/secrets",
        json={"name": "OPENAI_API_KEY", "value": "sk-secret"},
    )
    assert unauth_create.status_code == 401


def test_admin_secrets_list_works_for_logged_in_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    _login_and_csrf(client)

    response = client.get("/api/admin/v1/secrets")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {"secrets": []}


def test_admin_secrets_create_update_list_delete_never_returns_secret_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    client = _client_from_base_dir(base_dir)
    csrf_token = _login_and_csrf(client)

    create_response = client.post(
        "/api/admin/v1/secrets",
        json={"name": "OPENAI_API_KEY", "value": "sk-live-123", "key_id": "k1"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert set(created.keys()) == {"name", "key_id", "created_at", "updated_at"}
    assert created["name"] == "OPENAI_API_KEY"
    assert created["key_id"] == "k1"
    assert "value" not in created
    assert "encrypted_payload" not in created
    assert "ciphertext" not in created

    update_response = client.post(
        "/api/admin/v1/secrets",
        json={"name": "OPENAI_API_KEY", "value": "sk-live-456", "key_id": None},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert set(updated.keys()) == {"name", "key_id", "created_at", "updated_at"}
    assert updated["name"] == "OPENAI_API_KEY"
    assert updated["key_id"] is None
    assert "value" not in updated
    assert "encrypted_payload" not in updated
    assert "ciphertext" not in updated

    list_response = client.get("/api/admin/v1/secrets")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert "secrets" in listed
    assert len(listed["secrets"]) == 1
    only_item = listed["secrets"][0]
    assert set(only_item.keys()) == {"name", "key_id", "created_at", "updated_at"}
    assert only_item["name"] == "OPENAI_API_KEY"
    assert "value" not in only_item
    assert "encrypted_payload" not in only_item
    assert "ciphertext" not in only_item

    conn = connect_metadata_db(base_dir / "server.db")
    try:
        row = conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
            ("OPENAI_API_KEY",),
        ).fetchone()
        assert row is not None
        encrypted_payload = str(row[0])
        assert "sk-live-456" not in encrypted_payload
        assert "ciphertext" in encrypted_payload
    finally:
        conn.close()

    delete_response = client.delete(
        "/api/admin/v1/secrets/OPENAI_API_KEY",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True}

    list_after_delete = client.get("/api/admin/v1/secrets")
    assert list_after_delete.status_code == 200
    assert list_after_delete.json() == {"secrets": []}


def test_admin_secrets_post_and_delete_require_csrf_with_session_cookie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    post_missing_csrf = client.post(
        "/api/admin/v1/secrets",
        json={"name": "OPENAI_API_KEY", "value": "sk-live-123"},
    )
    assert post_missing_csrf.status_code == 403

    post_ok = client.post(
        "/api/admin/v1/secrets",
        json={"name": "OPENAI_API_KEY", "value": "sk-live-123"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert post_ok.status_code == 200

    delete_missing_csrf = client.delete("/api/admin/v1/secrets/OPENAI_API_KEY")
    assert delete_missing_csrf.status_code == 403

    delete_ok = client.delete(
        "/api/admin/v1/secrets/OPENAI_API_KEY",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert delete_ok.status_code == 200


def test_admin_secrets_missing_or_corrupt_key_fails_closed_with_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    client = _client_from_base_dir(base_dir)
    csrf_token = _login_and_csrf(client)

    key_path = base_dir / "secrets.key"
    key_path.unlink()

    missing_key_response = client.post(
        "/api/admin/v1/secrets",
        json={"name": "MISSING_KEY_SECRET", "value": "plain-value"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert missing_key_response.status_code == 503
    assert "secrets.key" in str(missing_key_response.json()).lower()

    conn = connect_metadata_db(base_dir / "server.db")
    try:
        row = conn.execute(
            "SELECT encrypted_payload FROM encrypted_secret_metadata WHERE secret_name = ?",
            ("MISSING_KEY_SECRET",),
        ).fetchone()
        assert row is None
    finally:
        conn.close()

    key_path.write_bytes(b"corrupt")
    corrupt_key_response = client.post(
        "/api/admin/v1/secrets",
        json={"name": "CORRUPT_KEY_SECRET", "value": "plain-value"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert corrupt_key_response.status_code == 503
    assert "secrets.key" in str(corrupt_key_response.json()).lower()


def test_mcp_bearer_token_cannot_access_admin_secrets_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))

    response = client.get(
        "/api/admin/v1/secrets",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )

    assert response.status_code in {401, 403}


def test_mcp_bearer_token_cannot_mutate_admin_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))

    response = client.post(
        "/api/admin/v1/secrets",
        json={"name": "OPENAI_API_KEY", "value": "sk-live-123"},
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )

    assert response.status_code in {401, 403}


@pytest.mark.parametrize(
    "bad_name",
    ["", "1INVALID", "INVALID-NAME", "INVALID NAME", "A" * 129],
)
def test_admin_secrets_post_rejects_invalid_secret_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_name: str
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    response = client.post(
        "/api/admin/v1/secrets",
        json={"name": bad_name, "value": "some-value"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 422


def test_admin_secrets_post_rejects_empty_or_oversized_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    empty_response = client.post(
        "/api/admin/v1/secrets",
        json={"name": "VALID_NAME", "value": ""},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert empty_response.status_code == 422

    oversized_response = client.post(
        "/api/admin/v1/secrets",
        json={"name": "VALID_NAME", "value": "x" * 65537},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert oversized_response.status_code == 422


def test_admin_secrets_post_rejects_oversized_key_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    response = client.post(
        "/api/admin/v1/secrets",
        json={"name": "VALID_NAME", "value": "value", "key_id": "k" * 257},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 422


@pytest.mark.parametrize("bad_name", ["BAD-NAME", "A" * 129])
def test_admin_secrets_delete_rejects_invalid_secret_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_name: str
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    response = client.delete(
        f"/api/admin/v1/secrets/{bad_name}",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 422


def test_admin_secrets_fail_closed_error_does_not_leak_paths_or_internals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    client = _client_from_base_dir(base_dir)
    csrf_token = _login_and_csrf(client)

    key_path = base_dir / "secrets.key"
    key_path.unlink()

    response = client.post(
        "/api/admin/v1/secrets",
        json={"name": "MISSING_KEY_SECRET", "value": "plain-value"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 503
    payload = str(response.json()).lower()
    assert "secrets.key" in payload
    assert str(base_dir).lower() not in payload
    assert "reason:" not in payload
    assert "traceback" not in payload
