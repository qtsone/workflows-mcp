from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.engine.llm_config import LLMConfig
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos import SQLiteLLMConfigRepository
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase4-admin-password"


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


def _sample_config() -> LLMConfig:
    config = LLMConfig(
        providers={
            "openai-cloud": {
                "type": "openai",
                "api_url": "https://api.openai.com/v1/chat/completions",
                "api_key_secret": "OPENAI_API_KEY",
                "timeout": 120,
                "max_retries": 4,
                "retry_delay": 1.5,
                "extra_headers": {"x-tenant": "ops"},
            },
            "anthropic-cloud": {
                "type": "anthropic",
                "api_url": "https://api.anthropic.com/v1/messages",
                "api_key_secret": "ANTHROPIC_API_KEY",
            },
        },
        profiles={
            "quick": {
                "provider": "openai-cloud",
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 1200,
                "description": "fast and cheap",
            },
            "deep": {
                "provider": "anthropic-cloud",
                "model": "claude-sonnet-4",
                "temperature": 0.9,
                "max_tokens": 4000,
            },
        },
        default_profile="quick",
    )
    config.validate_profile_provider_references()
    return config


def _sample_yaml() -> str:
    return """
version: "1.0"
providers:
  openai-cloud:
    type: openai
    api_url: https://api.openai.com/v1/chat/completions
    api_key_secret: OPENAI_API_KEY
    timeout: 120
    max_retries: 4
    retry_delay: 1.5
    extra_headers:
      x-tenant: ops
profiles:
  quick:
    provider: openai-cloud
    model: gpt-4o-mini
    temperature: 0.2
    max_tokens: 1200
default_profile: quick
"""


def _semantic_dump(config: LLMConfig) -> dict[str, object]:
    return config.model_dump(mode="python", exclude_none=True)


def test_load_config_returns_empty_when_no_llm_rows(tmp_path: Path) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteLLMConfigRepository(conn)

        loaded = repo.load_config()
        assert loaded.providers == {}
        assert loaded.profiles == {}
        assert loaded.default_profile is None
    finally:
        conn.close()


def test_replace_and_load_roundtrip_persists_sqlite_source_of_truth(tmp_path: Path) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteLLMConfigRepository(conn)
        expected = _sample_config()

        repo.replace_config(expected)
        loaded = repo.load_config()

        assert _semantic_dump(loaded) == _semantic_dump(expected)
    finally:
        conn.close()


def test_preview_yaml_validates_without_persisting(tmp_path: Path) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteLLMConfigRepository(conn)

        previewed = repo.preview_yaml(_sample_yaml())
        assert previewed.default_profile == "quick"

        loaded_after_preview = repo.load_config()
        assert loaded_after_preview.providers == {}
        assert loaded_after_preview.profiles == {}
        assert loaded_after_preview.default_profile is None
    finally:
        conn.close()


def test_import_yaml_persists_normalized_config(tmp_path: Path) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteLLMConfigRepository(conn)

        imported = repo.import_yaml(_sample_yaml())
        loaded = repo.load_config()

        assert _semantic_dump(imported) == _semantic_dump(loaded)
    finally:
        conn.close()


def test_export_yaml_emits_sqlite_config_yaml(tmp_path: Path) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteLLMConfigRepository(conn)
        expected = _sample_config()

        repo.replace_config(expected)
        exported_yaml = repo.export_yaml()
        parsed = repo.preview_yaml(exported_yaml)

        assert _semantic_dump(parsed) == _semantic_dump(expected)
    finally:
        conn.close()


def test_preview_yaml_rejects_non_mapping_yaml_and_does_not_persist(tmp_path: Path) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteLLMConfigRepository(conn)

        with pytest.raises(ValueError, match="must parse to a mapping"):
            repo.preview_yaml("- quick\n- deep\n")

        loaded = repo.load_config()
        assert loaded.providers == {}
        assert loaded.profiles == {}
        assert loaded.default_profile is None
    finally:
        conn.close()


def test_import_yaml_rejects_malformed_yaml_and_does_not_overwrite_existing_config(
    tmp_path: Path,
) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteLLMConfigRepository(conn)
        baseline = _sample_config()
        repo.replace_config(baseline)

        with pytest.raises(yaml.YAMLError):
            repo.import_yaml("providers: [unterminated")

        loaded = repo.load_config()
        assert _semantic_dump(loaded) == _semantic_dump(baseline)
    finally:
        conn.close()


def test_preview_yaml_rejects_profile_provider_mismatch(tmp_path: Path) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteLLMConfigRepository(conn)

        raw_yaml = """
version: "1.0"
providers:
  openai-cloud:
    type: openai
profiles:
  broken:
    provider: missing-provider
    model: gpt-4o-mini
"""
        with pytest.raises(ValueError, match="references unknown provider"):
            repo.preview_yaml(raw_yaml)
    finally:
        conn.close()


def test_preview_yaml_rejects_default_profile_mismatch(tmp_path: Path) -> None:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    try:
        migrate_metadata_db(conn)
        repo = SQLiteLLMConfigRepository(conn)

        raw_yaml = """
version: "1.0"
providers:
  openai-cloud:
    type: openai
profiles:
  quick:
    provider: openai-cloud
    model: gpt-4o-mini
default_profile: does-not-exist
"""
        with pytest.raises(ValueError, match="default_profile 'does-not-exist' not found"):
            repo.preview_yaml(raw_yaml)
    finally:
        conn.close()


def test_admin_llm_routes_require_auth_and_logged_in_get_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))

    unauth_get = client.get("/api/admin/v1/llm/config")
    assert unauth_get.status_code == 401

    unauth_export = client.get("/api/admin/v1/llm/export")
    assert unauth_export.status_code == 401

    _login_and_csrf(client)
    auth_get = client.get("/api/admin/v1/llm/config")
    assert auth_get.status_code == 200
    assert auth_get.json() == {
        "version": "1.0",
        "providers": {},
        "profiles": {},
        "default_profile": None,
    }


def test_admin_llm_mutating_routes_require_csrf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)
    body = _sample_config().model_dump(mode="json", exclude_none=False)

    put_missing_csrf = client.put("/api/admin/v1/llm/config", json=body)
    assert put_missing_csrf.status_code == 403

    preview_missing_csrf = client.post(
        "/api/admin/v1/llm/preview",
        json={"raw_yaml": _sample_yaml()},
    )
    assert preview_missing_csrf.status_code == 403

    import_missing_csrf = client.post("/api/admin/v1/llm/import", json={"raw_yaml": _sample_yaml()})
    assert import_missing_csrf.status_code == 403

    put_ok = client.put("/api/admin/v1/llm/config", json=body, headers={"X-CSRF-Token": csrf_token})
    assert put_ok.status_code == 200


def test_admin_llm_put_persists_to_sqlite_and_response_has_secret_references_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    client = _client_from_base_dir(base_dir)
    csrf_token = _login_and_csrf(client)
    payload = _sample_config().model_dump(mode="json", exclude_none=False)

    response = client.put(
        "/api/admin/v1/llm/config",
        json=payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    response_payload = response.json()
    assert response_payload["providers"]["openai-cloud"]["api_key_secret"] == "OPENAI_API_KEY"
    assert response_payload["providers"]["anthropic-cloud"]["api_key_secret"] == "ANTHROPIC_API_KEY"
    assert "api_key_value" not in str(response_payload)
    assert "sk-" not in str(response_payload)

    conn = connect_metadata_db(base_dir / "server.db")
    try:
        repo = SQLiteLLMConfigRepository(conn)
        persisted = repo.load_config()
        assert _semantic_dump(persisted) == _semantic_dump(LLMConfig(**response_payload))
    finally:
        conn.close()


def test_admin_llm_preview_validates_without_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    client = _client_from_base_dir(base_dir)
    csrf_token = _login_and_csrf(client)

    before_response = client.get("/api/admin/v1/llm/config")
    assert before_response.status_code == 200
    assert before_response.json()["providers"] == {}

    preview_response = client.post(
        "/api/admin/v1/llm/preview",
        json={"raw_yaml": _sample_yaml()},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert preview_response.status_code == 200
    assert preview_response.json()["default_profile"] == "quick"

    after_response = client.get("/api/admin/v1/llm/config")
    assert after_response.status_code == 200
    assert after_response.json()["providers"] == {}
    assert after_response.json()["profiles"] == {}
    assert after_response.json()["default_profile"] is None


def test_admin_llm_import_persists_and_export_returns_semantic_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = _bootstrap_base_dir(tmp_path, monkeypatch)
    client = _client_from_base_dir(base_dir)
    csrf_token = _login_and_csrf(client)

    import_response = client.post(
        "/api/admin/v1/llm/import",
        json={"raw_yaml": _sample_yaml()},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert import_response.status_code == 200
    imported = LLMConfig(**import_response.json())

    get_response = client.get("/api/admin/v1/llm/config")
    assert get_response.status_code == 200
    loaded = LLMConfig(**get_response.json())
    assert _semantic_dump(loaded) == _semantic_dump(imported)

    export_response = client.get("/api/admin/v1/llm/export")
    assert export_response.status_code == 200
    exported_raw_yaml = export_response.json()["raw_yaml"]
    exported = yaml.safe_load(exported_raw_yaml)
    assert isinstance(exported, dict)
    assert _semantic_dump(LLMConfig(**exported)) == _semantic_dump(imported)


def test_admin_llm_mcp_bearer_is_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))

    response = client.get(
        "/api/admin/v1/llm/config",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert response.status_code == 403


def test_admin_llm_preview_malformed_yaml_returns_sanitized_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    response = client.post(
        "/api/admin/v1/llm/preview",
        json={"raw_yaml": "providers: [unterminated"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "VALIDATION_FAILED"
    assert isinstance(payload["error"].get("message"), str)
    assert "detail" not in payload
    body = str(payload)
    assert "Traceback" not in body
    assert "/Users/" not in body
    assert "yaml." not in body
    assert "YAMLError" not in body


def test_admin_llm_import_invalid_provider_reference_returns_sanitized_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    raw_yaml = """
version: "1.0"
providers:
  openai-cloud:
    type: openai
profiles:
  broken:
    provider: missing-provider
    model: gpt-4o-mini
"""
    response = client.post(
        "/api/admin/v1/llm/import",
        json={"raw_yaml": raw_yaml},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "VALIDATION_FAILED"
    assert isinstance(payload["error"].get("message"), str)
    assert "detail" not in payload
    body = str(payload)
    assert "Traceback" not in body
    assert "/Users/" not in body
    assert "ValueError" not in body
    assert "Profile 'broken'" not in body


def test_admin_llm_import_invalid_default_profile_returns_sanitized_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    raw_yaml = """
version: "1.0"
providers:
  openai-cloud:
    type: openai
profiles:
  quick:
    provider: openai-cloud
    model: gpt-4o-mini
default_profile: does-not-exist
"""
    response = client.post(
        "/api/admin/v1/llm/import",
        json={"raw_yaml": raw_yaml},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 422
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "VALIDATION_FAILED"
    assert isinstance(payload["error"].get("message"), str)
    assert "detail" not in payload
    body = str(payload)
    assert "Traceback" not in body
    assert "/Users/" not in body
    assert "ValidationError" not in body
    assert "does-not-exist" not in body


def test_admin_llm_preview_oversized_raw_yaml_rejected_by_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_from_base_dir(_bootstrap_base_dir(tmp_path, monkeypatch))
    csrf_token = _login_and_csrf(client)

    oversized = "a" * 65537
    response = client.post(
        "/api/admin/v1/llm/preview",
        json={"raw_yaml": oversized},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 422
