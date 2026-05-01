"""HTTP startup integration tests for Task 7.

Verifies that ``build_app()`` composes the full HTTP application surface
and that the resulting FastAPI app exposes the required public endpoints.
Also verifies startup behavior around legacy bootstrap token storage.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from workflows_mcp.auth import TokenStore
from workflows_mcp.server import build_app

# ---------------------------------------------------------------------------
# Bootstrap token startup behavior
# ---------------------------------------------------------------------------


def test_build_app_succeeds_without_legacy_bootstrap_token_or_auth_store(
    tmp_path: Path,
) -> None:
    """build_app() must not fail when WORKFLOWS_BOOTSTRAP_TOKEN/auth.json are absent.

    V1 admin auth/bootstrap state is server.db + secrets.key driven, so legacy
    token store is no longer a startup precondition for no-arg runs.
    """
    app = build_app(base_dir=tmp_path)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert (tmp_path / "auth.json").exists() is False


def test_build_app_succeeds_when_token_store_exists(tmp_path: Path) -> None:
    """build_app() must still succeed when legacy auth.json already exists."""
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("a" * 40)  # write a valid hash — no env var needed

    app = build_app(base_dir=tmp_path)
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_unbootstrapped_app_rejects_admin_session_endpoint_without_session(
    tmp_path: Path,
) -> None:
    """Unbootstrapped server must fail closed on admin endpoint access.

    With no legacy auth.json and no admin UI session cookie, admin v1 routes must
    deny access (401/403), never succeed.
    """
    client = TestClient(build_app(base_dir=tmp_path), raise_server_exceptions=False)

    response = client.get("/api/admin/v1/auth/session")

    assert (tmp_path / "auth.json").exists() is False
    assert response.status_code in {401, 403}


# ---------------------------------------------------------------------------
# Public surface availability
# ---------------------------------------------------------------------------


def test_http_server_exposes_health(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("b" * 40)

    client = TestClient(build_app(base_dir=tmp_path))
    assert client.get("/health").status_code == 200


def test_http_server_exposes_openapi(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("c" * 40)

    client = TestClient(build_app(base_dir=tmp_path))
    assert client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------------------
# build_app() return type
# ---------------------------------------------------------------------------


def test_build_app_returns_fastapi_instance(tmp_path: Path) -> None:
    """build_app() must return a FastAPI instance (proper annotation, not Any)."""
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("d" * 40)

    app = build_app(base_dir=tmp_path)
    assert isinstance(app, FastAPI)


# ---------------------------------------------------------------------------
# WORKFLOWS_PORT: invalid value must not crash with an unhandled ValueError
# ---------------------------------------------------------------------------


def test_main_rejects_invalid_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """main() must log a clear error and exit cleanly for non-integer WORKFLOWS_PORT.

    The error must be caught before calling uvicorn so the log message is
    actionable (not a raw ValueError traceback from int() conversion).
    """
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("e" * 40)

    monkeypatch.setenv("WORKFLOWS_PORT", "not-a-number")
    monkeypatch.setenv("WORKFLOWS_LOG_LEVEL", "WARNING")

    from workflows_mcp import server

    with (
        patch.object(server, "build_app", return_value=None),
        caplog.at_level(logging.ERROR),
        pytest.raises(SystemExit) as exc_info,
    ):
        server.main()

    assert exc_info.value.code == 1
    # Must emit a clear config-level message, not a raw int() ValueError traceback
    assert any("WORKFLOWS_PORT" in record.message for record in caplog.records)
