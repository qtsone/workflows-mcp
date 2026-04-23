"""HTTP startup integration tests for Task 7.

Verifies that ``build_app()`` composes the full HTTP application surface
and that the resulting FastAPI app exposes the required public endpoints.
Also verifies fail-fast bootstrap token enforcement per spec §7.4.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.auth import BootstrapTokenError, TokenStore
from workflows_mcp.server import build_app

# ---------------------------------------------------------------------------
# Bootstrap token enforcement (spec §7.4: fail-fast on first start)
# ---------------------------------------------------------------------------


def test_build_app_raises_on_missing_bootstrap_token(tmp_path: Path) -> None:
    """build_app() must raise BootstrapTokenError when no token store exists
    and WORKFLOWS_BOOTSTRAP_TOKEN is not set, per spec §7.4."""
    # tmp_path has no auth.json and no env var is present in CI/test runs
    with pytest.raises(BootstrapTokenError):
        build_app(base_dir=tmp_path)


def test_build_app_succeeds_when_token_store_exists(tmp_path: Path) -> None:
    """build_app() must succeed without WORKFLOWS_BOOTSTRAP_TOKEN when an
    existing token store is already in place (subsequent starts)."""
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("a" * 40)  # write a valid hash — no env var needed

    app = build_app(base_dir=tmp_path)
    client = TestClient(app)
    assert client.get("/health").status_code == 200


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
