"""Integration tests for the protected /mcp HTTP endpoint.

Covers the acceptance criteria from remediation Task 3 (plan section "Task 4"):
- /mcp returns 401 without valid bearer auth.
- /mcp returns 409 when authenticated but service is not ready.
- /mcp provides at least one real delegated operation path when ready.
- All error responses use the stable ErrorEnvelope shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.auth import TokenStore
from workflows_mcp.http_app import create_app
from workflows_mcp.http_models import ReadinessState

_VALID_TOKEN = "a" * 40


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _NotReadyProbe:
    async def check(self) -> tuple[bool, list[str]]:
        return False, ["postgresql_dsn_missing"]


class _ReadyProbe:
    async def check(self) -> tuple[bool, list[str]]:
        return True, []


class _FakeReadinessService:
    def __init__(self, *, ready: bool, base_dir: Path) -> None:
        self._ready = ready
        self._base_dir = base_dir

    async def evaluate(self):  # noqa: ANN201
        class _Report:
            pass

        report = _Report()
        if self._ready:
            report.state = ReadinessState.READY  # type: ignore[attr-defined]
            report.blockers = []  # type: ignore[attr-defined]
        else:
            report.state = ReadinessState.UNCONFIGURED  # type: ignore[attr-defined]
            report.blockers = ["llm_config"]  # type: ignore[attr-defined]
        return report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def token_store(tmp_path: Path) -> TokenStore:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token(_VALID_TOKEN)
    return store


@pytest.fixture()
def not_ready_client(tmp_path: Path, token_store: TokenStore) -> TestClient:
    readiness = _FakeReadinessService(ready=False, base_dir=tmp_path)
    app = create_app(readiness_service=readiness, token_store=token_store)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def ready_client(tmp_path: Path, token_store: TokenStore) -> TestClient:
    readiness = _FakeReadinessService(ready=True, base_dir=tmp_path)
    app = create_app(readiness_service=readiness, token_store=token_store)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# AUTH tests (unauthenticated → 401)
# ---------------------------------------------------------------------------


def test_mcp_requires_auth_no_header(not_ready_client: TestClient) -> None:
    """POST /mcp without Authorization header must return 401."""
    response = not_ready_client.post("/mcp", json={})
    assert response.status_code == 401


def test_mcp_requires_auth_wrong_token(not_ready_client: TestClient) -> None:
    """POST /mcp with an invalid token must return 401."""
    response = not_ready_client.post(
        "/mcp",
        headers={"Authorization": "Bearer wrong-token"},
        json={},
    )
    assert response.status_code == 401


def test_mcp_401_uses_error_envelope(not_ready_client: TestClient) -> None:
    """401 from /mcp must use the stable ErrorEnvelope shape."""
    response = not_ready_client.post("/mcp", json={})
    assert response.status_code == 401
    payload = response.json()
    assert "error" in payload
    assert set(payload["error"]) >= {"code", "message", "request_id"}
    assert payload["error"]["code"] == "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# READINESS gate tests (authenticated but not ready → 409)
# ---------------------------------------------------------------------------


def test_mcp_returns_409_when_not_ready(not_ready_client: TestClient) -> None:
    """POST /mcp when authenticated but service not ready must return 409."""
    response = not_ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        json={},
    )
    assert response.status_code == 409


def test_mcp_409_uses_error_envelope(not_ready_client: TestClient) -> None:
    """409 from /mcp must use the stable ErrorEnvelope with CONFIG_REQUIRED code."""
    response = not_ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        json={},
    )
    assert response.status_code == 409
    payload = response.json()
    assert "error" in payload
    assert set(payload["error"]) >= {"code", "message", "request_id"}
    assert payload["error"]["code"] == "CONFIG_REQUIRED"


def test_mcp_409_includes_readiness_details(not_ready_client: TestClient) -> None:
    """409 from /mcp must include readiness_state in error details."""
    response = not_ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        json={},
    )
    payload = response.json()
    assert payload["error"]["details"] is not None
    assert "readiness_state" in payload["error"]["details"]


# ---------------------------------------------------------------------------
# Delegated operation tests (authenticated + ready → 200)
#
# method=schema delegates to tools_memory.memory_schema_payload() — a real
# production function already used by the MCP memory(operation="schema") path.
# It requires no DB connectivity, making it the cleanest real adapter callable
# from the HTTP layer without the MCP lifespan context.
# ---------------------------------------------------------------------------


def test_mcp_schema_returns_200_when_ready(ready_client: TestClient) -> None:
    """POST /mcp with method=schema must return 200 when service is ready."""
    response = ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        json={"method": "schema"},
    )
    assert response.status_code == 200


def test_mcp_schema_returns_real_adapter_output(ready_client: TestClient) -> None:
    """POST /mcp with method=schema must return output from memory_schema_payload().

    Verifies the endpoint delegates to the real adapter rather than a stub.
    The response must contain the version and operations keys produced by the
    existing tools_memory.memory_schema_payload() function.
    """
    from workflows_mcp.tools_memory import memory_schema_payload

    response = ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        json={"method": "schema"},
    )
    payload = response.json()
    expected = memory_schema_payload()
    # The /mcp response wraps the adapter output under a "result" key.
    assert "result" in payload
    assert payload["result"]["version"] == expected["version"]
    assert payload["result"]["operations"] == expected["operations"]


def test_mcp_unknown_method_returns_400(ready_client: TestClient) -> None:
    """POST /mcp with an unknown method must return 400 with a deterministic error."""
    response = ready_client.post(
        "/mcp",
        headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
        json={"method": "unknown_method"},
    )
    assert response.status_code == 400
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "METHOD_NOT_FOUND"


def test_mcp_build_app_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build_app() wires /mcp correctly — auth guard works end-to-end."""
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
    from workflows_mcp.server import build_app

    app = build_app(base_dir=tmp_path / ".workflows")
    client = TestClient(app, raise_server_exceptions=False)

    # No auth → 401
    response = client.post("/mcp", json={})
    assert response.status_code == 401

    # Authenticated but not ready (fresh base_dir has no llm-config.yml) → 409
    response = client.post(
        "/mcp",
        headers={"Authorization": "Bearer 0123456789abcdef0123456789abcdef"},
        json={},
    )
    assert response.status_code == 409
