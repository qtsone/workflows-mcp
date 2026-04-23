"""HTTP startup integration tests for Task 7.

Verifies that ``build_app()`` composes the full HTTP application surface
and that the resulting FastAPI app exposes the required public endpoints.
"""

from fastapi.testclient import TestClient

from workflows_mcp.server import build_app


def test_http_server_exposes_health() -> None:
    client = TestClient(build_app())
    assert client.get("/health").status_code == 200


def test_http_server_exposes_openapi() -> None:
    client = TestClient(build_app())
    assert client.get("/openapi.json").status_code == 200
