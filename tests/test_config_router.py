from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.server import build_app


def test_legacy_config_routes_are_unmounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "a" * 40)
    client = TestClient(build_app(base_dir=tmp_path / ".workflows"), raise_server_exceptions=False)

    legacy_paths = (
        ("GET", "/config"),
        ("GET", "/config/status"),
        ("POST", "/config/validate"),
        ("POST", "/config/apply"),
        ("POST", "/config/credentials/rotate"),
        ("POST", "/config/credentials/revoke"),
    )
    for method, path in legacy_paths:
        response = client.request(method, path, json={"profiles": []})
        assert response.status_code == 404
