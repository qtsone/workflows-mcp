"""Snapshot test: OpenAPI schema must match the committed snapshot.

Run the generator when the schema changes:

    uv run python tests/generate_openapi_snapshot.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "openapi.json"


def _build_schema() -> dict:
    """Build the OpenAPI schema from the live app."""
    import os

    from workflows_mcp.auth import TokenStore
    from workflows_mcp.server import build_app

    token = "0123456789abcdef0123456789abcdef"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Pre-seed token store so build_app() doesn't fail.
        store = TokenStore(tmp_path / "auth.json")
        store.write_token(token)
        # WORKFLOWS_BOOTSTRAP_TOKEN must be set for build_app() bootstrap check.
        old = os.environ.get("WORKFLOWS_BOOTSTRAP_TOKEN")
        os.environ["WORKFLOWS_BOOTSTRAP_TOKEN"] = token
        try:
            app = build_app(base_dir=tmp_path)
        finally:
            if old is None:
                os.environ.pop("WORKFLOWS_BOOTSTRAP_TOKEN", None)
            else:
                os.environ["WORKFLOWS_BOOTSTRAP_TOKEN"] = old
    return app.openapi()


class TestOpenAPISchema:
    def test_snapshot_exists(self) -> None:
        """The committed snapshot file must exist."""
        assert _SNAPSHOT_PATH.exists(), (
            f"OpenAPI snapshot not found at {_SNAPSHOT_PATH}. "
            "Run: uv run python tests/generate_openapi_snapshot.py"
        )

    def test_bearer_auth_scheme_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The OpenAPI spec must declare a BearerAuth security scheme."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        schema = _build_schema()
        components = schema.get("components", {})
        security_schemes = components.get("securitySchemes", {})
        assert "BearerAuth" in security_schemes, (
            f"BearerAuth not found in securitySchemes. Got: {list(security_schemes)}"
        )
        bearer = security_schemes["BearerAuth"]
        assert bearer.get("type") == "http"
        assert bearer.get("scheme") == "bearer"

    def test_protected_routes_declare_security(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All protected routes must declare security: [BearerAuth]."""
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        schema = _build_schema()
        paths = schema.get("paths", {})

        protected_paths = [p for p in paths if p.startswith("/config") or p == "/mcp"]
        assert protected_paths, "No protected paths found in schema"

        for path, path_item in paths.items():
            if not (path.startswith("/config") or path == "/mcp"):
                continue
            for method, operation in path_item.items():
                if method not in ("get", "post", "put", "patch", "delete"):
                    continue
                security = operation.get("security")
                assert security is not None, (
                    f"{method.upper()} {path} is missing 'security' declaration"
                )
                assert {"BearerAuth": []} in security, (
                    f"{method.upper()} {path} does not include BearerAuth in security"
                )

    def test_schema_matches_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Live schema must match the committed snapshot.

        If this test fails, regenerate with:
            uv run python tests/generate_openapi_snapshot.py
        """
        assert _SNAPSHOT_PATH.exists(), (
            f"OpenAPI snapshot not found at {_SNAPSHOT_PATH}. "
            "Run: uv run python tests/generate_openapi_snapshot.py"
        )
        monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")
        current = _build_schema()
        snapshot = json.loads(_SNAPSHOT_PATH.read_text())

        assert current == snapshot, (
            "OpenAPI schema has drifted. Run: uv run python tests/generate_openapi_snapshot.py"
        )
