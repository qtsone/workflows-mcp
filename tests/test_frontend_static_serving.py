from __future__ import annotations

import re
import shutil
import logging
import subprocess
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.auth import TokenStore
from workflows_mcp.http.static import FrontendAssetsMissingError, packaged_admin_static_dir
from workflows_mcp.http_app import create_app
from workflows_mcp.http_models import ReadinessState
from workflows_mcp.server import build_app


class FakeReadinessReport:
    def __init__(self, state: ReadinessState) -> None:
        self.state = state
        self.blockers: list[str] = []


class FakeReadiness:
    def __init__(self, state: ReadinessState) -> None:
        self._state = state

    async def evaluate(self) -> FakeReadinessReport:
        return FakeReadinessReport(self._state)


@pytest.fixture()
def token_store(tmp_path: Path) -> TokenStore:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("a" * 40)
    return store


def test_ui_routes_serve_index_when_assets_present(tmp_path: Path, token_store: TokenStore) -> None:
    static_dir = tmp_path / "admin-static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>ui-shell</body></html>", encoding="utf-8")

    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
        frontend_static_dir=static_dir,
    )
    client = TestClient(app)

    for route in [
        "/",
        "/login",
        "/watchers",
        "/sync",
        "/mcp-clients",
        "/admin/projects/abc/details",
    ]:
        response = client.get(route)
        assert response.status_code == 200
        assert "ui-shell" in response.text


def test_assets_are_served_from_assets_directory_when_present(
    tmp_path: Path, token_store: TokenStore
) -> None:
    static_dir = tmp_path / "admin-static"
    assets_dir = static_dir / "assets"
    assets_dir.mkdir(parents=True)
    (static_dir / "index.html").write_text("<html><body>ui-shell</body></html>", encoding="utf-8")
    (assets_dir / "main.js").write_text("console.info('ok');", encoding="utf-8")
    (assets_dir / "main.css").write_text("body { color: #111; }", encoding="utf-8")

    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
        frontend_static_dir=static_dir,
    )
    client = TestClient(app)

    js_response = client.get("/assets/main.js")
    css_response = client.get("/assets/main.css")

    assert js_response.status_code == 200
    assert "console.info('ok');" in js_response.text
    assert css_response.status_code == 200
    assert "body { color: #111; }" in css_response.text


def test_reserved_routes_are_not_swallowed_by_spa_fallback(
    tmp_path: Path, token_store: TokenStore
) -> None:
    static_dir = tmp_path / "admin-static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>ui-shell</body></html>", encoding="utf-8")

    app = create_app(
        readiness_service=FakeReadiness(ReadinessState.READY),
        token_store=token_store,
        frontend_static_dir=static_dir,
    )
    client = TestClient(app)

    reserved_paths = [
        "/api/admin/v1/auth/session",
        "/api/not-a-ui-route",
        "/mcp",
        "/openapi.json",
    ]

    for path in reserved_paths:
        response = client.get(path)
        assert not (response.status_code == 200 and "ui-shell" in response.text)


def test_packaged_static_dir_points_to_package_static_admin_location() -> None:
    expected_suffix = Path("workflows_mcp") / "static" / "admin"
    assert packaged_admin_static_dir().as_posix().endswith(expected_suffix.as_posix())


def test_build_app_production_mode_requires_frontend_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("b" * 40)

    missing_static_dir = tmp_path / "missing-static"
    monkeypatch.setenv("WORKFLOWS_FRONTEND_MODE", "production")
    monkeypatch.setattr(
        "workflows_mcp.http.static.packaged_admin_static_dir",
        lambda: missing_static_dir,
    )

    with pytest.raises(FrontendAssetsMissingError) as exc_info:
        build_app(base_dir=tmp_path)

    message = str(exc_info.value)
    assert "Frontend admin assets are required" in message
    assert str(missing_static_dir / "index.html") in message


def test_build_app_production_mode_serves_packaged_assets_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("d" * 40)

    packaged_static = tmp_path / "package-static"
    packaged_static.mkdir()
    (packaged_static / "index.html").write_text(
        "<html><body>package-ui</body></html>",
        encoding="utf-8",
    )

    monkeypatch.setenv("WORKFLOWS_FRONTEND_MODE", "production")
    monkeypatch.setattr(
        "workflows_mcp.http.static.packaged_admin_static_dir",
        lambda: packaged_static,
    )

    app = build_app(base_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/login")
    assert response.status_code == 200
    assert "package-ui" in response.text


def test_build_app_production_mode_raises_when_packaged_assets_are_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("e" * 40)

    packaged_static = tmp_path / "package-static"
    packaged_static.mkdir()

    monkeypatch.setenv("WORKFLOWS_FRONTEND_MODE", "production")
    monkeypatch.setattr(
        "workflows_mcp.http.static.packaged_admin_static_dir",
        lambda: packaged_static,
    )

    with pytest.raises(FrontendAssetsMissingError) as exc_info:
        build_app(base_dir=tmp_path)

    message = str(exc_info.value)
    assert "Frontend admin assets are required" in message
    assert str(packaged_static / "index.html") in message


def test_build_app_default_mode_allows_missing_frontend_assets_and_health_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("c" * 40)

    missing_static_dir = tmp_path / "missing-static"
    monkeypatch.delenv("WORKFLOWS_FRONTEND_MODE", raising=False)
    monkeypatch.setattr(
        "workflows_mcp.http.static.packaged_admin_static_dir",
        lambda: missing_static_dir,
    )

    app = build_app(base_dir=tmp_path)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/login").status_code == 404


def test_build_app_default_mode_logs_actionable_warning_when_frontend_assets_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("f" * 40)

    missing_static_dir = tmp_path / "missing-static"
    monkeypatch.delenv("WORKFLOWS_FRONTEND_MODE", raising=False)
    monkeypatch.setattr(
        "workflows_mcp.http.static.packaged_admin_static_dir",
        lambda: missing_static_dir,
    )

    with caplog.at_level(logging.WARNING):
        app = build_app(base_dir=tmp_path)

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/login").status_code == 404
    assert "uv run workflows-mcp --build-ui" in caplog.text
    assert "cd web && npm install && npm run build" in caplog.text


@contextmanager
def with_fresh_frontend_build_assets() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    admin_dir = repo_root / "src" / "workflows_mcp" / "static" / "admin"
    backup_root = Path(tempfile.mkdtemp(prefix="frontend-admin-backup-"))
    backup_dir = backup_root / "admin"

    if admin_dir.exists():
        shutil.copytree(admin_dir, backup_dir)

    if admin_dir.exists():
        shutil.rmtree(admin_dir)

    build = subprocess.run(
        ["npm", "run", "build"],
        cwd=repo_root / "web",
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    assert (admin_dir / "index.html").exists()
    assert (admin_dir / "assets").is_dir()
    assert any((admin_dir / "assets").iterdir())

    try:
        yield repo_root
    finally:
        if admin_dir.exists():
            shutil.rmtree(admin_dir)

        if backup_dir.exists():
            shutil.copytree(backup_dir, admin_dir)

        shutil.rmtree(backup_root)


def _iter_admin_static_file_contents(admin_dir: Path) -> list[tuple[Path, str]]:
    contents: list[tuple[Path, str]] = []
    for path in admin_dir.rglob("*"):
        if not path.is_file():
            continue
        contents.append((path, path.read_text(encoding="utf-8", errors="ignore")))
    return contents


def test_build_distributions_include_frontend_admin_static_assets() -> None:
    with with_fresh_frontend_build_assets() as repo_root:
        dist_dir = repo_root / ".pytest_dist"
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
        dist_dir.mkdir(parents=True)

        build = subprocess.run(
            ["uv", "build", "--out-dir", str(dist_dir)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert build.returncode == 0, build.stderr

        wheel_files = list(dist_dir.glob("*.whl"))
        sdist_files = list(dist_dir.glob("*.tar.gz"))
        assert wheel_files, "Expected wheel artifact from uv build"
        assert sdist_files, "Expected sdist artifact from uv build"

        with zipfile.ZipFile(wheel_files[0]) as wheel_zip:
            wheel_names = wheel_zip.namelist()

        assert "workflows_mcp/static/admin/index.html" in wheel_names
        assert any(
            name.startswith("workflows_mcp/static/admin/assets/") and not name.endswith("/")
            for name in wheel_names
        )

        with tarfile.open(sdist_files[0], "r:gz") as sdist_tar:
            sdist_names = sdist_tar.getnames()

        assert any(
            name.endswith("src/workflows_mcp/static/admin/index.html")
            for name in sdist_names
        )
        assert any(
            "/src/workflows_mcp/static/admin/assets/" in name and not name.endswith("/")
            for name in sdist_names
        )


def test_vite_build_outputs_to_packaged_admin_static_directory() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    vite_config = repo_root / "web" / "vite.config.ts"
    config_text = vite_config.read_text(encoding="utf-8")

    assert "../src/workflows_mcp/static/admin" in config_text


def test_packaged_project_onboarding_assets_allow_optional_default_location() -> None:
    with with_fresh_frontend_build_assets() as repo_root:
        admin_dir = repo_root / "src" / "workflows_mcp" / "static" / "admin"
        js_content = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in sorted((admin_dir / "assets").glob("*.js"))
        )
        css_content = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in sorted((admin_dir / "assets").glob("*.css"))
        )

        assert "project-default-wing" in js_content
        assert "project-default-room" in js_content
        assert not re.search(r"id:`project-default-wing`[^}]+required:!0", js_content)
        assert not re.search(r"id:`project-default-room`[^}]+required:!0", js_content)
        assert re.search(r"\.project-form__actions\{[^}]*bottom:-1rem", css_content)
        assert re.search(r"\.project-form__actions\{[^}]*margin:0 -1rem -1rem", css_content)


def test_frontend_bundle_does_not_leak_sensitive_env_literals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_server_secret = "SLICE5_SERVER_SECRET_SHOULD_NOT_LEAK_7f13d7"
    sentinel_private_key = "-----BEGIN PRIVATE KEY-----\nSLICE5_PRIVATE_KEY_SHOULD_NOT_LEAK"
    sentinel_bearer_token = "Bearer sk_slice5_token_should_not_leak_12345"

    monkeypatch.setenv("WORKFLOW_SECRET_SLICE5_TEST", sentinel_server_secret)
    monkeypatch.setenv("SLICE5_PRIVATE_KEY", sentinel_private_key)
    monkeypatch.setenv("SLICE5_BEARER_TOKEN", sentinel_bearer_token)

    with with_fresh_frontend_build_assets() as repo_root:
        admin_dir = repo_root / "src" / "workflows_mcp" / "static" / "admin"
        file_contents = _iter_admin_static_file_contents(admin_dir)

        leaked_literals: list[tuple[Path, str]] = []
        for path, content in file_contents:
            for sensitive_literal in [
                sentinel_server_secret,
                sentinel_private_key,
                sentinel_bearer_token,
            ]:
                if sensitive_literal in content:
                    leaked_literals.append((path, sensitive_literal))
        assert not leaked_literals, f"Found leaked sensitive literal(s): {leaked_literals}"

        private_key_marker_hits = [
            path for path, content in file_contents if "-----BEGIN PRIVATE KEY-----" in content
        ]
        assert not private_key_marker_hits, (
            "Found private key marker in bundled frontend assets: "
            f"{private_key_marker_hits}"
        )

        # Guard against obvious token-shaped bearer values while allowing harmless scheme text.
        bearer_token_pattern = re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}")
        bearer_token_hits = [
            path for path, content in file_contents if bearer_token_pattern.search(content)
        ]
        assert not bearer_token_hits, (
            "Found bearer token-like literal in bundled frontend assets: "
            f"{bearer_token_hits}"
        )
