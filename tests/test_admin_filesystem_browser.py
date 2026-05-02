from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.server import build_app

_MCP_BOOTSTRAP_TOKEN = "0123456789abcdef0123456789abcdef01234567"
_ADMIN_PASSWORD = "phase2-admin-password"


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    base_dir = tmp_path / ".workflows"
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", _MCP_BOOTSTRAP_TOKEN)
    bootstrap_if_needed(
        config_dir=base_dir,
        host="127.0.0.1",
        port=8000,
        admin_password=_ADMIN_PASSWORD,
    )
    return TestClient(build_app(base_dir=base_dir), raise_server_exceptions=False)


def _login(client: TestClient) -> None:
    response = client.post("/api/admin/v1/auth/login", json={"password": _ADMIN_PASSWORD})
    assert response.status_code == 200


def _error_code(payload: dict[str, object]) -> str | None:
    detail = payload.get("detail")
    if isinstance(detail, dict):
        code = detail.get("code")
        if isinstance(code, str):
            return code
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str):
            return code
    return None


def test_entries_browser_lists_configured_root(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    alpha = root / "alpha"
    beta = root / "beta"
    file_path = root / "not-a-directory.txt"
    alpha.mkdir(parents=True)
    beta.mkdir()
    file_path.write_text("not listed")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    response = app_client.get("/api/admin/v1/filesystem/entries")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "root": str(root.resolve()),
        "path": str(root.resolve()),
        "parent": None,
        "can_go_up": False,
        "entries": [
            {
                "name": "alpha",
                "path": str(alpha.resolve()),
                "type": "directory",
                "selectable": True,
            },
            {
                "name": "beta",
                "path": str(beta.resolve()),
                "type": "directory",
                "selectable": True,
            },
            {
                "name": "not-a-directory.txt",
                "path": str(file_path.resolve()),
                "type": "file",
                "selectable": False,
            },
        ],
    }


def test_entries_browser_requires_admin_session(app_client: TestClient) -> None:
    unauthenticated = app_client.get("/api/admin/v1/filesystem/entries")
    assert unauthenticated.status_code == 401

    bearer = app_client.get(
        "/api/admin/v1/filesystem/entries",
        headers={"Authorization": f"Bearer {_MCP_BOOTSTRAP_TOKEN}"},
    )
    assert bearer.status_code == 403


def test_entries_browser_rejects_relative_path_before_resolve(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    root.mkdir(parents=True)
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    response = app_client.get(
        "/api/admin/v1/filesystem/entries",
        params={"path": "relative/path/that-does-not-exist"},
    )

    assert response.status_code == 400
    assert _error_code(response.json()) == "filesystem_path_malformed"


def test_entries_browser_blank_root_defaults_to_slash(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", "   ")
    _login(app_client)

    response = app_client.get("/api/admin/v1/filesystem/entries")

    assert response.status_code == 200
    payload = response.json()
    assert payload["root"] == "/"
    assert payload["path"] == "/"


def test_entries_browser_invalid_root_fails_closed(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_root = tmp_path / "missing-root"
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(missing_root))
    _login(app_client)

    response = app_client.get("/api/admin/v1/filesystem/entries")

    assert response.status_code == 409
    payload = response.json()
    assert _error_code(payload) == "filesystem_browsing_root_invalid"
    assert str(missing_root) not in str(payload)


def test_entries_browser_lists_child_with_safe_parent(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    child = root / "child"
    nested = child / "nested"
    nested.mkdir(parents=True)
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    response = app_client.get(
        "/api/admin/v1/filesystem/entries",
        params={"path": str(child)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["root"] == str(root.resolve())
    assert payload["path"] == str(child.resolve())
    assert payload["parent"] == str(root.resolve())
    assert payload["can_go_up"] is True
    assert payload["entries"] == [
        {"name": "nested", "path": str(nested.resolve()), "type": "directory", "selectable": True}
    ]


def test_entries_browser_rejects_outside_root(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    response = app_client.get("/api/admin/v1/filesystem/entries", params={"path": str(outside)})

    assert response.status_code == 403
    payload = response.json()
    assert _error_code(payload) == "filesystem_path_outside_root"
    assert str(outside) not in str(payload)


def test_entries_browser_reports_not_found_and_not_directory(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    root.mkdir()
    file_path = root / "file.txt"
    file_path.write_text("x")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    missing = app_client.get(
        "/api/admin/v1/filesystem/entries",
        params={"path": str(root / "missing")},
    )
    assert missing.status_code == 404
    assert _error_code(missing.json()) == "filesystem_path_not_found"

    not_directory = app_client.get(
        "/api/admin/v1/filesystem/entries",
        params={"path": str(file_path)},
    )
    assert not_directory.status_code == 422
    assert _error_code(not_directory.json()) == "filesystem_path_not_directory"


def test_entries_browser_excludes_symlink_to_outside_and_rejects_direct_request(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    outside = tmp_path / "outside"
    safe = root / "safe"
    root.mkdir()
    outside.mkdir()
    safe.mkdir()
    escape = root / "escape"
    try:
        escape.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation unavailable on this platform")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    listing = app_client.get("/api/admin/v1/filesystem/entries")
    assert listing.status_code == 200
    assert listing.json()["entries"] == [
        {"name": "safe", "path": str(safe.resolve()), "type": "directory", "selectable": True}
    ]

    direct = app_client.get("/api/admin/v1/filesystem/entries", params={"path": str(escape)})
    assert direct.status_code == 403
    assert _error_code(direct.json()) == "filesystem_path_outside_root"


def test_entries_browser_direct_unreadable_directory_returns_sanitized_403(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    unreadable = root / "unreadable"
    unreadable.mkdir(parents=True)
    unreadable.chmod(0)
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)
    try:
        response = app_client.get(
            "/api/admin/v1/filesystem/entries",
            params={"path": str(unreadable)},
        )
    finally:
        unreadable.chmod(0o700)

    if response.status_code == 200:
        pytest.skip("current platform/user can list chmod(0) directory")
    assert response.status_code == 403
    payload = response.json()
    assert _error_code(payload) == "filesystem_access_denied"
    assert str(unreadable) not in str(payload)


def test_entries_browser_defaults_root_to_slash_when_env_missing(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORKFLOWS_SCAN_ROOT", raising=False)
    _login(app_client)

    response = app_client.get("/api/admin/v1/filesystem/entries")

    assert response.status_code == 200
    payload = response.json()
    assert payload["root"] == "/"
    assert payload["path"] == "/"
    assert payload["parent"] is None
    assert payload["can_go_up"] is False
    assert isinstance(payload["entries"], list)


def test_entries_folder_mode_returns_typed_entries(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    alpha = root / "alpha"
    zeta = root / "zeta"
    text_file = root / "note.txt"
    alpha.mkdir(parents=True)
    zeta.mkdir()
    text_file.write_text("hello")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    response = app_client.get(
        "/api/admin/v1/filesystem/entries",
        params={"selection_type": "folder"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["root"] == str(root.resolve())
    assert payload["path"] == str(root.resolve())
    assert payload["entries"] == [
        {
            "name": "alpha",
            "path": str(alpha.resolve()),
            "type": "directory",
            "selectable": True,
        },
        {
            "name": "zeta",
            "path": str(zeta.resolve()),
            "type": "directory",
            "selectable": True,
        },
        {
            "name": "note.txt",
            "path": str(text_file.resolve()),
            "type": "file",
            "selectable": False,
        },
    ]


def test_entries_file_mode_filters_extensions_case_insensitive(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    nested = root / "nested"
    keep_py = root / "main.PY"
    keep_md = root / "README.md"
    skip_txt = root / "skip.txt"
    nested.mkdir(parents=True)
    keep_py.write_text("print('x')")
    keep_md.write_text("# readme")
    skip_txt.write_text("nope")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    response = app_client.get(
        "/api/admin/v1/filesystem/entries",
        params=[("selection_type", "file"), ("extensions", "py,MD")],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["entries"] == [
        {
            "name": "nested",
            "path": str(nested.resolve()),
            "type": "directory",
            "selectable": False,
        },
        {
            "name": "main.PY",
            "path": str(keep_py.resolve()),
            "type": "file",
            "selectable": True,
        },
        {
            "name": "README.md",
            "path": str(keep_md.resolve()),
            "type": "file",
            "selectable": True,
        },
    ]


def test_entries_browser_rejects_outside_root_when_scan_root_set(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    response = app_client.get("/api/admin/v1/filesystem/entries", params={"path": str(outside)})

    assert response.status_code == 403
    assert _error_code(response.json()) == "filesystem_path_outside_root"


def test_entries_browser_excludes_symlink_escape_and_rejects_direct_request(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "scan-root"
    outside = tmp_path / "outside"
    safe = root / "safe"
    root.mkdir()
    outside.mkdir()
    safe.mkdir()
    escape = root / "escape"
    try:
        escape.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation unavailable on this platform")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(root))
    _login(app_client)

    listing = app_client.get("/api/admin/v1/filesystem/entries")
    assert listing.status_code == 200
    assert listing.json()["entries"] == [
        {
            "name": "safe",
            "path": str(safe.resolve()),
            "type": "directory",
            "selectable": True,
        }
    ]

    direct = app_client.get("/api/admin/v1/filesystem/entries", params={"path": str(escape)})
    assert direct.status_code == 403
    assert _error_code(direct.json()) == "filesystem_path_outside_root"
