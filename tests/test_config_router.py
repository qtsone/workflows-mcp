from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.config_service import ConfigService
from workflows_mcp.server import build_app


@pytest.fixture()
def config_service(tmp_path: Path) -> ConfigService:
    return ConfigService(base_dir=tmp_path / ".workflows")


def test_validate_rejects_invalid_payload(config_service: ConfigService) -> None:
    errors = config_service.validate_payload({"profiles": "bad"})
    assert "profiles" in errors


def test_validate_accepts_valid_payload(config_service: ConfigService) -> None:
    errors = config_service.validate_payload({"profiles": []})
    assert errors == {}


async def test_apply_creates_llm_config(config_service: ConfigService, tmp_path: Path) -> None:
    await config_service.apply_payload({"profiles": []})
    assert (tmp_path / ".workflows" / "llm-config.yml").exists()


async def test_apply_creates_base_dir_if_missing(tmp_path: Path) -> None:
    base_dir = tmp_path / "nested" / ".workflows"
    service = ConfigService(base_dir=base_dir)
    await service.apply_payload({"profiles": []})
    assert (base_dir / "llm-config.yml").exists()


async def test_apply_preserves_last_known_good_on_invalid_payload(
    config_service: ConfigService, tmp_path: Path
) -> None:
    await config_service.apply_payload({"profiles": []})
    original_mtime = (tmp_path / ".workflows" / "llm-config.yml").stat().st_mtime

    errors = config_service.validate_payload({"profiles": "bad"})
    assert errors

    new_mtime = (tmp_path / ".workflows" / "llm-config.yml").stat().st_mtime
    assert new_mtime == original_mtime


def test_legacy_config_routes_are_unmounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "a" * 40)
    client = TestClient(
        build_app(base_dir=tmp_path / ".workflows"), raise_server_exceptions=False
    )

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
