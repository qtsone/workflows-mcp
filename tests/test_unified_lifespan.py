from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from workflows_mcp.http.dependencies import get_resources
from workflows_mcp.server import app_lifespan, build_app, mcp


@pytest.fixture(autouse=True)
def _bootstrap_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "x" * 40)


def test_build_app_attaches_unified_resources_container(tmp_path: Path) -> None:
    app = build_app(base_dir=tmp_path)

    assert hasattr(app.state, "resources")
    resources = app.state.resources

    assert resources.workflow_registry is not None
    assert resources.executor_registry is not None
    assert resources.llm_config_loader is not None
    assert resources.metadata is not None
    assert resources.app_context is not None

    assert resources.app_context.registry is resources.workflow_registry
    assert resources.app_context.executor_registry is resources.executor_registry
    assert resources.app_context.llm_config_loader is resources.llm_config_loader
    assert resources.app_context.io_queue is resources.io_queue
    assert resources.app_context.job_queue is resources.job_queue
    assert resources.app_context.max_recursion_depth == resources.max_recursion_depth
    assert resources.job_queue is not None
    assert Path(resources.job_queue._store._db_path) == resources.metadata.base_dir / "server.db"

    client = TestClient(app)
    assert client.get("/health").status_code == 200


@pytest.mark.asyncio
async def test_app_lifespan_uses_shared_resource_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    import workflows_mcp.server as server_module
    from workflows_mcp.http.lifespan import build_resources

    called = {"count": 0}

    def _tracking_build_resources(*, base_dir: Path):
        called["count"] += 1
        return build_resources(base_dir=base_dir)

    monkeypatch.setattr(server_module, "build_resources", _tracking_build_resources)
    monkeypatch.setenv("WORKFLOWS_IO_QUEUE_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_JOB_QUEUE_ENABLED", "false")

    async with app_lifespan(mcp) as app_context:
        assert app_context.registry is not None
        assert app_context.executor_registry is not None
        assert app_context.llm_config_loader is not None

    assert called["count"] == 1


def test_get_resources_returns_app_state_resources(tmp_path: Path) -> None:
    app = build_app(base_dir=tmp_path)
    expected = app.state.resources

    request_like = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(resources=expected)),
    )

    assert get_resources(request_like) is expected
