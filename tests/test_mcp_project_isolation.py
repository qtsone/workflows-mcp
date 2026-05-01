from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from workflows_mcp.auth import TokenStore
from workflows_mcp.engine.memory_scope_resolver import SyncContextCandidate
from workflows_mcp.http.lifespan import build_resources
from workflows_mcp.http_app import create_app
from workflows_mcp.http_models import ReadinessState
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.projects_repo import ProjectCreate, SQLiteProjectsRepository
from workflows_mcp.metadata.repos.run_history_repo import SQLiteRunHistoryRepository
from workflows_mcp.metadata.repos.tokens_repo import SQLiteTokensRepository


class _FakeReadinessService:
    async def evaluate(self):  # noqa: ANN201
        class _Report:
            pass

        report = _Report()
        report.state = ReadinessState.READY  # type: ignore[attr-defined]
        report.blockers = []  # type: ignore[attr-defined]
        return report


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _build_app_with_resources(tmp_path: Path):
    token_store = TokenStore(tmp_path / "auth.json")
    token_store.write_token("a" * 40)
    app = create_app(readiness_service=_FakeReadinessService(), token_store=token_store)
    resources = build_resources(base_dir=tmp_path / ".workflows")
    app.state.resources = resources
    return app, resources


def _create_project(
    projects_repo: SQLiteProjectsRepository,
    *,
    name: str,
    slug: str,
    palace: str,
    root: Path,
):
    return projects_repo.create(
        ProjectCreate(
            name=name,
            slug=slug,
            palace=palace,
            default_wing="core",
            default_room="main",
            fs_root=str(root),
            fs_allowlist=[],
        )
    )


def _get_run_record_from_job_queue_db(resources, run_id: str):
    assert resources.job_queue is not None
    db_path = resources.job_queue._store._db_path
    conn = connect_metadata_db(db_path)
    try:
        return SQLiteRunHistoryRepository(conn).get_run(run_id)
    finally:
        conn.close()


@pytest.mark.anyio
async def test_sqlite_mcp_token_can_auth_and_select_bound_project(tmp_path: Path) -> None:
    app, resources = _build_app_with_resources(tmp_path)
    projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
    project = _create_project(
        projects_repo,
        name="Project Alpha",
        slug="alpha",
        palace="palace-alpha",
        root=tmp_path,
    )
    created_token = SQLiteTokensRepository(resources.metadata_db_conn).create(
        label="alpha-token",
        project_ids=[project.id],
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {created_token.token_secret}"},
    ) as http_client:
        async with streamable_http_client(
            "http://127.0.0.1/mcp",
            http_client=http_client,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("select", {"project": project.slug})
                payload = result.structuredContent
                assert payload["status"] == "selected"
                active = payload["active_project"]
                assert isinstance(active, dict)
                assert active["project_id"] == project.id


@pytest.mark.anyio
async def test_token_a_cannot_select_token_b_project(tmp_path: Path) -> None:
    app, resources = _build_app_with_resources(tmp_path)
    projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
    project_a = _create_project(
        projects_repo,
        name="Project A",
        slug="a",
        palace="palace-a",
        root=tmp_path,
    )
    project_b = _create_project(
        projects_repo,
        name="Project B",
        slug="b",
        palace="palace-b",
        root=tmp_path,
    )
    tokens_repo = SQLiteTokensRepository(resources.metadata_db_conn)
    token_a = tokens_repo.create(label="token-a", project_ids=[project_a.id])
    tokens_repo.create(label="token-b", project_ids=[project_b.id])

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {token_a.token_secret}"},
    ) as http_client:
        async with streamable_http_client(
            "http://127.0.0.1/mcp",
            http_client=http_client,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("select", {"project": project_b.slug})
                payload = result.structuredContent
                assert payload["error"]["code"] == "MEM_SELECT_NOT_FOUND"


@pytest.mark.anyio
async def test_session_teardown_clears_state_and_later_session_does_not_inherit(
    tmp_path: Path,
) -> None:
    app, resources = _build_app_with_resources(tmp_path)
    projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
    project = _create_project(
        projects_repo,
        name="Project",
        slug="teardown",
        palace="palace-teardown",
        root=tmp_path,
    )
    token = SQLiteTokensRepository(resources.metadata_db_conn).create(
        label="teardown-token",
        project_ids=[project.id],
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {token.token_secret}"},
    ) as http_client:
        async with streamable_http_client(
            "http://127.0.0.1/mcp",
            http_client=http_client,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await session.call_tool("select", {"project": project.slug})

    session_ids = list(resources.app_context._transport_sessions.keys())
    assert session_ids
    session_id = session_ids[0]
    mapped_session, owner_token_id = resources.app_context._transport_sessions[session_id]
    assert owner_token_id == token.id
    candidate = SyncContextCandidate(
        scope={
            "palace": project.palace,
            "wing": project.default_wing,
            "room": project.default_room,
        },
        scope_key_value=f"{project.palace}:{project.default_wing}:{project.default_room}",
        checkpoint_data={},
        source="stored_checkpoint",
    )
    resources.app_context.set_active_context(mapped_session, candidate)
    resources.app_context.register_onboard_context_candidate(
        mapped_session,
        candidate,
    )

    assert (
        resources.app_context.clear_session_state_by_transport_id(
            session_id,
            requester_token_id=token.id,
        )
        is True
    )
    assert resources.app_context._transport_sessions == {}
    assert resources.app_context.get_active_project(mapped_session) is None
    assert resources.app_context.get_active_context(mapped_session) is None
    assert resources.app_context.list_allowed_projects(mapped_session) == []
    assert resources.app_context.list_onboard_context_candidates(mapped_session) == []


@pytest.mark.anyio
async def test_token_a_cannot_clear_token_b_transport_session_by_delete(tmp_path: Path) -> None:
    app, resources = _build_app_with_resources(tmp_path)
    projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
    project_a = _create_project(
        projects_repo,
        name="Project A",
        slug="delete-a",
        palace="palace-delete-a",
        root=tmp_path,
    )
    project_b = _create_project(
        projects_repo,
        name="Project B",
        slug="delete-b",
        palace="palace-delete-b",
        root=tmp_path,
    )
    tokens_repo = SQLiteTokensRepository(resources.metadata_db_conn)
    token_a = tokens_repo.create(label="delete-token-a", project_ids=[project_a.id])
    token_b = tokens_repo.create(label="delete-token-b", project_ids=[project_b.id])

    transport = httpx.ASGITransport(app=app)
    # Create B session and capture transport session id + mapped state.
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {token_b.token_secret}"},
    ) as http_client_b:
        async with streamable_http_client(
            "http://127.0.0.1/mcp",
            http_client=http_client_b,
        ) as (read_stream_b, write_stream_b, _):
            async with ClientSession(read_stream_b, write_stream_b) as session_b:
                await session_b.initialize()
                await session_b.call_tool("select", {"project": project_b.slug})

    matching = [
        sid
        for sid, (_session, owner_token_id) in resources.app_context._transport_sessions.items()
        if owner_token_id == token_b.id
    ]
    assert matching
    victim_session_id = matching[0]
    victim_session, victim_owner_token_id = resources.app_context._transport_sessions[
        victim_session_id
    ]
    assert victim_owner_token_id == token_b.id
    assert resources.app_context.get_active_project(victim_session) is not None

    # Token A attempts to clear token B session via DELETE + forged mcp-session-id.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={
            "Authorization": f"Bearer {token_a.token_secret}",
            "mcp-session-id": victim_session_id,
        },
    ) as http_client_a:
        response = await http_client_a.delete("/mcp")
        assert response.status_code < 500

    # Must remain intact: cross-token cleanup is denied.
    assert victim_session_id in resources.app_context._transport_sessions
    assert resources.app_context.get_active_project(victim_session) is not None


@pytest.mark.anyio
async def test_single_project_token_auto_sets_active_project_for_session(tmp_path: Path) -> None:
    app, resources = _build_app_with_resources(tmp_path)
    projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
    project = _create_project(
        projects_repo,
        name="Single",
        slug="single",
        palace="palace-single",
        root=tmp_path,
    )
    token = SQLiteTokensRepository(resources.metadata_db_conn).create(
        label="single-token",
        project_ids=[project.id],
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {token.token_secret}"},
    ) as http_client:
        async with streamable_http_client(
            "http://127.0.0.1/mcp",
            http_client=http_client,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("select", {"project": project.slug})
                payload = result.structuredContent
                assert payload["status"] == "selected"


@pytest.mark.anyio
async def test_execute_workflow_async_records_selected_project_and_token_id(tmp_path: Path) -> None:
    app, resources = _build_app_with_resources(tmp_path)
    projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
    project_a = _create_project(
        projects_repo,
        name="Project A",
        slug="proj-a",
        palace="palace-a",
        root=tmp_path,
    )
    project_b = _create_project(
        projects_repo,
        name="Project B",
        slug="proj-b",
        palace="palace-b",
        root=tmp_path,
    )
    token = SQLiteTokensRepository(resources.metadata_db_conn).create(
        label="multi-project-token",
        project_ids=[project_a.id, project_b.id],
    )

    assert resources.job_queue is not None
    assert Path(resources.job_queue._store._db_path) == resources.metadata.base_dir / "server.db"
    await resources.job_queue.start()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1",
            headers={"Authorization": f"Bearer {token.token_secret}"},
        ) as http_client:
            async with streamable_http_client(
                "http://127.0.0.1/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    select_result = await session.call_tool("select", {"project": project_b.slug})
                    select_payload = select_result.structuredContent
                    assert select_payload["status"] == "selected"

                    exec_result = await session.call_tool(
                        "execute_workflow",
                        {
                            "workflow": "nonexistent-workflow",
                            "mode": "async",
                            "inputs": {},
                        },
                    )
                    assert not exec_result.isError, exec_result.content[0].text
                    exec_payload = exec_result.structuredContent
                    job_id = exec_payload["job_id"]

        run = _get_run_record_from_job_queue_db(resources, job_id)
        assert run is not None
        assert run.project_id == project_b.id
        assert run.token_id == token.id
    finally:
        await resources.job_queue.stop(wait_for_completion=False)


@pytest.mark.anyio
async def test_direct_job_queue_submission_defaults_project_and_token_to_null(
    tmp_path: Path,
) -> None:
    app, resources = _build_app_with_resources(tmp_path)
    assert app is not None
    assert resources.job_queue is not None

    await resources.job_queue.start()
    try:
        job_id = await resources.job_queue.submit_job("nonexistent-workflow", inputs={})
        run = _get_run_record_from_job_queue_db(resources, job_id)
        assert run is not None
        assert run.project_id is None
        assert run.token_id is None
    finally:
        await resources.job_queue.stop(wait_for_completion=False)


@pytest.mark.anyio
async def test_direct_job_queue_submission_with_unknown_project_or_token_fails_fk(
    tmp_path: Path,
) -> None:
    app, resources = _build_app_with_resources(tmp_path)
    assert app is not None
    assert resources.job_queue is not None

    await resources.job_queue.start()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            await resources.job_queue.submit_job(
                "nonexistent-workflow",
                inputs={},
                project_id="missing-project",
                token_id="missing-token",
            )
    finally:
        await resources.job_queue.stop(wait_for_completion=False)
