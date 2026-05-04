from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows_mcp import server
from workflows_mcp.context import AppContext
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.workflow_source_loader import (
    WorkflowSourceReloadError,
    reload_registry_from_source_paths,
)
from workflows_mcp.http.lifespan import build_resources, stop_resources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos.projects_repo import ProjectCreate, SQLiteProjectsRepository
from workflows_mcp.metadata.repos.workflow_sources_repo import (
    DuplicateWorkflowSourceError,
    InvalidWorkflowSourcePathError,
    SQLiteWorkflowSourcesRepository,
    WorkflowSourceCreate,
    WorkflowSourceNotFoundError,
    WorkflowSourceProjectNotFoundError,
)


def _repos(
    tmp_path: Path,
) -> tuple[object, SQLiteProjectsRepository, SQLiteWorkflowSourcesRepository]:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    migrate_metadata_db(conn)
    return conn, SQLiteProjectsRepository(conn), SQLiteWorkflowSourcesRepository(conn)


def _create_project(projects_repo: SQLiteProjectsRepository, *, slug: str = "repo-test") -> str:
    project = projects_repo.create(
        ProjectCreate(
            name="Repo Test",
            slug=slug,
            palace=f"palace.{slug}",
            default_wing="wing-a",
            default_room="room-a",
            fs_root="/tmp",
            fs_allowlist=[],
        )
    )
    return project.id


def test_create_get_list_and_delete_workflow_source(tmp_path: Path) -> None:
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo)
        source_dir = tmp_path / "sources" / "a"
        source_dir.mkdir(parents=True)

        created = sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=f"{source_dir}/..//a")
        )
        expected_path = str(source_dir.expanduser().resolve(strict=False))

        assert created.project_id == project_id
        assert created.source_path == expected_path
        assert created.discovered_at
        assert created.last_loaded_at is None
        assert created.status is None
        assert created.error_message is None

        fetched = sources_repo.get(created.source_id)
        assert fetched == created

        listed = sources_repo.list(project_id=project_id)
        assert [item.source_id for item in listed] == [created.source_id]

        all_listed = sources_repo.list()
        assert [item.source_id for item in all_listed] == [created.source_id]

        sources_repo.delete(created.source_id)
        assert sources_repo.list(project_id=project_id) == []
        with pytest.raises(WorkflowSourceNotFoundError):
            sources_repo.get(created.source_id)
    finally:
        conn.close()


def test_create_rejects_missing_or_non_directory_path(tmp_path: Path) -> None:
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo)
        missing = tmp_path / "does-not-exist"
        file_path = tmp_path / "not-a-dir.txt"
        file_path.write_text("x", encoding="utf-8")

        with pytest.raises(InvalidWorkflowSourcePathError):
            sources_repo.create(
                WorkflowSourceCreate(project_id=project_id, source_path=str(missing))
            )

        with pytest.raises(InvalidWorkflowSourcePathError):
            sources_repo.create(
                WorkflowSourceCreate(project_id=project_id, source_path=str(file_path))
            )
    finally:
        conn.close()


def test_create_rejects_missing_project(tmp_path: Path) -> None:
    conn, _projects_repo, sources_repo = _repos(tmp_path)
    try:
        source_dir = tmp_path / "wf"
        source_dir.mkdir()
        with pytest.raises(WorkflowSourceProjectNotFoundError):
            sources_repo.create(
                WorkflowSourceCreate(project_id="missing-project", source_path=str(source_dir))
            )
    finally:
        conn.close()


def test_create_rejects_duplicate_source_path_for_same_project(tmp_path: Path) -> None:
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo)
        source_dir = tmp_path / "wf" / "nested"
        source_dir.mkdir(parents=True)

        sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(source_dir))
        )
        with pytest.raises(DuplicateWorkflowSourceError):
            sources_repo.create(
                WorkflowSourceCreate(project_id=project_id, source_path=f"{source_dir}/../nested")
            )
    finally:
        conn.close()


def test_direct_sql_duplicate_insert_fails_with_integrity_error(tmp_path: Path) -> None:
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo)
        source_dir = tmp_path / "wf" / "direct"
        source_dir.mkdir(parents=True)
        normalized = str(source_dir.expanduser().resolve(strict=False))

        sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(source_dir))
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO workflow_sources (source_id, project_id, source_path, checksum)
                VALUES (?, ?, ?, ?)
                """,
                ("direct-duplicate", project_id, normalized, None),
            )
    finally:
        conn.close()


def test_create_maps_db_unique_constraint_to_duplicate_workflow_source_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo)
        source_dir = tmp_path / "wf" / "race"
        source_dir.mkdir(parents=True)

        sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(source_dir))
        )

        monkeypatch.setattr(
            sources_repo,
            "_assert_not_duplicate_path",
            lambda *, project_id, normalized_source_path: None,
        )

        with pytest.raises(DuplicateWorkflowSourceError):
            sources_repo.create(
                WorkflowSourceCreate(project_id=project_id, source_path=str(source_dir))
            )
    finally:
        conn.close()


def test_delete_missing_source_raises_not_found(tmp_path: Path) -> None:
    conn, _projects_repo, sources_repo = _repos(tmp_path)
    try:
        with pytest.raises(WorkflowSourceNotFoundError):
            sources_repo.delete("missing-source")
    finally:
        conn.close()


def test_update_reload_state_upserts_and_cascade_delete(tmp_path: Path) -> None:
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo)
        source_dir = tmp_path / "wf"
        source_dir.mkdir()

        created = sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(source_dir))
        )
        updated = sources_repo.update_reload_state(
            created.source_id,
            status="loaded",
            error_message=None,
        )

        assert updated.status == "loaded"
        assert updated.last_loaded_at is not None
        assert updated.error_message is None

        failed = sources_repo.update_reload_state(
            created.source_id,
            status="failed",
            error_message="boom",
        )
        assert failed.status == "failed"
        assert failed.error_message == "boom"
        assert failed.last_loaded_at is not None

        sources_repo.delete(created.source_id)
        reload_state_row = conn.execute(
            "SELECT 1 FROM workflow_reload_state WHERE source_id = ?",
            (created.source_id,),
        ).fetchone()
        assert reload_state_row is None

        with pytest.raises(WorkflowSourceNotFoundError):
            sources_repo.update_reload_state(
                created.source_id,
                status="loaded",
                error_message=None,
            )
    finally:
        conn.close()


def _write_workflow_yaml(directory: Path, *, filename: str, name: str) -> Path:
    target = directory / filename
    target.write_text(
        "\n".join(
            [
                f"name: {name}",
                f"description: workflow {name}",
                "blocks:",
                "  - id: run",
                "    type: Shell",
                "    inputs:",
                "      command: \"printf 'ok'\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def _write_invalid_workflow_yaml(directory: Path, *, filename: str) -> Path:
    target = directory / filename
    target.write_text("name: bad\nblocks: [", encoding="utf-8")
    return target


def test_reload_registry_from_source_paths_replaces_registry_exactly(tmp_path: Path) -> None:
    registry = WorkflowRegistry()
    src_one = tmp_path / "src-one"
    src_two = tmp_path / "src-two"
    src_one.mkdir()
    src_two.mkdir()

    _write_workflow_yaml(src_one, filename="a.yaml", name="wf-a")
    _write_workflow_yaml(src_two, filename="b.yaml", name="wf-b")
    summary = reload_registry_from_source_paths(registry, [src_one, src_two])

    assert summary.workflow_count == 2
    assert summary.source_count == 2
    assert sorted(summary.workflow_names) == ["wf-a", "wf-b"]
    assert sorted(registry.list_names()) == ["wf-a", "wf-b"]

    _write_workflow_yaml(src_one, filename="c.yaml", name="wf-c")
    summary_2 = reload_registry_from_source_paths(registry, [src_one])

    assert summary_2.workflow_count == 2
    assert sorted(summary_2.workflow_names) == ["wf-a", "wf-c"]
    assert sorted(registry.list_names()) == ["wf-a", "wf-c"]
    assert not registry.exists("wf-b")


def test_reload_registry_from_source_paths_empty_sources_clears_registry(tmp_path: Path) -> None:
    registry = WorkflowRegistry()
    source = tmp_path / "source"
    source.mkdir()
    _write_workflow_yaml(source, filename="a.yaml", name="wf-a")
    reload_registry_from_source_paths(registry, [source])
    assert sorted(registry.list_names()) == ["wf-a"]

    summary = reload_registry_from_source_paths(registry, [])

    assert summary.workflow_count == 0
    assert summary.source_count == 0
    assert summary.workflow_names == []
    assert registry.list_names() == []


def test_reload_registry_from_source_paths_duplicate_name_is_atomic(tmp_path: Path) -> None:
    registry = WorkflowRegistry()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _write_workflow_yaml(baseline, filename="base.yaml", name="baseline")
    reload_registry_from_source_paths(registry, [baseline])

    src_one = tmp_path / "src-one"
    src_two = tmp_path / "src-two"
    src_one.mkdir()
    src_two.mkdir()
    _write_workflow_yaml(src_one, filename="a.yaml", name="dup")
    _write_workflow_yaml(src_two, filename="b.yaml", name="dup")

    with pytest.raises(WorkflowSourceReloadError) as exc:
        reload_registry_from_source_paths(registry, [src_one, src_two])

    assert exc.value.code == "workflow_duplicate_name"
    assert sorted(registry.list_names()) == ["baseline"]


def test_reload_registry_from_source_paths_invalid_yaml_is_atomic(tmp_path: Path) -> None:
    registry = WorkflowRegistry()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _write_workflow_yaml(baseline, filename="base.yaml", name="baseline")
    reload_registry_from_source_paths(registry, [baseline])

    bad = tmp_path / "bad"
    bad.mkdir()
    _write_invalid_workflow_yaml(bad, filename="broken.yaml")

    with pytest.raises(WorkflowSourceReloadError) as exc:
        reload_registry_from_source_paths(registry, [bad])

    assert exc.value.code == "workflow_invalid_definition"
    assert sorted(registry.list_names()) == ["baseline"]


def test_reload_registry_from_source_paths_invalid_source_path_is_atomic(tmp_path: Path) -> None:
    registry = WorkflowRegistry()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _write_workflow_yaml(baseline, filename="base.yaml", name="baseline")
    reload_registry_from_source_paths(registry, [baseline])

    missing = tmp_path / "missing-dir"

    with pytest.raises(WorkflowSourceReloadError) as exc:
        reload_registry_from_source_paths(registry, [missing])

    assert exc.value.code == "workflow_source_invalid_path"
    assert sorted(registry.list_names()) == ["baseline"]


@pytest.mark.asyncio
async def test_server_load_workflows_ignores_env_and_uses_sqlite_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_source = tmp_path / "env-source"
    sqlite_source = tmp_path / "sqlite-source"
    env_source.mkdir()
    sqlite_source.mkdir()
    _write_workflow_yaml(env_source, filename="env.yaml", name="wf-env")
    _write_workflow_yaml(sqlite_source, filename="sqlite.yaml", name="wf-sqlite")

    monkeypatch.setenv("WORKFLOWS_TEMPLATE_PATHS", str(env_source))
    resources = build_resources(base_dir=tmp_path)
    try:
        projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        project_id = _create_project(projects_repo, slug="reload-sqlite")
        sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(sqlite_source))
        )

        summary = server.load_workflows(resources)

        assert summary.workflow_names == ["wf-sqlite"]
        assert resources.workflow_registry.list_names() == ["wf-sqlite"]
        assert "wf-env" not in resources.workflow_registry.list_names()
        reloaded_sources = sources_repo.list()
        assert [(source.status, source.error_message) for source in reloaded_sources] == [
            ("loaded", None)
        ]
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_server_load_workflows_marks_all_configured_sources_failed_on_error(
    tmp_path: Path,
) -> None:
    good_source = tmp_path / "sqlite-good"
    bad_source = tmp_path / "sqlite-bad"
    good_source.mkdir()
    bad_source.mkdir()
    _write_workflow_yaml(good_source, filename="good.yaml", name="wf-good")

    resources = build_resources(base_dir=tmp_path)
    try:
        projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        project_id = _create_project(projects_repo, slug="reload-failure")
        sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(good_source))
        )

        initial = server.load_workflows(resources)
        assert initial.workflow_names == ["wf-good"]
        assert resources.workflow_registry.list_names() == ["wf-good"]

        _write_invalid_workflow_yaml(bad_source, filename="broken.yaml")
        sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(bad_source))
        )

        with pytest.raises(WorkflowSourceReloadError) as exc:
            server.load_workflows(resources)

        assert exc.value.code == "workflow_invalid_definition"
        assert resources.workflow_registry.list_names() == ["wf-good"]
        reloaded_sources = sources_repo.list()
        assert [source.status for source in reloaded_sources] == ["failed", "failed"]
        assert all(
            source.error_message is not None
            and "Invalid workflow definition" in source.error_message
            for source in reloaded_sources
        )
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_server_load_workflows_empty_sources_clears_registry(tmp_path: Path) -> None:
    source = tmp_path / "sqlite-source"
    source.mkdir()
    _write_workflow_yaml(source, filename="a.yaml", name="wf-a")
    resources = build_resources(base_dir=tmp_path)
    try:
        projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        project_id = _create_project(projects_repo, slug="reload-empty")
        created = sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(source))
        )

        initial = server.load_workflows(resources)
        assert initial.workflow_names == ["wf-a"]
        assert resources.workflow_registry.list_names() == ["wf-a"]

        sources_repo.delete(created.source_id)
        reloaded = server.load_workflows(resources)

        assert reloaded.workflow_count == 0
        assert resources.workflow_registry.list_names() == []
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_reload_workflows_tool_uses_context_callback_and_stable_failure() -> None:
    from workflows_mcp.tools import reload_workflows

    app_ctx = AppContext(
        registry=WorkflowRegistry(),
        executor_registry=object(),
        llm_config_loader=object(),
        io_queue=None,
    )
    callback_calls = {"count": 0}

    def _reload_callback() -> SimpleNamespace:
        callback_calls["count"] += 1
        return SimpleNamespace(
            source_count=2,
            workflow_count=3,
            workflow_names=["a", "b", "c"],
            source_paths=["/tmp/a", "/tmp/b"],
        )

    app_ctx.reload_workflows = _reload_callback
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=app_ctx))

    result = await reload_workflows(ctx=ctx)
    data = result.structuredContent

    assert callback_calls["count"] == 1
    assert data["status"] == "success"
    assert data["total"] == 3
    assert data["source_count"] == 2
    assert data["workflow_names"] == ["a", "b", "c"]

    app_ctx.reload_workflows = None
    failed = await reload_workflows(ctx=ctx)
    failed_data = failed.structuredContent

    assert failed_data["status"] == "failure"
    assert failed_data["message"] == "Workflow reload is not available in current server context."
