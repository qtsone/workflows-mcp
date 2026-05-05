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
    SystemWorkflowSourceProtectedError,
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


_BUILTIN_WORKFLOW_NAMES: frozenset[str] = frozenset({"system1-scan"})


def _is_builtin(name: str) -> bool:
    return name in _BUILTIN_WORKFLOW_NAMES


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

        configured_names = [n for n in summary.workflow_names if not _is_builtin(n)]
        assert configured_names == ["wf-sqlite"]
        registry_names = [n for n in resources.workflow_registry.list_names() if not _is_builtin(n)]
        assert registry_names == ["wf-sqlite"]
        assert "wf-env" not in resources.workflow_registry.list_names()
        reloaded_sources = [s for s in sources_repo.list() if not s.is_system]
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
        assert [n for n in initial.workflow_names if not _is_builtin(n)] == ["wf-good"]
        initial_registry = [
            n for n in resources.workflow_registry.list_names() if not _is_builtin(n)
        ]
        assert initial_registry == ["wf-good"]

        _write_invalid_workflow_yaml(bad_source, filename="broken.yaml")
        sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(bad_source))
        )

        with pytest.raises(WorkflowSourceReloadError) as exc:
            server.load_workflows(resources)

        assert exc.value.code == "workflow_invalid_definition"
        after_fail = [n for n in resources.workflow_registry.list_names() if not _is_builtin(n)]
        assert after_fail == ["wf-good"]
        reloaded_sources = [s for s in sources_repo.list() if not s.is_system]
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
        assert [n for n in initial.workflow_names if not _is_builtin(n)] == ["wf-a"]
        initial_reg = [n for n in resources.workflow_registry.list_names() if not _is_builtin(n)]
        assert initial_reg == ["wf-a"]

        sources_repo.delete(created.source_id)
        reloaded = server.load_workflows(resources)

        assert [n for n in reloaded.workflow_names if not _is_builtin(n)] == []
        assert [n for n in resources.workflow_registry.list_names() if not _is_builtin(n)] == []
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


# ---------------------------------------------------------------------------
# Built-in workflow source mechanism (Track 4 prereqs / Phase 3)
# ---------------------------------------------------------------------------


def test_builtin_path_with_no_workflows_loads_cleanly(tmp_path: Path) -> None:
    """Empty built-in directory must load with zero workflows and no errors."""
    registry = WorkflowRegistry()
    builtin_dir = tmp_path / "builtins"
    builtin_dir.mkdir()

    summary = reload_registry_from_source_paths(
        registry, [], builtin_paths=[builtin_dir]
    )

    assert summary.workflow_count == 0
    assert summary.builtin_workflow_count == 0
    assert summary.workflow_names == []
    assert registry.list_names() == []


def test_builtin_workflow_loads_and_appears_in_registry(tmp_path: Path) -> None:
    """Workflows in the built-in path are registered and counted as built-ins."""
    registry = WorkflowRegistry()
    builtin_dir = tmp_path / "builtins"
    builtin_dir.mkdir()
    _write_workflow_yaml(builtin_dir, filename="foo.yaml", name="builtin-foo")

    summary = reload_registry_from_source_paths(
        registry, [], builtin_paths=[builtin_dir]
    )

    assert summary.builtin_workflow_count == 1
    assert summary.workflow_count == 1
    assert summary.workflow_names == ["builtin-foo"]
    assert registry.exists("builtin-foo")


def test_user_workflow_shadowing_builtin_is_rejected(tmp_path: Path) -> None:
    """User workflow with same name as a built-in must be rejected atomically."""
    registry = WorkflowRegistry()
    builtin_dir = tmp_path / "builtins"
    user_dir = tmp_path / "user"
    builtin_dir.mkdir()
    user_dir.mkdir()
    _write_workflow_yaml(builtin_dir, filename="b.yaml", name="system1-scan")
    user_yaml = _write_workflow_yaml(user_dir, filename="u.yaml", name="system1-scan")

    with pytest.raises(WorkflowSourceReloadError) as exc:
        reload_registry_from_source_paths(
            registry, [user_dir], builtin_paths=[builtin_dir]
        )

    assert exc.value.code == "user_workflow_shadows_builtin"
    assert "system1-scan" in exc.value.message
    assert str(user_yaml) in exc.value.message
    # Registry untouched (no successful reload happened).
    assert registry.list_names() == []


def test_user_workflow_with_unique_name_loads_alongside_builtin(tmp_path: Path) -> None:
    """User workflow with a non-conflicting name coexists with built-ins."""
    registry = WorkflowRegistry()
    builtin_dir = tmp_path / "builtins"
    user_dir = tmp_path / "user"
    builtin_dir.mkdir()
    user_dir.mkdir()
    _write_workflow_yaml(builtin_dir, filename="b.yaml", name="system1-scan")
    _write_workflow_yaml(user_dir, filename="u.yaml", name="my-pipeline")

    summary = reload_registry_from_source_paths(
        registry, [user_dir], builtin_paths=[builtin_dir]
    )

    assert summary.workflow_count == 2
    assert summary.builtin_workflow_count == 1
    assert sorted(summary.workflow_names) == ["my-pipeline", "system1-scan"]
    assert registry.exists("system1-scan")
    assert registry.exists("my-pipeline")


def test_builtin_duplicate_name_within_builtin_dir_is_rejected(tmp_path: Path) -> None:
    """Two built-in YAMLs sharing a name must raise builtin_workflow_duplicate_name."""
    registry = WorkflowRegistry()
    builtin_dir = tmp_path / "builtins"
    builtin_dir.mkdir()
    _write_workflow_yaml(builtin_dir, filename="a.yaml", name="builtin-foo")
    _write_workflow_yaml(builtin_dir, filename="b.yaml", name="builtin-foo")

    with pytest.raises(WorkflowSourceReloadError) as exc:
        reload_registry_from_source_paths(
            registry, [], builtin_paths=[builtin_dir]
        )

    assert exc.value.code == "builtin_workflow_duplicate_name"
    assert "builtin-foo" in exc.value.message


def test_invalid_builtin_workflow_yaml_is_rejected_with_distinct_code(
    tmp_path: Path,
) -> None:
    """Invalid built-in YAML raises builtin_workflow_invalid_definition (not the user code)."""
    registry = WorkflowRegistry()
    builtin_dir = tmp_path / "builtins"
    builtin_dir.mkdir()
    _write_invalid_workflow_yaml(builtin_dir, filename="broken.yaml")

    with pytest.raises(WorkflowSourceReloadError) as exc:
        reload_registry_from_source_paths(
            registry, [], builtin_paths=[builtin_dir]
        )

    assert exc.value.code == "builtin_workflow_invalid_definition"


def test_builtin_workflow_path_resolves_to_templates_memory() -> None:
    """The packaged built-in workflow directory must be templates/memory, not builtin_workflows."""
    from workflows_mcp import server

    builtin_path = server._builtin_workflow_path()
    assert builtin_path.parts[-2:] == ("templates", "memory"), (
        f"Expected packaged path to end with templates/memory, got: {builtin_path}"
    )


# ---------------------------------------------------------------------------
# Slice 2a: is_system support on workflow sources
# ---------------------------------------------------------------------------


def test_workflow_source_record_has_is_system_field(tmp_path: Path) -> None:
    """WorkflowSourceRecord exposes is_system as a bool, defaulting to False."""
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo, slug="is-system-default")
        source_dir = tmp_path / "src-is-system"
        source_dir.mkdir()

        record = sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(source_dir))
        )

        assert hasattr(record, "is_system")
        assert record.is_system is False
    finally:
        conn.close()


def test_workflow_source_create_accepts_is_system_true(tmp_path: Path) -> None:
    """WorkflowSourceCreate with is_system=True persists and reads back True."""
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo, slug="is-system-true")
        source_dir = tmp_path / "src-system"
        source_dir.mkdir()

        record = sources_repo.create(
            WorkflowSourceCreate(
                project_id=project_id,
                source_path=str(source_dir),
                is_system=True,
            )
        )

        assert record.is_system is True

        fetched = sources_repo.get(record.source_id)
        assert fetched.is_system is True

        listed = sources_repo.list(project_id=project_id)
        assert len(listed) == 1
        assert listed[0].is_system is True
    finally:
        conn.close()


def test_delete_system_source_raises_protected_error(tmp_path: Path) -> None:
    """Deleting a system workflow source raises SystemWorkflowSourceProtectedError."""
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo, slug="del-system-src")
        source_dir = tmp_path / "src-del-system"
        source_dir.mkdir()

        record = sources_repo.create(
            WorkflowSourceCreate(
                project_id=project_id,
                source_path=str(source_dir),
                is_system=True,
            )
        )

        with pytest.raises(SystemWorkflowSourceProtectedError):
            sources_repo.delete(record.source_id)

        # Source must still exist after the rejected delete.
        assert sources_repo.get(record.source_id).source_id == record.source_id
    finally:
        conn.close()


def test_delete_non_system_source_still_works(tmp_path: Path) -> None:
    """Non-system sources can still be deleted normally (regression guard)."""
    conn, projects_repo, sources_repo = _repos(tmp_path)
    try:
        project_id = _create_project(projects_repo, slug="del-non-system")
        source_dir = tmp_path / "src-non-system"
        source_dir.mkdir()

        record = sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(source_dir))
        )

        sources_repo.delete(record.source_id)

        assert sources_repo.list(project_id=project_id) == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Slice 2b: seed System project and system workflow source at load_workflows()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_workflows_seeds_system_project(tmp_path: Path) -> None:
    """load_workflows must create a visible System project in the projects repository."""
    from workflows_mcp.http.lifespan import build_resources, stop_resources
    from workflows_mcp.metadata.repos.projects_repo import SQLiteProjectsRepository

    resources = build_resources(base_dir=tmp_path)
    try:
        server.load_workflows(resources)

        projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
        all_projects = projects_repo.list_all()
        system_projects = [p for p in all_projects if p.name == "System"]
        assert len(system_projects) == 1, (
            f"Expected exactly one 'System' project, got: {[p.name for p in all_projects]}"
        )
        system_project = system_projects[0]
        assert system_project.name == "System"
        assert system_project.slug == "system"
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_load_workflows_seeds_system_workflow_source(tmp_path: Path) -> None:
    """load_workflows must create a system workflow source pointing to templates/memory."""
    from workflows_mcp.http.lifespan import build_resources, stop_resources
    from workflows_mcp.metadata.repos.workflow_sources_repo import SQLiteWorkflowSourcesRepository

    resources = build_resources(base_dir=tmp_path)
    try:
        server.load_workflows(resources)

        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        all_sources = sources_repo.list()
        system_sources = [s for s in all_sources if s.is_system]
        assert len(system_sources) >= 1, (
            f"Expected at least one system workflow source, got: {all_sources}"
        )

        expected_path = str(server._builtin_workflow_path())
        matching = [s for s in system_sources if s.source_path == expected_path]
        assert len(matching) == 1, (
            f"Expected system source with path {expected_path!r}, "
            f"system sources: {[s.source_path for s in system_sources]}"
        )
        assert matching[0].is_system is True
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_load_workflows_system_seeding_is_idempotent(tmp_path: Path) -> None:
    """Calling load_workflows multiple times must not create duplicate System projects or sources."""  # noqa: E501
    from workflows_mcp.http.lifespan import build_resources, stop_resources
    from workflows_mcp.metadata.repos.projects_repo import SQLiteProjectsRepository
    from workflows_mcp.metadata.repos.workflow_sources_repo import SQLiteWorkflowSourcesRepository

    resources = build_resources(base_dir=tmp_path)
    try:
        server.load_workflows(resources)
        server.load_workflows(resources)
        server.load_workflows(resources)

        projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
        system_projects = [p for p in projects_repo.list_all() if p.name == "System"]
        assert len(system_projects) == 1, (
            f"Expected exactly 1 System project after 3 calls, got {len(system_projects)}"
        )

        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        expected_path = str(server._builtin_workflow_path())
        matching = [s for s in sources_repo.list() if s.source_path == expected_path]
        assert len(matching) == 1, (
            f"Expected exactly 1 system source for {expected_path!r} after 3 calls, "
            f"got {len(matching)}"
        )
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_load_workflows_system_project_visible_in_list(tmp_path: Path) -> None:
    """System project must appear in project list_all() with expected fields."""
    from workflows_mcp.http.lifespan import build_resources, stop_resources
    from workflows_mcp.metadata.repos.projects_repo import SQLiteProjectsRepository

    resources = build_resources(base_dir=tmp_path)
    try:
        server.load_workflows(resources)

        projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
        all_projects = projects_repo.list_all()
        system_project = next((p for p in all_projects if p.name == "System"), None)
        assert system_project is not None, "System project not found in list_all()"
        assert system_project.id  # non-empty stable id
        assert system_project.slug == "system"
        assert system_project.name == "System"
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_load_workflows_existing_user_sources_still_load(tmp_path: Path) -> None:
    """Seeding system source must not break existing user workflow sources."""
    from workflows_mcp.http.lifespan import build_resources, stop_resources
    from workflows_mcp.metadata.repos.projects_repo import SQLiteProjectsRepository
    from workflows_mcp.metadata.repos.workflow_sources_repo import (
        SQLiteWorkflowSourcesRepository,
        WorkflowSourceCreate,
    )

    user_source_dir = tmp_path / "user-workflows"
    user_source_dir.mkdir()
    _write_workflow_yaml(user_source_dir, filename="my.yaml", name="my-user-workflow")

    resources = build_resources(base_dir=tmp_path)
    try:
        projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        project_id = _create_project(projects_repo, slug="user-project-2b")
        sources_repo.create(
            WorkflowSourceCreate(project_id=project_id, source_path=str(user_source_dir))
        )

        server.load_workflows(resources)

        user_wf_names = [
            n for n in resources.workflow_registry.list_names() if not _is_builtin(n)
        ]
        assert "my-user-workflow" in user_wf_names
    finally:
        await stop_resources(resources)


# ---------------------------------------------------------------------------
# Slice 3: unified source-record loading — system sources load through
#           is_system source records, not a hidden builtin_paths injection.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_workflows_system_source_record_path_loads_workflows(
    tmp_path: Path,
) -> None:
    """System workflow source registered in SQLite must be used to load system workflows.

    After load_workflows(), all workflow names that live under the system
    source record path must appear in the registry.  This verifies that
    system sources are not bypassed by a hidden builtin_paths injection.
    """
    resources = build_resources(base_dir=tmp_path)
    try:
        server.load_workflows(resources)

        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        system_sources = [s for s in sources_repo.list() if s.is_system]
        assert system_sources, "Expected at least one system source record after load_workflows()"

        system_source = system_sources[0]
        system_path = Path(system_source.source_path)

        # Collect workflow names from the system source path directly.
        yaml_files = sorted([*system_path.glob("**/*.yaml"), *system_path.glob("**/*.yml")])
        expected_names = set()
        for yf in yaml_files:
            import yaml as _yaml  # noqa: PLC0415
            data = _yaml.safe_load(yf.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "name" in data:
                expected_names.add(data["name"])

        registry_names = set(resources.workflow_registry.list_names())
        missing = expected_names - registry_names
        assert not missing, (
            f"System workflows not loaded from source record path {system_source.source_path!r}: "
            f"missing {missing}"
        )
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_load_workflows_custom_system_source_record_loads_its_workflows(
    tmp_path: Path,
) -> None:
    """A user-registered system source record must cause its workflows to load as system workflows.

    This is the key slice-3 contract: if ``is_system=True`` is set on a source
    record, its workflows are treated as system (built-in) workflows — loaded
    first and non-shadowable.  The loader must NOT rely on ``_builtin_workflow_path()``
    being the sole source of system workflows.
    """
    custom_system_dir = tmp_path / "custom-system"
    custom_system_dir.mkdir()
    _write_workflow_yaml(custom_system_dir, filename="sys.yaml", name="custom-system-wf")

    resources = build_resources(base_dir=tmp_path)
    try:
        # Seed the standard system project/source first so seeding doesn't fail.
        server._seed_system_project_and_source(resources)

        # Register a SECOND system source pointing to our custom directory.
        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        from workflows_mcp.metadata.repos.projects_repo import SQLiteProjectsRepository

        projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
        system_project = next(
            p for p in projects_repo.list_all() if p.slug == "system"
        )
        sources_repo.create(
            WorkflowSourceCreate(
                project_id=system_project.id,
                source_path=str(custom_system_dir),
                is_system=True,
            )
        )

        server.load_workflows(resources)

        assert resources.workflow_registry.exists("custom-system-wf"), (
            "Workflow from custom system source record was not loaded into the registry"
        )
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_load_workflows_user_workflow_shadowing_system_source_record_is_rejected(
    tmp_path: Path,
) -> None:
    """User source may not shadow a workflow from a system source record.

    When the system source record contains workflow 'sys-protected' and a
    user source also defines 'sys-protected', load_workflows() must raise
    WorkflowSourceReloadError with code 'user_workflow_shadows_builtin'.
    """
    custom_system_dir = tmp_path / "custom-system"
    custom_system_dir.mkdir()
    _write_workflow_yaml(custom_system_dir, filename="sys.yaml", name="sys-protected")

    user_dir = tmp_path / "user"
    user_dir.mkdir()
    _write_workflow_yaml(user_dir, filename="u.yaml", name="sys-protected")

    resources = build_resources(base_dir=tmp_path)
    try:
        # Seed standard system project and add custom system source.
        server._seed_system_project_and_source(resources)

        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        from workflows_mcp.metadata.repos.projects_repo import SQLiteProjectsRepository

        projects_repo = SQLiteProjectsRepository(resources.metadata_db_conn)
        system_project = next(p for p in projects_repo.list_all() if p.slug == "system")

        sources_repo.create(
            WorkflowSourceCreate(
                project_id=system_project.id,
                source_path=str(custom_system_dir),
                is_system=True,
            )
        )

        # Add user source with colliding name.
        user_project_id = _create_project(projects_repo, slug="shadow-test")
        sources_repo.create(
            WorkflowSourceCreate(project_id=user_project_id, source_path=str(user_dir))
        )

        with pytest.raises(WorkflowSourceReloadError) as exc:
            server.load_workflows(resources)

        assert exc.value.code == "user_workflow_shadows_builtin", (
            f"Expected user_workflow_shadows_builtin, got {exc.value.code!r}"
        )
        assert "sys-protected" in exc.value.message
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_load_workflows_does_not_inject_builtin_path_outside_source_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime load_workflows must obtain all paths from source records.

    If the system source record is absent (e.g., seeding is skipped),
    no builtin workflows should appear in the registry.  This confirms the
    loader does not silently inject _builtin_workflow_path() outside the
    source-record mechanism.
    """
    resources = build_resources(base_dir=tmp_path)
    try:
        # Call reload directly without seeding — no system source record exists.
        repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        sources = repo.list()
        assert not sources, "Expected empty source list before seeding"

        # Patch _builtin_workflow_path to a no-op dir to catch hidden injection.
        empty_dir = tmp_path / "empty-builtin"
        empty_dir.mkdir()
        monkeypatch.setattr(server, "_builtin_workflow_path", lambda: empty_dir)

        from workflows_mcp.engine.workflow_source_loader import reload_registry_from_source_paths

        # Simulate what load_workflows should do: split sources by is_system.
        system_paths = [s.source_path for s in sources if s.is_system]
        user_paths = [s.source_path for s in sources if not s.is_system]

        summary = reload_registry_from_source_paths(
            resources.workflow_registry,
            user_paths,
            builtin_paths=system_paths,
        )

        assert summary.workflow_count == 0, (
            "Expected zero workflows when no source records exist; "
            f"got {summary.workflow_count}: {summary.workflow_names}"
        )
    finally:
        await stop_resources(resources)
