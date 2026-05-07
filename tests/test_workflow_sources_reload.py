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
from workflows_mcp.metadata.repos.workflow_sources_repo import (
    DuplicateWorkflowSourceError,
    InvalidWorkflowSourcePathError,
    SQLiteWorkflowSourcesRepository,
    WorkflowSourceCreate,
    WorkflowSourceNotFoundError,
)


def _repos(
    tmp_path: Path,
) -> tuple[object, SQLiteWorkflowSourcesRepository]:
    conn = connect_metadata_db(tmp_path / "metadata.db")
    migrate_metadata_db(conn)
    return conn, SQLiteWorkflowSourcesRepository(conn)


def test_create_get_list_and_delete_workflow_source(tmp_path: Path) -> None:
    conn, sources_repo = _repos(tmp_path)
    try:
        source_dir = tmp_path / "sources" / "a"
        source_dir.mkdir(parents=True)

        created = sources_repo.create(
            WorkflowSourceCreate(source_path=f"{source_dir}/..//a")
        )
        expected_path = str(source_dir.expanduser().resolve(strict=False))

        assert created.source_path == expected_path
        assert created.discovered_at
        assert created.last_loaded_at is None
        assert created.status is None
        assert created.error_message is None

        fetched = sources_repo.get(created.source_id)
        assert fetched == created

        listed = sources_repo.list()
        assert [item.source_id for item in listed] == [created.source_id]

        sources_repo.delete(created.source_id)
        assert sources_repo.list() == []
        with pytest.raises(WorkflowSourceNotFoundError):
            sources_repo.get(created.source_id)
    finally:
        conn.close()


def test_create_rejects_missing_or_non_directory_path(tmp_path: Path) -> None:
    conn, sources_repo = _repos(tmp_path)
    try:
        missing = tmp_path / "does-not-exist"
        file_path = tmp_path / "not-a-dir.txt"
        file_path.write_text("x", encoding="utf-8")

        with pytest.raises(InvalidWorkflowSourcePathError):
            sources_repo.create(WorkflowSourceCreate(source_path=str(missing)))

        with pytest.raises(InvalidWorkflowSourcePathError):
            sources_repo.create(WorkflowSourceCreate(source_path=str(file_path)))
    finally:
        conn.close()


def test_create_rejects_duplicate_source_path(tmp_path: Path) -> None:
    conn, sources_repo = _repos(tmp_path)
    try:
        source_dir = tmp_path / "wf" / "nested"
        source_dir.mkdir(parents=True)

        sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))
        with pytest.raises(DuplicateWorkflowSourceError):
            sources_repo.create(WorkflowSourceCreate(source_path=f"{source_dir}/../nested"))
    finally:
        conn.close()


def test_direct_sql_duplicate_insert_fails_with_integrity_error(tmp_path: Path) -> None:
    conn, sources_repo = _repos(tmp_path)
    try:
        source_dir = tmp_path / "wf" / "direct"
        source_dir.mkdir(parents=True)
        normalized = str(source_dir.expanduser().resolve(strict=False))

        sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO workflow_sources (source_id, source_path, checksum)
                VALUES (?, ?, ?)
                """,
                ("direct-duplicate", normalized, None),
            )
    finally:
        conn.close()


def test_create_maps_db_unique_constraint_to_duplicate_workflow_source_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn, sources_repo = _repos(tmp_path)
    try:
        source_dir = tmp_path / "wf" / "race"
        source_dir.mkdir(parents=True)

        sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))

        monkeypatch.setattr(
            sources_repo,
            "_assert_not_duplicate_path",
            lambda *, normalized_source_path: None,
        )

        with pytest.raises(DuplicateWorkflowSourceError):
            sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))
    finally:
        conn.close()


def test_delete_missing_source_raises_not_found(tmp_path: Path) -> None:
    conn, sources_repo = _repos(tmp_path)
    try:
        with pytest.raises(WorkflowSourceNotFoundError):
            sources_repo.delete("missing-source")
    finally:
        conn.close()


def test_update_reload_state_upserts_and_cascade_delete(tmp_path: Path) -> None:
    conn, sources_repo = _repos(tmp_path)
    try:
        source_dir = tmp_path / "wf"
        source_dir.mkdir()

        created = sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))
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


_BUILTIN_WORKFLOW_NAMES: frozenset[str] = frozenset(
    {
        "project-memory-sync",
        "system1-project-sync",
        "system1-scan",
        "system2-derive",
        "system2-project-sync",
        "system2-verify-lifecycle",
    }
)


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
        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        sources_repo.create(WorkflowSourceCreate(source_path=str(sqlite_source)))

        summary = server.load_workflows(resources)

        configured_names = [n for n in summary.workflow_names if not _is_builtin(n)]
        assert configured_names == ["wf-sqlite"]
        registry_names = [n for n in resources.workflow_registry.list_names() if not _is_builtin(n)]
        assert registry_names == ["wf-sqlite"]
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
        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        sources_repo.create(WorkflowSourceCreate(source_path=str(good_source)))

        initial = server.load_workflows(resources)
        assert [n for n in initial.workflow_names if not _is_builtin(n)] == ["wf-good"]
        initial_registry = [
            n for n in resources.workflow_registry.list_names() if not _is_builtin(n)
        ]
        assert initial_registry == ["wf-good"]

        _write_invalid_workflow_yaml(bad_source, filename="broken.yaml")
        sources_repo.create(WorkflowSourceCreate(source_path=str(bad_source)))

        with pytest.raises(WorkflowSourceReloadError) as exc:
            server.load_workflows(resources)

        assert exc.value.code == "workflow_invalid_definition"
        after_fail = [n for n in resources.workflow_registry.list_names() if not _is_builtin(n)]
        assert after_fail == ["wf-good"]
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
        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        created = sources_repo.create(WorkflowSourceCreate(source_path=str(source)))

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
# Slice 2b: load_workflows does NOT seed a System project (v10 behavior)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_workflows_existing_user_sources_still_load(tmp_path: Path) -> None:
    """System project removal must not break existing user workflow sources."""
    from workflows_mcp.http.lifespan import build_resources, stop_resources
    from workflows_mcp.metadata.repos.workflow_sources_repo import (
        SQLiteWorkflowSourcesRepository,
        WorkflowSourceCreate,
    )

    user_source_dir = tmp_path / "user-workflows"
    user_source_dir.mkdir()
    _write_workflow_yaml(user_source_dir, filename="my.yaml", name="my-user-workflow")

    resources = build_resources(base_dir=tmp_path)
    try:
        sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
        sources_repo.create(WorkflowSourceCreate(source_path=str(user_source_dir)))

        server.load_workflows(resources)

        user_wf_names = [
            n for n in resources.workflow_registry.list_names() if not _is_builtin(n)
        ]
        assert "my-user-workflow" in user_wf_names
    finally:
        await stop_resources(resources)


# ---------------------------------------------------------------------------
# Slice 3: runtime builtin path — system workflows come from _builtin_workflow_path().
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_workflows_runtime_builtin_path_loads_system_workflows(
    tmp_path: Path,
) -> None:
    """Builtin workflows in the package templates/memory directory load via runtime path.

    Under the new design, load_workflows() synthesizes the builtin path at
    runtime from _builtin_workflow_path() rather than reading is_system=1 rows
    from SQLite.  All YAML files found under that path must appear in the registry.
    """
    resources = build_resources(base_dir=tmp_path)
    try:
        server.load_workflows(resources)

        builtin_path = server._builtin_workflow_path()
        yaml_files = sorted([*builtin_path.glob("**/*.yaml"), *builtin_path.glob("**/*.yml")])
        expected_names: set[str] = set()
        for yf in yaml_files:
            import yaml as _yaml  # noqa: PLC0415
            data = _yaml.safe_load(yf.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "name" in data:
                expected_names.add(data["name"])

        registry_names = set(resources.workflow_registry.list_names())
        missing = expected_names - registry_names
        assert not missing, (
            f"System workflows not loaded from runtime path {builtin_path!r}: missing {missing}"
        )
    finally:
        await stop_resources(resources)


@pytest.mark.asyncio
async def test_load_workflows_user_workflow_shadowing_builtin_is_rejected(
    tmp_path: Path,
) -> None:
    """User source may not shadow a workflow from the runtime builtin path."""
    user_dir = tmp_path / "user"
    user_dir.mkdir()

    resources = build_resources(base_dir=tmp_path)
    try:
        # Find a known builtin workflow name to attempt shadowing.
        builtin_path = server._builtin_workflow_path()
        yaml_files = [*builtin_path.glob("**/*.yaml"), *builtin_path.glob("**/*.yml")]
        if not yaml_files:
            pytest.skip("No builtin YAMLs present; cannot test shadowing")

        import yaml as _yaml  # noqa: PLC0415
        data = _yaml.safe_load(yaml_files[0].read_text(encoding="utf-8"))
        builtin_name = data.get("name") if isinstance(data, dict) else None
        if not builtin_name:
            pytest.skip("Could not determine builtin workflow name")

        _write_workflow_yaml(user_dir, filename="shadow.yaml", name=builtin_name)

        SQLiteWorkflowSourcesRepository(resources.metadata_db_conn).create(
            WorkflowSourceCreate(source_path=str(user_dir))
        )

        with pytest.raises(WorkflowSourceReloadError) as exc:
            server.load_workflows(resources)

        assert exc.value.code == "user_workflow_shadows_builtin", (
            f"Expected user_workflow_shadows_builtin, got {exc.value.code!r}"
        )
        assert builtin_name in exc.value.message
    finally:
        await stop_resources(resources)


# ---------------------------------------------------------------------------
# System 2 built-in non-shadowing regression (Task 15)
# ---------------------------------------------------------------------------


def test_user_workflow_shadowing_system2_derive_is_rejected(tmp_path: Path) -> None:
    """User workflow named system2-derive must be rejected as it shadows a built-in."""
    registry = WorkflowRegistry()
    builtin_dir = tmp_path / "builtins"
    user_dir = tmp_path / "user"
    builtin_dir.mkdir()
    user_dir.mkdir()
    _write_workflow_yaml(builtin_dir, filename="s2d.yaml", name="system2-derive")
    user_yaml = _write_workflow_yaml(user_dir, filename="u.yaml", name="system2-derive")

    with pytest.raises(WorkflowSourceReloadError) as exc:
        reload_registry_from_source_paths(
            registry, [user_dir], builtin_paths=[builtin_dir]
        )

    assert exc.value.code == "user_workflow_shadows_builtin"
    assert "system2-derive" in exc.value.message
    assert str(user_yaml) in exc.value.message
    assert registry.list_names() == []


def test_user_workflow_shadowing_system2_verify_lifecycle_is_rejected(tmp_path: Path) -> None:
    """User workflow named system2-verify-lifecycle must be rejected as it shadows a built-in."""
    registry = WorkflowRegistry()
    builtin_dir = tmp_path / "builtins"
    user_dir = tmp_path / "user"
    builtin_dir.mkdir()
    user_dir.mkdir()
    _write_workflow_yaml(builtin_dir, filename="s2vl.yaml", name="system2-verify-lifecycle")
    user_yaml = _write_workflow_yaml(user_dir, filename="u.yaml", name="system2-verify-lifecycle")

    with pytest.raises(WorkflowSourceReloadError) as exc:
        reload_registry_from_source_paths(
            registry, [user_dir], builtin_paths=[builtin_dir]
        )

    assert exc.value.code == "user_workflow_shadows_builtin"
    assert "system2-verify-lifecycle" in exc.value.message
    assert str(user_yaml) in exc.value.message
    assert registry.list_names() == []


def test_system2_derive_and_verify_lifecycle_load_alongside_user_workflows(
    tmp_path: Path,
) -> None:
    """system2-derive and system2-verify-lifecycle coexist with non-conflicting user workflows."""
    registry = WorkflowRegistry()
    builtin_dir = tmp_path / "builtins"
    user_dir = tmp_path / "user"
    builtin_dir.mkdir()
    user_dir.mkdir()
    _write_workflow_yaml(builtin_dir, filename="s2d.yaml", name="system2-derive")
    _write_workflow_yaml(builtin_dir, filename="s2vl.yaml", name="system2-verify-lifecycle")
    _write_workflow_yaml(user_dir, filename="u.yaml", name="my-custom-pipeline")

    summary = reload_registry_from_source_paths(
        registry, [user_dir], builtin_paths=[builtin_dir]
    )

    assert summary.builtin_workflow_count == 2
    assert summary.workflow_count == 3
    assert sorted(summary.workflow_names) == [
        "my-custom-pipeline",
        "system2-derive",
        "system2-verify-lifecycle",
    ]
    assert registry.exists("system2-derive")
    assert registry.exists("system2-verify-lifecycle")
    assert registry.exists("my-custom-pipeline")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Runtime builtin path loads without any DB row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_workflows_loads_builtins_from_runtime_path_without_system_row(
    tmp_path: Path,
) -> None:
    """Builtin workflows load from _builtin_workflow_path() without any DB row."""
    resources = build_resources(base_dir=tmp_path)
    try:
        server.load_workflows(resources)

        builtin_path = server._builtin_workflow_path()
        yaml_files = [*builtin_path.glob("**/*.yaml"), *builtin_path.glob("**/*.yml")]

        if yaml_files:
            loaded_names = set(resources.workflow_registry.list_names())
            assert len(loaded_names) > 0, (
                "Expected builtin workflows in registry; runtime path has YAMLs but registry empty"
            )
        else:
            sources_repo = SQLiteWorkflowSourcesRepository(resources.metadata_db_conn)
            user_sources = sources_repo.list()
            if not user_sources:
                assert resources.workflow_registry.list_names() == []
    finally:
        await stop_resources(resources)


# ---------------------------------------------------------------------------
# v10: Repository operates without project_id
# ---------------------------------------------------------------------------


def test_repo_create_does_not_require_project_id(tmp_path: Path) -> None:
    """WorkflowSourceCreate must work without project_id after v10."""
    conn, sources_repo = _repos(tmp_path)
    source_dir = tmp_path / "wf"
    source_dir.mkdir()
    try:
        record = sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))
        assert record.source_id
        assert record.source_path == str(source_dir.resolve())
    finally:
        conn.close()


def test_repo_create_enforces_global_unique_source_path(tmp_path: Path) -> None:
    """Creating a second source with the same path raises DuplicateWorkflowSourceError."""
    conn, sources_repo = _repos(tmp_path)
    source_dir = tmp_path / "wf"
    source_dir.mkdir()
    try:
        sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))
        with pytest.raises(DuplicateWorkflowSourceError):
            sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))
    finally:
        conn.close()


def test_repo_list_returns_no_project_id_field(tmp_path: Path) -> None:
    """Listed WorkflowSourceRecord must not carry a project_id attribute after v10."""
    conn, sources_repo = _repos(tmp_path)
    source_dir = tmp_path / "wf"
    source_dir.mkdir()
    try:
        sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))
        records = sources_repo.list()
        assert len(records) == 1
        assert not hasattr(records[0], "project_id"), (
            "WorkflowSourceRecord must not have project_id after v10"
        )
    finally:
        conn.close()


def test_repo_get_by_path_works_globally_without_project_id(tmp_path: Path) -> None:
    """get_by_path with no project_id arg finds the source globally."""
    conn, sources_repo = _repos(tmp_path)
    source_dir = tmp_path / "wf"
    source_dir.mkdir()
    try:
        created = sources_repo.create(WorkflowSourceCreate(source_path=str(source_dir)))
        found = sources_repo.get_by_path(str(source_dir))
        assert found is not None
        assert found.source_id == created.source_id
    finally:
        conn.close()


def test_load_workflows_does_not_seed_system_project(tmp_path: Path) -> None:
    """load_workflows must not create a project with slug=system and palace=__system__."""
    import asyncio

    from workflows_mcp import server
    from workflows_mcp.http.lifespan import build_resources, stop_resources

    resources = build_resources(base_dir=tmp_path)
    try:
        server.load_workflows(resources)
        row = resources.metadata_db_conn.execute(
            "SELECT 1 FROM projects WHERE slug = ? AND palace = ?",
            ("system", "__system__"),
        ).fetchone()
        assert row is None, "load_workflows must not seed a System project row"
    finally:
        asyncio.run(stop_resources(resources))
