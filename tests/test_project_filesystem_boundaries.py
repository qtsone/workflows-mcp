"""Project filesystem boundary enforcement tests (Phase 6 Slice E)."""

from __future__ import annotations

from pathlib import Path

import pytest

from workflows_mcp.context import SessionProjectContext
from workflows_mcp.engine.executors_file import run_readfiles_scan
from workflows_mcp.memory.memory_service import MemoryContractError
from workflows_mcp.security.filesystem_boundaries import (
    normalize_project_roots,
    validate_path_within_effective_boundary,
)
from workflows_mcp.tools_memory import ScanConfig, _run_scan


def _project(
    *,
    fs_root: str | None,
    fs_allowlist: tuple[str, ...] = (),
) -> SessionProjectContext:
    return SessionProjectContext(
        project_id="p1",
        slug="proj",
        palace="proj",
        default_wing=None,
        default_room=None,
        source="session_selected",
        fs_root=fs_root,
        fs_allowlist=fs_allowlist,
    )


def test_session_project_context_models_filesystem_policy() -> None:
    project = _project(fs_root="/tmp/proj", fs_allowlist=("/tmp/shared",))
    assert project.fs_root == "/tmp/proj"
    assert tuple(project.fs_allowlist) == ("/tmp/shared",)


def test_validate_path_within_effective_boundary_enforces_global_and_project_roots(
    tmp_path: Path,
) -> None:
    global_root = tmp_path / "global"
    project_root = global_root / "project"
    allow_root = global_root / "allow"
    outside_project = global_root / "outside-project"
    outside_global = tmp_path / "outside-global"

    for p in (project_root, allow_root, outside_project, outside_global):
        p.mkdir(parents=True, exist_ok=True)

    roots = normalize_project_roots(
        fs_root=str(project_root),
        fs_allowlist=(str(allow_root),),
    )

    assert validate_path_within_effective_boundary(
        project_root / "a.txt",
        global_root=global_root,
        project_roots=roots,
    )
    assert validate_path_within_effective_boundary(
        allow_root / "b.txt",
        global_root=global_root,
        project_roots=roots,
    )
    assert not validate_path_within_effective_boundary(
        outside_project / "c.txt",
        global_root=global_root,
        project_roots=roots,
    )
    assert not validate_path_within_effective_boundary(
        outside_global / "d.txt",
        global_root=global_root,
        project_roots=roots,
    )


@pytest.mark.asyncio
async def test_scan_root_outside_project_boundary_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_root = tmp_path / "global"
    project_root = global_root / "project"
    outside = global_root / "outside"
    project_root.mkdir(parents=True)
    outside.mkdir(parents=True)
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(global_root))

    scan = ScanConfig(path="x.txt", root=str(outside))
    with pytest.raises(MemoryContractError) as exc:
        await _run_scan(scan, active_project=_project(fs_root=str(project_root)))
    assert exc.value.code == "MEM_SCAN_PATH_OUT_OF_ROOT"
    assert "project boundary" in exc.value.message


@pytest.mark.asyncio
async def test_scan_path_traversal_outside_project_boundary_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_root = tmp_path / "global"
    project_root = global_root / "project"
    secret_root = global_root / "secret"
    project_root.mkdir(parents=True)
    secret_root.mkdir(parents=True)
    (secret_root / "secret.txt").write_text("secret")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(global_root))

    scan = ScanConfig(path="../secret/secret.txt", root=str(project_root))
    with pytest.raises(MemoryContractError) as exc:
        await _run_scan(scan, active_project=_project(fs_root=str(project_root)))
    assert exc.value.code == "MEM_SCAN_PATH_OUT_OF_ROOT"


@pytest.mark.asyncio
async def test_symlink_escape_outside_project_boundary_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_root = tmp_path / "global"
    project_root = global_root / "project"
    outside = global_root / "outside"
    project_root.mkdir(parents=True)
    outside.mkdir(parents=True)
    (outside / "real.txt").write_text("outside")
    (project_root / "escape.txt").symlink_to(outside / "real.txt")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(global_root))

    scan = ScanConfig(path="escape.txt", root=str(project_root))
    with pytest.raises(MemoryContractError) as exc:
        await _run_scan(scan, active_project=_project(fs_root=str(project_root)))
    assert exc.value.code == "MEM_SCAN_PATH_OUT_OF_ROOT"


@pytest.mark.asyncio
async def test_allowlist_outside_fs_root_is_accepted_when_under_global_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_root = tmp_path / "global"
    project_root = global_root / "project"
    allow_root = global_root / "allow"
    project_root.mkdir(parents=True)
    allow_root.mkdir(parents=True)
    (allow_root / "ok.txt").write_text("ok")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(global_root))

    scan = ScanConfig(path=None, patterns=["ok.txt"], root=str(allow_root))
    files, _snapshot = await _run_scan(
        scan,
        active_project=_project(
            fs_root=str(project_root),
            fs_allowlist=(str(allow_root),),
        ),
    )
    assert [f["path"] for f in files] == ["ok.txt"]


@pytest.mark.asyncio
async def test_global_ceiling_rejects_even_if_project_allowlist_contains_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_root = tmp_path / "global"
    project_root = global_root / "project"
    global_root.mkdir(parents=True)
    project_root.mkdir(parents=True)
    outside_global = tmp_path / "outside-global"
    outside_global.mkdir(parents=True)
    (outside_global / "x.txt").write_text("x")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(global_root))

    scan = ScanConfig(path="x.txt", root=str(outside_global))
    with pytest.raises(MemoryContractError) as exc:
        await _run_scan(
            scan,
            active_project=_project(
                fs_root=str(project_root),
                fs_allowlist=(str(outside_global),),
            ),
        )
    assert exc.value.code == "MEM_SCAN_PATH_OUT_OF_ROOT"


@pytest.mark.asyncio
async def test_workflow_file_scan_is_not_auto_sandboxed_by_project_fs_root_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: workflow file reads remain governed by global ceiling only."""
    global_root = tmp_path / "global"
    project_root = global_root / "project"
    workflow_root = global_root / "workflow-area"
    project_root.mkdir(parents=True)
    workflow_root.mkdir(parents=True)
    (workflow_root / "wf.txt").write_text("workflow")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(global_root))

    # This path is outside project_root but still under global ceiling.
    # run_readfiles_scan (workflow-facing file execution helper) must keep working.
    files = await run_readfiles_scan(
        path=None,
        base_path=str(workflow_root),
        patterns=["wf.txt"],
    )
    assert [f["path"] for f in files] == ["wf.txt"]


@pytest.mark.asyncio
async def test_run_scan_without_active_project_uses_global_ceiling_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_root = tmp_path / "global"
    inside = global_root / "inside"
    outside = tmp_path / "outside"
    inside.mkdir(parents=True)
    outside.mkdir(parents=True)
    (inside / "ok.txt").write_text("ok")
    (outside / "nope.txt").write_text("nope")
    monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(global_root))

    inside_scan = ScanConfig(path=None, patterns=["ok.txt"], root=str(inside))
    files, _ = await _run_scan(inside_scan, active_project=None)
    assert [f["path"] for f in files] == ["ok.txt"]

    outside_scan = ScanConfig(path="nope.txt", root=str(outside))
    with pytest.raises(MemoryContractError) as exc:
        await _run_scan(outside_scan, active_project=None)
    assert exc.value.code == "MEM_SCAN_PATH_OUT_OF_ROOT"
