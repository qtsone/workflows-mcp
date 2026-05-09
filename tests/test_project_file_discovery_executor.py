from __future__ import annotations

from pathlib import Path

import pytest

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.executors_project_files import (
    ProjectFilesExecutor,
    ProjectFilesInput,
)


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.asyncio
async def test_project_files_discovers_supported_files_with_ignore_and_allowlist(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repo"
    allowed_root = project_root / "src"
    _write(project_root / ".gitignore", "ignored.py\n")
    _write(allowed_root / "app.py", "def main():\n    return 1\n")
    _write(allowed_root / "notes.txt", "unsupported")
    _write(project_root / "ignored.py", "print('ignored')\n")
    _write(project_root / "outside.py", "print('outside')\n")

    output = await ProjectFilesExecutor().execute(
        ProjectFilesInput(
            project_root=str(project_root),
            fs_allowlist=[str(allowed_root)],
            default_wing="platform",
            default_room="runtime",
            default_compartment="forge",
        ),
        Execution(),
    )

    assert output.count == 1
    assert output.files == [
        {
            "file_path": str(allowed_root / "app.py"),
            "repo_relative_path": "src/app.py",
            "language": "python",
            "topology_override": None,
        }
    ]
    assert output.skipped_unsupported == 1
    assert output.skipped_disallowed == 1


@pytest.mark.asyncio
async def test_project_files_candidate_paths_limit_dirty_sync_scope(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    _write(project_root / "src" / "changed.py", "def changed():\n    return 1\n")
    _write(project_root / "src" / "unchanged.py", "def unchanged():\n    return 1\n")

    output = await ProjectFilesExecutor().execute(
        ProjectFilesInput(
            project_root=str(project_root),
            fs_allowlist=[str(project_root)],
            candidate_paths=["src/changed.py"],
            default_wing="default-wing",
            default_room="default-room",
            default_compartment="forge",
        ),
        Execution(),
    )

    assert [file["repo_relative_path"] for file in output.files] == ["src/changed.py"]
    assert output.count == 1


@pytest.mark.asyncio
async def test_project_files_blank_defaults_do_not_emit_synthesized_topology_override(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repo"
    _write(project_root / "src" / "app.py", "def main():\n    return 1\n")

    output = await ProjectFilesExecutor().execute(
        ProjectFilesInput(
            project_root=str(project_root),
            fs_allowlist=[str(project_root)],
            default_wing="",
            default_room="",
            default_compartment="forge",
        ),
        Execution(),
    )

    assert output.count == 1
    assert output.files[0]["repo_relative_path"] == "src/app.py"
    assert output.files[0]["topology_override"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wing", "room"),
    [
        ("default-wing", "runtime"),
        ("platform", "default-room"),
        ("default", "runtime"),
        ("code", "default"),
    ],
)
async def test_project_files_placeholder_defaults_do_not_emit_topology_override(
    tmp_path: Path,
    wing: str,
    room: str,
) -> None:
    project_root = tmp_path / "repo"
    _write(project_root / "src" / "app.py", "def main():\n    return 1\n")

    output = await ProjectFilesExecutor().execute(
        ProjectFilesInput(
            project_root=str(project_root),
            fs_allowlist=[str(project_root)],
            default_wing=wing,
            default_room=room,
            default_compartment="forge",
        ),
        Execution(),
    )

    assert output.count == 1
    assert output.files[0]["topology_override"] is None
