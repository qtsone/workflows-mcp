"""Project file discovery executor for ADR-013 project sync workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from workflows_mcp.watcher.ignore import WatcherIgnorePolicy
from workflows_mcp.watcher.scanner import scan_project_files

from .block import BlockInput, BlockOutput
from .execution import Execution
from .executor_base import BlockExecutor, ExecutorCapabilities, ExecutorSecurityLevel
from .treesitter_languages import detect_language

DEFAULT_PROJECT_WING = "default-wing"
DEFAULT_PROJECT_ROOM = "default-room"


class ProjectFilesInput(BlockInput):
    """Discover files eligible for System 1 project sync."""

    project_root: str = Field(description="Absolute project root to scan.")
    fs_allowlist: list[str] = Field(
        default_factory=list,
        description="Allowed absolute filesystem roots for this project.",
    )
    candidate_paths: list[str] = Field(
        default_factory=list,
        description="Optional project-relative paths limiting dirty-file sync scope.",
    )
    scope: Literal["dirty", "reconcile", "rebuild"] = Field(
        default="rebuild",
        description="Sync scope for diagnostics; discovery policy is selected by candidate_paths.",
    )
    include_supported_only: bool = Field(
        default=True,
        description="When true, unsupported TreeSitter languages are skipped.",
    )
    default_wing: str | None = Field(default=None)
    default_room: str | None = Field(default=None)
    default_compartment: str = Field(description="Default project compartment.")


class ProjectFilesOutput(BlockOutput):
    """Project file manifest for system1-scan for_each blocks."""

    files: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    skipped_unsupported: int = 0
    skipped_disallowed: int = 0
    skipped_missing: int = 0


def _path_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _allowed_roots(raw_paths: list[str]) -> list[Path]:
    return [Path(path).expanduser().resolve(strict=False) for path in raw_paths]


def _is_allowed(absolute_path: Path, allowed_roots: list[Path]) -> bool:
    if not allowed_roots:
        return True
    resolved = absolute_path.expanduser().resolve(strict=False)
    return any(resolved == allowed or _path_within(resolved, allowed) for allowed in allowed_roots)


def _candidate_relative_paths(root: Path, candidate_paths: list[str]) -> list[Path]:
    relative_paths: list[Path] = []
    for raw_path in candidate_paths:
        if not raw_path or raw_path == ".":
            continue
        candidate = Path(raw_path)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve(strict=False).relative_to(root)
            except ValueError:
                continue
        relative_paths.append(Path(candidate.as_posix()))
    return sorted(set(relative_paths), key=lambda path: path.as_posix())


class ProjectFilesExecutor(BlockExecutor):
    """Discover project files using watcher ignore and allowlist policy."""

    type_name: ClassVar[str] = "ProjectFiles"
    input_type: ClassVar[type[BlockInput]] = ProjectFilesInput
    output_type: ClassVar[type[BlockOutput]] = ProjectFilesOutput
    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_read_files=True)

    async def execute(  # type: ignore[override]
        self,
        inputs: ProjectFilesInput,
        _context: Execution,
    ) -> ProjectFilesOutput:
        root = Path(inputs.project_root).expanduser().resolve(strict=True)
        policy = WatcherIgnorePolicy.from_project_root(root)
        allowed_roots = _allowed_roots(inputs.fs_allowlist)
        topology_override = {
            "wing": (inputs.default_wing or "").strip() or DEFAULT_PROJECT_WING,
            "room": (inputs.default_room or "").strip() or DEFAULT_PROJECT_ROOM,
            "compartment": inputs.default_compartment,
            "override_reason": "Project default topology for System 1 project sync",
            "applied_by": "admin_sync",
        }

        if inputs.candidate_paths:
            relative_paths = _candidate_relative_paths(root, inputs.candidate_paths)
        else:
            relative_paths = scan_project_files(root, policy=policy)

        files: list[dict[str, Any]] = []
        skipped_unsupported = 0
        skipped_disallowed = 0
        skipped_missing = 0

        for relative_path in relative_paths:
            absolute_path = root / relative_path
            if not absolute_path.is_file():
                skipped_missing += 1
                continue
            try:
                if policy.is_ignored(absolute_path):
                    skipped_disallowed += 1
                    continue
            except OSError:
                skipped_disallowed += 1
                continue
            if not _is_allowed(absolute_path, allowed_roots):
                skipped_disallowed += 1
                continue

            language = detect_language(relative_path.as_posix())
            if inputs.include_supported_only and language == "unsupported":
                skipped_unsupported += 1
                continue

            files.append(
                {
                    "file_path": str(absolute_path),
                    "repo_relative_path": relative_path.as_posix(),
                    "language": language,
                    "topology_override": topology_override,
                }
            )

        return ProjectFilesOutput(
            files=files,
            count=len(files),
            skipped_unsupported=skipped_unsupported,
            skipped_disallowed=skipped_disallowed,
            skipped_missing=skipped_missing,
        )
