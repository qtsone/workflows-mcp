from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pathspec import PathSpec

_IGNORE_FILENAMES: tuple[str, ...] = (".gitignore", ".workflowsignore")
_DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    ".gitignore",
    ".workflowsignore",
    ".git/",
    "**/.git/",
    "node_modules/",
    "**/node_modules/",
    ".venv/",
    "**/.venv/",
    "dist/",
    "**/dist/",
    "build/",
    "**/build/",
    "__pycache__/",
    "**/__pycache__/",
    ".pytest_cache/",
    "**/.pytest_cache/",
    ".mypy_cache/",
    "**/.mypy_cache/",
    ".ruff_cache/",
    "**/.ruff_cache/",
)


@dataclass(frozen=True, slots=True)
class WatcherIgnorePolicy:
    project_root: Path
    spec: PathSpec

    @classmethod
    def from_project_root(cls, root: Path) -> WatcherIgnorePolicy:
        project_root = root.resolve(strict=True)

        patterns: list[str] = list(_DEFAULT_IGNORE_PATTERNS)
        for filename in _IGNORE_FILENAMES:
            ignore_path = project_root / filename
            if not ignore_path.exists() or not ignore_path.is_file():
                continue
            patterns.extend(ignore_path.read_text(encoding="utf-8").splitlines())

        return cls(project_root=project_root, spec=PathSpec.from_lines("gitwildmatch", patterns))

    def is_ignored(self, path: Path) -> bool:
        resolved = path.resolve()
        try:
            relative_path = resolved.relative_to(self.project_root)
        except ValueError:
            return True

        return self.spec.match_file(relative_path.as_posix())
