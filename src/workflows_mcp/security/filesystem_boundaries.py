"""Reusable filesystem boundary helpers for project-scoped operations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def normalize_project_roots(
    *,
    fs_root: str,
    fs_allowlist: Sequence[str] | None,
) -> tuple[Path, ...]:
    """Resolve and normalize effective project roots.

    Effective roots are always ``[fs_root] + allowlist`` with duplicates removed.
    """
    resolved: list[Path] = [Path(fs_root).expanduser().resolve()]
    for raw in fs_allowlist or ():
        if raw is None:
            continue
        candidate = str(raw).strip()
        if not candidate:
            continue
        resolved.append(Path(candidate).expanduser().resolve())

    dedup: list[Path] = []
    seen: set[Path] = set()
    for item in resolved:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return tuple(dedup)


def path_within_any_root(path: Path, roots: Sequence[Path]) -> bool:
    """Return True when path resolves inside at least one root."""
    resolved = path.expanduser().resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def validate_path_within_effective_boundary(
    candidate_path: Path,
    *,
    global_root: Path,
    project_roots: Sequence[Path] | None,
) -> bool:
    """Validate candidate path against global ceiling and optional project roots.

    Rules:
    - Path must always be under ``global_root``.
    - When ``project_roots`` are provided and non-empty, path must also be under
      at least one project root.
    - When ``project_roots`` is None/empty, only global ceiling is enforced.
    """
    resolved = candidate_path.expanduser().resolve()
    resolved_global = global_root.expanduser().resolve()

    try:
        resolved.relative_to(resolved_global)
    except ValueError:
        return False

    if not project_roots:
        return True
    return path_within_any_root(resolved, project_roots)
