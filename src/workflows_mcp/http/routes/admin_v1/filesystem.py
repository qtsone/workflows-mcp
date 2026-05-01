from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    require_current_admin_session,
)

router = APIRouter(prefix="/filesystem")


class ErrorDetail(BaseModel):
    code: str
    message: str


class FilesystemEntryResponse(BaseModel):
    name: str
    path: str
    type: Literal["directory", "file"]
    selectable: bool


class FilesystemEntriesResponse(BaseModel):
    root: str
    path: str
    parent: str | None
    can_go_up: bool
    entries: list[FilesystemEntryResponse]


def _raise(status_code: int, code: str, message: str) -> NoReturn:
    raise HTTPException(
        status_code=status_code,
        detail=ErrorDetail(code=code, message=message).model_dump(),
    )


def _configured_browsing_root() -> Path:
    raw = os.environ.get("WORKFLOWS_SCAN_ROOT", "").strip()
    if not raw:
        return Path("/")
    try:
        root = Path(raw).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        _raise(
            status.HTTP_409_CONFLICT,
            "filesystem_browsing_root_invalid",
            "Server folder browsing root is not available.",
        )
    if not root.is_dir():
        _raise(
            status.HTTP_409_CONFLICT,
            "filesystem_browsing_root_invalid",
            "Server folder browsing root is not available.",
        )
    return root


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_requested_path(raw_path: str | None, root: Path) -> Path:
    if raw_path is None or raw_path.strip() == "":
        return root
    if not Path(raw_path).expanduser().is_absolute():
        _raise(
            status.HTTP_400_BAD_REQUEST,
            "filesystem_path_malformed",
            "Path must be absolute.",
        )
    try:
        candidate = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        _raise(
            status.HTTP_404_NOT_FOUND,
            "filesystem_path_not_found",
            "Directory was not found.",
        )
    if not _is_within_root(candidate, root):
        _raise(
            status.HTTP_403_FORBIDDEN,
            "filesystem_path_outside_root",
            "Directory is outside the configured browsing root.",
        )
    if not candidate.is_dir():
        _raise(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "filesystem_path_not_directory",
            "Path is not a directory.",
        )
    return candidate


def _normalize_extensions(extensions: list[str] | None) -> set[str]:
    normalized: set[str] = set()
    if not extensions:
        return normalized
    for raw in extensions:
        for part in raw.split(","):
            value = part.strip().lower()
            if not value:
                continue
            if value.startswith("."):
                value = value[1:]
            if value:
                normalized.add(value)
    return normalized


def _safe_child_entries(
    path: Path,
    root: Path,
    *,
    selection_type: Literal["folder", "file"],
    extensions: set[str],
) -> list[FilesystemEntryResponse]:
    directories: list[FilesystemEntryResponse] = []
    files: list[FilesystemEntryResponse] = []
    try:
        children = list(path.iterdir())
    except PermissionError:
        _raise(
            status.HTTP_403_FORBIDDEN,
            "filesystem_access_denied",
            "Directory cannot be accessed.",
        )
    except OSError:
        _raise(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "filesystem_listing_failed",
            "Directory listing failed.",
        )

    for child in children:
        try:
            resolved = child.resolve(strict=True)
            if not _is_within_root(resolved, root):
                continue

            if resolved.is_dir():
                try:
                    next(resolved.iterdir(), None)
                except PermissionError:
                    continue
                except OSError:
                    continue
                directories.append(
                    FilesystemEntryResponse(
                        name=child.name,
                        path=str(resolved),
                        type="directory",
                        selectable=selection_type == "folder",
                    )
                )
                continue

            if not resolved.is_file():
                continue

            if selection_type == "file":
                suffix = resolved.suffix.lower().lstrip(".")
                if suffix not in extensions:
                    continue

            files.append(
                FilesystemEntryResponse(
                    name=child.name,
                    path=str(resolved),
                    type="file",
                    selectable=selection_type == "file",
                )
            )
        except (OSError, RuntimeError):
            continue

    directories_sorted = sorted(directories, key=lambda item: item.name.lower())
    files_sorted = sorted(files, key=lambda item: item.name.lower())
    return directories_sorted + files_sorted


@router.get(
    "/entries",
    response_model=FilesystemEntriesResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_entries(
    path: Annotated[str | None, Query(max_length=4096)] = None,
    selection_type: Annotated[Literal["folder", "file"], Query()] = "folder",
    extensions: Annotated[list[str] | None, Query()] = None,
    _current: CurrentAdminSession = Depends(require_current_admin_session),
) -> FilesystemEntriesResponse:
    root = _configured_browsing_root()
    current = _resolve_requested_path(path, root)
    parent: str | None = None
    if current != root:
        resolved_parent = current.parent.resolve(strict=True)
        if _is_within_root(resolved_parent, root):
            parent = str(resolved_parent)
    normalized_extensions = _normalize_extensions(extensions)
    return FilesystemEntriesResponse(
        root=str(root),
        path=str(current),
        parent=parent,
        can_go_up=parent is not None,
        entries=_safe_child_entries(
            current,
            root,
            selection_type=selection_type,
            extensions=normalized_extensions,
        ),
    )
