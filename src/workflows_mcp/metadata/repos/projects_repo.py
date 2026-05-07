from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection


class ProjectRepositoryError(RuntimeError):
    """Base class for deterministic project repository errors."""


class DuplicateProjectSlugError(ProjectRepositoryError):
    """Raised when creating/updating a project with a duplicate slug."""


class DuplicateProjectPalaceError(ProjectRepositoryError):
    """Raised when creating/updating a project with a duplicate palace."""


class ProjectNotFoundError(ProjectRepositoryError):
    """Raised when an operation targets a missing project id."""


class ProjectPalaceImmutableError(ProjectRepositoryError):
    """Raised when attempting to mutate a project's palace after creation."""


class ProjectPersistenceError(ProjectRepositoryError):
    """Raised when a write succeeds but the persisted row cannot be reloaded."""


@dataclass(frozen=True)
class ProjectCreate:
    name: str
    slug: str
    palace: str
    default_wing: str | None
    default_room: str | None
    fs_root: str
    fs_allowlist: list[str] | None = None
    system2_enabled: bool = False


@dataclass(frozen=True)
class ProjectUpdate:
    name: str | None = None
    slug: str | None = None
    palace: str | None = None
    default_wing: str | None = None
    default_room: str | None = None
    fs_root: str | None = None
    fs_allowlist: list[str] | None = None
    system2_enabled: bool | None = None


@dataclass(frozen=True)
class ProjectRecord:
    id: str
    name: str
    slug: str
    palace: str
    default_wing: str | None
    default_room: str | None
    fs_root: str
    fs_allowlist: list[str]
    system2_enabled: bool
    created_at: str
    updated_at: str


def _normalize_fs_path(raw: str) -> str:
    return str(Path(raw).expanduser().resolve(strict=False))


def _encode_allowlist(paths: list[str] | None) -> str | None:
    if paths is None:
        return None
    normalized = [_normalize_fs_path(path) for path in paths]
    return json.dumps(normalized, sort_keys=False)


def _decode_allowlist(value: str | None) -> list[str]:
    if value is None or value == "":
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        return []
    result: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            result.append(item)
    return result


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _db_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return False


class SQLiteProjectsRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def create(self, data: ProjectCreate) -> ProjectRecord:
        project_id = str(uuid.uuid4())
        fs_root = _normalize_fs_path(data.fs_root)
        allowlist_json = _encode_allowlist(data.fs_allowlist)

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                INSERT INTO projects (
                    id,
                    name,
                    slug,
                    palace,
                    default_wing,
                    default_room,
                    fs_root,
                    fs_allowlist_json,
                    system2_enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    data.name,
                    data.slug,
                    data.palace,
                    data.default_wing,
                    data.default_room,
                    fs_root,
                    allowlist_json,
                    1 if data.system2_enabled else 0,
                ),
            )
            row = self._conn.execute(
                """
                SELECT id, name, slug, palace, default_wing, default_room, fs_root,
                       fs_allowlist_json, system2_enabled, created_at, updated_at
                FROM projects
                WHERE id = ?
                """,
                (project_id,),
            ).fetchone()
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            self._raise_constraint_error(exc)
        except Exception:
            self._conn.rollback()
            raise

        if row is None:
            raise ProjectPersistenceError(
                f"project create persisted but row reload failed for id={project_id}"
            )
        return self._row_to_record(row)

    def get_by_slug(self, slug: str) -> ProjectRecord | None:
        row = self._conn.execute(
            """
            SELECT id, name, slug, palace, default_wing, default_room, fs_root,
                   fs_allowlist_json, system2_enabled, created_at, updated_at
            FROM projects
            WHERE slug = ?
            """,
            (slug,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_or_create(self, data: ProjectCreate) -> ProjectRecord:
        """Return the existing project matching slug, or create it idempotently."""
        existing = self.get_by_slug(data.slug)
        if existing is not None:
            return existing
        return self.create(data)

    def get_by_id(self, project_id: str) -> ProjectRecord | None:
        row = self._conn.execute(
            """
            SELECT id, name, slug, palace, default_wing, default_room, fs_root,
                   fs_allowlist_json, system2_enabled, created_at, updated_at
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_all(self) -> list[ProjectRecord]:
        rows = self._conn.execute(
            """
            SELECT id, name, slug, palace, default_wing, default_room, fs_root,
                   fs_allowlist_json, system2_enabled, created_at, updated_at
            FROM projects
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update(self, project_id: str, data: ProjectUpdate) -> ProjectRecord:
        current = self.get_by_id(project_id)
        if current is None:
            raise ProjectNotFoundError(f"project not found: {project_id}")
        if data.palace is not None and data.palace != current.palace:
            palace_collision = self._conn.execute(
                "SELECT 1 FROM projects WHERE palace = ? AND id != ?",
                (data.palace, project_id),
            ).fetchone()
            if palace_collision is not None:
                raise DuplicateProjectPalaceError("project palace already exists")
            raise ProjectPalaceImmutableError("palace is immutable after project creation")

        name = data.name if data.name is not None else current.name
        slug = data.slug if data.slug is not None else current.slug
        default_wing = data.default_wing if data.default_wing is not None else current.default_wing
        default_room = data.default_room if data.default_room is not None else current.default_room
        fs_root = _normalize_fs_path(data.fs_root) if data.fs_root is not None else current.fs_root
        fs_allowlist_json = (
            _encode_allowlist(data.fs_allowlist)
            if data.fs_allowlist is not None
            else json.dumps(current.fs_allowlist)
        )
        system2_enabled = (
            data.system2_enabled
            if data.system2_enabled is not None
            else current.system2_enabled
        )

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                UPDATE projects
                SET name = ?,
                    slug = ?,
                    default_wing = ?,
                    default_room = ?,
                    fs_root = ?,
                    fs_allowlist_json = ?,
                    system2_enabled = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    name,
                    slug,
                    default_wing,
                    default_room,
                    fs_root,
                    fs_allowlist_json,
                    1 if system2_enabled else 0,
                    project_id,
                ),
            )
            row = self._conn.execute(
                """
                SELECT id, name, slug, palace, default_wing, default_room, fs_root,
                       fs_allowlist_json, system2_enabled, created_at, updated_at
                FROM projects
                WHERE id = ?
                """,
                (project_id,),
            ).fetchone()
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            self._raise_constraint_error(exc)
        except Exception:
            self._conn.rollback()
            raise

        if row is None:
            raise ProjectPersistenceError(
                f"project update persisted but row reload failed for id={project_id}"
            )
        return self._row_to_record(row)

    def delete(self, project_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row | tuple[object, ...]) -> ProjectRecord:
        return ProjectRecord(
            id=str(row[0]),
            name=str(row[1]),
            slug=str(row[2]),
            palace=str(row[3]),
            default_wing=_optional_text(row[4]),
            default_room=_optional_text(row[5]),
            fs_root=str(row[6]),
            fs_allowlist=_decode_allowlist(str(row[7]) if row[7] is not None else None),
            system2_enabled=_db_bool(row[8]),
            created_at=str(row[9]),
            updated_at=str(row[10]),
        )

    @staticmethod
    def _raise_constraint_error(exc: sqlite3.IntegrityError) -> None:
        message = str(exc)
        if "projects.slug" in message:
            raise DuplicateProjectSlugError("project slug already exists") from exc
        if "projects.palace" in message:
            raise DuplicateProjectPalaceError("project palace already exists") from exc
        raise
