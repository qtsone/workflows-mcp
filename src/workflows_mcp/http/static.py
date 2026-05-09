from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


logger = logging.getLogger(__name__)


class FrontendAssetsMissingError(RuntimeError):
    """Raised when frontend assets are required but missing."""


_RESERVED_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "/mcp",
        "/health",
        "/ready",
        "/docs",
        "/openapi.json",
    }
)


def is_reserved_path(path: str) -> bool:
    """Return ``True`` if *path* is reserved for API/ops/docs routes."""
    normalized = path if path.startswith("/") else f"/{path}"
    normalized = normalized.rstrip("/") or "/"
    return (
        normalized == "/api"
        or normalized.startswith("/api/")
        or normalized in _RESERVED_EXACT_PATHS
    )


def packaged_admin_static_dir() -> Path:
    """Return default packaged admin static directory path."""
    return Path(__file__).resolve().parents[1] / "static" / "admin"


def resolve_admin_static_dir(*, static_dir: Path | None = None) -> Path:
    """Resolve effective static directory with packaged-asset precedence."""
    if static_dir is not None:
        return static_dir

    return packaged_admin_static_dir()


def validate_admin_static_assets(static_dir: Path) -> Path:
    """Validate admin SPA assets and return resolved ``index.html`` path."""
    index_path = static_dir / "index.html"
    if not index_path.is_file():
        raise FrontendAssetsMissingError(
            "Frontend admin assets are required but index.html is missing at "
            f"{index_path}. Build and package frontend assets before starting "
            "production HTTP server."
        )
    return index_path


def install_admin_static(
    app: FastAPI,
    *,
    static_dir: Path | None = None,
    require_assets: bool = False,
) -> None:
    """Install guarded SPA fallback route when frontend assets are available."""
    resolved_static_dir = resolve_admin_static_dir(static_dir=static_dir)
    guidance = (
        "Frontend admin assets are missing; UI routes are disabled in default mode. "
        "Run `uv run workflows-mcp --build-ui` or `cd web && npm install && npm run build`."
    )

    if not resolved_static_dir.exists():
        if require_assets:
            validate_admin_static_assets(resolved_static_dir)
        else:
            logger.warning(guidance)
        return

    index_path = resolved_static_dir / "index.html"
    if not index_path.is_file():
        if require_assets:
            validate_admin_static_assets(resolved_static_dir)
        else:
            logger.warning(guidance)
        return

    index_path = validate_admin_static_assets(resolved_static_dir)

    assets_dir = resolved_static_dir / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir, html=False),
            name="admin-assets",
        )

    async def admin_spa_index() -> FileResponse:
        return FileResponse(index_path)

    spa_routes = [
        "/",
        "/login",
        "/setup",
        "/setup/{ui_path:path}",
        "/projects",
        "/projects/{ui_path:path}",
        "/database",
        "/database/{ui_path:path}",
        "/llm",
        "/llm/{ui_path:path}",
        "/secrets",
        "/secrets/{ui_path:path}",
        "/watchers",
        "/watchers/{ui_path:path}",
        "/sync",
        "/sync/{ui_path:path}",
        "/mcp-clients",
        "/mcp-clients/{ui_path:path}",
        "/workflows",
        "/workflows/{ui_path:path}",
        "/runs",
        "/runs/{ui_path:path}",
        "/admin",
        "/admin/{ui_path:path}",
    ]

    for route in spa_routes:
        app.add_api_route(route, admin_spa_index, methods=["GET"], include_in_schema=False)
