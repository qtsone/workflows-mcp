from __future__ import annotations

from fastapi import APIRouter

from .auth import router as auth_router
from .database import router as database_router
from .filesystem import router as filesystem_router
from .llm import router as llm_router
from .mcp_clients import router as mcp_clients_router
from .projects import router as projects_router
from .runs import router as runs_router
from .secrets import router as secrets_router
from .sync import router as sync_router
from .watchers import router as watchers_router
from .workflows import router as workflows_router

router = APIRouter(prefix="/api/admin/v1", tags=["admin-v1"])

router.include_router(auth_router)
router.include_router(secrets_router)
router.include_router(llm_router)
router.include_router(database_router)
router.include_router(filesystem_router)
router.include_router(projects_router)
router.include_router(runs_router)
router.include_router(mcp_clients_router)
router.include_router(watchers_router)
router.include_router(sync_router)
router.include_router(workflows_router)
