from __future__ import annotations

from sqlite3 import Connection
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from workflows_mcp.engine.llm_config import LLMConfig
from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos import SQLiteLLMConfigRepository

router = APIRouter(prefix="/llm")


class RawYAMLRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"raw_yaml": "version: \"1.0\"\nproviders: {}\nprofiles: {}"}]
        }
    )

    raw_yaml: Annotated[str, StringConstraints(min_length=1, max_length=65536)]


class RawYAMLResponse(BaseModel):
    raw_yaml: str


def _repo(resources: AppResources) -> tuple[SQLiteLLMConfigRepository, Connection]:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    return SQLiteLLMConfigRepository(conn), conn


def _invalid_config_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="Invalid LLM configuration input.",
    )


@router.get(
    "/config",
    response_model=LLMConfig,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def get_llm_config(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> LLMConfig:
    repo, conn = _repo(resources)
    try:
        return repo.load_config()
    finally:
        conn.close()


@router.put(
    "/config",
    response_model=LLMConfig,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def put_llm_config(
    body: LLMConfig,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> LLMConfig:
    repo, conn = _repo(resources)
    try:
        try:
            repo.replace_config(body)
        except ValueError as error:
            raise _invalid_config_error() from error
        return repo.load_config()
    finally:
        conn.close()


@router.post(
    "/preview",
    response_model=LLMConfig,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def preview_llm_config(
    body: RawYAMLRequest,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> LLMConfig:
    repo, conn = _repo(resources)
    try:
        try:
            return repo.preview_yaml(body.raw_yaml)
        except (yaml.YAMLError, ValidationError, ValueError) as error:
            raise _invalid_config_error() from error
    finally:
        conn.close()


@router.post(
    "/import",
    response_model=LLMConfig,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def import_llm_config(
    body: RawYAMLRequest,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> LLMConfig:
    repo, conn = _repo(resources)
    try:
        try:
            return repo.import_yaml(body.raw_yaml)
        except (yaml.YAMLError, ValidationError, ValueError) as error:
            raise _invalid_config_error() from error
    finally:
        conn.close()


@router.get(
    "/export",
    response_model=RawYAMLResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def export_llm_config(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> RawYAMLResponse:
    repo, conn = _repo(resources)
    try:
        return RawYAMLResponse(raw_yaml=repo.export_yaml())
    finally:
        conn.close()
