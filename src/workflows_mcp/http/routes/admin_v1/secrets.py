from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict, StringConstraints

from workflows_mcp.http.dependencies import (
    CurrentAdminSession,
    get_resources,
    require_admin_csrf,
    require_current_admin_session,
)
from workflows_mcp.http.lifespan import AppResources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.repos.secrets_repo import SecretMetadata, SQLiteSecretsRepository
from workflows_mcp.security.crypto import SecretCryptoError, SecretKeyError

router = APIRouter(prefix="/secrets")

SECRET_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
SecretName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=SECRET_NAME_PATTERN),
]
SecretValue = Annotated[str, StringConstraints(min_length=1, max_length=65536)]
KeyId = Annotated[str, StringConstraints(max_length=256)]


class SecretMetadataResponse(BaseModel):
    name: str
    key_id: str | None
    created_at: str
    updated_at: str


class SecretsListResponse(BaseModel):
    secrets: list[SecretMetadataResponse]


class UpsertSecretRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"name": "OPENAI_API_KEY", "key_id": "k1"}]}
    )

    name: SecretName
    value: SecretValue
    key_id: KeyId | None = None


class DeleteSecretResponse(BaseModel):
    deleted: bool


def _to_response(metadata: SecretMetadata) -> SecretMetadataResponse:
    return SecretMetadataResponse(
        name=metadata.name,
        key_id=metadata.key_id,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
    )


def _secret_store_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "Secret storage is unavailable. Check that secrets.key exists, "
            "is valid, and is readable."
        ),
    )


@router.get(
    "",
    response_model=SecretsListResponse,
    openapi_extra={"security": [{"AdminSessionCookie": []}]},
)
async def list_secrets(
    _current: CurrentAdminSession = Depends(require_current_admin_session),
    resources: AppResources = Depends(get_resources),
) -> SecretsListResponse:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    try:
        repo = SQLiteSecretsRepository(
            conn=conn,
            key_path=resources.metadata.base_dir / "secrets.key",
        )
        metadata = repo.list_metadata()
        return SecretsListResponse(secrets=[_to_response(item) for item in metadata])
    finally:
        conn.close()


@router.post(
    "",
    response_model=SecretMetadataResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def upsert_secret(
    body: UpsertSecretRequest,
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> SecretMetadataResponse:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    try:
        repo = SQLiteSecretsRepository(
            conn=conn,
            key_path=resources.metadata.base_dir / "secrets.key",
        )
        try:
            metadata = repo.upsert_secret(name=body.name, value=body.value, key_id=body.key_id)
        except (SecretKeyError, SecretCryptoError) as error:
            raise _secret_store_unavailable() from error
        return _to_response(metadata)
    finally:
        conn.close()


@router.delete(
    "/{name}",
    response_model=DeleteSecretResponse,
    openapi_extra={"security": [{"AdminSessionCookie": [], "CsrfToken": []}]},
)
async def delete_secret(
    name: Annotated[
        str,
        Path(min_length=1, max_length=128, pattern=SECRET_NAME_PATTERN),
    ],
    _current: CurrentAdminSession = Depends(require_admin_csrf),
    resources: AppResources = Depends(get_resources),
) -> DeleteSecretResponse:
    conn = connect_metadata_db(resources.metadata.base_dir / "server.db")
    try:
        repo = SQLiteSecretsRepository(
            conn=conn,
            key_path=resources.metadata.base_dir / "secrets.key",
        )
        try:
            deleted = repo.delete_secret(name)
        except (SecretKeyError, SecretCryptoError) as error:
            raise _secret_store_unavailable() from error
        return DeleteSecretResponse(deleted=deleted)
    finally:
        conn.close()
