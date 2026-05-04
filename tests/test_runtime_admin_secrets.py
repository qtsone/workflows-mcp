from __future__ import annotations

import secrets
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.executors_image import ImageGenExecutor, ImageGenInput
from workflows_mcp.engine.executors_llm import (
    LLMCallExecutor,
    LLMCallInput,
    compute_embedding,
    compute_embedding_batch,
)
from workflows_mcp.engine.llm_config import LLMConfig, ProfileConfig, ProviderConfig
from workflows_mcp.engine.schema import WorkflowSchema
from workflows_mcp.engine.workflow_runner import WorkflowRunner
from workflows_mcp.http.lifespan import AppResources, build_resources
from workflows_mcp.metadata.repos.llm_repo import SQLiteLLMConfigRepository
from workflows_mcp.metadata.repos.secrets_repo import SQLiteSecretsRepository


def _write_valid_key(path: Path) -> None:
    path.write_bytes(secrets.token_bytes(32))


def _build_resources_with_key(base_dir: Path) -> AppResources:
    base_dir.mkdir(parents=True, exist_ok=True)
    _write_valid_key(base_dir / "secrets.key")
    return build_resources(base_dir=base_dir)


def _runtime_secret_workflow(secret_name: str) -> WorkflowSchema:
    return WorkflowSchema(
        name="runtime-secret",
        description="Resolve a runtime secret from workflow outputs",
        blocks=[
            {
                "id": "noop",
                "type": "Shell",
                "inputs": {"command": "true"},
            }
        ],
        outputs={"token": {"value": f"{{{{secrets.{secret_name}}}}}"}},
    )


def _runtime_secret_block_input_workflow(secret_name: str) -> WorkflowSchema:
    return WorkflowSchema(
        name="runtime-secret-block-input",
        description="Resolve a runtime secret from block inputs",
        blocks=[
            {
                "id": "use_secret",
                "type": "Shell",
                "inputs": {"command": f"printf '%s' '{{{{secrets.{secret_name}}}}}'"},
            }
        ],
    )


def _execution_with_runtime_context(resources: AppResources) -> Execution:
    execution = Execution(inputs={}, metadata=None, blocks={})
    execution.set_execution_context(resources.app_context.create_execution_context())
    return execution


def _store_profile_and_secret(
    resources: AppResources,
    *,
    profile_name: str,
    model: str,
    secret_name: str,
    secret_value: str,
) -> None:
    SQLiteSecretsRepository(
        conn=resources.metadata_db_conn,
        key_path=resources.metadata.base_dir / "secrets.key",
    ).upsert_secret(secret_name, secret_value)
    SQLiteLLMConfigRepository(resources.metadata_db_conn).replace_config(
        LLMConfig(
            providers={
                "sqlite-provider": ProviderConfig(
                    type="openai",
                    api_url="http://localhost:65535/v1",
                    api_key_secret=secret_name,
                )
            },
            profiles={
                profile_name: ProfileConfig(
                    provider="sqlite-provider",
                    model=model,
                )
            },
            default_profile=profile_name,
        )
    )


def test_app_context_and_child_context_preserve_runtime_secret_provider(tmp_path: Path) -> None:
    resources = _build_resources_with_key(tmp_path / ".workflows")
    try:
        context = resources.app_context.create_execution_context()
        child = context.create_child_context(
            parent_execution=Execution(inputs={}, metadata=None, blocks={}),
            workflow_name="runtime-secret",
        )

        assert context.secret_provider is resources.app_context.secret_provider
        assert child.secret_provider is resources.app_context.secret_provider
    finally:
        resources.metadata_db_conn.close()


@pytest.mark.asyncio
async def test_llm_profile_api_key_secret_resolves_from_admin_sqlite_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_name = "RUNTIME_PROFILE_API_KEY"
    monkeypatch.delenv(f"WORKFLOW_SECRET_{secret_name}", raising=False)
    resources = _build_resources_with_key(tmp_path / ".workflows")
    try:
        _store_profile_and_secret(
            resources,
            profile_name="sqlite-llm",
            model="gpt-4o-mini",
            secret_name=secret_name,
            secret_value="sqlite-llm-secret",
        )

        resolved = await LLMCallExecutor()._resolve_profile_to_inputs(
            LLMCallInput(profile="sqlite-llm", prompt="hello"),
            _execution_with_runtime_context(resources),
        )

        assert resolved.api_key == "sqlite-llm-secret"
    finally:
        resources.metadata_db_conn.close()


@pytest.mark.asyncio
async def test_embedding_profile_api_key_secret_resolves_from_admin_sqlite_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_name = "RUNTIME_EMBEDDING_API_KEY"
    monkeypatch.delenv(f"WORKFLOW_SECRET_{secret_name}", raising=False)
    resources = _build_resources_with_key(tmp_path / ".workflows")
    try:
        _store_profile_and_secret(
            resources,
            profile_name="embedding",
            model="text-embedding-3-small",
            secret_name=secret_name,
            secret_value="sqlite-embedding-secret",
        )
        mock_response = Mock()
        mock_response.data = [Mock(index=0, embedding=[0.1, 0.2])]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = Mock(prompt_tokens=1, total_tokens=1)

        with patch("workflows_mcp.engine.executors_llm.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            embedding, model, dimensions, usage = await compute_embedding(
                "hello",
                _execution_with_runtime_context(resources),
            )

        assert embedding == [0.1, 0.2]
        assert model == "text-embedding-3-small"
        assert dimensions == 2
        assert usage == {"prompt_tokens": 1, "total_tokens": 1}
        assert mock_cls.call_args.kwargs["api_key"] == "sqlite-embedding-secret"
    finally:
        resources.metadata_db_conn.close()


@pytest.mark.asyncio
async def test_embedding_batch_profile_api_key_secret_resolves_from_admin_sqlite_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_name = "RUNTIME_BATCH_EMBEDDING_API_KEY"
    monkeypatch.delenv(f"WORKFLOW_SECRET_{secret_name}", raising=False)
    resources = _build_resources_with_key(tmp_path / ".workflows")
    try:
        _store_profile_and_secret(
            resources,
            profile_name="embedding",
            model="text-embedding-3-small",
            secret_name=secret_name,
            secret_value="sqlite-batch-embedding-secret",
        )
        mock_response = Mock()
        mock_response.data = [
            Mock(index=0, embedding=[0.1, 0.2]),
            Mock(index=1, embedding=[0.3, 0.4]),
        ]
        mock_response.model = "text-embedding-3-small"
        mock_response.usage = None

        with patch("workflows_mcp.engine.executors_llm.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            embeddings, _, dimensions, usage = await compute_embedding_batch(
                ["hello", "world"],
                _execution_with_runtime_context(resources),
            )

        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert dimensions == 2
        assert usage is None
        assert mock_cls.call_args.kwargs["api_key"] == "sqlite-batch-embedding-secret"
    finally:
        resources.metadata_db_conn.close()


@pytest.mark.asyncio
async def test_image_profile_api_key_secret_resolves_from_admin_sqlite_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_name = "RUNTIME_IMAGE_API_KEY"
    monkeypatch.delenv(f"WORKFLOW_SECRET_{secret_name}", raising=False)
    resources = _build_resources_with_key(tmp_path / ".workflows")
    try:
        _store_profile_and_secret(
            resources,
            profile_name="sqlite-image",
            model="dall-e-3",
            secret_name=secret_name,
            secret_value="sqlite-image-secret",
        )

        resolved = await ImageGenExecutor()._resolve_profile_to_inputs(
            ImageGenInput(profile="sqlite-image", prompt="draw a test image"),
            _execution_with_runtime_context(resources),
        )

        assert resolved.api_key == "sqlite-image-secret"
    finally:
        resources.metadata_db_conn.close()


@pytest.mark.asyncio
async def test_workflow_resolves_secret_saved_in_admin_sqlite_store(tmp_path: Path) -> None:
    resources = _build_resources_with_key(tmp_path / ".workflows")
    try:
        SQLiteSecretsRepository(
            conn=resources.metadata_db_conn,
            key_path=resources.metadata.base_dir / "secrets.key",
        ).upsert_secret("ADMIN_RUNTIME_TOKEN", "sqlite-secret-value")

        workflow = _runtime_secret_workflow("ADMIN_RUNTIME_TOKEN")
        context = resources.app_context.create_execution_context()

        result = await WorkflowRunner().execute(workflow=workflow, context=context)

        assert result.status == "success"
        assert result.execution.outputs["token"] == "sqlite-secret-value"
    finally:
        resources.metadata_db_conn.close()


@pytest.mark.parametrize(
    ("key_mutation", "expected_error"),
    [
        pytest.param("missing", "Secret key file is missing", id="missing-key"),
        pytest.param(
            "wrong-key",
            "Encrypted secret payload failed authentication",
            id="undecryptable-payload",
        ),
    ],
)
@pytest.mark.asyncio
async def test_workflow_output_secret_key_failures_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key_mutation: str,
    expected_error: str,
) -> None:
    secret_name = "ADMIN_RUNTIME_TOKEN"
    secret_value = "sqlite-secret-value"
    monkeypatch.delenv(f"WORKFLOW_SECRET_{secret_name}", raising=False)
    resources = _build_resources_with_key(tmp_path / ".workflows")
    try:
        key_path = resources.metadata.base_dir / "secrets.key"
        SQLiteSecretsRepository(
            conn=resources.metadata_db_conn,
            key_path=key_path,
        ).upsert_secret(secret_name, secret_value)
        if key_mutation == "missing":
            key_path.unlink()
        else:
            key_path.write_bytes(b"x" * 32)

        result = await WorkflowRunner().execute(
            workflow=_runtime_secret_workflow(secret_name),
            context=resources.app_context.create_execution_context(),
        )

        assert result.status == "failure"
        assert result.error is not None
        assert expected_error in result.error
        assert secret_value not in result.error
        assert secret_value not in str(result.execution.outputs)
    finally:
        resources.metadata_db_conn.close()


@pytest.mark.parametrize(
    ("key_mutation", "expected_error"),
    [
        pytest.param("missing", "Secret key file is missing", id="missing-key"),
        pytest.param(
            "wrong-key",
            "Encrypted secret payload failed authentication",
            id="undecryptable-payload",
        ),
    ],
)
@pytest.mark.asyncio
async def test_block_input_secret_key_failures_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key_mutation: str,
    expected_error: str,
) -> None:
    secret_name = "ADMIN_RUNTIME_TOKEN"
    secret_value = "sqlite-secret-value"
    monkeypatch.delenv(f"WORKFLOW_SECRET_{secret_name}", raising=False)
    resources = _build_resources_with_key(tmp_path / ".workflows")
    try:
        key_path = resources.metadata.base_dir / "secrets.key"
        SQLiteSecretsRepository(
            conn=resources.metadata_db_conn,
            key_path=key_path,
        ).upsert_secret(secret_name, secret_value)
        if key_mutation == "missing":
            key_path.unlink()
        else:
            key_path.write_bytes(b"x" * 32)

        result = await WorkflowRunner().execute(
            workflow=_runtime_secret_block_input_workflow(secret_name),
            context=resources.app_context.create_execution_context(),
        )

        assert result.status == "failure"
        assert result.error is not None
        assert expected_error in result.error
        assert secret_value not in result.error
        assert secret_value not in str(result.execution.blocks)
    finally:
        resources.metadata_db_conn.close()


@pytest.mark.asyncio
async def test_env_secret_takes_precedence_over_admin_sqlite_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORKFLOW_SECRET_ADMIN_RUNTIME_TOKEN", "env-secret-value")
    resources = _build_resources_with_key(tmp_path / ".workflows")
    try:
        SQLiteSecretsRepository(
            conn=resources.metadata_db_conn,
            key_path=resources.metadata.base_dir / "secrets.key",
        ).upsert_secret("ADMIN_RUNTIME_TOKEN", "sqlite-secret-value")

        workflow = _runtime_secret_workflow("ADMIN_RUNTIME_TOKEN")
        context = resources.app_context.create_execution_context()

        result = await WorkflowRunner().execute(workflow=workflow, context=context)
        secret_keys = await context.secret_provider.list_secret_keys()

        assert result.status == "success"
        assert result.execution.outputs["token"] == "env-secret-value"
        assert "admin_runtime_token" in secret_keys
        assert "env-secret-value" not in secret_keys
        assert "sqlite-secret-value" not in secret_keys
    finally:
        resources.metadata_db_conn.close()
