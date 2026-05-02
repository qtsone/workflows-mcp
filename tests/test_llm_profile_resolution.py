"""Unit tests for LLM profile resolution and fallback logic."""

import sqlite3
from pathlib import Path

import pytest

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.execution_context import ExecutionContext
from workflows_mcp.engine.executors_image import ImageGenExecutor, ImageGenInput
from workflows_mcp.engine.executors_llm import LLMCallExecutor, LLMCallInput
from workflows_mcp.engine.llm_config import (
    LLMConfig,
    LLMConfigLoader,
    ProfileConfig,
    ProviderConfig,
)
from workflows_mcp.http.lifespan import build_resources
from workflows_mcp.metadata.db import connect_metadata_db
from workflows_mcp.metadata.migrations import migrate_metadata_db
from workflows_mcp.metadata.repos import SQLiteLLMConfigRepository


class TestProfileResolution:
    """Test profile resolution without making API calls."""

    def test_profile_exists_uses_it(self) -> None:
        """Scenario 1: Existing profile is used directly."""
        config = LLMConfig(
            profiles={"standard": ProfileConfig(provider="openai-cloud", model="gpt-4o")},
            providers={"openai-cloud": ProviderConfig(type="openai")},
            default_profile="standard",
        )
        loader = LLMConfigLoader()
        loader._config = config

        executor = LLMCallExecutor()
        inputs = LLMCallInput(profile="standard", prompt="test")

        resolved = executor._resolve_profile_with_fallback(inputs, loader)
        assert resolved == "standard"

    def test_profile_missing_falls_back_to_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Scenario 2: Fallback to default_profile when requested profile missing."""
        config = LLMConfig(
            profiles={"standard": ProfileConfig(provider="openai-cloud", model="gpt-4o")},
            providers={"openai-cloud": ProviderConfig(type="openai")},
            default_profile="standard",
        )
        loader = LLMConfigLoader()
        loader._config = config

        executor = LLMCallExecutor()
        inputs = LLMCallInput(profile="cloud.small", prompt="test")

        resolved = executor._resolve_profile_with_fallback(inputs, loader)
        assert resolved == "standard"
        assert "Profile 'cloud.small' not found" in caplog.text
        assert "Falling back to default_profile 'standard'" in caplog.text

    def test_profile_missing_no_default_errors(self) -> None:
        """Scenario 3: Error when profile missing and no default_profile."""
        config = LLMConfig(
            profiles={"standard": ProfileConfig(provider="openai-cloud", model="gpt-4o")},
            providers={"openai-cloud": ProviderConfig(type="openai")},
            default_profile=None,
        )
        loader = LLMConfigLoader()
        loader._config = config

        executor = LLMCallExecutor()
        inputs = LLMCallInput(profile="cloud.small", prompt="test")

        with pytest.raises(
            ValueError, match="Profile 'cloud.small' not found and no default_profile set"
        ):
            executor._resolve_profile_with_fallback(inputs, loader)

    def test_direct_provider_bypasses_profiles(self) -> None:
        """Scenario 4: Direct provider/model bypasses profile system."""
        config = LLMConfig(profiles={}, providers={})
        loader = LLMConfigLoader()
        loader._config = config

        executor = LLMCallExecutor()
        inputs = LLMCallInput(provider="openai", model="gpt-4o", api_key="test", prompt="test")

        resolved = executor._resolve_profile_with_fallback(inputs, loader)
        assert resolved is None

    def test_no_profile_no_provider_errors(self) -> None:
        """Scenario 5: Error when neither profile nor provider specified."""
        config = LLMConfig(
            profiles={"standard": ProfileConfig(provider="openai-cloud", model="gpt-4o")},
            providers={"openai-cloud": ProviderConfig(type="openai")},
            default_profile="standard",
        )
        loader = LLMConfigLoader()
        loader._config = config

        executor = LLMCallExecutor()
        inputs = LLMCallInput(prompt="test")

        with pytest.raises(ValueError, match="LLM configuration required"):
            executor._resolve_profile_with_fallback(inputs, loader)


class TestProfileValidation:
    """Test Pydantic model validation."""

    def test_both_profile_and_provider_errors(self) -> None:
        """Scenario 6: Specifying both profile and provider raises error."""
        with pytest.raises(ValueError, match="Cannot specify both 'profile' and 'provider'"):
            LLMCallInput(profile="standard", provider="openai", model="gpt-4o", prompt="test")

    def test_provider_without_model_valid(self) -> None:
        """Scenario 7: Provider without model is valid (model is optional)."""
        inputs = LLMCallInput(provider="openai", prompt="test")
        assert inputs.provider == "openai"
        assert inputs.model is None
        assert inputs.profile is None

    def test_profile_only_valid(self) -> None:
        """Profile specified alone is valid."""
        inputs = LLMCallInput(profile="standard", prompt="test")
        assert inputs.profile == "standard"
        assert inputs.provider is None

    def test_provider_and_model_valid(self) -> None:
        """Provider + model specified is valid."""
        inputs = LLMCallInput(provider="openai", model="gpt-4o", api_key="test", prompt="test")
        assert inputs.provider == "openai"
        assert inputs.model == "gpt-4o"
        assert inputs.profile is None

    def test_neither_specified_valid(self) -> None:
        """Neither profile nor provider is valid (errors at execution)."""
        inputs = LLMCallInput(prompt="test")
        assert inputs.profile is None
        assert inputs.provider is None


class TestImageProfileResolution:
    """Test ImageGen profile model resolution without making API calls."""

    def _execution_with_loader(self, loader: LLMConfigLoader) -> Execution:
        context = ExecutionContext(
            workflow_registry=None,  # type: ignore[arg-type]
            executor_registry=None,  # type: ignore[arg-type]
            llm_config_loader=loader,
            io_queue=None,
        )
        execution = Execution()
        execution.set_execution_context(context)
        return execution

    @pytest.mark.asyncio
    async def test_profile_only_uses_profile_image_model(self) -> None:
        loader = LLMConfigLoader()
        loader._config = LLMConfig(
            providers={"openai-cloud": ProviderConfig(type="openai")},
            profiles={
                "image": ProfileConfig(provider="openai-cloud", model="gpt-image-1")
            },
            default_profile="image",
        )
        inputs = ImageGenInput(profile="image", prompt="paint a red square")

        resolved = await ImageGenExecutor()._resolve_profile_to_inputs(
            inputs, self._execution_with_loader(loader)
        )

        assert resolved.model == "gpt-image-1"

    def test_direct_image_generation_defaults_to_dall_e_3(self) -> None:
        inputs = ImageGenInput(provider="openai", prompt="paint a red square")

        assert inputs.model == "dall-e-3"

    @pytest.mark.asyncio
    async def test_explicit_model_with_profile_remains_inline_override(self) -> None:
        loader = LLMConfigLoader()
        loader._config = LLMConfig(
            providers={"openai-cloud": ProviderConfig(type="openai")},
            profiles={
                "image": ProfileConfig(provider="openai-cloud", model="gpt-image-1")
            },
            default_profile="image",
        )
        inputs = ImageGenInput(
            profile="image",
            model="dall-e-3",
            prompt="paint a red square",
        )

        resolved = await ImageGenExecutor()._resolve_profile_to_inputs(
            inputs, self._execution_with_loader(loader)
        )

        assert resolved.model == "dall-e-3"


class TestSQLiteBackedLoader:
    def _seed_sqlite_llm_config(
        self,
        db_path: Path,
        *,
        profile_name: str = "sqlite-profile",
    ) -> None:
        conn = connect_metadata_db(db_path)
        try:
            migrate_metadata_db(conn)
            repo = SQLiteLLMConfigRepository(conn)
            repo.replace_config(
                LLMConfig(
                    providers={
                        "openai-cloud": ProviderConfig(
                            type="openai",
                            api_url="https://api.openai.com/v1/chat/completions",
                        )
                    },
                    profiles={
                        profile_name: ProfileConfig(provider="openai-cloud", model="gpt-4o-mini")
                    },
                    default_profile=profile_name,
                )
            )
        finally:
            conn.close()

    def test_sqlite_backed_loader_resolves_profile_from_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "server.db"
        self._seed_sqlite_llm_config(db_path)

        loader = LLMConfigLoader(metadata_db_path=db_path)
        resolved = loader.resolve_profile("sqlite-profile")

        assert resolved is not None
        assert resolved.model == "gpt-4o-mini"
        assert resolved.provider == "openai"

    def test_sqlite_backed_loader_ignores_workflows_llm_config_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "server.db"
        self._seed_sqlite_llm_config(db_path, profile_name="db-only")

        yaml_path = tmp_path / "llm-config.yml"
        yaml_path.write_text(
            """
version: "1.0"
providers:
  from-yaml:
    type: openai
profiles:
  yaml-only:
    provider: from-yaml
    model: gpt-4o
default_profile: yaml-only
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setenv("WORKFLOWS_LLM_CONFIG", str(yaml_path))

        loader = LLMConfigLoader(metadata_db_path=db_path)
        resolved = loader.resolve_profile("db-only")

        assert resolved is not None
        assert resolved.model == "gpt-4o-mini"

    def test_loader_without_metadata_db_ignores_legacy_yaml_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        yaml_path = tmp_path / "llm-config.yml"
        yaml_path.write_text(
            """
version: "1.0"
providers:
  from-yaml:
    type: openai
profiles:
  yaml-only:
    provider: from-yaml
    model: gpt-4o
default_profile: yaml-only
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setenv("WORKFLOWS_LLM_CONFIG", str(yaml_path))

        loader = LLMConfigLoader()
        config = loader.load_config()

        assert config == LLMConfig()
        with pytest.raises(ValueError, match="No profiles configured"):
            loader.resolve_profile("yaml-only")

    def test_build_resources_wires_loader_to_base_dir_server_db(self, tmp_path: Path) -> None:
        base_dir = tmp_path / ".workflows"
        db_path = base_dir / "server.db"
        self._seed_sqlite_llm_config(db_path, profile_name="from-http")

        resources = build_resources(base_dir=base_dir)
        resolved = resources.llm_config_loader.resolve_profile("from-http")

        assert resolved is not None
        assert resolved.model == "gpt-4o-mini"

    def test_sqlite_backed_loader_reflects_db_updates_between_calls(self, tmp_path: Path) -> None:
        db_path = tmp_path / "server.db"
        self._seed_sqlite_llm_config(db_path, profile_name="initial")

        loader = LLMConfigLoader(metadata_db_path=db_path)
        first = loader.resolve_profile("initial")
        assert first is not None
        assert first.model == "gpt-4o-mini"

        conn = connect_metadata_db(db_path)
        try:
            repo = SQLiteLLMConfigRepository(conn)
            repo.replace_config(
                LLMConfig(
                    providers={"openai-cloud": ProviderConfig(type="openai")},
                    profiles={
                        "updated": ProfileConfig(
                            provider="openai-cloud",
                            model="gpt-4.1-mini",
                        )
                    },
                    default_profile="updated",
                )
            )
        finally:
            conn.close()

        second = loader.resolve_profile("updated")
        assert second is not None
        assert second.model == "gpt-4.1-mini"

    def test_sqlite_backed_loader_handles_existing_db_without_llm_schema(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "server.db"
        conn = sqlite3.connect(db_path)
        conn.close()

        loader = LLMConfigLoader(metadata_db_path=db_path)
        config = loader.load_config()

        assert config == LLMConfig()

        with pytest.raises(ValueError, match="No profiles configured"):
            loader.resolve_profile("whatever")
