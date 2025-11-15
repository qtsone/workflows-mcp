"""Unit tests for LLM profile resolution and fallback logic."""

import pytest

from workflows_mcp.engine.executors_llm import LLMCallExecutor, LLMCallInput
from workflows_mcp.engine.llm_config import (
    LLMConfig,
    LLMConfigLoader,
    ProfileConfig,
    ProviderConfig,
)


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
