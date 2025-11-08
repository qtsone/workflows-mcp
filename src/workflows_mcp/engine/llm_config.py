"""LLM configuration management for profile-based provider/model selection.

This module implements the File-Based Profile Configuration system as described in
docs/WORKFLOWS-AS-AGENTS.md. It provides hierarchical LLM configuration through:

1. Providers: Infrastructure definitions (reusable provider configs)
2. Profiles: Named configurations (like AWS instance sizes) that reference providers
3. Hierarchical resolution with inline parameter overrides

Configuration file location priority:
1. Explicit path passed to LLMConfigLoader
2. WORKFLOWS_LLM_CONFIG environment variable
3. Standard location: ~/.workflows/llm-config.yml
4. Built-in defaults (if no config file found)

Example config file:
```yaml
version: "1.0"

providers:
  openai-cloud:
    type: openai
    api_url: "https://api.openai.com/v1/chat/completions"
    api_key_secret: "OPENAI_API_KEY"
    timeout: 120
    max_retries: 3

  anthropic-cloud:
    type: anthropic
    api_url: "https://api.anthropic.com/v1/messages"
    api_key_secret: "ANTHROPIC_API_KEY"
    timeout: 120

profiles:
  quick:
    provider: openai-cloud
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 2000

  standard:
    provider: openai-cloud
    model: gpt-4o
    temperature: 0.7
    max_tokens: 4000

  deep:
    provider: anthropic-cloud
    model: claude-sonnet-4
    temperature: 1.0
    max_tokens: 8000

default_profile: standard
```

Architecture:
- Load config once during app startup (singleton pattern)
- Validate schema using Pydantic models
- Resolve profiles with inline parameter overrides
- Backward compatible (works without config file)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ===========================================================================
# Configuration Models
# ===========================================================================


class ProviderConfig(BaseModel):
    """Provider infrastructure definition (reusable).

    Defines connection details and defaults for an LLM provider endpoint.
    """

    type: str = Field(description="Provider type (openai, anthropic, gemini, ollama, azure-openai)")
    api_url: str | None = Field(
        default=None,
        description="API endpoint URL (optional, uses provider defaults if not specified)",
    )
    api_key_secret: str | None = Field(
        default=None,
        description=(
            "Secret key name for API authentication. "
            "Maps to WORKFLOW_SECRET_{KEY_NAME} environment variable."
        ),
    )
    model: str | None = Field(
        default=None,
        description="Default model for this provider (optional, can be overridden in profile)",
    )
    timeout: int = Field(
        default=60,
        ge=1,
        le=1800,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed requests",
    )
    retry_delay: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
        description="Initial retry delay in seconds (exponential backoff)",
    )
    # Azure OpenAI specific fields
    deployment_name: str | None = Field(
        default=None,
        description="[Azure OpenAI] Deployment name",
    )
    api_version: str | None = Field(
        default=None,
        description="[Azure OpenAI] API version",
    )


class ProfileConfig(BaseModel):
    """Profile tier definition (like AWS EC2 instance sizes).

    Defines a named configuration that references a provider and specifies
    model parameters. Profiles enable right-sizing models for different tasks.
    """

    provider: str = Field(description="Provider name (must reference a key in providers section)")
    model: str = Field(
        description="Model name (e.g., gpt-4o, claude-sonnet-4, gemini-2.0-flash-exp)"
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=128000,
        description="Maximum tokens to generate",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of profile use case",
    )


class LLMConfig(BaseModel):
    """Root LLM configuration model.

    Validates the complete llm-config.yml structure with schema versioning.
    """

    version: str = Field(
        default="1.0",
        description="Configuration schema version",
    )
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="Provider infrastructure definitions (reusable)",
    )
    profiles: dict[str, ProfileConfig] = Field(
        default_factory=dict,
        description="Profile tier definitions (reference providers)",
    )
    default_profile: str | None = Field(
        default=None,
        description="Default profile name when none specified",
    )

    @field_validator("default_profile")
    @classmethod
    def validate_default_profile(cls, v: str | None, info: Any) -> str | None:
        """Validate that default_profile references an existing profile."""
        if v is not None:
            # Access profiles from the model data
            profiles = info.data.get("profiles", {})
            if v not in profiles:
                raise ValueError(
                    f"default_profile '{v}' not found in profiles. "
                    f"Available profiles: {', '.join(profiles.keys())}"
                )
        return v

    def validate_profile_provider_references(self) -> None:
        """Validate that all profiles reference existing providers.

        This is a post-initialization validation because it requires access to
        both profiles and providers after the model is fully constructed.

        Raises:
            ValueError: If any profile references a non-existent provider
        """
        for profile_name, profile_config in self.profiles.items():
            if profile_config.provider not in self.providers:
                available = ", ".join(self.providers.keys())
                raise ValueError(
                    f"Profile '{profile_name}' references unknown provider "
                    f"'{profile_config.provider}'. Available providers: {available}"
                )


# ===========================================================================
# Resolved Configuration (After Profile Resolution)
# ===========================================================================


class ResolvedLLMConfig(BaseModel):
    """Fully resolved LLM configuration after profile resolution.

    This model represents the final configuration that will be used by
    LLMCallExecutor, with all profile lookups resolved and inline overrides applied.
    """

    provider: str = Field(description="Provider type (openai, anthropic, gemini, ollama)")
    model: str = Field(description="Model name")
    api_url: str | None = Field(default=None, description="API endpoint URL")
    api_key_secret: str | None = Field(
        default=None,
        description="Secret key name (maps to WORKFLOW_SECRET_{KEY_NAME})",
    )
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=2.0, description="Initial retry delay in seconds")
    temperature: float | None = Field(default=None, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    system_instructions: str | None = Field(default=None, description="System instructions")
    # Azure OpenAI specific
    deployment_name: str | None = Field(default=None)
    api_version: str | None = Field(default=None)


# ===========================================================================
# Configuration Loader
# ===========================================================================


class LLMConfigLoader:
    """Loader for LLM configuration from YAML file.

    This class implements the File-Based Profile Configuration system with:
    - Hierarchical configuration file location
    - Schema validation using Pydantic
    - Profile resolution with inline parameter overrides
    - Backward compatibility (works without config file)

    Usage:
        ```python
        # Load config during app startup
        loader = LLMConfigLoader()
        config = loader.load_config()

        # Resolve profile for a workflow block
        resolved = loader.resolve_profile(
            profile="standard",
            inline_overrides={"temperature": 1.0, "max_tokens": 8000}
        )
        ```

    Thread Safety:
        This class is thread-safe for reading (load_config() caches result).
        Config is loaded once during initialization and reused.
    """

    def __init__(self, config_path: str | Path | None = None):
        """Initialize config loader with optional explicit path.

        Args:
            config_path: Explicit path to config file (optional).
                If not provided, uses environment variable or standard location.
        """
        self._config: LLMConfig | None = None
        self._explicit_path = Path(config_path) if config_path else None

    def get_config_path(self) -> Path | None:
        """Determine config file path using priority order.

        Priority:
        1. Explicit path passed to constructor
        2. WORKFLOWS_LLM_CONFIG environment variable
        3. Standard location: ~/.workflows/llm-config.yml

        Returns:
            Path to config file, or None if file doesn't exist
        """
        # Priority 1: Explicit path
        if self._explicit_path:
            if self._explicit_path.exists():
                return self._explicit_path
            logger.warning(f"Explicit LLM config path does not exist: {self._explicit_path}")
            return None

        # Priority 2: Environment variable
        env_path_str = os.getenv("WORKFLOWS_LLM_CONFIG")
        if env_path_str:
            env_path = Path(env_path_str).expanduser()
            if env_path.exists():
                return env_path
            logger.warning(f"WORKFLOWS_LLM_CONFIG path does not exist: {env_path}")
            return None

        # Priority 3: Standard location
        standard_path = Path.home() / ".workflows" / "llm-config.yml"
        if standard_path.exists():
            return standard_path

        return None

    def load_config(self) -> LLMConfig:
        """Load and validate LLM configuration from file.

        This method caches the loaded config for reuse. Call once during app startup.

        Returns:
            Validated LLMConfig instance (may be empty if no config file found)

        Raises:
            ValueError: If config file is invalid or fails validation
            yaml.YAMLError: If YAML parsing fails
        """
        if self._config is not None:
            return self._config

        config_path = self.get_config_path()

        if config_path is None:
            logger.info(
                "No LLM config file found. Using backward-compatible mode "
                "(workflows must specify provider/model directly)."
            )
            self._config = LLMConfig()
            return self._config

        logger.info(f"Loading LLM config from: {config_path}")

        try:
            # Load YAML file
            with open(config_path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            if not isinstance(raw_config, dict):
                raise ValueError("Config file must contain a YAML dictionary")

            # Validate with Pydantic
            config = LLMConfig(**raw_config)

            # Post-validation: check profile provider references
            config.validate_profile_provider_references()

            # Log summary
            logger.info(
                f"Loaded LLM config: {len(config.providers)} providers, "
                f"{len(config.profiles)} profiles"
            )
            if config.default_profile:
                logger.info(f"Default profile: {config.default_profile}")

            self._config = config
            return config

        except (yaml.YAMLError, ValueError) as e:
            raise ValueError(f"Failed to load LLM config from {config_path}: {e}")

    def resolve_profile(
        self,
        profile: str | None = None,
        inline_overrides: dict[str, Any] | None = None,
    ) -> ResolvedLLMConfig | None:
        """Resolve profile to fully configured LLM parameters.

        Resolution logic:
        1. If profile is None → return None (caller should use direct provider/model)
        2. Load profile from config
        3. Load provider referenced by profile
        4. Merge provider defaults + profile parameters
        5. Apply inline overrides (highest priority)

        Args:
            profile: Profile name to resolve (e.g., "quick", "standard", "deep")
            inline_overrides: Parameter overrides from workflow block (optional)

        Returns:
            ResolvedLLMConfig with all parameters, or None if profile not specified

        Raises:
            ValueError: If profile or provider not found in config
        """
        # Ensure config is loaded
        config = self.load_config()

        # If no profile specified, return None (backward compatibility)
        if profile is None:
            return None

        # Check if profile exists
        if profile not in config.profiles:
            if not config.profiles:
                # No profiles configured - provide setup guidance
                raise ValueError(
                    f"Profile '{profile}' not found. No profiles configured.\n"
                    f"Create ~/.workflows/llm-config.yml or use direct provider/model.\n"
                    f"See: https://github.com/qtsone/workflows-mcp#-llm-integration"
                )
            else:
                # Config exists but profile not found
                available = ", ".join(config.profiles.keys())
                raise ValueError(f"Profile '{profile}' not found. Available profiles: {available}")

        profile_config = config.profiles[profile]

        # Check if provider exists
        if profile_config.provider not in config.providers:
            available = ", ".join(config.providers.keys()) if config.providers else "none"
            raise ValueError(
                f"Provider '{profile_config.provider}' not found in LLM config. "
                f"Available providers: {available}"
            )

        provider_config = config.providers[profile_config.provider]

        # Build resolved config by merging provider + profile + inline overrides
        # Priority: inline_overrides > profile > provider
        inline = inline_overrides or {}

        return ResolvedLLMConfig(
            provider=inline.get("provider", provider_config.type),
            model=inline.get("model", profile_config.model),
            api_url=inline.get("api_url", provider_config.api_url),
            api_key_secret=inline.get("api_key_secret", provider_config.api_key_secret),
            timeout=inline.get("timeout", provider_config.timeout),
            max_retries=inline.get("max_retries", provider_config.max_retries),
            retry_delay=inline.get("retry_delay", provider_config.retry_delay),
            temperature=inline.get("temperature", profile_config.temperature),
            max_tokens=inline.get("max_tokens", profile_config.max_tokens),
            system_instructions=inline.get("system_instructions"),
            deployment_name=inline.get("deployment_name", provider_config.deployment_name),
            api_version=inline.get("api_version", provider_config.api_version),
        )

    def get_default_profile(self) -> str | None:
        """Get the default profile name from config.

        Returns:
            Default profile name, or None if not configured
        """
        config = self.load_config()
        return config.default_profile
