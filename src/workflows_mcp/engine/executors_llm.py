"""LLM call executor with retry logic and schema validation.

Supports multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama) with automatic
retry, JSON schema validation, and token usage tracking. All providers use client-side
validation with feedback loop for maximum compatibility with OpenAI-like servers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from typing import Any, ClassVar, cast

import httpx
import jsonschema
import openai
from openai import AsyncOpenAI
from pydantic import Field, field_validator, model_validator

from .block import BlockInput, BlockOutput
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)
from .interpolation import (
    interpolatable_enum_validator,
    interpolatable_numeric_validator,
    resolve_interpolatable_enum,
    resolve_interpolatable_numeric,
)
from .llm_config import LLMConfigLoader

logger = logging.getLogger(__name__)

# ===========================================================================
# Type Definitions
# ===========================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


# ===========================================================================
# LLMCall Executor
# ===========================================================================


class LLMCallInput(BlockInput):
    """Input schema for LLMCall block.

    Supports two configuration modes:
    - Profile-based: Load settings from ~/.workflows/llm-config.yml
    - Direct: Specify provider and model inline

    Variables are resolved by VariableResolver before execution. Schema validation
    uses OpenAI native structured outputs (strict mode) or client-side validation
    with retry feedback for other providers.
    """

    profile: str | None = Field(
        default=None,
        description=(
            "Profile name from ~/.workflows/llm-config.yml (e.g., 'cloud', 'local', 'default'). "
            "If specified, provider/model are loaded from config. "
            "Mutually exclusive with direct provider/model specification."
        ),
    )

    provider: LLMProvider | str | None = Field(
        default=None,
        description=(
            "LLM provider (enum or interpolation string). "
            "Required if profile not specified. Ignored if profile specified."
        ),
    )

    # Validator allows enum values, valid strings, and interpolation strings
    _validate_provider = field_validator("provider", mode="before")(
        interpolatable_enum_validator(LLMProvider)
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022, gemini-2.0-flash-exp). "
            "Required if profile not specified. Can override profile model if both specified."
        ),
    )
    prompt: str = Field(
        description="User prompt to send to the LLM",
    )
    system_instructions: str | None = Field(
        default=None,
        description="System instructions (optional)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (pre-resolved from {{secrets.PROVIDER_API_KEY}})",
    )
    api_url: str | None = Field(
        default=None,
        description="Custom API endpoint URL (optional, for custom deployments)",
    )
    response_schema: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "JSON Schema for expected response structure "
            "(dict or JSON string, enables validation and retry)"
        ),
    )
    max_retries: int | str = Field(
        default=3,
        description="Maximum number of retry attempts (or interpolation string)",
    )
    retry_delay: float | str = Field(
        default=2.0,
        description="Initial retry delay in seconds (exponential backoff, or interpolation string)",
    )
    timeout: int | str = Field(
        default=60,
        description="Request timeout in seconds (or interpolation string)",
    )
    temperature: float | str | None = Field(
        default=None,
        description="Sampling temperature 0.0-2.0 (or interpolation string)",
    )
    max_tokens: int | str | None = Field(
        default=None,
        description="Maximum tokens to generate (or interpolation string)",
    )

    # Validators for numeric fields with interpolation support
    _validate_max_retries = field_validator("max_retries", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=10)
    )
    _validate_retry_delay = field_validator("retry_delay", mode="before")(
        interpolatable_numeric_validator(float, ge=0.1, le=60.0)
    )
    _validate_timeout = field_validator("timeout", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=1800)
    )
    _validate_temperature = field_validator("temperature", mode="before")(
        interpolatable_numeric_validator(float, ge=0.0, le=2.0)
    )
    _validate_max_tokens = field_validator("max_tokens", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=128000)
    )
    validation_prompt_template: str = Field(
        default=(
            "Your previous response failed JSON schema validation.\n\n"
            "Error: {validation_error}\n\n"
            "Expected schema:\n{schema}\n\n"
            "Please provide a valid response that conforms to the schema."
        ),
        description="Template for validation retry prompt",
    )

    @field_validator("response_schema")
    @classmethod
    def validate_response_schema(cls, v: dict[str, Any] | str | None) -> dict[str, Any] | None:
        """Parse and validate response schema.

        Accepts dict (JSON Schema), str (JSON string), or None (no validation).

        Raises:
            ValueError: Invalid JSON or missing 'type' field
        """
        if v is None:
            return None

        # Parse string to dict if needed
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"response_schema string is not valid JSON: {e}")

        if not isinstance(v, dict):
            raise ValueError(
                f"response_schema must be a dict or JSON string, got {type(v).__name__}"
            )

        if "type" not in v:
            raise ValueError("response_schema must have a 'type' field")

        return v

    @model_validator(mode="after")
    def validate_profile_or_provider_model(self) -> LLMCallInput:
        """Validate LLM configuration - profile and provider are mutually exclusive.

        Valid configurations:
        1. profile specified (resolved from config, may fallback to default_profile)
        2. provider specified (model is optional, can be empty string)
        3. neither specified (will error at execution with better context)

        Invalid:
        - Both profile and provider specified (ambiguous)

        Raises:
            ValueError: Invalid configuration
        """
        # Error: Both profile and provider specified
        if self.profile is not None and self.provider is not None:
            raise ValueError(
                "Cannot specify both 'profile' and 'provider'. Choose one:\n"
                "  - Use 'profile' for config-based setup, OR\n"
                "  - Use 'provider' (+ optional 'model') for direct specification"
            )

        # OK: Profile specified (validated at execution)
        # OK: Provider specified (model is optional)
        # OK: Neither specified (error at execution with better context)
        return self


class LLMCallOutput(BlockOutput):
    """Output schema for LLMCall block.

    Unified response structure for clean workflow integration:

    - No schema: response = {"content": "raw text"}
    - Schema validation succeeds: response = {validated JSON structure}
    - Schema validation fails: response = {"content": "raw text"}
      + metadata.validation_failed = True
    - Request fails: response = {} + success = False

    Fields:
    - response: Always a dict (never None) containing either validated JSON
      or raw text
    - success: Whether the LLM API call succeeded
    - metadata: Execution details (attempts, validation_failed,
      validation_error, model, usage, etc.)
    """

    response: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Response dictionary. Contains validated JSON structure if schema "
            "provided and validation succeeded. "
            "Contains {'content': 'raw text'} if no schema or validation failed. "
            "Empty dict {} if request failed completely."
        ),
    )
    success: bool = Field(
        default=False,
        description=(
            "True if LLM API call succeeded (response received from provider). "
            "False if request failed (network error, timeout, API error, etc.). "
            "Independent of schema validation - "
            "check metadata.validation_failed for validation status."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Execution metadata including: "
            "attempts (int), "
            "validation_failed (bool, if schema validation failed), "
            "validation_error (str, error message if validation_failed=true), "
            "model (str), usage (dict), finish_reason (str), etc. "
            "Empty dict if request failed before execution."
        ),
    )


class LLMCallExecutor(BlockExecutor):
    """Executor for LLMCall blocks.

    Supports multiple LLM providers with automatic retry, exponential backoff, and
    client-side schema validation for maximum compatibility. All providers validate
    responses client-side (including OpenAI-compatible servers like LM Studio).
    Token usage tracking and null safety checks included for all providers.

    Example:
        ```yaml
        - id: extract_data
          type: LLMCall
          inputs:
            provider: openai
            model: gpt-4o
            api_key: "{{secrets.OPENAI_API_KEY}}"
            prompt: "Extract: {{inputs.text}}"
            response_schema:
              type: object
              required: [entities]
              properties:
                entities: {type: array, items: {type: string}}
        ```

    Outputs:
        - response: Always a dict containing either:
            * Validated JSON structure if schema provided and validation succeeded
            * {"content": "raw text"} if no schema or validation failed
        - success: True if API call succeeded (independent of validation)
        - metadata: Contains attempts, validation_failed (if applicable),
                   validation_error (if failed), model, usage, finish_reason
    """

    type_name: ClassVar[str] = "LLMCall"
    input_type: ClassVar[type[BlockInput]] = LLMCallInput
    output_type: ClassVar[type[BlockOutput]] = LLMCallOutput
    examples: ClassVar[str] = """```yaml
- id: summarize
  type: LLMCall
  inputs:
    profile: default
    prompt: "Summarize this text: {{inputs.text}}"
```"""

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_network=True)

    async def execute(  # type: ignore[override]
        self, inputs: LLMCallInput, context: Execution
    ) -> LLMCallOutput:
        """Execute LLM call with retry logic and profile fallback.

        Resolves profile configuration with fallback to default_profile, calls provider
        API with exponential backoff retry, and validates responses client-side for
        maximum compatibility.

        Raises:
            ValueError: Invalid configuration
            httpx.*: Network errors after retry exhaustion
        """
        # Get execution context for config access
        execution_context = context.execution_context
        if execution_context is None:
            raise ValueError("ExecutionContext not available. Cannot resolve LLM configuration.")

        llm_config_loader = execution_context.llm_config_loader

        # Step 1: Resolve profile (with fallback logic)
        effective_profile = self._resolve_profile_with_fallback(inputs, llm_config_loader)

        # Step 2: Profile resolution (if profile determined)
        effective_inputs = inputs
        profile_fallback_occurred = False

        if effective_profile is not None:
            # Track if fallback occurred (profile requested but different one used)
            profile_fallback_occurred = (
                inputs.profile is not None and inputs.profile != effective_profile
            )

            # Create new input with resolved profile
            inputs_with_profile = inputs.model_copy(update={"profile": effective_profile})
            effective_inputs = await self._resolve_profile_to_inputs(inputs_with_profile, context)

        # Step 3: Resolve interpolatable numeric fields to their actual types
        max_retries = resolve_interpolatable_numeric(
            effective_inputs.max_retries, int, "max_retries", ge=1, le=10
        )
        retry_delay = resolve_interpolatable_numeric(
            effective_inputs.retry_delay, float, "retry_delay", ge=0.1, le=60.0
        )
        timeout = resolve_interpolatable_numeric(
            effective_inputs.timeout, int, "timeout", ge=1, le=1800
        )
        temperature = (
            resolve_interpolatable_numeric(
                effective_inputs.temperature, float, "temperature", ge=0.0, le=2.0
            )
            if effective_inputs.temperature is not None
            else None
        )
        max_tokens = (
            resolve_interpolatable_numeric(
                effective_inputs.max_tokens, int, "max_tokens", ge=1, le=128000
            )
            if effective_inputs.max_tokens is not None
            else None
        )

        attempts = 0
        last_error: Exception | None = None
        validation_error: str | None = None

        # Determine if we need schema validation
        needs_validation = effective_inputs.response_schema is not None

        # Prepare schema ONCE before retry loop - this is the schema LLM receives
        # CRITICAL: Use the SAME prepared schema for validation to avoid mismatch
        # where LLM generates null (allowed by prepared schema) but validation
        # uses original schema (which doesn't allow null)
        prepared_schema: dict[str, Any] | None = None
        if needs_validation:
            prepared_schema = self._prepare_schema_for_openai(effective_inputs.response_schema)

        for attempt in range(max_retries):
            attempts += 1

            try:
                # Build prompt (append validation feedback if retry due to validation failure)
                prompt = effective_inputs.prompt
                if validation_error and attempt > 0:
                    # APPEND validation feedback to original prompt instead of replacing
                    # This preserves context so LLM knows what task it's performing
                    prompt = (
                        effective_inputs.prompt
                        + "\n\n"
                        + effective_inputs.validation_prompt_template.format(
                            validation_error=validation_error,
                            schema=json.dumps(prepared_schema, indent=2),
                        )
                    )

                # Make LLM API call
                response_text, provider_metadata = await self._call_provider(
                    inputs=effective_inputs,
                    prompt=prompt,
                    timeout=timeout,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=context,
                    prepared_schema=prepared_schema,
                )

                # Validate response if schema provided (client-side for all providers)
                if needs_validation:
                    try:
                        # Use the SAME prepared schema that was sent to the LLM
                        validated_response = self._validate_response(
                            response_text=response_text,
                            schema=cast(dict[str, Any], prepared_schema),
                        )

                        # Success - return validated JSON structure directly
                        metadata = {
                            "attempts": attempts,
                            **provider_metadata,
                        }
                        # Add fallback info if applicable
                        if profile_fallback_occurred:
                            metadata["profile_fallback"] = {
                                "requested": inputs.profile,
                                "resolved": effective_profile,
                            }

                        return LLMCallOutput(
                            response=validated_response,
                            success=True,
                            metadata=metadata,
                        )
                    except ValueError as e:
                        # Validation failed
                        validation_error = str(e)
                        last_error = e

                        # If this is the last attempt, return with validation failure
                        if attempt == max_retries - 1:
                            metadata = {
                                "attempts": attempts,
                                "validation_failed": True,
                                "validation_error": validation_error,
                                **provider_metadata,
                            }
                            # Add fallback info if applicable
                            if profile_fallback_occurred:
                                metadata["profile_fallback"] = {
                                    "requested": inputs.profile,
                                    "resolved": effective_profile,
                                }

                            return LLMCallOutput(
                                response={"content": response_text},
                                success=True,  # API call succeeded, validation failed
                                metadata=metadata,
                            )

                        # Otherwise, wait and retry with feedback
                        delay = retry_delay * (2**attempt)
                        await asyncio.sleep(delay)
                        continue
                else:
                    # No schema provided - return raw text in content key
                    metadata = {
                        "attempts": attempts,
                        **provider_metadata,
                    }
                    # Add fallback info if applicable
                    if profile_fallback_occurred:
                        metadata["profile_fallback"] = {
                            "requested": inputs.profile,
                            "resolved": effective_profile,
                        }

                    return LLMCallOutput(
                        response={"content": response_text},
                        success=True,
                        metadata=metadata,
                    )

            except (
                httpx.TimeoutException,
                httpx.HTTPStatusError,
                httpx.NetworkError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.APIStatusError,
            ) as e:
                last_error = e

                # If this is the last attempt, raise
                if attempt == max_retries - 1:
                    raise

                # Otherwise, wait and retry with exponential backoff
                delay = retry_delay * (2**attempt)
                await asyncio.sleep(delay)
                continue

            except Exception:
                # Unrecoverable error, raise immediately
                raise

        # Should never reach here, but handle gracefully
        if last_error:
            raise last_error
        raise RuntimeError("LLM call failed after all retry attempts")

    async def _resolve_profile_to_inputs(
        self, inputs: LLMCallInput, context: Execution
    ) -> LLMCallInput:
        """Resolve profile configuration from ~/.workflows/llm-config.yml.

        Loads profile, merges with inline overrides, resolves API key from secrets,
        and returns merged LLMCallInput.

        Raises:
            ValueError: Profile not found or ExecutionContext unavailable
        """
        # Get llm_config_loader from execution context
        execution_context = context.execution_context
        if execution_context is None:
            raise ValueError(
                "ExecutionContext not available. Cannot resolve LLM profile. "
                "This is a system error - execution context should always be set."
            )

        llm_config_loader = execution_context.llm_config_loader

        # Build inline overrides from inputs (filter out None values)
        inline_overrides = {
            key: value
            for key, value in {
                "provider": inputs.provider,
                "model": inputs.model,
                "api_url": inputs.api_url,
                "api_key": inputs.api_key,
                "timeout": inputs.timeout,
                "max_retries": inputs.max_retries,
                "retry_delay": inputs.retry_delay,
                "temperature": inputs.temperature,
                "max_tokens": inputs.max_tokens,
                "system_instructions": inputs.system_instructions,
            }.items()
            if value is not None
        }

        # Resolve profile
        resolved_config = llm_config_loader.resolve_profile(
            profile=inputs.profile, inline_overrides=inline_overrides
        )

        if resolved_config is None:
            # This shouldn't happen because model_validator ensures profile is not None when called
            raise ValueError("Profile resolution returned None unexpectedly")

        # Resolve API key from secrets if api_key_secret is specified
        api_key = None
        if resolved_config.api_key_secret:
            # Get secret provider (use default EnvVarSecretProvider)
            from .secrets import EnvVarSecretProvider

            secret_provider = EnvVarSecretProvider()
            api_key = await secret_provider.get_secret(resolved_config.api_key_secret)

        # Create new LLMCallInput with resolved values
        # Preserve original prompt and validation settings
        return LLMCallInput(
            profile=None,  # Clear profile (already resolved)
            provider=resolved_config.provider,
            model=resolved_config.model,
            prompt=inputs.prompt,
            system_instructions=resolved_config.system_instructions or inputs.system_instructions,
            api_key=api_key,
            api_url=resolved_config.api_url,
            response_schema=inputs.response_schema,
            max_retries=resolved_config.max_retries,
            retry_delay=resolved_config.retry_delay,
            timeout=resolved_config.timeout,
            temperature=resolved_config.temperature,
            max_tokens=resolved_config.max_tokens,
            validation_prompt_template=inputs.validation_prompt_template,
        )

    def _resolve_profile_with_fallback(
        self,
        inputs: LLMCallInput,
        llm_config_loader: LLMConfigLoader,
    ) -> str | None:
        """Resolve profile with fallback to default_profile.

        Resolution logic:
        1. Direct provider/model specified → None (bypass profiles)
        2. Profile exists in config → use it
        3. Profile missing + default_profile exists → WARN and use default_profile
        4. Profile missing + no default_profile → ERROR
        5. No profile and no provider → ERROR (explicit required)

        Args:
            inputs: LLMCall inputs with profile/provider configuration
            llm_config_loader: Config loader with profile definitions

        Returns:
            Profile name to use, or None for direct provider config

        Raises:
            ValueError: Missing or invalid configuration
        """
        config = llm_config_loader.load_config()

        # Case 1: Direct provider/model bypasses profiles
        if inputs.provider is not None:
            return None

        # Case 2: Profile specified
        if inputs.profile is not None:
            # Profile exists - use it
            if inputs.profile in config.profiles:
                return inputs.profile

            # Profile missing - try fallback to default_profile
            if config.default_profile is not None:
                logger.warning(
                    f"Profile '{inputs.profile}' not found in config. "
                    f"Falling back to default_profile '{config.default_profile}'. "
                    f"Available profiles: {', '.join(config.profiles.keys())}"
                )
                return config.default_profile

            # No fallback available - error
            available = ", ".join(config.profiles.keys()) if config.profiles else "none"
            raise ValueError(
                f"Profile '{inputs.profile}' not found and no default_profile set.\n"
                f"Available profiles: {available}\n"
                f"Either:\n"
                f"  1. Add '{inputs.profile}' profile to ~/.workflows/llm-config.yml, OR\n"
                f"  2. Set 'default_profile' in ~/.workflows/llm-config.yml"
            )

        # Case 3: Neither profile nor provider - error (explicit required)
        raise ValueError(
            "LLM configuration required. Either:\n"
            "  1. Specify 'profile' in LLMCall block, OR\n"
            "  2. Specify 'provider' and 'model' directly"
        )

    async def _call_provider(
        self,
        inputs: LLMCallInput,
        prompt: str,
        timeout: int,
        temperature: float | None,
        max_tokens: int | None,
        context: Execution,
        prepared_schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Call LLM provider API.

        Args:
            prepared_schema: Pre-prepared schema for OpenAI-compatible providers.
                If provided, used directly instead of re-preparing from inputs.

        Raises:
            ValueError: Invalid provider or configuration
            httpx.*: Network/API errors
        """
        # Validate and coerce provider to enum
        # At this point, variables like {{inputs.llm_provider}} are already resolved
        # Profile resolution ensures provider is not None
        if inputs.provider is None:
            raise ValueError(
                "provider must be specified. This should not happen after profile resolution."
            )
        provider = resolve_interpolatable_enum(inputs.provider, LLMProvider, "provider")

        if provider == LLMProvider.OPENAI:
            return await self._call_openai(
                inputs, prompt, timeout, temperature, max_tokens, prepared_schema
            )
        elif provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(inputs, prompt, timeout, temperature, max_tokens)
        elif provider == LLMProvider.GEMINI:
            return await self._call_gemini(inputs, prompt, timeout, temperature, max_tokens)
        elif provider == LLMProvider.OLLAMA:
            return await self._call_ollama(inputs, prompt, timeout, temperature, max_tokens)
        else:
            # This case should be unreachable due to the Enum validation
            raise ValueError(f"Unsupported provider: {provider}")

    # Whitelist of supported JSON schema keywords for OpenAI's format
    SUPPORTED_KEYWORDS: ClassVar[set[str]] = {
        "type",
        "properties",
        "items",
        "required",
        "enum",
        "description",
        "additionalProperties",
        "anyOf",
    }

    def _prepare_schema_for_openai(self, schema: Any) -> Any:
        """Recursively simplifies and strictifies a JSON schema for OpenAI.

        This function processes a JSON schema to make it compliant with
        OpenAI's strict requirements for 'response_format'. It does this by:
        1.  Stripping unsupported keywords to reduce complexity.
        2.  Enforcing 'additionalProperties: false' on objects that are not
            explicitly defined as maps (i.e., don't have a schema in
            'additionalProperties' already).
        3.  Converting 'additionalProperties: true' to a string-typed schema
            (OpenAI strict mode doesn't allow 'true').
        4.  Converting empty 'items: {}' to a string-typed schema for arrays.
        5.  Ensuring ALL properties are in the 'required' array (OpenAI strict mode
            requirement). Properties not originally required are made nullable.
        """
        if not isinstance(schema, dict):
            return schema

        # Prune unsupported keywords
        simplified_schema = {
            key: value for key, value in schema.items() if key in self.SUPPORTED_KEYWORDS
        }

        # Enforce strictness for objects
        if simplified_schema.get("type") == "object":
            # If 'properties' are defined, recurse into them
            if "properties" in simplified_schema:
                original_required = set(simplified_schema.get("required", []))
                all_property_keys = list(simplified_schema["properties"].keys())

                new_properties = {}
                for k, v in simplified_schema["properties"].items():
                    processed = self._prepare_schema_for_openai(v)

                    # If property was not originally required, make it nullable
                    if k not in original_required:
                        processed = self._make_nullable(processed)

                    new_properties[k] = processed

                simplified_schema["properties"] = new_properties

                # OpenAI strict mode: ALL properties must be in required array
                simplified_schema["required"] = all_property_keys

            # Handle additionalProperties
            addl_props = simplified_schema.get("additionalProperties")
            if isinstance(addl_props, dict) and addl_props:
                # If it's a non-empty schema dict, recurse into it
                simplified_schema["additionalProperties"] = self._prepare_schema_for_openai(
                    addl_props
                )
            elif addl_props is True or (isinstance(addl_props, dict) and not addl_props):
                # Convert 'true' or empty dict {} to a string-typed schema
                # This allows arbitrary string-valued properties (best we can do
                # while remaining OpenAI strict-mode compliant)
                simplified_schema["additionalProperties"] = {"type": "string"}
            elif "additionalProperties" not in simplified_schema:
                # If not set at all, forbid extra properties
                simplified_schema["additionalProperties"] = False

        # Recurse into array items
        elif simplified_schema.get("type") == "array":
            items = simplified_schema.get("items")
            if isinstance(items, dict) and items:
                # Non-empty items schema - recurse
                simplified_schema["items"] = self._prepare_schema_for_openai(items)
            elif items == {} or items is None or "items" not in simplified_schema:
                # Empty items {} or missing items - use string type as fallback
                simplified_schema["items"] = {"type": "string"}

        # Recurse into anyOf options
        if "anyOf" in simplified_schema:
            simplified_schema["anyOf"] = [
                self._prepare_schema_for_openai(option) for option in simplified_schema["anyOf"]
            ]

        return simplified_schema

    def _make_nullable(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Make a schema nullable using anyOf union with null type.

        OpenAI strict mode requires all properties in the required array.
        To make a property optional, we use anyOf: [{original_schema}, {type: "null"}].

        Uses anyOf syntax instead of type arrays for broader compatibility
        with LLM providers (e.g., LMStudio doesn't support type: ["string", "null"]).
        """
        if not isinstance(schema, dict):
            return schema

        schema = schema.copy()

        # If already has anyOf, add null option if not present
        if "anyOf" in schema:
            has_null = any(
                isinstance(opt, dict) and opt.get("type") == "null" for opt in schema["anyOf"]
            )
            if not has_null:
                schema["anyOf"] = list(schema["anyOf"]) + [{"type": "null"}]
            return schema

        current_type = schema.get("type")

        if current_type is None:
            # No type specified, use nullable string as default
            schema["anyOf"] = [{"type": "string"}, {"type": "null"}]
            return schema

        if isinstance(current_type, list):
            # Already a list of types, convert to anyOf format
            type_options = [{"type": t} for t in current_type]
            if not any(opt["type"] == "null" for opt in type_options):
                type_options.append({"type": "null"})
            del schema["type"]
            schema["anyOf"] = type_options
            return schema

        # Single type - build anyOf with the type schema and null
        # For complex types (object, array), include their sub-schemas
        type_schema: dict[str, Any] = {"type": current_type}

        # Move type-specific keywords into the type schema
        if current_type == "object":
            for key in ["properties", "required", "additionalProperties"]:
                if key in schema:
                    type_schema[key] = schema.pop(key)
        elif current_type == "array":
            if "items" in schema:
                type_schema["items"] = schema.pop("items")
        elif current_type == "string":
            if "enum" in schema:
                type_schema["enum"] = schema.pop("enum")

        # Remove type from top level
        del schema["type"]

        # Add anyOf with original type and null
        schema["anyOf"] = [type_schema, {"type": "null"}]

        return schema

    async def _call_openai(
        self,
        inputs: LLMCallInput,
        prompt: str,
        timeout: int,
        temperature: float | None,
        max_tokens: int | None,
        prepared_schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Call OpenAI API using official AsyncOpenAI client.

        Uses OpenAI's official Python library with built-in retry logic, better error
        handling, and native structured outputs support. Compatible with OpenAI-like
        servers (LM Studio, vLLM, etc.) via custom base_url.

        Args:
            prepared_schema: Pre-prepared schema. If provided, used directly instead
                of re-preparing from inputs.response_schema. This ensures the same
                schema is used for both the API call and validation.

        Raises:
            ValueError: Null content, refusal, or unexpected format
            openai.*: OpenAI SDK exceptions (APIConnectionError, RateLimitError, etc.)
        """
        # Build messages list
        messages: list[dict[str, str]] = []
        if inputs.system_instructions:
            messages.append({"role": "system", "content": inputs.system_instructions})
        messages.append({"role": "user", "content": prompt})

        # Initialize AsyncOpenAI client with custom settings
        client_kwargs: dict[str, Any] = {
            "timeout": float(timeout),
            "max_retries": 0,  # We handle retries at executor level
        }

        if inputs.api_key:
            client_kwargs["api_key"] = inputs.api_key
        else:
            # Set a dummy key for OpenAI-compatible servers that don't require auth
            client_kwargs["api_key"] = "sk-no-key-required"

        if inputs.api_url:
            # OpenAI SDK appends "/chat/completions" to base_url automatically
            # Strip it if user provided full endpoint URL (common mistake)
            base_url = inputs.api_url
            if base_url.endswith("/chat/completions"):
                base_url = base_url.rsplit("/chat/completions", 1)[0]
            client_kwargs["base_url"] = base_url

        # Prepare completion parameters (required parameters only)
        completion_kwargs: dict[str, Any] = {
            "model": inputs.model or "",
            "messages": messages,
        }

        # Detect reasoning models (o1, o3, o4, gpt-5 series)
        # These models don't support temperature and other sampling parameters
        model_name = (inputs.model or "").lower()
        is_reasoning_model = any(
            pattern in model_name
            for pattern in ["o1-", "o1_", "o3-", "o3_", "o4-", "o4_", "gpt-5", "gpt5"]
        )

        # Add temperature only if user specified it AND model supports it
        if temperature is not None and not is_reasoning_model:
            completion_kwargs["temperature"] = temperature

        # Add max_tokens only if user specified it
        # Use max_completion_tokens (new standard for modern models)
        if max_tokens is not None:
            completion_kwargs["max_completion_tokens"] = max_tokens

        # Native schema validation (OpenAI Structured Outputs)
        # Use pre-prepared schema if provided, otherwise prepare from inputs
        schema_for_api = prepared_schema
        if schema_for_api is None and inputs.response_schema:
            schema_for_api = self._prepare_schema_for_openai(inputs.response_schema)

        if schema_for_api:
            completion_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "strict": True,
                    "schema": schema_for_api,
                },
            }

        # Create client and make request
        async with AsyncOpenAI(**client_kwargs) as client:
            # Try with max_completion_tokens (new standard), fallback to max_tokens for older models
            try:
                response = await client.chat.completions.create(**completion_kwargs)
            except openai.BadRequestError as e:
                # Check if error is about max_completion_tokens not being supported
                error_msg = str(e)
                if (
                    max_tokens is not None
                    and "max_completion_tokens" in error_msg
                    and "not supported" in error_msg.lower()
                ):
                    # Retry with max_tokens for older models
                    completion_kwargs.pop("max_completion_tokens", None)
                    completion_kwargs["max_tokens"] = max_tokens
                    response = await client.chat.completions.create(**completion_kwargs)
                else:
                    # Re-raise for other errors
                    raise

            # Extract content with null safety
            message = response.choices[0].message
            content = message.content

            if content is None:
                # Handle refusal or tool call scenarios
                if message.refusal:
                    raise ValueError(f"OpenAI refused request: {message.refusal}")
                # Check for tool calls
                if message.tool_calls:
                    raise ValueError("OpenAI returned tool calls instead of text content")
                raise ValueError("OpenAI returned null content (unexpected response format)")

            response_text = content

            # Build provider metadata
            provider_metadata: dict[str, Any] = {
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            }

            # Add usage stats if available
            if response.usage:
                provider_metadata["usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            # Add reasoning if available (for models like o1 that support CoT)
            if hasattr(message, "reasoning") and message.reasoning:
                provider_metadata["reasoning"] = message.reasoning

            return response_text, provider_metadata

    async def _call_anthropic(
        self,
        inputs: LLMCallInput,
        prompt: str,
        timeout: int,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Call Anthropic API with null safety.

        Raises:
            ValueError: Empty content or null text
            httpx.*: Network/API errors
        """
        url = inputs.api_url or "https://api.anthropic.com/v1/messages"

        body: dict[str, Any] = {
            "model": inputs.model,
            "messages": [{"role": "user", "content": prompt}],
            # max_tokens is required by Anthropic API - use 4096 as reasonable default
            # (higher than docs example of 1024 to allow longer responses)
            "max_tokens": max_tokens or 4096,
        }

        if inputs.system_instructions:
            body["system"] = inputs.system_instructions
        if temperature is not None:
            body["temperature"] = temperature

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if inputs.api_key:
            headers["x-api-key"] = inputs.api_key

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()

            data = response.json()

            # Extract content with null safety
            content_blocks = data.get("content", [])
            if not content_blocks:
                raise ValueError("Anthropic returned empty content array")

            text_content = content_blocks[0].get("text")
            if text_content is None:
                # Check for refusal
                stop_reason = data.get("stop_reason")
                if stop_reason == "end_turn":
                    raise ValueError("Anthropic returned null text content")
                raise ValueError(f"Anthropic returned null content (stop_reason: {stop_reason})")

            response_text = text_content

            provider_metadata = {
                "model": data.get("model"),
                "usage": data.get("usage", {}),
                "stop_reason": data.get("stop_reason"),
            }

            return response_text, provider_metadata

    async def _call_gemini(
        self,
        inputs: LLMCallInput,
        prompt: str,
        timeout: int,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Call Google Gemini API with null safety.

        Raises:
            ValueError: Missing API key, empty content, or null text
            httpx.*: Network/API errors
        """
        if not inputs.api_key:
            raise ValueError("api_key is required for Gemini provider")

        base_url = "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base_url}/models/{inputs.model}:generateContent?key={inputs.api_key}"

        contents = [{"parts": [{"text": prompt}], "role": "user"}]

        body: dict[str, Any] = {
            "contents": contents,
        }

        if inputs.system_instructions:
            body["system_instruction"] = {"parts": [{"text": inputs.system_instructions}]}

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        if generation_config:
            body["generationConfig"] = generation_config

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            data = response.json()

            # Extract content with null safety
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini returned empty candidates array")

            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            if not parts:
                # Check for content filtering or safety blocks
                finish_reason = candidate.get("finishReason")
                if finish_reason and finish_reason != "STOP":
                    raise ValueError(f"Gemini blocked content (finishReason: {finish_reason})")
                raise ValueError("Gemini returned empty parts array")

            text_content = parts[0].get("text")
            if text_content is None:
                raise ValueError("Gemini returned null text content")

            response_text = text_content

            provider_metadata = {
                "model": inputs.model,
                "usage": data.get("usageMetadata", {}),
                "finish_reason": candidate.get("finishReason"),
            }

            return response_text, provider_metadata

    async def _call_ollama(
        self,
        inputs: LLMCallInput,
        prompt: str,
        timeout: int,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Call Ollama local API with null safety.

        Raises:
            ValueError: Null response
            httpx.*: Network/API errors
        """
        url = inputs.api_url or "http://localhost:11434/api/generate"

        # Combine system instructions and prompt for Ollama
        full_prompt = prompt
        if inputs.system_instructions:
            full_prompt = f"{inputs.system_instructions}\n\n{prompt}"

        body: dict[str, Any] = {
            "model": inputs.model,
            "prompt": full_prompt,
            "stream": False,
        }

        if temperature is not None:
            body["options"] = {"temperature": temperature}

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            data = response.json()

            # Extract content with null safety
            response_text = data.get("response")
            if response_text is None:
                raise ValueError("Ollama returned null response")

            provider_metadata = {
                "model": data.get("model"),
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            }

            return response_text, provider_metadata

    @staticmethod
    def _validate_response(response_text: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Validate response against JSON Schema (client-side).

        Used by all providers for maximum compatibility with OpenAI-like servers.

        Raises:
            ValueError: Invalid JSON, non-dict response, or schema validation failure
        """
        # Try to parse as JSON
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Response is not valid JSON: {e}")

        # Ensure it's a dict (schema requires type: object)
        if not isinstance(response, dict):
            raise ValueError(f"Response is not a JSON object, got {type(response).__name__}")

        # Validate against schema using jsonschema
        try:
            jsonschema.validate(instance=response, schema=schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Response does not match schema: {e.message}")

        return response


# ===========================================================================
# Embedding Executor
# ===========================================================================


class EmbeddingInput(BlockInput):
    """Input schema for Embedding block.

    Generates vector embeddings for text using OpenAI-compatible embedding API.
    Works with OpenAI, LMStudio, Ollama (--api-compat), LocalAI, vLLM, and other
    servers implementing the /v1/embeddings endpoint.

    Defaults to 'embedding' profile from ~/.workflows/llm-config.yml.
    """

    profile: str = Field(
        default="embedding",
        description="Profile name from ~/.workflows/llm-config.yml (defaults to 'embedding')",
    )

    model: str | None = Field(
        default=None,
        description="Override embedding model (uses profile model if not specified)",
    )

    text: str = Field(
        description="Text to generate embedding for",
    )

    api_key: str | None = Field(
        default=None,
        description="Override API key (uses profile api_key_secret if not specified)",
    )

    api_url: str | None = Field(
        default=None,
        description="Override API endpoint URL (uses profile api_url if not specified)",
    )

    timeout: int | str = Field(
        default=30,
        description="Request timeout in seconds",
    )

    _validate_timeout = field_validator("timeout", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=300)
    )


class EmbeddingOutput(BlockOutput):
    """Output schema for Embedding block.

    Returns the embedding vector and metadata.
    """

    embedding: list[float] = Field(
        default_factory=list,
        description="Embedding vector (list of floats)",
    )

    dimensions: int = Field(
        default=0,
        description="Number of dimensions in the embedding",
    )

    success: bool = Field(
        default=False,
        description="Whether the embedding generation succeeded",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (model, usage, etc.)",
    )


class EmbeddingExecutor(BlockExecutor):
    """Executor for generating text embeddings using OpenAI-compatible API.

    Works with any server implementing the OpenAI embeddings endpoint:
    - OpenAI API (api.openai.com)
    - LMStudio (localhost:1234)
    - Ollama with OpenAI compatibility (localhost:11434/v1)
    - LocalAI, vLLM, and other OpenAI-compatible servers

    Configuration via ~/.workflows/llm-config.yml profile (defaults to 'embedding'):
    ```yaml
    profiles:
      embedding:
        provider: openai  # or ollama, local, etc.
        model: text-embedding-3-small
        api_url: http://localhost:1234/v1  # for LMStudio
        api_key_secret: OPENAI_API_KEY
    ```

    Example:
        ```yaml
        - id: embed_query
          type: Embedding
          inputs:
            text: "{{inputs.query}}"
        ```

    Outputs:
        - embedding: List of floats representing the embedding vector
        - dimensions: Number of dimensions (e.g., 1536 for text-embedding-3-small)
        - success: Whether the API call succeeded
        - metadata: Contains model, usage, etc.
    """

    type_name: ClassVar[str] = "Embedding"
    input_type: ClassVar[type[BlockInput]] = EmbeddingInput
    output_type: ClassVar[type[BlockOutput]] = EmbeddingOutput
    examples: ClassVar[str] = """```yaml
- id: embed_text
  type: Embedding
  inputs:
    text: "Search for authentication bugs"
```"""

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_network=True)

    async def execute(  # type: ignore[override]
        self, inputs: EmbeddingInput, context: Execution
    ) -> EmbeddingOutput:
        """Execute embedding generation using OpenAI-compatible API.

        Resolves profile configuration and calls the embeddings endpoint.
        Works with any OpenAI-compatible server (OpenAI, LMStudio, Ollama, etc.).

        Raises:
            ValueError: Invalid configuration
            httpx.*: Network errors
        """
        execution_context = context.execution_context
        if execution_context is None:
            raise ValueError("ExecutionContext not available")

        llm_config_loader = execution_context.llm_config_loader

        # Start with input overrides
        model = inputs.model
        api_key = inputs.api_key
        api_url = inputs.api_url

        # Build inline overrides from inputs (filter out None values)
        inline_overrides = {
            key: value
            for key, value in {
                "model": inputs.model,
                "api_url": inputs.api_url,
            }.items()
            if value is not None
        }

        # Try to resolve profile (returns ResolvedLLMConfig with merged provider+profile values)
        resolved_config = None
        config = llm_config_loader.load_config()

        effective_profile: str | None = inputs.profile
        if effective_profile not in config.profiles:
            if config.default_profile and config.default_profile in config.profiles:
                logger.warning(
                    f"Embedding profile '{inputs.profile}' not found, "
                    f"using default_profile '{config.default_profile}'"
                )
                effective_profile = config.default_profile
            else:
                effective_profile = None

        if effective_profile is not None:
            resolved_config = llm_config_loader.resolve_profile(
                profile=effective_profile, inline_overrides=inline_overrides
            )

        if resolved_config:
            # Use resolved values as defaults (inputs override)
            model = model or resolved_config.model
            api_url = api_url or resolved_config.api_url

            # Resolve API key from secrets if not provided
            if api_key is None and resolved_config.api_key_secret:
                from .secrets import EnvVarSecretProvider

                secret_provider = EnvVarSecretProvider()
                api_key = await secret_provider.get_secret(resolved_config.api_key_secret)

        # Default model if still not set
        if model is None:
            model = "text-embedding-3-small"

        # Resolve timeout
        timeout = resolve_interpolatable_numeric(inputs.timeout, int, "timeout", ge=1, le=300)

        try:
            # Use OpenAI SDK which works with any OpenAI-compatible server
            client = AsyncOpenAI(
                api_key=api_key or "not-required",  # Some local servers don't need API key
                base_url=api_url,  # None = default OpenAI endpoint
                timeout=float(timeout),
            )

            response = await client.embeddings.create(
                model=model,
                input=inputs.text,
            )

            embedding = response.data[0].embedding
            dimensions = len(embedding)

            metadata: dict[str, Any] = {
                "model": response.model,
            }

            # Include usage if available (some servers may not return it)
            if response.usage:
                metadata["usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return EmbeddingOutput(
                embedding=embedding,
                dimensions=dimensions,
                success=True,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return EmbeddingOutput(
                embedding=[],
                dimensions=0,
                success=False,
                metadata={"error": str(e)},
            )
