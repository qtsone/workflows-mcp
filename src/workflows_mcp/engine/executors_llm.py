"""LLM call executor with retry logic and schema validation.

Architecture (ADR-006):
- Execute returns output directly (no Result wrapper)
- Raises exceptions for failures
- Uses Execution context (not dict)
- Orchestrator creates Metadata based on success/exceptions

Features:
- Multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama)
- Automatic retry with exponential backoff
- JSON Schema validation with feedback loop
- Pre-resolved inputs ({{secrets.KEY}} resolved by VariableResolver)
- Token usage tracking
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any, ClassVar

import httpx
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

# ===========================================================================
# Type Definitions
# ===========================================================================


class LLMProvider(str, Enum):
    """Enum for supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


# ===========================================================================
# LLMCall Executor
# ===========================================================================


class LLMCallInput(BlockInput):
    """Input model for LLMCall executor.

    All inputs are pre-resolved by VariableResolver:
    - {{secrets.OPENAI_API_KEY}} → actual key value
    - {{inputs.prompt}} → actual prompt text
    - {{blocks.previous.outputs.data}} → actual data

    Configuration Modes:
    1. Profile-based (NEW): Specify profile name, optionally override parameters
    2. Direct specification (backward compatible): Specify provider + model directly

    Examples:
        # Profile-based (recommended)
        profile: quick
        prompt: "..."

        # Profile with overrides
        profile: standard
        temperature: 1.0
        max_tokens: 8000
        prompt: "..."

        # Direct specification (backward compatible)
        provider: openai
        model: gpt-4o
        api_key: "{{secrets.OPENAI_API_KEY}}"
        prompt: "..."
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
        """Validate and parse response_schema if provided.

        Accepts:
        - dict: Direct JSON Schema object
        - str: JSON string that will be parsed into a dict
        - None: No schema validation

        Returns:
            Parsed dict or None

        Raises:
            ValueError: If string is not valid JSON or schema is invalid
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
        """Validate that either profile OR (provider + model) is specified.

        Resolution logic:
        - If profile is specified: Use profile, ignore provider/model (unless overriding)
        - If profile is NOT specified: provider and model are REQUIRED

        This validator runs AFTER all field validators, so we have access to
        all resolved field values.

        Returns:
            self

        Raises:
            ValueError: If neither profile nor (provider + model) is specified
        """
        if self.profile is None:
            # No profile - provider and model are required
            if self.provider is None:
                raise ValueError(
                    "Either 'profile' OR 'provider' must be specified. "
                    "Use profile for config-based setup, or direct provider+model."
                )
            if self.model is None:
                raise ValueError(
                    "When 'provider' is specified without 'profile', 'model' is required."
                )

        return self


class LLMCallOutput(BlockOutput):
    """Output model for LLMCall executor.

    All fields have defaults to support graceful degradation when LLM calls fail.
    A default-constructed instance represents a failed/crashed LLM call.

    Clean separation (aligned with industry standards):
    - response: Raw LLM response text (empty string if crashed)
    - response_json: Parsed JSON dict (empty if validation failed or not JSON)
    - success: Overall operation success (False if crashed)
    - metadata: Execution details (empty dict if crashed before execution)

    Design rationale:
    - response_json is always a dict (never None) to prevent downstream errors
    - Empty dict {} indicates: validation failed, response wasn't JSON, or crashed
    - Use success + metadata.validation_passed to distinguish scenarios
    """

    response: str = Field(
        default="",
        description="Raw LLM response text (empty string if request failed or crashed)",
    )
    response_json: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parsed JSON response. Empty dict if validation failed, response is plain text, "
            "or request crashed. Check success/metadata.validation_passed to distinguish scenarios."
        ),
    )
    success: bool = Field(
        default=False,
        description=(
            "True if LLM call succeeded and passed validation (if schema provided), "
            "False if crashed"
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Execution metadata: attempts (int), validation_passed (bool), "
            "validation_error (str, if failed), model (str), usage (dict), "
            "finish_reason (str), etc. Empty dict if crashed before execution."
        ),
    )


class LLMCallExecutor(BlockExecutor):
    """
    LLM call executor with retry logic and schema validation.

    Architecture (ADR-006):
    - Returns LLMCallOutput directly
    - Raises exceptions for unrecoverable failures
    - Uses Execution context
    - All inputs pre-resolved by VariableResolver

    Features:
    - Multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama)
    - Automatic retry with exponential backoff
    - JSON Schema validation with feedback loop
    - Token usage tracking
    - Configurable timeouts and retry behavior

    Usage:
        ```yaml
        - id: call_gpt4
          type: LLMCall
          inputs:
            provider: openai
            model: gpt-4o
            api_key: "{{secrets.OPENAI_API_KEY}}"  # Pre-resolved
            prompt: "{{inputs.user_question}}"
            response_schema:  # Can be dict or JSON string
              type: object
              required: [answer, confidence]
              properties:
                answer: {type: string}
                confidence: {type: number}
            max_retries: 3

        # Access outputs:
        # {{blocks.call_gpt4.response}} - Raw text
        # {{blocks.call_gpt4.response_json.answer}} - Parsed JSON field
        # {{blocks.call_gpt4.succeeded}} - Boolean status
        # {{blocks.call_gpt4.metadata.attempts}} - Retry count
        ```

    Security:
    - API keys passed as pre-resolved inputs (from secrets system)
    - SSL verification enabled by default
    - No arbitrary code execution
    - Response size limited by httpx defaults
    """

    type_name: ClassVar[str] = "LLMCall"
    input_type: ClassVar[type[BlockInput]] = LLMCallInput
    output_type: ClassVar[type[BlockOutput]] = LLMCallOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_network=True)

    async def execute(  # type: ignore[override]
        self, inputs: LLMCallInput, context: Execution
    ) -> LLMCallOutput:
        """Execute LLM call with retry logic and optional schema validation.

        Profile Resolution:
        - If inputs.profile is specified: Load config from ~/.workflows/llm-config.yml
        - Merge profile config with inline parameter overrides from inputs
        - If inputs.profile is None: Use direct provider/model specification (backward compatible)

        Returns:
            LLMCallOutput with response, validation status, and metadata

        Raises:
            ValueError: Invalid inputs, configuration, or profile not found
            httpx.TimeoutException: All retry attempts timed out
            httpx.HTTPStatusError: HTTP error from provider
            Exception: Other unrecoverable errors
        """
        # Step 1: Profile Resolution (if profile specified)
        effective_inputs = inputs
        if inputs.profile is not None:
            effective_inputs = await self._resolve_profile_to_inputs(inputs, context)

        # Step 2: Resolve interpolatable numeric fields to their actual types
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

        for attempt in range(max_retries):
            attempts += 1

            try:
                # Build prompt (add validation feedback if retry due to validation failure)
                prompt = effective_inputs.prompt
                if validation_error and attempt > 0:
                    prompt = effective_inputs.validation_prompt_template.format(
                        validation_error=validation_error,
                        schema=json.dumps(effective_inputs.response_schema, indent=2),
                    )

                # Make LLM API call
                response_text, provider_metadata = await self._call_provider(
                    inputs=effective_inputs,
                    prompt=prompt,
                    timeout=timeout,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=context,
                )

                # Validate response if schema provided
                if needs_validation:
                    try:
                        validated_response = self._validate_response(
                            response_text=response_text,
                            schema=effective_inputs.response_schema,  # type: ignore[arg-type]
                        )

                        # Success! Return validated response
                        return LLMCallOutput(
                            response=response_text,
                            response_json=validated_response,
                            success=True,
                            metadata={
                                "attempts": attempts,
                                "validation_passed": True,
                                **provider_metadata,
                            },
                        )
                    except ValueError as e:
                        # Validation failed
                        validation_error = str(e)
                        last_error = e

                        # If this is the last attempt, return with validation failure
                        if attempt == max_retries - 1:
                            return LLMCallOutput(
                                response=response_text,
                                response_json={},  # Empty dict - validation failed
                                success=False,
                                metadata={
                                    "attempts": attempts,
                                    "validation_passed": False,
                                    "validation_error": validation_error,
                                    **provider_metadata,
                                },
                            )

                        # Otherwise, wait and retry with feedback
                        delay = retry_delay * (2**attempt)
                        await asyncio.sleep(delay)
                        continue
                else:
                    # No validation needed, but opportunistically parse JSON
                    response_json = {}
                    try:
                        parsed = json.loads(response_text)
                        # Only accept dicts (schema requires type: object)
                        if isinstance(parsed, dict):
                            response_json = parsed
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON or not a dict, that's fine - keep empty dict
                        pass

                    return LLMCallOutput(
                        response=response_text,
                        response_json=response_json,
                        success=True,
                        metadata={
                            "attempts": attempts,
                            "validation_passed": True,  # No schema = no validation needed
                            **provider_metadata,
                        },
                    )

            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.NetworkError) as e:
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
        """Resolve profile configuration to effective inputs.

        This method:
        1. Loads profile config from llm_config_loader (via ExecutionContext)
        2. Merges profile config with inline parameter overrides from inputs
        3. Resolves API key from secrets if api_key_secret is specified
        4. Returns new LLMCallInput with all values resolved

        Args:
            inputs: Original inputs with profile specified
            context: Execution context (provides access to llm_config_loader and secrets)

        Returns:
            New LLMCallInput with profile-resolved values

        Raises:
            ValueError: If profile not found or llm_config_loader not available
        """
        # Get llm_config_loader from execution context
        execution_context = context._internal.execution_context
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

    async def _call_provider(
        self,
        inputs: LLMCallInput,
        prompt: str,
        timeout: int,
        temperature: float | None,
        max_tokens: int | None,
        context: Execution,
    ) -> tuple[str, dict[str, Any]]:
        """Call the specified LLM provider.

        Args:
            inputs: Validated input model (variables already resolved by VariableResolver)
            prompt: Processed prompt (may include validation feedback)
            timeout: Resolved timeout in seconds
            temperature: Resolved temperature parameter (or None)
            max_tokens: Resolved max tokens parameter (or None)
            context: Execution context

        Returns:
            Tuple of (response_text, provider_metadata)

        Raises:
            ValueError: Invalid provider or configuration
            httpx exceptions: Network/API errors
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
            return await self._call_openai(inputs, prompt, timeout, temperature, max_tokens)
        elif provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(inputs, prompt, timeout, temperature, max_tokens)
        elif provider == LLMProvider.GEMINI:
            return await self._call_gemini(inputs, prompt, timeout, temperature, max_tokens)
        elif provider == LLMProvider.OLLAMA:
            return await self._call_ollama(inputs, prompt, timeout, temperature, max_tokens)
        else:
            # This case should be unreachable due to the Enum validation
            raise ValueError(f"Unsupported provider: {provider}")

    async def _call_openai(
        self,
        inputs: LLMCallInput,
        prompt: str,
        timeout: int,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Call OpenAI API with optional native schema validation."""
        url = inputs.api_url or "https://api.openai.com/v1/chat/completions"

        messages = []
        if inputs.system_instructions:
            messages.append({"role": "system", "content": inputs.system_instructions})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": inputs.model,
            "messages": messages,
        }

        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        # Native schema validation (OpenAI Structured Outputs)
        # Send schema to API if provided - compatible APIs will use it
        if inputs.response_schema:
            # Type narrowing: validator ensures this is dict (never str at runtime)
            assert isinstance(inputs.response_schema, dict), "Schema must be dict after validation"
            # OpenAI requires additionalProperties: false for strict mode
            schema = inputs.response_schema.copy()
            if "additionalProperties" not in schema:
                schema["additionalProperties"] = False

            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "strict": True,
                    "schema": schema,
                },
            }

        headers = {"Content-Type": "application/json"}
        if inputs.api_key:
            headers["Authorization"] = f"Bearer {inputs.api_key}"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()

            data = response.json()
            response_text = data["choices"][0]["message"]["content"]

            provider_metadata = {
                "model": data.get("model"),
                "usage": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason"),
                "native_schema_sent": inputs.response_schema is not None,
            }

            return response_text, provider_metadata

    async def _call_anthropic(
        self,
        inputs: LLMCallInput,
        prompt: str,
        timeout: int,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Call Anthropic API."""
        url = inputs.api_url or "https://api.anthropic.com/v1/messages"

        body: dict[str, Any] = {
            "model": inputs.model,
            "messages": [{"role": "user", "content": prompt}],
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
            response_text = data["content"][0]["text"]

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
        """Call Google Gemini API."""
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
            response_text = data["candidates"][0]["content"]["parts"][0]["text"]

            provider_metadata = {
                "model": inputs.model,
                "usage": data.get("usageMetadata", {}),
                "finish_reason": data["candidates"][0].get("finishReason"),
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
        """Call Ollama local API."""
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
            response_text = data["response"]

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
        """Validate LLM response against JSON Schema.

        Args:
            response_text: Raw text response from LLM
            schema: JSON Schema to validate against

        Returns:
            Parsed and validated JSON object

        Raises:
            ValueError: If response is not valid JSON or doesn't match schema
        """
        # Try to parse as JSON
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Response is not valid JSON: {e}")

        # Ensure it's a dict (schema requires type: object)
        if not isinstance(response_json, dict):
            raise ValueError(f"Response is not a JSON object, got {type(response_json).__name__}")

        # Validate against schema using jsonschema
        try:
            import jsonschema

            jsonschema.validate(instance=response_json, schema=schema)
        except ImportError:
            # jsonschema not installed, skip strict validation
            # Just return the parsed JSON
            pass
        except jsonschema.ValidationError as e:
            raise ValueError(f"Response does not match schema: {e.message}")

        return response_json
