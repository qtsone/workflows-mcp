"""Image generation executor using OpenAI DALL-E and compatible providers.

Supports generation, editing, and variations with DALL-E 2, DALL-E 3, and
OpenAI-compatible image generation endpoints.
Includes capability to save generated images directly to disk.
"""

from __future__ import annotations

import base64
import logging
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx
from openai import AsyncOpenAI
from pydantic import Field, field_validator, model_validator

from .block import BlockInput, BlockOutput
from .block_utils import PathResolver
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


# Model capability registry - easily extensible for new models and providers
# Maps model name patterns to supported operations and parameters
MODEL_CAPABILITIES = {
    "dall-e-3": {
        "operations": {"generate"},
        "params": {"response_format", "quality", "style"},
    },
    "dall-e-2": {
        "operations": {"generate", "edit", "variation"},
        "params": {"response_format"},
    },
    "gpt-image-": {
        "operations": {"generate", "edit"},
        "params": set(),
    },
}


def _get_model_capabilities(model: str | None) -> dict[str, set[str]]:
    """Determine which operations and parameters a model supports.

    Returns dict with:
        - 'operations': set of supported operations (generate, edit, variation)
        - 'params': set of supported optional params (response_format, quality, style)
    """
    if not model:
        # Default to dall-e-3 capabilities
        return MODEL_CAPABILITIES["dall-e-3"]

    # Check for exact matches first
    if model in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model]

    # Check for prefix matches (e.g., gpt-image-1 matches gpt-image-)
    for pattern, capabilities in MODEL_CAPABILITIES.items():
        if pattern.endswith("-") and model.startswith(pattern):
            return capabilities

    # Unknown models: assume DALL-E 3 capabilities for compatibility
    logger.warning(
        f"Unknown model '{model}'. Assuming DALL-E 3 capabilities. "
        "Add to MODEL_CAPABILITIES if incorrect."
    )
    return MODEL_CAPABILITIES["dall-e-3"]


class ImageProvider(str, Enum):
    """Supported image generation providers."""

    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"


class ImageGenInput(BlockInput):
    """Input model for ImageGen executor."""

    prompt: str | None = Field(
        default=None,
        description="Text prompt (required for generate/edit, not used for variation)",
    )
    profile: str | None = Field(
        default=None,
        description=(
            "Profile name from ~/.workflows/llm-config.yml. "
            "If specified, provider/model are loaded from config. "
            "Mutually exclusive with direct provider/model specification."
        ),
    )
    provider: ImageProvider | str | None = Field(
        default=None,
        description=(
            "Image provider (openai, openai_compatible). Required if profile not specified."
        ),
    )
    model: str | None = Field(
        default="dall-e-3",
        description="Model to use (dall-e-3, dall-e-2, or custom model name)",
    )
    api_url: str | None = Field(
        default=None,
        description="Custom API endpoint URL (required for openai_compatible)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (pre-resolved from {{secrets.OPENAI_API_KEY}})",
    )
    operation: Literal["generate", "edit", "variation"] = Field(
        default="generate",
        description="Operation to perform",
    )
    size: str = Field(
        default="1024x1024",
        description="Image size (e.g., 1024x1024, 256x256, 512x512)",
    )
    quality: Literal["standard", "hd"] | None = Field(
        default="standard",
        description="Image quality (dall-e-3 only)",
    )
    style: Literal["vivid", "natural"] | None = Field(
        default="vivid",
        description="Image style (dall-e-3 only)",
    )
    response_format: Literal["url", "b64_json"] = Field(
        default="url",
        description="Format of the response",
    )
    n: int | str = Field(
        default=1,
        description="Number of images to generate (supports interpolation)",
    )
    image: str | None = Field(
        default=None,
        description="Path to base image (required for edit/variation)",
    )
    mask: str | None = Field(
        default=None,
        description="Path to mask image (optional for edit)",
    )
    output_file: str | None = Field(
        default=None,
        description="Path to save the generated image(s). If n>1, appends index.",
    )

    _validate_provider = field_validator("provider", mode="before")(
        interpolatable_enum_validator(ImageProvider)
    )
    _validate_n = field_validator("n", mode="before")(
        interpolatable_numeric_validator(int, ge=1, le=10)
    )

    @model_validator(mode="after")
    def validate_profile_or_provider_model(self) -> ImageGenInput:
        """Validate configuration - profile and provider are mutually exclusive."""
        if self.profile is not None and self.provider is not None:
            raise ValueError(
                "Cannot specify both 'profile' and 'provider'. Choose one:\n"
                "  - Use 'profile' for config-based setup, OR\n"
                "  - Use 'provider' (+ optional 'model') for direct specification"
            )
        return self

    @model_validator(mode="after")
    def validate_operation_requirements(self) -> ImageGenInput:
        """Validate operation-specific requirements."""
        if self.operation in ("generate", "edit") and not self.prompt:
            raise ValueError(
                f"Operation '{self.operation}' requires a prompt. "
                "Variation operations do not use prompts."
            )
        return self


class ImageGenOutput(BlockOutput):
    """Output model for ImageGen executor."""

    urls: list[str] = Field(
        default_factory=list,
        description="List of image URLs",
    )
    b64_json: list[str] = Field(
        default_factory=list,
        description="List of base64 encoded image data",
    )
    revised_prompts: list[str] = Field(
        default_factory=list,
        description="List of revised prompts (dall-e-3)",
    )
    saved_files: list[str] = Field(
        default_factory=list,
        description="List of paths where images were saved",
    )
    success: bool = Field(
        default=False,
        description="Whether the operation succeeded",
    )
    provider_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata from the provider response",
    )


class ImageGenExecutor(BlockExecutor):
    """Executor for Image Generation.

    Supports:
    - OpenAI DALL-E (generate, edit, variation)
    - OpenAI-compatible endpoints (e.g., local SD, other providers)
    - Saving generated images to disk

    Example:
        ```yaml
        - id: gen_image
          type: ImageGen
          inputs:
            prompt: "A futuristic city"
            provider: openai
            model: dall-e-3
            output_file: "{{tmp}}/city.png"
        ```

    Example (Custom Provider):
        ```yaml
        - id: gen_custom
          type: ImageGen
          inputs:
            prompt: "A futuristic city"
            provider: openai_compatible
            api_url: "http://localhost:8000/v1"
            model: "sd-xl"
            output_file: "{{tmp}}/custom.png"
        ```
    """

    type_name: ClassVar[str] = "ImageGen"
    input_type: ClassVar[type[BlockInput]] = ImageGenInput
    output_type: ClassVar[type[BlockOutput]] = ImageGenOutput
    examples: ClassVar[str] = """```yaml
- id: generate-image
  type: ImageGen
  inputs:
    profile: default
    operation: generate
    prompt: "A beautiful sunset over mountains"
```"""

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(
        can_network=True,
        can_read_files=True,
        can_write_files=True,
    )

    async def execute(  # type: ignore[override]
        self, inputs: ImageGenInput, context: Execution
    ) -> ImageGenOutput:
        """Execute image generation."""
        # Get execution context for config access
        execution_context = context.execution_context
        if execution_context is None:
            raise ValueError("ExecutionContext not available. Cannot resolve configuration.")

        llm_config_loader = execution_context.llm_config_loader

        # Step 1: Resolve profile (with fallback logic)
        effective_profile = self._resolve_profile_with_fallback(inputs, llm_config_loader)

        # Step 2: Profile resolution (if profile determined)
        effective_inputs = inputs
        if effective_profile is not None:
            # Create new input with resolved profile
            inputs_with_profile = inputs.model_copy(update={"profile": effective_profile})
            effective_inputs = await self._resolve_profile_to_inputs(inputs_with_profile, context)

        # Default to OPENAI if provider is still None
        if effective_inputs.provider is None:
            if effective_inputs.profile is None:
                effective_inputs.provider = ImageProvider.OPENAI

        n = resolve_interpolatable_numeric(effective_inputs.n, int, "n", ge=1, le=10)
        provider = resolve_interpolatable_enum(effective_inputs.provider, ImageProvider, "provider")

        if provider == ImageProvider.OPENAI:
            return await self._call_openai(effective_inputs, n, base_url=None)
        elif provider == ImageProvider.OPENAI_COMPATIBLE:
            if not effective_inputs.api_url:
                # If api_url is missing, maybe it's in the profile?
                # effective_inputs should have it if resolved.
                raise ValueError("api_url is required for openai_compatible provider")
            return await self._call_openai(effective_inputs, n, base_url=effective_inputs.api_url)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _resolve_profile_to_inputs(
        self, inputs: ImageGenInput, context: Execution
    ) -> ImageGenInput:
        """Resolve profile configuration from ~/.workflows/llm-config.yml."""
        execution_context = context.execution_context
        if execution_context is None:
            raise ValueError("ExecutionContext not available.")

        llm_config_loader = execution_context.llm_config_loader

        # Build inline overrides
        inline_overrides = {
            key: value
            for key, value in {
                "provider": inputs.provider,
                "model": inputs.model,
                "api_url": inputs.api_url,
                "api_key": inputs.api_key,
            }.items()
            if value is not None
        }

        # Resolve profile
        resolved_config = llm_config_loader.resolve_profile(
            profile=inputs.profile, inline_overrides=inline_overrides
        )

        if resolved_config is None:
            raise ValueError("Profile resolution returned None unexpectedly")

        # Resolve API key from secrets
        api_key = None
        if resolved_config.api_key_secret:
            from .secrets import EnvVarSecretProvider

            secret_provider = EnvVarSecretProvider()
            api_key = await secret_provider.get_secret(resolved_config.api_key_secret)

        # Map provider type
        provider_type = self._map_provider_type(resolved_config.provider, resolved_config.api_url)

        return ImageGenInput(
            profile=None,
            provider=provider_type,
            model=resolved_config.model,
            prompt=inputs.prompt,
            api_key=api_key,
            api_url=resolved_config.api_url,
            operation=inputs.operation,
            size=inputs.size,
            quality=inputs.quality,
            style=inputs.style,
            response_format=inputs.response_format,
            n=inputs.n,
            image=inputs.image,
            mask=inputs.mask,
            output_file=inputs.output_file,
        )

    def _resolve_profile_with_fallback(
        self,
        inputs: ImageGenInput,
        llm_config_loader: LLMConfigLoader,
    ) -> str | None:
        """Resolve profile with fallback to default_profile."""
        # 1. Direct provider specified -> None
        if inputs.provider is not None:
            return None

        # 2. Profile specified -> use it
        if inputs.profile is not None:
            return inputs.profile

        # 3. Neither specified -> try default profile
        default_profile = llm_config_loader.get_default_profile()
        if default_profile:
            logger.warning(
                f"No provider/profile specified for ImageGen. "
                f"Using default profile: {default_profile}"
            )
            return default_profile

        # 4. No profile, no provider, no default -> Error
        # But wait, we want to support default behavior if user just wants to use OpenAI?
        # If we error here, we break "easy start".
        # But LLMCallExecutor errors here.
        # I'll return None and let execute handle it (defaulting to OpenAI).
        return None

    def _map_provider_type(self, provider_type: str, api_url: str | None) -> ImageProvider:
        """Map LLM config provider type to ImageProvider."""
        if provider_type == "openai" or provider_type == "azure-openai":
            if api_url:
                # Custom API URL - treat as OpenAI-compatible
                return ImageProvider.OPENAI_COMPATIBLE
            return ImageProvider.OPENAI

        # Assume any other provider type with an API URL is compatible
        if api_url:
            return ImageProvider.OPENAI_COMPATIBLE

        # Fallback
        return ImageProvider.OPENAI

    async def _call_openai(
        self, inputs: ImageGenInput, n: int, base_url: str | None
    ) -> ImageGenOutput:
        """Call OpenAI or compatible API."""
        # Resolve API key
        api_key = inputs.api_key
        if not api_key:
            # For compatible providers, key might not be needed, but client requires one
            api_key = "sk-no-key-required"

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Determine which operations and parameters this model supports
        capabilities = _get_model_capabilities(inputs.model)

        # Validate that the model supports this operation
        if inputs.operation not in capabilities["operations"]:
            supported_ops = ", ".join(sorted(capabilities["operations"]))
            raise ValueError(
                f"Model '{inputs.model}' does not support '{inputs.operation}' operation. "
                f"Supported operations: {supported_ops}"
            )

        try:
            response = None
            if inputs.operation == "generate":
                # Build kwargs with only supported parameters
                kwargs = {
                    "prompt": inputs.prompt,
                    "model": inputs.model,
                    "n": n,
                    "size": inputs.size,
                }

                # Add optional parameters only if model supports them
                if "response_format" in capabilities["params"]:
                    kwargs["response_format"] = inputs.response_format
                if "quality" in capabilities["params"] and inputs.quality:
                    kwargs["quality"] = inputs.quality
                if "style" in capabilities["params"] and inputs.style:
                    kwargs["style"] = inputs.style

                response = await client.images.generate(**kwargs)

            elif inputs.operation == "edit":
                if not inputs.image:
                    raise ValueError("Image path required for edit operation")

                image_path = Path(inputs.image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                mask_path = Path(inputs.mask) if inputs.mask else None
                if mask_path and not mask_path.exists():
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")

                # Open files
                with open(image_path, "rb") as img_file:
                    mask_file = open(mask_path, "rb") if mask_path else None
                    try:
                        edit_kwargs = {
                            "image": img_file,
                            "prompt": inputs.prompt,
                            "model": inputs.model,
                            "n": n,
                            "size": inputs.size,
                        }
                        # Only include mask if provided
                        if mask_file:
                            edit_kwargs["mask"] = mask_file
                        if "response_format" in capabilities["params"]:
                            edit_kwargs["response_format"] = inputs.response_format

                        response = await client.images.edit(**edit_kwargs)
                    finally:
                        if mask_file:
                            mask_file.close()

            elif inputs.operation == "variation":
                if not inputs.image:
                    raise ValueError("Image path required for variation operation")

                image_path = Path(inputs.image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                with open(image_path, "rb") as img_file:
                    variation_kwargs = {
                        "image": img_file,
                        "model": inputs.model,
                        "n": n,
                        "size": inputs.size,
                    }
                    if "response_format" in capabilities["params"]:
                        variation_kwargs["response_format"] = inputs.response_format

                    response = await client.images.create_variation(**variation_kwargs)

            else:
                raise ValueError(f"Unknown operation: {inputs.operation}")

            # Process response
            urls = []
            b64_json = []
            revised_prompts = []
            saved_files = []

            for i, data in enumerate(response.data):
                if data.url:
                    urls.append(data.url)
                if data.b64_json:
                    b64_json.append(data.b64_json)
                if data.revised_prompt:
                    revised_prompts.append(data.revised_prompt)

                # Save to file if requested
                if inputs.output_file:
                    # Determine filename
                    output_path = Path(inputs.output_file)
                    if n > 1:
                        # Append index: image.png -> image_0.png
                        stem = output_path.stem
                        suffix = output_path.suffix
                        output_path = output_path.with_name(f"{stem}_{i}{suffix}")

                    # Resolve path
                    path_result = PathResolver.resolve_and_validate(
                        str(output_path), allow_traversal=True
                    )
                    if not path_result.is_success:
                        logger.warning(f"Invalid output path {output_path}: {path_result.error}")
                        continue

                    resolved_path = path_result.value
                    assert resolved_path is not None

                    # Get content
                    content_bytes = None
                    if data.b64_json:
                        content_bytes = base64.b64decode(data.b64_json)
                    elif data.url:
                        # Download URL
                        async with httpx.AsyncClient() as http_client:
                            resp = await http_client.get(data.url)
                            resp.raise_for_status()
                            content_bytes = resp.content

                    if content_bytes:
                        # Write to file
                        # Ensure parent exists
                        if not resolved_path.parent.exists():
                            resolved_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(resolved_path, "wb") as f:
                            f.write(content_bytes)

                        saved_files.append(str(resolved_path))

            return ImageGenOutput(
                urls=urls,
                b64_json=b64_json,
                revised_prompts=revised_prompts,
                saved_files=saved_files,
                success=True,
                provider_metadata={"created": response.created},
            )

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
