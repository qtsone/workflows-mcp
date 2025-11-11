"""Tests for LLMCall executor.

Tests cover:
- Basic LLM calls for each provider
- Retry logic with exponential backoff
- JSON Schema validation with feedback loop
- Error handling and timeout behavior
- Token usage tracking
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import openai
import pytest

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.executors_llm import LLMCallExecutor, LLMCallInput, LLMCallOutput


class TestLLMCallExecutor:
    """Test suite for LLMCallExecutor."""

    @pytest.fixture
    def executor(self):
        """Create LLMCall executor instance."""
        return LLMCallExecutor()

    @pytest.fixture
    def mock_context(self):
        """Create mock execution context."""
        return Mock(spec=Execution)

    @pytest.mark.asyncio
    async def test_openai_basic_call(self, executor, mock_context):
        """Test basic OpenAI API call without schema validation."""
        inputs = LLMCallInput(
            provider="openai",
            model="gpt-4o",
            prompt="What is 2+2?",
            api_key="sk-test-key",
            timeout=10,
        )

        # Mock the OpenAI completion response
        mock_completion = Mock()
        mock_completion.choices = [
            Mock(
                message=Mock(content="The answer is 4.", refusal=None, tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_completion.model = "gpt-4o"
        mock_completion.usage = Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        with patch("workflows_mcp.engine.executors_llm.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_client_class.return_value = mock_client

            result = await executor.execute(inputs, mock_context)

            assert isinstance(result, LLMCallOutput)
            assert result.success is True
            assert result.response == {"content": "The answer is 4."}
            assert result.metadata["attempts"] == 1
            assert "validation_failed" not in result.metadata  # No schema requested
            assert result.metadata["model"] == "gpt-4o"
            assert result.metadata["usage"]["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_anthropic_basic_call(self, executor, mock_context):
        """Test basic Anthropic API call."""
        inputs = LLMCallInput(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            prompt="Hello!",
            api_key="sk-ant-test",
            system_instructions="You are a helpful assistant.",
            timeout=10,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "Hello! How can I help you today?"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
            },
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await executor.execute(inputs, mock_context)

            assert result.success is True
            assert result.response == {"content": "Hello! How can I help you today?"}
            assert result.metadata["attempts"] == 1
            assert result.metadata["model"] == "claude-3-5-sonnet-20241022"

    @pytest.mark.asyncio
    async def test_gemini_basic_call(self, executor, mock_context):
        """Test basic Gemini API call."""
        inputs = LLMCallInput(
            provider="gemini",
            model="gemini-2.0-flash-exp",
            prompt="Explain quantum computing",
            api_key="test-gemini-key",
            timeout=10,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Quantum computing uses quantum mechanics..."}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 50,
                "totalTokenCount": 65,
            },
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await executor.execute(inputs, mock_context)

            assert result.success is True
            assert "Quantum computing" in result.response["content"]
            assert result.metadata["attempts"] == 1

    @pytest.mark.asyncio
    async def test_ollama_basic_call(self, executor, mock_context):
        """Test basic Ollama local API call."""
        inputs = LLMCallInput(
            provider="ollama",
            model="llama2",
            prompt="Hello",
            api_url="http://localhost:11434/api/generate",
            timeout=60,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama2",
            "response": "Hello! How are you?",
            "total_duration": 1234567890,
            "load_duration": 123456,
            "prompt_eval_count": 5,
            "eval_count": 10,
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await executor.execute(inputs, mock_context)

            assert result.success is True
            assert result.response == {"content": "Hello! How are you?"}
            assert result.metadata["attempts"] == 1

    @pytest.mark.asyncio
    async def test_schema_validation_success(self, executor, mock_context):
        """Test successful JSON schema validation."""
        schema = {
            "type": "object",
            "required": ["answer", "confidence"],
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }

        inputs = LLMCallInput(
            provider="openai",
            model="gpt-4o",
            prompt="What is 2+2?",
            api_key="sk-test-key",
            response_schema=schema,
            timeout=10,
        )

        # Mock response with valid JSON matching schema
        valid_json_response = json.dumps({"answer": "4", "confidence": 0.99})

        mock_completion = Mock()
        mock_completion.choices = [
            Mock(
                message=Mock(content=valid_json_response, refusal=None, tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_completion.model = "gpt-4o"
        mock_completion.usage = Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        with patch("workflows_mcp.engine.executors_llm.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_client_class.return_value = mock_client

            result = await executor.execute(inputs, mock_context)

            assert result.success is True
            assert "validation_failed" not in result.metadata
            assert result.response["answer"] == "4"
            assert result.response["confidence"] == 0.99
            assert result.metadata["attempts"] == 1

    @pytest.mark.asyncio
    async def test_schema_validation_retry(self, executor, mock_context):
        """Test schema validation retry with feedback loop (Anthropic, no native schema)."""
        schema = {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        }

        inputs = LLMCallInput(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            prompt="What is 2+2?",
            api_key="sk-ant-test",
            response_schema=schema,
            max_retries=2,
            retry_delay=0.1,  # Fast retry for testing
            timeout=10,
        )

        # First response: invalid JSON
        # Second response: valid JSON
        responses = [
            "This is not JSON",  # First attempt - invalid
            json.dumps({"answer": "4"}),  # Second attempt - valid
        ]

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "content": [{"text": responses[call_count]}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            }
            call_count += 1
            return response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=mock_post)

            result = await executor.execute(inputs, mock_context)

            assert result.success is True
            assert "validation_failed" not in result.metadata
            assert result.metadata["attempts"] == 2
            assert result.response["answer"] == "4"

    @pytest.mark.asyncio
    async def test_schema_validation_failure_exhausted_retries(self, executor, mock_context):
        """Test schema validation failure after all retries exhausted (Anthropic)."""
        schema = {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        }

        inputs = LLMCallInput(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            prompt="What is 2+2?",
            api_key="sk-ant-test",
            response_schema=schema,
            max_retries=2,
            retry_delay=0.1,
            timeout=10,
        )

        # All responses invalid
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "This is not JSON"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await executor.execute(inputs, mock_context)

            # API call succeeded, but validation failed
            assert result.success is True
            assert result.metadata.get("validation_failed") is True
            assert result.response == {"content": "This is not JSON"}
            assert result.metadata["attempts"] == 2
            assert "validation_error" in result.metadata

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, executor, mock_context):
        """Test retry logic on timeout."""
        inputs = LLMCallInput(
            provider="openai",
            model="gpt-4o",
            prompt="Test",
            api_key="sk-test-key",
            max_retries=3,
            retry_delay=0.1,
            timeout=10,
        )

        # First two calls timeout, third succeeds
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Raise APITimeoutError instead of httpx.TimeoutException
                raise openai.APITimeoutError(request=Mock())

            mock_completion = Mock()
            mock_completion.choices = [
                Mock(
                    message=Mock(content="Success!", refusal=None, tool_calls=None),
                    finish_reason="stop",
                )
            ]
            mock_completion.model = "gpt-4o"
            mock_completion.usage = Mock(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
            return mock_completion

        with patch("workflows_mcp.engine.executors_llm.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
            mock_client_class.return_value = mock_client

            result = await executor.execute(inputs, mock_context)

            assert result.success is True
            assert result.response == {"content": "Success!"}
            assert result.metadata["attempts"] == 3

    @pytest.mark.asyncio
    async def test_timeout_exhausted_retries(self, executor, mock_context):
        """Test timeout with all retries exhausted."""
        inputs = LLMCallInput(
            provider="openai",
            model="gpt-4o",
            prompt="Test",
            api_key="sk-test-key",
            max_retries=2,
            retry_delay=0.1,
            timeout=10,
        )

        async def mock_create(*args, **kwargs):
            raise openai.APITimeoutError(request=Mock())

        with patch("workflows_mcp.engine.executors_llm.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
            mock_client_class.return_value = mock_client

            # OpenAI SDK errors are raised directly after retry exhaustion
            with pytest.raises(openai.APITimeoutError):
                await executor.execute(inputs, mock_context)

    @pytest.mark.asyncio
    async def test_http_error_retry(self, executor, mock_context):
        """Test retry on HTTP errors."""
        inputs = LLMCallInput(
            provider="openai",
            model="gpt-4o",
            prompt="Test",
            api_key="sk-test-key",
            max_retries=3,
            retry_delay=0.1,
            timeout=10,
        )

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                # Raise APIStatusError for HTTP 500
                raise openai.APIStatusError(
                    message="Server error",
                    response=Mock(status_code=500),
                    body=None,
                )
            else:
                mock_completion = Mock()
                mock_completion.choices = [
                    Mock(
                        message=Mock(content="Success after retry!", refusal=None, tool_calls=None),
                        finish_reason="stop",
                    )
                ]
                mock_completion.model = "gpt-4o"
                mock_completion.usage = Mock(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                )
                return mock_completion

        with patch("workflows_mcp.engine.executors_llm.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
            mock_client_class.return_value = mock_client

            result = await executor.execute(inputs, mock_context)

            assert result.success is True
            assert result.response == {"content": "Success after retry!"}
            assert result.metadata["attempts"] == 2

    @pytest.mark.asyncio
    async def test_validate_response_not_json_object(self, executor):
        """Test validation failure when response is not a JSON object."""
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        # Test with JSON array instead of object
        with pytest.raises(ValueError, match="not a JSON object"):
            executor._validate_response('["array", "not", "object"]', schema)

        # Test with JSON primitive
        with pytest.raises(ValueError, match="not a JSON object"):
            executor._validate_response('"string"', schema)

    @pytest.mark.asyncio
    async def test_invalid_provider(self, executor, mock_context):
        """Test error handling for invalid provider."""
        inputs = LLMCallInput(
            provider="openai",  # Will be invalid after we mock
            model="test-model",
            prompt="Test",
            timeout=10,
        )

        # Override provider to invalid value
        inputs.provider = "invalid_provider"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Invalid provider"):
            await executor.execute(inputs, mock_context)

    def test_executor_metadata(self, executor):
        """Test executor type name and capabilities."""
        assert executor.type_name == "LLMCall"
        assert executor.input_type == LLMCallInput
        assert executor.output_type == LLMCallOutput
        assert executor.capabilities.can_network is True
