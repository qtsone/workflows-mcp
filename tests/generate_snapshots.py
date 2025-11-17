#!/usr/bin/env python3
"""Generate comprehensive execution snapshots for test workflows.

This script executes each test workflow via MCP protocol with detailed
response format and saves the complete execution details as JSON snapshot
files for test comparison and validation.

The detailed format includes:
- Complete block execution details (inputs, outputs, metadata)
- Execution timing and status information
- Full workflow metadata and state

Snapshots are NORMALIZED before saving to ensure stability:
- Dynamic fields (timestamps, IPs, secrets) are replaced with placeholders
- This ensures regenerating snapshots shows no diffs between runs
- Tests compare actual (normalized) responses against normalized snapshots
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from pytest_httpserver import HTTPServer
from test_secrets import setup_test_secrets
from test_utils import normalize_dynamic_fields
from werkzeug.wrappers import Response

# Configure test secrets (required for secrets-related workflows)
setup_test_secrets()

# Test workflows directory
TEST_WORKFLOWS_DIR = Path(__file__).parent / "workflows"
os.environ["WORKFLOWS_TEMPLATE_PATHS"] = str(TEST_WORKFLOWS_DIR)

# Snapshots output directory
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)

# HTTP workflows that need base_url injected
HTTP_WORKFLOWS = {
    "http-basic-get",
    "secrets-http-auth",
    "secrets-multiple-blocks",
    "core-secrets-management-test",
}

# LLM workflows that need api_url injected
LLM_WORKFLOWS = {
    "executor-llm-operations-test",
}


def create_httpbin_mock() -> HTTPServer:
    """Create HTTP mock server that mimics httpbin.org behavior."""
    httpserver = HTTPServer(host="127.0.0.1", port=0)
    httpserver.start()

    # GET endpoint - echoes back request information
    def get_handler(request):
        """Echo request information like httpbin.org/get."""
        data = {
            "args": dict(request.args),
            "headers": {k: v for k, v in request.headers},
            "origin": request.remote_addr or "127.0.0.1",
            "url": str(request.url),
        }
        return Response(json.dumps(data), content_type="application/json")

    httpserver.expect_request("/get").respond_with_handler(get_handler)

    # POST endpoint - echoes back JSON body and headers
    def post_handler(request):
        """Echo JSON body and headers like httpbin.org/post."""
        json_data = None
        if request.is_json:
            try:
                json_data = request.json
            except Exception:
                json_data = None

        data = {
            "json": json_data,
            "data": request.data.decode() if request.data else "",
            "headers": {k: v for k, v in request.headers},
            "origin": request.remote_addr or "127.0.0.1",
        }
        return Response(json.dumps(data), content_type="application/json")

    httpserver.expect_request("/post", method="POST").respond_with_handler(post_handler)

    # /headers endpoint - echoes back request headers (used in secrets tests)
    def headers_handler(request):
        """Echo request headers like httpbin.org/headers."""
        data = {"headers": {k: v for k, v in request.headers}}
        return Response(json.dumps(data), content_type="application/json")

    httpserver.expect_request("/headers").respond_with_handler(headers_handler)

    return httpserver


def create_llm_mock() -> HTTPServer:
    """Create LLM mock server that mimics OpenAI-compatible chat completion API."""
    llmserver = HTTPServer(host="127.0.0.1", port=0)
    llmserver.start()

    def chat_completion_handler(request):
        """Return OpenAI-compatible chat completion response."""
        # Parse request body
        if request.is_json and request.json is not None:
            request_data = dict(request.json)
        else:
            request_data = {}

        # Extract prompt from messages
        messages = request_data.get("messages", [])
        user_message = next((msg["content"] for msg in messages if msg.get("role") == "user"), "")

        # Generate response based on prompt
        if "capital" in user_message.lower() and "france" in user_message.lower():
            response_text = "Paris"
        elif "test" in user_message.lower():
            response_text = "This is a test response from the mock LLM server."
        else:
            response_text = f"Mock response to: {user_message[:50]}"

        # Return OpenAI-compatible response
        response_data = {
            "id": "chatcmpl-mock123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": request_data.get("model", "gpt-5-mini"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        return Response(json.dumps(response_data), content_type="application/json")

    # OpenAI SDK uses /v1/chat/completions or /chat/completions
    llmserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(
        chat_completion_handler
    )
    llmserver.expect_request("/chat/completions", method="POST").respond_with_handler(
        chat_completion_handler
    )

    return llmserver


@asynccontextmanager
async def get_mcp_client() -> AsyncIterator[ClientSession]:
    """Context manager providing MCP client connected to server via stdio."""
    # Suppress MCP server logs by setting log level
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "workflows_mcp"],
        env={
            **os.environ,
            "WORKFLOWS_TEMPLATE_PATHS": str(TEST_WORKFLOWS_DIR),
            "WORKFLOWS_LOG_LEVEL": "WARNING",  # Suppress INFO logs
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def generate_snapshot(
    workflow_name: str, client: ClientSession, inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute workflow with detailed format and return comprehensive response.

    Uses detailed response format to capture:
    - Complete block execution details
    - Full metadata about execution timing and state
    - Comprehensive outputs and block-level information

    Response is NORMALIZED before returning to ensure:
    - Dynamic fields (timestamps, IPs, secrets) replaced with placeholders
    - Snapshots are stable and regenerating shows no diffs
    - Tests can reliably compare actual vs expected responses
    """
    result = await client.call_tool(
        "execute_workflow",
        arguments={
            "workflow": workflow_name,
            "inputs": inputs or {},
            "debug": False,  # Minimal response (status + outputs/error only)
        },
    )

    # Extract text content from result
    content = result.content[0]
    if isinstance(content, TextContent):
        response: dict[str, Any] = json.loads(content.text)

        # Normalize dynamic fields (timestamps, IPs, secrets) for stable snapshots
        # Even minimal responses can contain dynamic data in workflow outputs
        # Example: variable-resolution-metadata outputs {{metadata.start_time}}
        normalized_response = normalize_dynamic_fields(response)

        return normalized_response
    else:
        raise ValueError(f"Expected TextContent, got {type(content)}")


async def main() -> None:
    """Generate all workflow snapshots."""
    print("=" * 80)
    print("WORKFLOW SNAPSHOT GENERATOR")
    print("=" * 80)
    print("\n🚀 Starting mock servers...", flush=True)

    # Create HTTP mock server for reliable testing (no external dependencies)
    httpserver = create_httpbin_mock()
    base_url = httpserver.url_for("/").rstrip("/")
    print(f"✅ HTTP mock server started at {base_url}")

    # Create LLM mock server for LLM workflow testing
    llmserver = create_llm_mock()
    llm_api_url = llmserver.url_for("/").rstrip("/")
    print(f"✅ LLM mock server started at {llm_api_url}\n")

    try:
        print("🚀 Starting MCP server...", flush=True)

        async with get_mcp_client() as client:
            # Get test workflows using tag filter (single source of truth)
            print("🔍 Discovering test workflows from MCP server (tag='test')...", flush=True)
            result = await client.call_tool("list_workflows", arguments={"tags": ["test"]})
            content = result.content[0]
            if isinstance(content, TextContent):
                # list_workflows returns a JSON list directly, not a dict
                all_workflows: list[str] = json.loads(content.text)
            else:
                raise ValueError(f"Expected TextContent, got {type(content)}")

            # Exclude interactive workflows (they require resume_workflow)
            # This matches pytest_generate_tests behavior in test_mcp_client.py
            print(
                f"📋 Found {len(all_workflows)} workflows with tag='test', filtering...",
                flush=True,
            )
            workflows: list[str] = []
            for wf_name in all_workflows:
                info_result = await client.call_tool(
                    "get_workflow_info", arguments={"workflow": wf_name, "format": "json"}
                )
                info_content = info_result.content[0]
                if not isinstance(info_content, TextContent):
                    continue

                workflow_info = json.loads(info_content.text)
                tags = workflow_info.get("tags", [])

                # Exclude workflows tagged as 'interactive' (cannot complete via normal execution)
                if "interactive" not in tags:
                    workflows.append(wf_name)
                else:
                    print(f"   ⏭️  Skipping interactive workflow: {wf_name}", flush=True)

            print(f"✅ Discovered {len(workflows)} non-interactive test workflows\n")
            print(f"📁 Snapshots will be saved to: {SNAPSHOTS_DIR}\n")
            print("=" * 80)
            print(f"EXECUTING {len(workflows)} WORKFLOWS")
            print("=" * 80 + "\n")

            success_count = 0
            failure_count = 0

            for i, workflow_name in enumerate(workflows, 1):
                print(f"[{i}/{len(workflows)}] {workflow_name}", flush=True)
                try:
                    # Prepare inputs for HTTP workflows
                    inputs: dict[str, Any] = {}
                    if workflow_name in HTTP_WORKFLOWS:
                        inputs["base_url"] = base_url
                    elif workflow_name in LLM_WORKFLOWS:
                        inputs["api_url"] = llm_api_url

                    if workflow_name == "secrets-redaction":
                        import hashlib

                        from test_secrets import TEST_SECRETS

                        # Calculate expected hash from actual test secret (single source of truth)
                        secret_value = TEST_SECRETS["WORKFLOW_SECRET_REDACTION_TEST"]
                        inputs["expected_hash"] = hashlib.sha256(secret_value.encode()).hexdigest()

                    # Execute workflow
                    response = await generate_snapshot(workflow_name, client, inputs)

                    # Save snapshot
                    snapshot_file = SNAPSHOTS_DIR / f"{workflow_name}.json"
                    with open(snapshot_file, "w") as f:
                        json.dump(response, f, indent=2)

                    # Show status
                    status = response.get("status", "unknown")
                    if status == "success":
                        print(f"    ✅ Success - Saved to {snapshot_file.name}")
                        success_count += 1
                    else:
                        print(f"    ⚠️  Status: {status}")
                        if "error" in response:
                            print(f"    Error: {response['error']}")
                        failure_count += 1
                    print()

                except Exception as e:
                    print(f"    ❌ Failed: {e}\n")
                    failure_count += 1

            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"✅ Successful: {success_count}")
            print(f"❌ Failed: {failure_count}")
            print(f"📊 Total: {len(workflows)}")
            print(f"📁 Snapshots saved to: {SNAPSHOTS_DIR}")
            print("=" * 80 + "\n")

    finally:
        # Clean up mock servers
        httpserver.stop()
        llmserver.stop()
        print("🛑 Mock servers stopped")


if __name__ == "__main__":
    asyncio.run(main())
