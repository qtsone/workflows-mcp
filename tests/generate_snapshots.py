#!/usr/bin/env python3
"""Generate comprehensive execution snapshots for test workflows.

This script executes each test workflow via MCP protocol with detailed
response format and saves the complete execution details as JSON snapshot
files for test comparison and validation.

The detailed format includes:
- Complete block execution details (inputs, outputs, metadata)
- Execution timing and status information
- Full workflow metadata and state
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

# Test workflows directory
TEST_WORKFLOWS_DIR = Path(__file__).parent / "workflows"
os.environ["WORKFLOWS_TEMPLATE_PATHS"] = str(TEST_WORKFLOWS_DIR)

# Snapshots output directory
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)


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


async def generate_snapshot(workflow_name: str, client: ClientSession) -> dict[str, Any]:
    """Execute workflow with detailed format and return comprehensive response.

    Uses detailed response format to capture:
    - Complete block execution details
    - Full metadata about execution timing and state
    - Comprehensive outputs and block-level information
    """
    result = await client.call_tool(
        "execute_workflow",
        arguments={
            "workflow": workflow_name,
            "inputs": {},
            "response_format": "detailed",
        },
    )

    # Extract text content from result
    content = result.content[0]
    if isinstance(content, TextContent):
        response: dict[str, Any] = json.loads(content.text)
        return response
    else:
        raise ValueError(f"Expected TextContent, got {type(content)}")


async def main() -> None:
    """Generate all workflow snapshots."""
    print("=" * 80)
    print("WORKFLOW SNAPSHOT GENERATOR")
    print("=" * 80)
    print("\n🚀 Starting MCP server...", flush=True)

    async with get_mcp_client() as client:
        # Get test workflows using tag filter (single source of truth)
        print("🔍 Discovering test workflows from MCP server (tag='test')...", flush=True)
        result = await client.call_tool("list_workflows", arguments={"tags": ["test"]})
        content = result.content[0]
        if isinstance(content, TextContent):
            # list_workflows returns a JSON list directly, not a dict
            workflows: list[str] = json.loads(content.text)
        else:
            raise ValueError(f"Expected TextContent, got {type(content)}")

        print(f"✅ Discovered {len(workflows)} test workflows\n")
        print(f"📁 Snapshots will be saved to: {SNAPSHOTS_DIR}\n")
        print("=" * 80)
        print(f"EXECUTING {len(workflows)} WORKFLOWS")
        print("=" * 80 + "\n")

        success_count = 0
        failure_count = 0

        for i, workflow_name in enumerate(workflows, 1):
            print(f"[{i}/{len(workflows)}] {workflow_name}", flush=True)
            try:
                # Execute workflow
                response = await generate_snapshot(workflow_name, client)

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


if __name__ == "__main__":
    asyncio.run(main())
