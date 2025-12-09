#!/usr/bin/env python3
"""Test nested Workflow blocks in for_each with pause/resume.

This test reproduces a bug where:
1. Parent workflow has for_each: sequential with Workflow blocks
2. Child workflow pauses (Prompt block)
3. User provides feedback/response
4. Response is incorrectly routed to parent instead of child

The bug causes:
- Child workflow never completes its approval flow
- Remaining for_each iterations are skipped
- Sub-workflows are never saved

Test Scenario:
1. Parent workflow with for_each: sequential calling child workflows
2. Child workflows have Prompt blocks that pause for approval
3. Resume first child with "yes"
4. Verify first child completes with approval
5. Second child should pause for its approval
6. Resume second child with "yes"
7. Verify both children completed successfully
"""

import json
from typing import Any

import pytest
from mcp.types import TextContent
from test_mcp_client import get_mcp_client


class TestNestedWorkflowPauseResume:
    """Test pause/resume for nested Workflow blocks in for_each loops."""

    @pytest.mark.asyncio
    async def test_nested_workflow_in_foreach_sequential_pause_resume(self) -> None:
        """Test that nested workflow pause/resume works correctly in for_each sequential.

        This is the core bug reproduction test:
        1. Start parent workflow with 2 items
        2. First child pauses for approval
        3. Resume with "yes" -> first child should complete
        4. Second child should pause for approval (not parent!)
        5. Resume with "yes" -> second child should complete
        6. Parent workflow should complete with all items processed
        """
        async with get_mcp_client() as client:
            # Step 1: Start parent workflow
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "nested-workflow-in-foreach-parent",
                    "inputs": {
                        "work_items": [
                            {"name": "item1", "value": "value1"},
                            {"name": "item2", "value": "value2"},
                        ]
                    },
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)

            # Verify workflow paused at first child
            assert exec_response["status"] == "paused", (
                f"Expected workflow to pause, got status: {exec_response.get('status')}"
            )
            assert "job_id" in exec_response, "Expected job_id in paused response"
            job_id = exec_response["job_id"]

            # The prompt should be from the CHILD workflow (item1)
            prompt = exec_response.get("prompt", "")
            assert "item1" in prompt, (
                f"Expected prompt to mention 'item1' (first child), got: {prompt}"
            )

            # Step 2: Resume first child with "yes"
            resume1_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "yes",
                    "debug": True,
                },
            )

            resume1_content = resume1_result.content[0]
            assert isinstance(resume1_content, TextContent)
            resume1_response: dict[str, Any] = json.loads(resume1_content.text)

            # Step 3: Should pause again for SECOND child (item2)
            # THIS IS WHERE THE BUG MANIFESTS:
            # - Bug behavior: status == "success" (parent completes, skipping item2)
            # - Correct behavior: status == "paused" with prompt for item2
            assert resume1_response["status"] == "paused", (
                f"Expected workflow to pause for item2, got: {resume1_response.get('status')}. "
                f"Resume response was incorrectly routed to parent workflow. "
                f"Full response: {resume1_response}"
            )

            # Verify prompt is for second child (item2)
            prompt2 = resume1_response.get("prompt", "")
            assert "item2" in prompt2, (
                f"Expected prompt to mention 'item2' (second child), got: {prompt2}. "
                f"This indicates for_each iteration was not correctly resumed."
            )

            job_id_2 = resume1_response.get("job_id", job_id)

            # Step 4: Resume second child with "yes"
            resume2_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id_2,
                    "response": "yes",
                    "debug": True,
                },
            )

            resume2_content = resume2_result.content[0]
            assert isinstance(resume2_content, TextContent)
            resume2_response: dict[str, Any] = json.loads(resume2_content.text)

            # Step 5: Now workflow should complete successfully
            assert resume2_response["status"] == "success", (
                f"Expected workflow to complete after both children approved, "
                f"got status: {resume2_response.get('status')}. "
                f"Error: {resume2_response.get('error')}"
            )

            # Verify outputs
            outputs = resume2_response.get("outputs", {})
            assert outputs.get("setup_completed") is True
            assert outputs.get("all_items_processed") is True
            assert outputs.get("finalize_completed") is True

    @pytest.mark.asyncio
    async def test_nested_workflow_single_item_pause_resume(self) -> None:
        """Test simpler case: single item for_each with nested workflow pause.

        This isolates the core pause/resume logic without multiple iterations.
        """
        async with get_mcp_client() as client:
            # Start with single item
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "nested-workflow-in-foreach-parent",
                    "inputs": {
                        "work_items": [
                            {"name": "single_item", "value": "single_value"},
                        ]
                    },
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)

            # Should pause for the single child
            assert exec_response["status"] == "paused"
            job_id = exec_response["job_id"]

            prompt = exec_response.get("prompt", "")
            assert "single_item" in prompt

            # Resume with approval
            resume_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "yes",
                    "debug": True,
                },
            )

            resume_content = resume_result.content[0]
            assert isinstance(resume_content, TextContent)
            resume_response: dict[str, Any] = json.loads(resume_content.text)

            # Should complete (only one item)
            assert resume_response["status"] == "success", (
                f"Expected success after single item approval, got: {resume_response}"
            )

            outputs = resume_response.get("outputs", {})
            assert outputs.get("all_items_processed") is True

    @pytest.mark.asyncio
    async def test_nested_workflow_feedback_iteration(self) -> None:
        """Test that feedback (non-approval response) triggers re-design iteration.

        This simulates the workflow-creator flow where:
        1. First response is feedback ("add more details")
        2. Second response is approval ("approve")
        """
        async with get_mcp_client() as client:
            # Start workflow
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "nested-workflow-in-foreach-parent",
                    "inputs": {
                        "work_items": [
                            {"name": "test_item", "value": "test_value"},
                        ]
                    },
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)

            assert exec_response["status"] == "paused"
            job_id = exec_response["job_id"]

            # Send denial (should complete with denied branch)
            resume_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "no",
                    "debug": True,
                },
            )

            resume_content = resume_result.content[0]
            assert isinstance(resume_content, TextContent)
            resume_response: dict[str, Any] = json.loads(resume_content.text)

            # Should complete (denial is a valid response that completes the workflow)
            assert resume_response["status"] == "success", (
                f"Expected success after denial, got: {resume_response}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
