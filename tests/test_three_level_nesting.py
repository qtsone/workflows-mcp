#!/usr/bin/env python3
"""Test 3-level nested workflow pause/resume.

This test reproduces a bug where:
1. Grandparent workflow has for_each calling parent workflows
2. Parent workflow has for_each calling child workflows AND blocks AFTER for_each
3. Child workflow has Prompt block that pauses
4. After child resumes, parent's remaining blocks (after for_each) don't run

Test Scenario:
    Grandparent
      ├── grandparent_setup (Shell)
      ├── process_parents (for_each → Parent)
      │     └── Parent
      │           ├── parent_setup (Shell)
      │           ├── process_children (for_each → Child)
      │           │     └── Child
      │           │           ├── child_setup (Shell)
      │           │           ├── child_approval (Prompt) ← PAUSES HERE
      │           │           └── child_complete (Shell)
      │           └── parent_complete (Shell) ← BUG: This doesn't run!
      └── grandparent_complete (Shell)

The bug causes parent_complete to never execute after child workflow resumes.
"""

import json
from typing import Any

import pytest
from mcp.types import TextContent
from test_mcp_client import get_mcp_client


class TestThreeLevelNestedWorkflow:
    """Test pause/resume for 3-level nested workflows."""

    @pytest.mark.asyncio
    async def test_three_level_nesting_blocks_after_foreach_run(self) -> None:
        """Test that blocks AFTER for_each in middle workflow run after child resumes.

        This is the core 3-level nesting bug reproduction test.
        """
        async with get_mcp_client() as client:
            # Step 1: Start grandparent workflow with minimal input
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "three-level-grandparent",
                    "inputs": {
                        "parents": [
                            {
                                "name": "parent1",
                                "children": ["child1"],
                            }
                        ]
                    },
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)

            # Step 2: Should pause at child workflow's Prompt
            assert exec_response["status"] == "paused", (
                f"Expected workflow to pause at child Prompt, got: {exec_response}"
            )
            assert "job_id" in exec_response
            job_id = exec_response["job_id"]

            # Prompt should be from the child workflow
            prompt = exec_response.get("prompt", "")
            assert "child1" in prompt, f"Expected prompt to mention 'child1', got: {prompt}"

            # Step 3: Resume child with approval
            resume_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "approved",
                    "debug": True,
                },
            )

            resume_content = resume_result.content[0]
            assert isinstance(resume_content, TextContent)
            resume_response: dict[str, Any] = json.loads(resume_content.text)

            # Step 4: Workflow should complete successfully
            assert resume_response["status"] == "success", (
                f"Expected workflow to complete after child approval, "
                f"got status: {resume_response.get('status')}. "
                f"Error: {resume_response.get('error')}"
            )

            # Step 5: Verify ALL blocks ran, including parent_complete
            outputs = resume_response.get("outputs", {})

            # Grandparent level
            assert outputs.get("grandparent_setup_completed") is True, (
                "grandparent_setup should have run"
            )
            assert outputs.get("all_parents_completed") is True, "all parents should have completed"
            assert outputs.get("grandparent_complete_ran") is True, (
                "grandparent_complete should have run"
            )

    @pytest.mark.asyncio
    async def test_three_level_multiple_children_sequential(self) -> None:
        """Test 3-level nesting with multiple children (sequential pause/resume).

        Each child pauses, and after all children complete,
        parent_complete should run.
        """
        async with get_mcp_client() as client:
            # Start with parent that has 2 children
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "three-level-grandparent",
                    "inputs": {
                        "parents": [
                            {
                                "name": "parent1",
                                "children": ["childA", "childB"],
                            }
                        ]
                    },
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)

            # Should pause at first child
            assert exec_response["status"] == "paused"
            job_id = exec_response["job_id"]
            prompt = exec_response.get("prompt", "")
            assert "childA" in prompt, f"Expected childA in prompt, got: {prompt}"

            # Resume first child
            resume1_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "approved A",
                    "debug": True,
                },
            )

            resume1_content = resume1_result.content[0]
            assert isinstance(resume1_content, TextContent)
            resume1_response: dict[str, Any] = json.loads(resume1_content.text)

            # Should pause at second child
            assert resume1_response["status"] == "paused", (
                f"Expected pause for childB, got: {resume1_response}"
            )
            job_id_2 = resume1_response.get("job_id", job_id)
            prompt2 = resume1_response.get("prompt", "")
            assert "childB" in prompt2, f"Expected childB in prompt, got: {prompt2}"

            # Resume second child
            resume2_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id_2,
                    "response": "approved B",
                    "debug": True,
                },
            )

            resume2_content = resume2_result.content[0]
            assert isinstance(resume2_content, TextContent)
            resume2_response: dict[str, Any] = json.loads(resume2_content.text)

            # Now should complete with all blocks having run
            assert resume2_response["status"] == "success", (
                f"Expected success after both children, got: {resume2_response}"
            )

            outputs = resume2_response.get("outputs", {})
            assert outputs.get("grandparent_complete_ran") is True

    @pytest.mark.asyncio
    async def test_three_level_multiple_parents_sequential(self) -> None:
        """Test 3-level nesting with multiple parents (each with children).

        This tests the full recursive flow:
        - 2 parents, each with 1 child
        - Each child pauses
        - After parent1's child completes, parent1_complete should run
        - Then parent2 starts and its child pauses
        """
        async with get_mcp_client() as client:
            exec_result = await client.call_tool(
                "execute_workflow",
                arguments={
                    "workflow": "three-level-grandparent",
                    "inputs": {
                        "parents": [
                            {"name": "parent1", "children": ["child1"]},
                            {"name": "parent2", "children": ["child2"]},
                        ]
                    },
                    "debug": True,
                },
            )

            exec_content = exec_result.content[0]
            assert isinstance(exec_content, TextContent)
            exec_response: dict[str, Any] = json.loads(exec_content.text)

            # Should pause at parent1's child
            assert exec_response["status"] == "paused"
            job_id = exec_response["job_id"]
            prompt = exec_response.get("prompt", "")
            assert "child1" in prompt and "parent1" in prompt

            # Resume parent1's child
            resume1_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id,
                    "response": "approved 1",
                    "debug": True,
                },
            )

            resume1_content = resume1_result.content[0]
            assert isinstance(resume1_content, TextContent)
            resume1_response: dict[str, Any] = json.loads(resume1_content.text)

            # Should pause at parent2's child
            # This also verifies parent1_complete ran (parent1 fully completed)
            assert resume1_response["status"] == "paused", (
                f"Expected pause for parent2's child, got: {resume1_response}"
            )
            job_id_2 = resume1_response.get("job_id", job_id)
            prompt2 = resume1_response.get("prompt", "")
            assert "child2" in prompt2 and "parent2" in prompt2, (
                f"Expected parent2/child2 in prompt, got: {prompt2}"
            )

            # Resume parent2's child
            resume2_result = await client.call_tool(
                "resume_workflow",
                arguments={
                    "job_id": job_id_2,
                    "response": "approved 2",
                    "debug": True,
                },
            )

            resume2_content = resume2_result.content[0]
            assert isinstance(resume2_content, TextContent)
            resume2_response: dict[str, Any] = json.loads(resume2_content.text)

            # Should complete fully
            assert resume2_response["status"] == "success", (
                f"Expected success, got: {resume2_response}"
            )

            outputs = resume2_response.get("outputs", {})
            assert outputs.get("grandparent_complete_ran") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
