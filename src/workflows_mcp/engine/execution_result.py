"""
Execution result monad for workflow operations.

Aligned with LoadResult pattern - provides type-safe result handling
with full execution context preservation for debugging.

Design Principles:
- Execution field ALWAYS present (never None)
- Status determines interpretation (success/failure/paused)
- Factory methods ensure valid state combinations
- to_response() is single source of truth for formatting
- Debug mode writes full details to /tmp file (minimal LLM context)
- Secrets are redacted from debug files to prevent leakage
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from .execution import Execution
from .secrets.redactor import SecretRedactor

logger = logging.getLogger(__name__)


@dataclass
class PauseData:
    """Pause-specific metadata for paused workflows."""

    checkpoint_id: str
    prompt: str
    metadata: dict[str, Any] | None = None

    @property
    def resume_message(self) -> str:
        """Generate user-friendly resume instruction."""
        return (
            f'Use resume_workflow(checkpoint_id: "{self.checkpoint_id}", '
            f'response: "<your-response>") to continue'
        )


@dataclass
class ExecutionResult:
    """
    Monad for workflow execution results (success/failure/paused).

    Inspired by LoadResult pattern - ensures execution context is ALWAYS
    preserved regardless of outcome. This enables:

    - Rich error debugging (partial execution state on failures)
    - Consistent response formatting (single code path for all statuses)
    - Type-safe result handling (factory methods prevent invalid states)
    - Secret redaction in debug files (prevents credential leakage)

    Example Usage:
        # Success
        result = ExecutionResult.success(execution, secret_redactor)

        # Failure with partial execution
        result = ExecutionResult.failure("Build failed", partial_execution, secret_redactor)

        # Paused for user input
        result = ExecutionResult.paused(checkpoint_id, prompt, execution, secret_redactor)

        # Format for MCP tool
        return result.to_response(debug=True)  # Secrets redacted in debug file
    """

    status: Literal["success", "failure", "paused"]
    execution: Execution  # ALWAYS present - complete execution context
    error: str | None = None
    pause_data: PauseData | None = None
    secret_redactor: SecretRedactor | None = None  # For debug file redaction

    # Factory Methods (Type-Safe Construction)

    @staticmethod
    def success(
        execution: Execution, secret_redactor: SecretRedactor | None = None
    ) -> ExecutionResult:
        """
        Create success result with complete execution.

        Args:
            execution: Complete execution context with outputs
            secret_redactor: Optional secret redactor for debug file sanitization

        Returns:
            ExecutionResult with status="success"
        """
        return ExecutionResult(
            status="success",
            execution=execution,
            error=None,
            pause_data=None,
            secret_redactor=secret_redactor,
        )

    @staticmethod
    def failure(
        error: str, partial_execution: Execution, secret_redactor: SecretRedactor | None = None
    ) -> ExecutionResult:
        """
        Create failure result with partial execution for debugging.

        Unlike traditional error handling that loses context, this preserves:
        - All completed blocks up to point of failure
        - Execution metadata (start time, wave count, etc.)
        - Partial outputs from successful blocks

        Args:
            error: Error message describing what failed
            partial_execution: Execution context up to failure point
            secret_redactor: Optional secret redactor for debug file sanitization

        Returns:
            ExecutionResult with status="failure" and full debug context
        """
        return ExecutionResult(
            status="failure",
            execution=partial_execution,
            error=error,
            pause_data=None,
            secret_redactor=secret_redactor,
        )

    @staticmethod
    def paused(
        checkpoint_id: str,
        prompt: str,
        execution: Execution,
        pause_metadata: dict[str, Any] | None = None,
        secret_redactor: SecretRedactor | None = None,
    ) -> ExecutionResult:
        """
        Create paused result with execution state for resume.

        Args:
            checkpoint_id: Checkpoint ID for resuming
            prompt: Prompt to show user
            execution: Current execution state
            pause_metadata: Optional pause-specific metadata
            secret_redactor: Optional secret redactor for debug file sanitization

        Returns:
            ExecutionResult with status="paused"
        """
        pause_data = PauseData(
            checkpoint_id=checkpoint_id,
            prompt=prompt,
            metadata=pause_metadata,
        )
        return ExecutionResult(
            status="paused",
            execution=execution,
            error=None,
            pause_data=pause_data,
            secret_redactor=secret_redactor,
        )

    # Formatting Methods

    def to_response(self, debug: bool = False) -> dict[str, Any]:
        """
        Format execution result for MCP tool response.

        Single source of truth for ALL response formatting across all statuses.
        Handles success, failure, and paused states consistently.

        Formatting Rules:
        - debug=False: Minimal response (status + essential fields only)
        - debug=True: Minimal response + logfile path (full details written to /tmp)

        Args:
            debug: Enable debug mode (writes full execution details to file)

        Returns:
            Dict ready for MCP tool return

        Examples:
            # Success - minimal (debug=False)
            {"status": "success", "outputs": {...}}

            # Success - debug mode (debug=True)
            {"status": "success", "outputs": {...}, "logfile": "/tmp/workflow-123.json"}

            # Failure - minimal (debug=False)
            {"status": "failure", "error": "..."}

            # Failure - debug mode (debug=True, includes full partial execution!)
            {"status": "failure", "error": "...", "logfile": "/tmp/workflow-123.json"}

            # Paused - minimal (debug=False)
            {"status": "paused", "checkpoint_id": "...", "prompt": "..."}

            # Paused - debug mode (debug=True)
            {"status": "paused", "checkpoint_id": "...", "prompt": "...",
             "logfile": "/tmp/workflow-123.json"}
        """
        # Base response (status always included)
        response: dict[str, Any] = {
            "status": self.status,
        }

        # Add status-specific essential fields
        if self.status == "success":
            response["outputs"] = self.execution.outputs

        elif self.status == "failure":
            response["error"] = self.error
            # Note: outputs is None for failures (not applicable)

        elif self.status == "paused":
            assert self.pause_data is not None
            response["checkpoint_id"] = self.pause_data.checkpoint_id
            response["prompt"] = self.pause_data.prompt
            response["message"] = self.pause_data.resume_message

        # Add debug logfile if requested (works for ALL statuses!)
        if debug:
            logfile_path = self._write_debug_file()
            response["logfile"] = logfile_path

        return response

    def _blocks_to_dict(self) -> dict[str, Any]:
        """Convert execution.blocks to dict format for JSON serialization."""
        blocks_dict = {}
        for block_id, block_exec in self.execution.blocks.items():
            if isinstance(block_exec, Execution):
                # Nested execution (from WorkflowExecutor)
                blocks_dict[block_id] = block_exec.model_dump()
            else:
                blocks_dict[block_id] = block_exec
        return blocks_dict

    def _metadata_to_dict(self) -> dict[str, Any]:
        """Convert execution workflow metadata to dict format for JSON serialization."""
        # Get workflow metadata from typed accessor
        return self.execution.workflow_metadata

    def _write_debug_file(self) -> str:
        """
        Write complete execution details to /tmp file for debugging.

        SECURITY: All secrets are redacted before writing to prevent leakage.

        Returns:
            Path to the created debug file

        File Format:
            JSON file containing full execution details:
            - status (success/failure/paused)
            - outputs (workflow outputs if success)
            - error (error message if failure)
            - pause_data (pause metadata if paused)
            - blocks (all block execution details - REDACTED)
            - metadata (workflow execution metadata)

        File Naming:
            /tmp/<workflow-name>-<timestamp-ms>.json
            Example: /tmp/python-ci-pipeline-1730000000123.json
        """
        # Extract workflow name from metadata
        metadata_dict = self._metadata_to_dict()
        workflow_name = metadata_dict.get("workflow_name", "workflow")

        # Sanitize workflow name for filename (replace invalid chars with hyphen)
        safe_workflow_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in workflow_name)

        # Generate filename with timestamp
        timestamp_ms = int(datetime.now().timestamp() * 1000)
        filename = f"/tmp/{safe_workflow_name}-{timestamp_ms}.json"

        # Build complete debug data (same structure as old "detailed" format)
        debug_data: dict[str, Any] = {
            "status": self.status,
            "outputs": self.execution.outputs if self.status == "success" else None,
            "error": self.error,
            "pause_data": self.pause_data.metadata if self.pause_data else None,
            "blocks": self._blocks_to_dict(),
            "metadata": metadata_dict,
        }

        # Add pause-specific fields if paused
        if self.status == "paused" and self.pause_data:
            debug_data["checkpoint_id"] = self.pause_data.checkpoint_id
            debug_data["prompt"] = self.pause_data.prompt

        # SECURITY: Redact secrets from entire debug data structure
        if self.secret_redactor:
            debug_data = self.secret_redactor.redact(debug_data)
            logger.debug("Secrets redacted from debug file")

        # Write to file with pretty formatting
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, default=str, ensure_ascii=False)

            logger.info(f"Debug file written: {filename}")
            return filename

        except Exception as e:
            # Fallback: log error and return error message instead of path
            logger.error(f"Failed to write debug file {filename}: {e}")
            return f"ERROR: Failed to write debug file: {e}"


__all__ = ["ExecutionResult", "PauseData"]
