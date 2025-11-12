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
class ExecutionState:
    """Runtime execution state needed to resume paused workflows.

    This contains all the information needed to reconstruct the workflow
    execution context and continue from where it paused.

    Unified Job Architecture:
    - Stored in Job.result["execution_state"] when workflow pauses
    - Replaces separate CheckpointState storage
    - Works for both sync and async execution modes
    """

    context: Execution  # Full execution context with block results
    completed_blocks: list[str]  # IDs of blocks that have finished
    current_wave_index: int  # Current position in DAG execution
    execution_waves: list[list[str]]  # DAG wave structure
    block_definitions: dict[str, Any]  # Block configuration data
    workflow_stack: list[dict[str, Any]]  # Stack for workflow composition
    paused_block_id: str  # ID of block that triggered pause
    workflow_name: str  # Name of workflow being executed
    runtime_inputs: dict[str, Any]  # Original inputs provided to workflow
    pause_metadata: dict[str, Any] | None = None  # Pause metadata from ExecutionPaused


@dataclass
class PauseData:
    """Pause-specific metadata for paused workflows."""

    prompt: str
    metadata: dict[str, Any] | None = None


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
    execution_state: ExecutionState | None = None  # Runtime state for resume (paused only)
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
        prompt: str,
        execution: Execution,
        execution_state: ExecutionState,
        pause_metadata: dict[str, Any] | None = None,
        secret_redactor: SecretRedactor | None = None,
    ) -> ExecutionResult:
        """
        Create paused result with execution state for resume.

        Unified Job Architecture:
        - execution_state contains all runtime state needed for resume
        - Stored in Job.result["execution_state"] when workflow pauses
        - No separate checkpoint storage needed

        Args:
            prompt: Prompt to show user
            execution: Current execution context
            execution_state: Complete runtime state for resume
            pause_metadata: Optional pause-specific metadata
            secret_redactor: Optional secret redactor for debug file sanitization

        Returns:
            ExecutionResult with status="paused"
        """
        pause_data = PauseData(
            prompt=prompt,
            metadata=pause_metadata,
        )
        return ExecutionResult(
            status="paused",
            execution=execution,
            error=None,
            pause_data=pause_data,
            execution_state=execution_state,
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
            # Note: job_id added by caller (tools.py) after creating Job
            {"status": "paused", "prompt": "..."}

            # Paused - debug mode (debug=True)
            # Note: job_id added by caller (tools.py) after creating Job
            {"status": "paused", "prompt": "...", "logfile": "/tmp/workflow-123.json"}
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
            response["prompt"] = self.pause_data.prompt
            # Note: checkpoint_id/job_id not included here - added by caller (tools.py)
            # Caller will create Job and return job_id for resume

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

    def _execution_state_to_dict(self) -> dict[str, Any] | None:
        """Convert ExecutionState to dict format for JSON serialization."""
        if not self.execution_state:
            return None

        return {
            "context": self.execution_state.context.model_dump(),
            "completed_blocks": self.execution_state.completed_blocks,
            "current_wave_index": self.execution_state.current_wave_index,
            "execution_waves": self.execution_state.execution_waves,
            "block_definitions": self.execution_state.block_definitions,
            "workflow_stack": self.execution_state.workflow_stack,
            "paused_block_id": self.execution_state.paused_block_id,
            "workflow_name": self.execution_state.workflow_name,
            "runtime_inputs": self.execution_state.runtime_inputs,
        }

    def _build_debug_data(self) -> dict[str, Any]:
        """
        Build debug data structure (canonical format for job persistence and debug files).

        This is the CANONICAL format for:
        - Job persistence (stored in job.result)
        - Debug files (written to /tmp/)
        - API responses (detailed format)

        Format:
        {
            "status": "success" | "failure" | "paused",
            "outputs": {...},         # Workflow outputs (success only)
            "error": "...",           # Error message (failure only)
            "blocks": {...},          # All block execution details
            "metadata": {...},        # Workflow metadata (timing, counts)
            "pause_data": {...},      # Pause metadata (paused only)
            "prompt": "...",          # Prompt text (paused only)
            "execution_state": {...}  # Runtime state for resume (paused only)
        }

        SECURITY: Secrets are automatically redacted.

        Returns:
            Dict containing complete execution details with secrets redacted
        """
        # Build complete debug data structure
        data: dict[str, Any] = {
            "status": self.status,
            "outputs": self.execution.outputs if self.status == "success" else None,
            "error": self.error,
            "pause_data": self.pause_data.metadata if self.pause_data else None,
            "blocks": self._blocks_to_dict(),
            "metadata": self._metadata_to_dict(),
        }

        # Add pause-specific fields if paused
        if self.status == "paused":
            if self.pause_data:
                data["prompt"] = self.pause_data.prompt
            if self.execution_state:
                data["execution_state"] = self._execution_state_to_dict()

        # SECURITY: Redact secrets from entire data structure
        if self.secret_redactor:
            data = self.secret_redactor.redact(data)
            logger.debug("Secrets redacted from debug data")

        return data

    def _write_debug_file(self) -> str:
        """
        Write complete execution details to /tmp file for debugging.

        Uses _build_debug_data() to ensure consistent format with job persistence.

        SECURITY: All secrets are redacted before writing to prevent leakage.

        Returns:
            Path to the created debug file

        File Format:
            JSON file containing full execution details (see _build_debug_data())

        File Naming:
            /tmp/<workflow-name>-<timestamp-ms>.json
            Example: /tmp/python-ci-pipeline-1730000000123.json
        """
        # Get debug data (with secrets already redacted)
        debug_data = self._build_debug_data()

        # Extract workflow name from metadata for filename
        metadata_dict = self._metadata_to_dict()
        workflow_name = metadata_dict.get("workflow_name", "workflow")

        # Sanitize workflow name for filename (replace invalid chars with hyphen)
        safe_workflow_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in workflow_name)

        # Generate filename with timestamp
        timestamp_ms = int(datetime.now().timestamp() * 1000)
        filename = f"/tmp/{safe_workflow_name}-{timestamp_ms}.json"

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


__all__ = ["ExecutionResult", "ExecutionState", "PauseData"]
