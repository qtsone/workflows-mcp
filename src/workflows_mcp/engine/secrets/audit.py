"""Secret access audit logging for compliance and security monitoring.

This module provides comprehensive audit logging for secret access operations,
enabling compliance reporting, security monitoring, and debugging of secret usage.

The audit log tracks every secret access with contextual information including
workflow name, block ID, secret key, timestamp, and access result (success/failure).

Features:
    - Structured event logging with Pydantic models
    - Filtering by workflow, block, or secret key
    - Export to JSON format for external analysis
    - MCP-safe logging (stderr only, no stdout)
    - In-memory storage with optional persistence

Example:
    >>> audit_log = SecretAuditLog()
    >>> await audit_log.log_access(
    ...     workflow_name="deploy-app",
    ...     block_id="get_db_credentials",
    ...     secret_key="database_password",
    ...     success=True
    ... )
    >>> events = audit_log.get_events(workflow_name="deploy-app")
    >>> await audit_log.export_to_file("audit.json")
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SecretAccessEvent(BaseModel):
    """Represents a single secret access event in the audit log.

    This model captures all relevant information about a secret access operation,
    including when it occurred, which workflow and block accessed it, which secret
    was accessed, and whether the access was successful.

    Attributes:
        timestamp: ISO 8601 timestamp of the access event
        workflow_name: Name of the workflow accessing the secret
        block_id: ID of the block accessing the secret
        secret_key: Key of the secret that was accessed
        success: Whether the access was successful
        error_message: Optional error message if access failed

    Example:
        >>> event = SecretAccessEvent(
        ...     timestamp="2025-01-15T10:30:00Z",
        ...     workflow_name="deploy-app",
        ...     block_id="get_credentials",
        ...     secret_key="database_password",
        ...     success=True
        ... )
        >>> print(event.model_dump_json(indent=2))
        {
          "timestamp": "2025-01-15T10:30:00Z",
          "workflow_name": "deploy-app",
          "block_id": "get_credentials",
          "secret_key": "database_password",
          "success": true,
          "error_message": null
        }
    """

    model_config = ConfigDict(frozen=True)

    timestamp: str = Field(description="ISO 8601 timestamp of the access event")
    workflow_name: str = Field(description="Name of the workflow accessing the secret")
    block_id: str = Field(description="ID of the block accessing the secret")
    secret_key: str = Field(description="Key of the secret that was accessed")
    success: bool = Field(description="Whether the access was successful")
    error_message: str | None = Field(
        default=None,
        description="Error message if access failed",
    )


class SecretAuditLog:
    """Audit log for secret access operations.

    The SecretAuditLog maintains an in-memory record of all secret access events,
    providing filtering, export, and query capabilities for security monitoring
    and compliance reporting.

    All logging to stderr is MCP-safe (no stdout usage).

    Attributes:
        events: List of all recorded access events

    Example:
        >>> import asyncio
        >>> audit_log = SecretAuditLog()
        >>>
        >>> # Log successful access
        >>> asyncio.run(audit_log.log_access(
        ...     workflow_name="deploy-prod",
        ...     block_id="fetch_db_creds",
        ...     secret_key="database_password",
        ...     success=True
        ... ))
        >>>
        >>> # Log failed access
        >>> asyncio.run(audit_log.log_access(
        ...     workflow_name="deploy-prod",
        ...     block_id="fetch_api_key",
        ...     secret_key="missing_key",
        ...     success=False,
        ...     error_message="Secret not found"
        ... ))
        >>>
        >>> # Query events
        >>> events = audit_log.get_events(workflow_name="deploy-prod")
        >>> print(f"Total events: {len(events)}")
        Total events: 2
        >>>
        >>> # Export to file
        >>> asyncio.run(audit_log.export_to_file("audit-2025-01.json"))
    """

    def __init__(self) -> None:
        """Initialize the audit log with an empty event list."""
        self.events: list[SecretAccessEvent] = []

    async def log_access(
        self,
        workflow_name: str,
        block_id: str,
        secret_key: str,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Log a secret access event.

        Records a secret access operation with full context including timestamp,
        workflow, block, secret key, and success status. Logs to stderr for
        MCP compatibility.

        Args:
            workflow_name: Name of the workflow accessing the secret
            block_id: ID of the block accessing the secret
            secret_key: Key of the secret that was accessed
            success: Whether the access was successful
            error_message: Optional error message if access failed

        Example:
            >>> # Log successful access
            >>> await audit_log.log_access(
            ...     workflow_name="ci-pipeline",
            ...     block_id="docker_login",
            ...     secret_key="docker_hub_token",
            ...     success=True
            ... )
            >>>
            >>> # Log failed access
            >>> await audit_log.log_access(
            ...     workflow_name="ci-pipeline",
            ...     block_id="aws_deploy",
            ...     secret_key="aws_access_key",
            ...     success=False,
            ...     error_message="Secret not found in provider"
            ... )
        """
        # Create event with current timestamp
        event = SecretAccessEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            workflow_name=workflow_name,
            block_id=block_id,
            secret_key=secret_key,
            success=success,
            error_message=error_message,
        )

        # Add to event list
        self.events.append(event)

        # Log to stderr (MCP-safe)
        log_level = logging.INFO if success else logging.WARNING
        status = "SUCCESS" if success else "FAILED"

        log_message = (
            f"Secret access [{status}]: workflow={workflow_name}, "
            f"block={block_id}, key={secret_key}"
        )

        if error_message:
            log_message += f", error={error_message}"

        logger.log(log_level, log_message)

    def get_events(
        self,
        workflow_name: str | None = None,
        block_id: str | None = None,
        secret_key: str | None = None,
    ) -> list[SecretAccessEvent]:
        """Query audit events with optional filters.

        Retrieves events matching the specified filters. If no filters are provided,
        returns all events. Multiple filters are combined with AND logic.

        Args:
            workflow_name: Optional filter by workflow name
            block_id: Optional filter by block ID
            secret_key: Optional filter by secret key

        Returns:
            List of events matching the filters

        Example:
            >>> # Get all events for a workflow
            >>> events = audit_log.get_events(workflow_name="deploy-prod")
            >>>
            >>> # Get events for a specific block
            >>> events = audit_log.get_events(
            ...     workflow_name="deploy-prod",
            ...     block_id="fetch_credentials"
            ... )
            >>>
            >>> # Get all accesses to a specific secret
            >>> events = audit_log.get_events(secret_key="database_password")
            >>>
            >>> # Get all events (no filters)
            >>> all_events = audit_log.get_events()
        """
        filtered_events = self.events

        # Apply workflow name filter
        if workflow_name is not None:
            filtered_events = [e for e in filtered_events if e.workflow_name == workflow_name]

        # Apply block ID filter
        if block_id is not None:
            filtered_events = [e for e in filtered_events if e.block_id == block_id]

        # Apply secret key filter
        if secret_key is not None:
            filtered_events = [e for e in filtered_events if e.secret_key == secret_key]

        return filtered_events

    async def export_to_file(self, file_path: str | Path) -> None:
        """Export audit events to a JSON file.

        Writes all audit events to a JSON file in a structured format suitable
        for external analysis, compliance reporting, or archival.

        Args:
            file_path: Path to the output JSON file

        Raises:
            IOError: If the file cannot be written

        Example:
            >>> # Export to file
            >>> await audit_log.export_to_file("audit-2025-01.json")
            >>>
            >>> # Export with Path object
            >>> from pathlib import Path
            >>> await audit_log.export_to_file(Path("logs/audit.json"))
        """
        file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert events to dictionaries
        events_data: list[dict[str, Any]] = [event.model_dump() for event in self.events]

        # Write to file with pretty formatting
        with file_path.open("w") as f:
            json.dump(
                {
                    "audit_log_version": "1.0",
                    "total_events": len(events_data),
                    "events": events_data,
                },
                f,
                indent=2,
            )

        logger.info(f"Exported {len(events_data)} audit events to {file_path}")

    def clear(self) -> None:
        """Clear all audit events from memory.

        Removes all events from the audit log. Use with caution as this operation
        cannot be undone unless events have been exported to a file first.

        Example:
            >>> # Export before clearing
            >>> await audit_log.export_to_file("backup-audit.json")
            >>> audit_log.clear()
            >>> print(len(audit_log.get_events()))
            0
        """
        event_count = len(self.events)
        self.events.clear()
        logger.info(f"Cleared {event_count} audit events from memory")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of audit log statistics.

        Provides aggregate statistics about the audit log including total events,
        success/failure counts, unique workflows, blocks, and secrets accessed.

        Returns:
            Dictionary containing audit log statistics

        Example:
            >>> summary = audit_log.get_summary()
            >>> print(json.dumps(summary, indent=2))
            {
              "total_events": 42,
              "successful_accesses": 38,
              "failed_accesses": 4,
              "unique_workflows": 5,
              "unique_blocks": 12,
              "unique_secrets": 8,
              "workflows": ["deploy-prod", "ci-pipeline", ...],
              "most_accessed_secret": "database_password"
            }
        """
        if not self.events:
            return {
                "total_events": 0,
                "successful_accesses": 0,
                "failed_accesses": 0,
                "unique_workflows": 0,
                "unique_blocks": 0,
                "unique_secrets": 0,
            }

        # Count successes and failures
        successful = sum(1 for e in self.events if e.success)
        failed = len(self.events) - successful

        # Collect unique values
        workflows = {e.workflow_name for e in self.events}
        blocks = {e.block_id for e in self.events}
        secrets = {e.secret_key for e in self.events}

        # Find most accessed secret
        secret_counts: dict[str, int] = {}
        for event in self.events:
            secret_counts[event.secret_key] = secret_counts.get(event.secret_key, 0) + 1

        most_accessed = max(secret_counts.items(), key=lambda x: x[1])[0] if secret_counts else None

        return {
            "total_events": len(self.events),
            "successful_accesses": successful,
            "failed_accesses": failed,
            "unique_workflows": len(workflows),
            "unique_blocks": len(blocks),
            "unique_secrets": len(secrets),
            "workflows": sorted(workflows),
            "most_accessed_secret": most_accessed,
        }
