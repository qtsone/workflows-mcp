#!/usr/bin/env python3
"""Shared test utilities for workflows-mcp test suite.

Provides common functionality for test execution and snapshot management:
- Dynamic field normalization for stable snapshot comparisons
- Diff formatting for actionable test failure messages
- Shared test patterns and helpers

Philosophy: DRY principle - define normalization logic once, use everywhere.
"""

import difflib
import json
import re
from typing import Any


def normalize_dynamic_fields(data: Any, path: str = "") -> Any:
    """Recursively normalize dynamic fields in workflow responses.

    Handles fields that change between test runs but don't affect
    functional correctness:
    - ISO 8601 timestamps → 'TIMESTAMP'
    - Execution times → 'EXECUTION_TIME'
    - Checkpoint IDs → 'CHECKPOINT_ID'
    - HTTP date headers → 'HTTP_DATE'
    - Amazon trace IDs → 'TRACE_ID'
    - HTTP origin/IP fields → 'HTTP_ORIGIN'
    - Content-Length headers → 'CONTENT_LENGTH'
    - Secret redaction markers → 'REDACTED_SECRET'
    - Environment variable values → 'ENV_VAR_VALUE'
    - File paths with /private/tmp → /tmp (macOS compatibility)
    - Block execution order → sorted by block ID (for parallel blocks)

    Args:
        data: Response data to normalize (dict, list, str, or primitive)
        path: Current path in data structure (for debugging)

    Returns:
        Normalized copy of data with dynamic fields replaced

    Examples:
        >>> normalize_dynamic_fields({"start_time": "2025-11-02T10:30:45.123456+00:00"})
        {"start_time": "TIMESTAMP"}

        >>> normalize_dynamic_fields({"stdout": "Token: ghp_abc123def456"})
        {"stdout": "Token: REDACTED_SECRET"}

        >>> normalize_dynamic_fields({"path": "/private/tmp/test.txt"})
        {"path": "/tmp/test.txt"}
    """
    if isinstance(data, dict):
        # Sort dictionary keys for consistent ordering (important for blocks dict)
        # This ensures parallel block execution order doesn't affect snapshots
        normalized = {}
        for key, value in sorted(data.items()):
            current_path = f"{path}.{key}" if path else key

            # Normalize known dynamic field names
            if key in ("start_time", "end_time", "created_at", "timestamp", "completed_at"):
                normalized[key] = "TIMESTAMP"
            elif key in (
                "execution_time_ms",
                "duration_ms",
                "execution_time",
                "execution_time_seconds",
            ):
                normalized[key] = "EXECUTION_TIME"
            elif key.lower() == "checkpoint_id" and isinstance(value, str):
                normalized[key] = "CHECKPOINT_ID"
            elif key.lower() == "date" and isinstance(value, str):
                # Normalize HTTP date headers (e.g., "Sat, 01 Nov 2025 12:26:49 GMT")
                normalized[key] = "HTTP_DATE"
            elif key.lower() == "origin" and isinstance(value, str):
                # Normalize HTTP origin/client IP (e.g., "192.168.1.1")
                normalized[key] = "HTTP_ORIGIN"
            elif key.lower() == "content-length" and isinstance(value, str):
                # Normalize content-length header (varies with response body size)
                normalized[key] = "CONTENT_LENGTH"
            else:
                normalized[key] = normalize_dynamic_fields(value, current_path)
        return normalized

    elif isinstance(data, list):
        return [normalize_dynamic_fields(item, f"{path}[{i}]") for i, item in enumerate(data)]

    elif isinstance(data, str):
        normalized_str = data

        # Normalize ISO 8601 timestamps in string values
        # Format: 2025-11-02T10:30:45.123456+00:00
        normalized_str = re.sub(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2}",
            "TIMESTAMP",
            normalized_str,
        )

        # Normalize Amazon trace IDs (X-Amzn-Trace-Id)
        # Format: Root=1-6905fc89-71dbb5d47ab07eeb01ec09c5
        normalized_str = re.sub(
            r"Root=1-[a-f0-9]{8}-[a-f0-9]{24}",
            "Root=TRACE_ID",
            normalized_str,
        )

        # Normalize host:port patterns (localhost, IP addresses with ports)
        # Pattern: localhost:8080, 127.0.0.1:8080, 192.168.1.1:8080, etc.
        # Must come before standalone IP/localhost normalization
        normalized_str = re.sub(
            r"\b(?:localhost|(?:\d{1,3}\.){3}\d{1,3}):\d{1,5}\b",
            "IP_ADDRESS:PORT",
            normalized_str,
        )

        # Normalize standalone IP addresses (IPv4 and IPv6)
        # IPv4: 192.168.1.1, 10.0.0.1, etc.
        normalized_str = re.sub(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "IP_ADDRESS",
            normalized_str,
        )
        # IPv6: 2001:db8::1, ::1, etc.
        normalized_str = re.sub(
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|::1\b",
            "IP_ADDRESS",
            normalized_str,
        )

        # Normalize standalone localhost (without port)
        normalized_str = re.sub(
            r"\blocalhost\b",
            "IP_ADDRESS",
            normalized_str,
        )

        # Normalize secret patterns (ADR-008 secrets management)
        # GitHub tokens: ghp_*, gho_*, ghs_*, etc.
        normalized_str = re.sub(
            r"gh[pso]_[a-zA-Z0-9]{36,}",
            "REDACTED_SECRET",
            normalized_str,
        )
        # Generic API keys: sk-*, pk-*, etc.
        normalized_str = re.sub(
            r"\b[sp]k-[a-zA-Z0-9]{32,}\b",
            "REDACTED_SECRET",
            normalized_str,
        )
        # AWS keys: AKIA*, etc.
        normalized_str = re.sub(
            r"\bAKIA[0-9A-Z]{16}\b",
            "REDACTED_SECRET",
            normalized_str,
        )
        # JWT tokens (base64 encoded with dots)
        normalized_str = re.sub(
            r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
            "REDACTED_SECRET",
            normalized_str,
        )

        # Normalize ***REDACTED*** markers (ensure consistency)
        normalized_str = re.sub(
            r"\*\*\*REDACTED\*\*\*",
            "REDACTED_SECRET",
            normalized_str,
        )

        # Normalize environment variable values that might contain secrets
        # Pattern: VAR_NAME="value" or VAR_NAME='value'
        normalized_str = re.sub(
            r'([A-Z_]+)=["\']([^"\']+)["\']',
            r'\1="ENV_VAR_VALUE"',
            normalized_str,
        )

        # Normalize checkpoint IDs in strings (format: pause_workflow-name_timestamp_hash)
        # Example: pause_interactive-simple-approval_1762091622230_391d77ce → CHECKPOINT_ID
        normalized_str = re.sub(
            r"pause_[\w-]+_\d+_[a-f0-9]{8}",
            "CHECKPOINT_ID",
            normalized_str,
        )

        # Normalize /private/tmp to /tmp (macOS vs Linux compatibility)
        # On macOS, /tmp is a symlink to /private/tmp
        # On Linux, /tmp is just /tmp
        normalized_str = normalized_str.replace("/private/tmp", "/tmp")

        return normalized_str

    else:
        # Primitives (int, float, bool, None) pass through unchanged
        return data


def format_diff(actual: dict[str, Any], expected: dict[str, Any], workflow_name: str) -> str:
    """Generate formatted diff between actual and expected responses.

    Creates a unified diff with actionable guidance for resolving mismatches.
    Used for snapshot test failures to show exactly what changed.

    Args:
        actual: Normalized actual response
        expected: Normalized expected response
        workflow_name: Name of workflow being tested

    Returns:
        Formatted diff string with actionable guidance

    Example output:
        ================================================================================
        Snapshot mismatch for workflow: secrets-basic
        ================================================================================
        --- expected/secrets-basic.json
        +++ actual/secrets-basic.json
        @@ -5,7 +5,7 @@
           "outputs": {
        -    "result": "success"
        +    "result": "failure"
           }
        ================================================================================
        To update snapshot if this change is intentional:
          uv run python tests/generate_snapshots.py
        ================================================================================
    """
    actual_json = json.dumps(actual, indent=2, sort_keys=True)
    expected_json = json.dumps(expected, indent=2, sort_keys=True)

    diff = difflib.unified_diff(
        expected_json.splitlines(keepends=True),
        actual_json.splitlines(keepends=True),
        fromfile=f"expected/{workflow_name}.json",
        tofile=f"actual/{workflow_name}.json",
        lineterm="",
    )

    diff_text = "".join(diff)

    return (
        f"\n{'=' * 80}\n"
        f"Snapshot mismatch for workflow: {workflow_name}\n"
        f"{'=' * 80}\n"
        f"{diff_text}\n"
        f"{'=' * 80}\n"
        f"To update snapshot if this change is intentional:\n"
        f"  uv run python tests/generate_snapshots.py\n"
        f"{'=' * 80}\n"
    )
