"""Async-local context variables for workflow execution.

Uses Python's contextvars module to provide thread-safe and async-safe
storage for execution-specific data that needs to be accessible across
function boundaries without explicit parameter passing.
"""

from contextvars import ContextVar
from typing import Any

# Block custom outputs context variable
# This allows passing custom output definitions from WorkflowExecutor
# to Shell executor without race conditions in parallel execution.
#
# Set by: WorkflowExecutor._execute_block() before calling orchestrator
# Read by: ShellExecutor.execute() when reading custom output files
#
# Why contextvars?
# - Thread-safe and async-safe by design
# - Each async task gets its own isolated copy
# - Automatic cleanup via token.reset()
# - No shared mutable state issues in parallel execution
block_custom_outputs: ContextVar[dict[str, Any] | None] = ContextVar(
    "block_custom_outputs", default=None
)
