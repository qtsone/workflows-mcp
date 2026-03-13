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

# Current block ID context variable
# Allows executors (e.g., LLMCallExecutor) to identify which block they
# are executing as, without explicit parameter passing.
#
# Set by: WorkflowRunner._execute_block() before calling executor.execute()
# Read by: Executors that emit log events (e.g., LLMCallExecutor)
current_block_id: ContextVar[str | None] = ContextVar("current_block_id", default=None)

# Current node ID context variable (globally unique UUID per block execution)
# Enables unambiguous DAG parent-child relationships: each block execution
# gets a UUID, and child workflows reference the parent's UUID via parent_node_id.
#
# Set by: WorkflowRunner._execute_block() and orchestrator.execute_iteration()
# Read by: WorkflowExecutor.execute() to set parent_node_id on child context
current_node_id: ContextVar[str | None] = ContextVar("current_node_id", default=None)

