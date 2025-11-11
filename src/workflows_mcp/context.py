"""Shared context types for MCP server.

This module contains context types used across server and tools modules,
separated to avoid circular imports.
"""

from dataclasses import dataclass

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from .engine import ExecutionContext, WorkflowRegistry
from .engine.executor_base import ExecutorRegistry
from .engine.io_queue import IOQueue
from .engine.job_queue import JobQueue
from .engine.llm_config import LLMConfigLoader


@dataclass
class AppContext:
    """Application context containing shared resources for MCP tools.

    This context is created during server startup and made available to all tools
    via dependency injection through the Context parameter.

    Post ADR-008: Stores shared resources for ExecutionContext creation.
    """

    registry: WorkflowRegistry
    executor_registry: ExecutorRegistry
    llm_config_loader: LLMConfigLoader
    io_queue: IOQueue | None  # Optional IO queue for serialized file operations
    job_queue: JobQueue | None = None  # Optional job queue for async execution
    max_recursion_depth: int = 50  # Default recursion depth limit

    def create_execution_context(self) -> ExecutionContext:
        """Create ExecutionContext for workflow execution.

        Returns:
            ExecutionContext with access to all shared resources and configured recursion limit
        """
        return ExecutionContext(
            workflow_registry=self.registry,
            executor_registry=self.executor_registry,
            llm_config_loader=self.llm_config_loader,
            io_queue=self.io_queue,
            parent=None,
            workflow_stack=[],
            max_recursion_depth=self.max_recursion_depth,
        )


# Type alias for MCP tool context parameter
AppContextType = Context[ServerSession, AppContext]


__all__ = ["AppContext", "AppContextType"]
