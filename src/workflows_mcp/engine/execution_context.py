"""
Execution context for dependency injection and fractal composition.

Provides access to:
- Workflow registry (for Workflow composition)
- Executor registry (for block execution)
- Parent execution (for nested workflows)

This enables:
- Fractal composition (workflows in workflows)
- Access to shared resources (registries)
- Execution isolation (each workflow gets its own context)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution import Execution
    from .executor_base import ExecutorRegistry
    from .io_queue import IOQueue
    from .llm_config import LLMConfigLoader
    from .registry import WorkflowRegistry
    from .schema import WorkflowSchema


class ExecutionContext:
    """
    Context providing dependencies for workflow execution.

    Injected into block executors to enable:
    - Workflow composition (Workflow needs registry access)
    - Block execution (all executors need executor registry)
    - Fractal nesting (parent execution tracking)
    - Recursion depth limiting (prevents infinite recursion)

    Design:
    - Immutable after creation (use create_child for nesting)
    - Contains references to shared resources (registries)
    - Tracks execution hierarchy (parent chain)
    - Enforces recursion depth limits (configurable)
    """

    def __init__(
        self,
        workflow_registry: WorkflowRegistry,
        executor_registry: ExecutorRegistry,
        llm_config_loader: LLMConfigLoader,
        io_queue: IOQueue | None,
        parent: Execution | None = None,
        workflow_stack: list[str] | None = None,
        max_recursion_depth: int = 50,
    ):
        """
        Initialize execution context with shared resources.

        Args:
            workflow_registry: Registry of workflow definitions
            executor_registry: Registry of block executors
            llm_config_loader: Loader for LLM profile configuration
            io_queue: IO queue for serialized file operations (optional)
            parent: Parent execution (for nested workflows)
            workflow_stack: Workflow execution stack (depth tracking)
            max_recursion_depth: Maximum allowed recursion depth (default: 50)
        """
        self.workflow_registry = workflow_registry
        self.executor_registry = executor_registry
        self.llm_config_loader = llm_config_loader
        self.io_queue = io_queue
        self.parent = parent
        self.workflow_stack = workflow_stack or []
        self.max_recursion_depth = max_recursion_depth

    def get_workflow(self, name: str) -> WorkflowSchema | None:
        """
        Get workflow by name (for Workflow composition).

        Args:
            name: Workflow name

        Returns:
            WorkflowSchema if found, None otherwise
        """
        return self.workflow_registry.get(name)

    def create_child_context(
        self,
        parent_execution: Execution,
        workflow_name: str,
    ) -> ExecutionContext:
        """
        Create child context for nested workflow execution.

        Preserves shared resources (registries, stores) while tracking
        parent execution and workflow stack for depth tracking and debugging.

        Args:
            parent_execution: Parent execution context
            workflow_name: Name of child workflow being executed

        Returns:
            New ExecutionContext with parent tracking and incremented depth

        Raises:
            RecursionDepthExceededError: If adding this workflow would exceed max_recursion_depth
        """
        return ExecutionContext(
            workflow_registry=self.workflow_registry,
            executor_registry=self.executor_registry,
            llm_config_loader=self.llm_config_loader,
            io_queue=self.io_queue,
            parent=parent_execution,
            workflow_stack=self.workflow_stack + [workflow_name],
            max_recursion_depth=self.max_recursion_depth,
        )

    def check_recursion_depth(self, workflow_name: str) -> None:
        """
        Check if adding this workflow would exceed recursion depth limit.

        This allows recursive workflows (A→A, A→B→A, etc.) up to max_recursion_depth.
        Replaces the previous circular dependency check which prevented ANY recursion.

        Args:
            workflow_name: Name of workflow about to be executed

        Raises:
            RecursionDepthExceededError: If executing this workflow would exceed depth limit
        """
        from .exceptions import RecursionDepthExceededError

        current_depth = len(self.workflow_stack)
        if current_depth >= self.max_recursion_depth:
            raise RecursionDepthExceededError(
                workflow_name=workflow_name,
                current_depth=current_depth + 1,  # Depth if we execute this workflow
                max_depth=self.max_recursion_depth,
                workflow_stack=self.workflow_stack,
            )


__all__ = ["ExecutionContext"]
