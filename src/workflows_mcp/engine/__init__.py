"""Workflow engine core components using executor pattern.

This package contains the modern workflow execution engine based on the executor pattern.

Key Components (Post ADR-008):

- WorkflowRunner: Stateless workflow executor (returns ExecutionResult)
- ExecutionResult: Monad for workflow results (success/failure/paused)
- ExecutionContext: Dependency injection container for workflow composition
- WorkflowExecutor: Block executor for workflow composition (fractal pattern)
- BlockOrchestrator: Exception handling and metadata creation
- BlockExecutor: Base class for executor implementations
- BlockInput/BlockOutput: Pydantic v2 base classes for I/O validation
- Execution: Fractal execution context model
- Metadata: Execution state tracking
- DAGResolver: Dependency resolution via Kahn's algorithm
- WorkflowRegistry: Registry for managing workflow definitions
- WorkflowSchema: Pydantic v2 schema for YAML validation (with execution_waves)
- LoadResult: Error monad for loader/registry safe file operations

Architecture (ADR-008 ExecutionResult Pattern):
- WorkflowRunner returns ExecutionResult (not WorkflowResponse)
- ExecutionResult.to_response() is single source of truth for formatting
- Partial execution preserved on failures (attached to exceptions)
- ExecutionContext enables workflow composition via dependency injection
- WorkflowSchema.execution_waves provides pre-computed DAG
- Executors return BaseModel directly (no wrapper classes)
- Exceptions for control flow (ExecutionPaused for pause)
- BlockOrchestrator wraps executor calls with exception handling
- Execution model provides fractal context (supports nested workflows)
- Type Safety: Pydantic models ensure correct I/O
"""

# Import executors (they auto-register in create_default_registry)
from . import (  # noqa: F401
    executors_core,  # Shell executor
    executors_file,  # File operation executors
    executors_interactive,  # Interactive executors
    executors_state,  # JSON state executors
    executors_workflow,  # Workflow executor (ADR-008)
)
from .block import BlockInput, BlockOutput
from .dag import DAGResolver
from .execution_context import ExecutionContext
from .execution_result import ExecutionResult, PauseData
from .executor_base import create_default_registry
from .executors_core import (
    ShellExecutor,
    ShellInput,
    ShellOutput,
)
from .executors_file import (
    CreateFileExecutor,
    CreateFileInput,
    CreateFileOutput,
    ReadFileExecutor,
    ReadFileInput,
    ReadFileOutput,
    RenderTemplateExecutor,
    RenderTemplateInput,
    RenderTemplateOutput,
)
from .executors_interactive import (
    PromptExecutor,
    PromptInput,
    PromptOutput,
)
from .executors_state import (
    MergeJSONStateExecutor,
    MergeJSONStateInput,
    MergeJSONStateOutput,
    ReadJSONStateExecutor,
    ReadJSONStateInput,
    ReadJSONStateOutput,
    WriteJSONStateExecutor,
    WriteJSONStateInput,
    WriteJSONStateOutput,
)
from .executors_workflow import (
    WorkflowExecutor,
    WorkflowInput,
)
from .load_result import LoadResult
from .loader import load_workflow_from_yaml
from .registry import WorkflowRegistry
from .schema import WorkflowSchema
from .workflow_runner import WorkflowRunner

__all__ = [
    # Core types (ADR-008)
    "LoadResult",
    "DAGResolver",
    "BlockInput",
    "BlockOutput",
    "create_default_registry",
    "WorkflowRunner",
    "ExecutionResult",
    "PauseData",
    "ExecutionContext",
    "WorkflowRegistry",
    "WorkflowSchema",
    "load_workflow_from_yaml",
    # Core Executors
    "ShellExecutor",
    "ShellInput",
    "ShellOutput",
    "WorkflowExecutor",
    "WorkflowInput",
    # File Executors
    "CreateFileExecutor",
    "CreateFileInput",
    "CreateFileOutput",
    "ReadFileExecutor",
    "ReadFileInput",
    "ReadFileOutput",
    "RenderTemplateExecutor",
    "RenderTemplateInput",
    "RenderTemplateOutput",
    # Interactive Executors
    "PromptExecutor",
    "PromptInput",
    "PromptOutput",
    # State Executors
    "ReadJSONStateExecutor",
    "ReadJSONStateInput",
    "ReadJSONStateOutput",
    "WriteJSONStateExecutor",
    "WriteJSONStateInput",
    "WriteJSONStateOutput",
    "MergeJSONStateExecutor",
    "MergeJSONStateInput",
    "MergeJSONStateOutput",
]
