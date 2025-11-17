"""Base executor architecture for ADR-006 unified execution model.

This module provides the foundation for the fractal/recursive executor pattern.
Executors are pure functions that implement block logic as stateless,
reusable components.

Key principles:
- Executors are stateless (singleton pattern)
- Execute returns BaseModel directly (no Result wrapper)
- Exceptions indicate execution failure
- Type safety through Pydantic models
"""

import importlib.util
import inspect
import sys
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, PrivateAttr

from .block import BlockInput, BlockOutput
from .execution import Execution
from .schema import InputType, WorkflowOutputSchema


class ExecutorSecurityLevel(Enum):
    """Security level classification for executors.

    Used for security policy enforcement and audit purposes.
    """

    SAFE = "safe"  # Read-only operations, no system access
    TRUSTED = "trusted"  # File I/O, safe commands
    PRIVILEGED = "privileged"  # Full system access (shell, git, network)


class ExecutorCapabilities(BaseModel):
    """Executor capability flags for security audit.

    Declares what system resources an executor can access.
    Used by security policies to restrict execution.
    """

    can_read_files: bool = False
    can_write_files: bool = False
    can_execute_commands: bool = False
    can_network: bool = False
    can_modify_state: bool = False


class BlockExecutor(ABC):
    """Base class for workflow block executors.

    Executors are pure functions that implement block logic. They are:
    - Stateless: No mutable state between executions
    - Reusable: Single instance serves all blocks of this type
    - Type-safe: Pydantic models for inputs and outputs
    - Testable: Pure functions with no side effects in tests

    Subclasses must:
    1. Set class attributes (type_name, input_type, output_type)
    2. Implement execute() method
    3. Optionally override security attributes

    Example:
        class ShellExecutor(BlockExecutor):
            type_name = "Shell"
            input_type = ShellInput
            output_type = ShellOutput
            security_level = ExecutorSecurityLevel.PRIVILEGED
            capabilities = ExecutorCapabilities(can_execute_commands=True)

            async def execute(self, inputs: ShellInput, context: Execution) -> ShellOutput:
                # Returns output directly (ADR-006)
                # Raises exceptions for failures
                result = await run_command(inputs.command)
                return ShellOutput(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.returncode
                )
    """

    # Class attributes (must be set by subclasses)
    type_name: ClassVar[str]  # Block type identifier (e.g., "Shell")
    input_type: ClassVar[type[BlockInput]]  # Pydantic input model
    output_type: ClassVar[type[BlockOutput]]  # Pydantic output model

    # Security attributes (can be overridden by subclasses)
    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.SAFE
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities()

    @abstractmethod
    async def execute(self, inputs: BlockInput, context: Execution) -> BaseModel:
        """Execute block logic with validated inputs.

        ADR-006: Returns output directly, raises exceptions for failures.

        Args:
            inputs: Validated input model instance (type matches input_type)
            context: Current execution context (fractal - same for all levels)

        Returns:
            Outputs (instance of self.output_type) on success

        Raises:
            Exception: Any exception indicates execution failure (status=FAILED)

        Example:
            async def execute(self, inputs: ShellInput, context: Execution) -> ShellOutput:
                # Run shell command
                process = await asyncio.create_subprocess_shell(
                    inputs.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()

                # Return typed output directly
                return ShellOutput(
                    stdout=stdout.decode(),
                    stderr=stderr.decode(),
                    exit_code=process.returncode
                )
        """
        pass

    async def resume(
        self,
        _inputs: BlockInput,
        _context: Execution,
        _response: str,
        _pause_metadata: dict[str, Any],
    ) -> BaseModel:
        """Resume execution after a pause (optional, for interactive blocks).

        This method is called when a workflow is resumed after a pause.
        Most executors don't need to implement this - it's only required
        for interactive executors that can pause workflow execution.

        Args:
            _inputs: Validated input model instance
            _context: Execution context (fractal)
            _response: Response from LLM to the pause prompt
            _pause_metadata: Metadata stored when the block paused

        Returns:
            Outputs (instance of self.output_type) on success

        Raises:
            NotImplementedError: By default (non-interactive executors)
            Exception: Any exception indicates execution failure
        """
        raise NotImplementedError(
            f"{self.type_name} executor does not support resume (not an interactive executor)"
        )

    def get_input_schema(self) -> dict[str, Any]:
        """Generate JSON Schema for input validation.

        Uses Pydantic's model_json_schema() to auto-generate schema from input_type.
        This schema is used for:
        - VS Code autocomplete
        - Pre-execution validation
        - MCP schema tools
        - Documentation generation

        Returns:
            JSON Schema dictionary for input model
        """
        return self.input_type.model_json_schema()

    def get_output_schema(self) -> dict[str, Any]:
        """Generate JSON Schema for output structure.

        Returns:
            JSON Schema dictionary for output model
        """
        return self.output_type.model_json_schema()

    def get_capabilities(self) -> dict[str, Any]:
        """Get executor capabilities for security audit.

        Returns:
            Dictionary with type, security_level, and capabilities
        """
        return {
            "type": self.type_name,
            "security_level": self.security_level.value,
            "capabilities": self.capabilities.model_dump(),
        }


class ExecutorRegistry(BaseModel):
    """
    Registry of executors.

    Maps executor type names to executor instances.
    """

    model_config = {"arbitrary_types_allowed": True}

    _executors: dict[str, BlockExecutor] = PrivateAttr(default_factory=dict)

    def register(self, executor: BlockExecutor) -> None:
        """Register executor using executor.type_name as key."""
        if executor.type_name in self._executors:
            raise ValueError(f"Executor already registered: {executor.type_name}")
        self._executors[executor.type_name] = executor

    def get(self, type_name: str) -> BlockExecutor:
        """Get executor by type name."""
        if type_name not in self._executors:
            available = list(self._executors.keys())
            raise ValueError(f"Unknown executor type: {type_name}. Available: {available}")
        return self._executors[type_name]

    def list_types(self) -> list[str]:
        """List registered executor types."""
        return list(self._executors.keys())

    def has(self, type_name: str) -> bool:
        """Check if executor type is registered."""
        return type_name in self._executors

    def generate_workflow_schema(self) -> dict[str, Any]:
        """Generate complete JSON Schema for workflow validation.

        This creates a comprehensive schema that includes:
        - All registered executor input schemas
        - Workflow structure validation
        - Conditional schemas per block type

        The schema can be used for:
        - VS Code YAML autocomplete
        - Pre-execution workflow validation
        - MCP schema tools for Claude
        - Automatic documentation generation

        Returns:
            Complete JSON Schema for workflow definitions

        Example:
            schema = registry.generate_workflow_schema()

            # Use with jsonschema library
            from jsonschema import validate
            validate(instance=workflow_dict, schema=schema)

            # Save for VS Code autocomplete
            with open("workflow-schema.json", "w") as f:
                json.dump(schema, f, indent=2)
        """
        # Collect all executor input schemas and shared type definitions
        definitions = {}
        block_types = []
        type_conditionals = []

        # Extract WorkflowOutputSchema and merge its $defs to root level
        # This ensures ValueType enum is accessible at the root for proper $ref resolution
        output_schema = WorkflowOutputSchema.model_json_schema()
        definitions.update(output_schema.pop("$defs", {}))

        for type_name, executor in self._executors.items():
            # Get the actual Pydantic schema from the executor
            input_schema = executor.get_input_schema()

            # Extract and merge nested $defs to root level (same as WorkflowOutputSchema)
            # This ensures types like LLMProvider are accessible at root for $ref
            # $ref uses absolute paths (#/$defs/Type), so nested $defs won't resolve
            nested_defs = input_schema.pop("$defs", {})
            definitions.update(nested_defs)

            # Store in definitions
            definitions[f"{type_name}Input"] = input_schema
            block_types.append(type_name)

            # Create conditional validation for this block type
            # If block.type == type_name, then inputs must match the executor's input schema
            type_conditionals.append(
                {
                    "if": {"properties": {"type": {"const": type_name}}},
                    "then": {
                        "properties": {
                            "inputs": {
                                # Reference the definition OR inline the schema
                                # Using inline to avoid $ref resolution issues in some validators
                                **input_schema,
                                "description": f"Inputs for {type_name} block",
                            }
                        }
                    },
                }
            )

        # Base block schema (common properties)
        base_block_schema = {
            "type": "object",
            "required": ["id", "type", "inputs"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique block identifier",
                },
                "type": {
                    "enum": block_types,
                    "description": "Block type",
                },
                "description": {
                    "type": "string",
                    "description": "Optional block description",
                },
                "inputs": {
                    "type": "object",
                    "description": "Block inputs (schema depends on type)",
                },
                "depends_on": {
                    "type": "array",
                    "description": ("Dependencies (block IDs or specs with 'required' flag)"),
                    "items": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": (
                                    'Block ID (shorthand for {"block": "id", "required": true})'
                                ),
                            },
                            {
                                "type": "object",
                                "required": ["block"],
                                "properties": {
                                    "block": {
                                        "type": "string",
                                        "description": ("Block ID of parent dependency"),
                                    },
                                    "required": {
                                        "type": "boolean",
                                        "default": True,
                                        "description": (
                                            "Skip this block if parent fails/skips (default: true)"
                                        ),
                                    },
                                },
                                "additionalProperties": False,
                            },
                        ]
                    },
                },
                "condition": {
                    "type": "string",
                    "description": "Conditional execution expression",
                },
                "outputs": {
                    "type": "object",
                    "description": "Custom file-based outputs (Shell blocks only)",
                    "patternProperties": {
                        ".*": {
                            "type": "object",
                            "required": ["type", "path"],
                            "properties": {
                                "type": {
                                    "$ref": "#/$defs/ValueType",
                                    "description": "Output value type",
                                },
                                "path": {
                                    "type": "string",
                                    "description": "File path to read",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Output description",
                                },
                                "required": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Whether this output is required",
                                },
                                "unsafe": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Allow absolute paths (default: false)",
                                },
                            },
                        }
                    },
                },
            },
            # Add all the type-specific conditionals
            "allOf": type_conditionals,
        }

        # Build complete workflow schema
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "MCP Workflow Definition",
            "type": "object",
            "required": ["name", "blocks"],
            "properties": {
                "name": {"type": "string", "description": "Workflow name"},
                "description": {
                    "type": "string",
                    "description": "Workflow description",
                },
                "inputs": {
                    "type": "object",
                    "description": "Workflow input parameters",
                    "patternProperties": {
                        ".*": {
                            "type": "object",
                            "properties": {
                                "type": {"enum": [t.value for t in InputType]},
                                "default": {},
                                "description": {"type": "string"},
                                "required": {"type": "boolean"},
                            },
                            "required": ["type"],
                        }
                    },
                },
                "outputs": {
                    "type": "object",
                    "description": "Workflow output expressions with type coercion",
                    "patternProperties": {".*": output_schema},
                },
                "blocks": {
                    "type": "array",
                    "description": "Workflow execution blocks",
                    "items": base_block_schema,
                },
            },
            "$defs": definitions,
        }

    def discover_entry_points(self, group: str = "mcp_workflows.executors") -> int:
        """Discover and register executors from entry points.

        This enables third-party packages to provide custom executors by declaring
        entry points in their pyproject.toml:

            [project.entry-points."mcp_workflows.executors"]
            custom_executor = "my_package.executors:CustomExecutor"

        Args:
            group: Entry point group name (default: "mcp_workflows.executors")

        Returns:
            Number of executors discovered and registered

        Example:
            # In third-party package pyproject.toml
            [project.entry-points."mcp_workflows.executors"]
            database = "my_pkg.executors:DatabaseExecutor"
            api_call = "my_pkg.executors:APICallExecutor"

            # In application startup
            registry.discover_entry_points()
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            return 0

        discovered = 0
        eps = entry_points()

        # Use Python 3.12+ entry_points() API
        group_eps = eps.select(group=group)

        for entry_point in group_eps:
            try:
                executor_class = entry_point.load()

                # Validate it's a BlockExecutor subclass
                if not (
                    inspect.isclass(executor_class) and issubclass(executor_class, BlockExecutor)
                ):
                    continue

                # Instantiate and register
                executor = executor_class()
                self.register(executor)
                discovered += 1

            except Exception:
                # Skip invalid entry points
                continue

        return discovered

    def discover_from_directories(
        self, directories: list[Path | str], pattern: str = "*_executor.py"
    ) -> int:
        """Discover and register executors from Python files in directories.

        This enables local executor plugins similar to pytest's conftest.py pattern.
        The registry will scan directories for Python files matching the pattern,
        import them, and register any BlockExecutor subclasses found.

        Args:
            directories: List of directories to scan for executor files
            pattern: Glob pattern for executor files (default: "*_executor.py")

        Returns:
            Number of executors discovered and registered

        Example:
            # Directory structure:
            # ~/.workflows-mcp/executors/
            # ├── database_executor.py  (contains DatabaseExecutor)
            # └── api_executor.py       (contains APICallExecutor)

            # Discover from user and project directories
            registry.discover_from_directories([
                Path.home() / ".workflows-mcp/executors",
                Path.cwd() / ".workflows/executors"
            ])
        """
        discovered = 0

        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists() or not dir_path.is_dir():
                continue

            # Find all matching Python files
            for file_path in dir_path.glob(pattern):
                if not file_path.is_file():
                    continue

                try:
                    # Import module from file
                    module_name = f"_executor_plugin_{file_path.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None or spec.loader is None:
                        continue

                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # Find all BlockExecutor subclasses in module
                    for _name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, BlockExecutor)
                            and obj is not BlockExecutor
                            and obj.__module__ == module_name
                        ):
                            # Instantiate and register
                            executor = obj()
                            self.register(executor)
                            discovered += 1

                except Exception:
                    # Skip invalid files
                    continue

        return discovered

    def discover_plugins(
        self,
        entry_point_group: str = "mcp_workflows.executors",
        plugin_directories: list[Path | str] | None = None,
    ) -> dict[str, int]:
        """Discover and register executors from all plugin sources.

        This is a convenience method that combines entry point and directory-based
        plugin discovery. It's the recommended way to load plugins at application startup.

        Args:
            entry_point_group: Entry point group name
            plugin_directories: Optional list of directories to scan for plugins.
                If None, uses default locations:
                - ~/.workflows-mcp/executors
                - ./.workflows/executors

        Returns:
            Dictionary with counts: {"entry_points": N, "directories": M}

        Example:
            # Load all plugins at startup
            counts = EXECUTOR_REGISTRY.discover_plugins()
            print(f"Loaded {counts['entry_points']} entry point executors")
            print(f"Loaded {counts['directories']} directory executors")
        """
        # Default plugin directories
        if plugin_directories is None:
            plugin_directories = [
                Path.home() / ".workflows-mcp/executors",
                Path.cwd() / ".workflows/executors",
            ]

        return {
            "entry_points": self.discover_entry_points(entry_point_group),
            "directories": self.discover_from_directories(plugin_directories),
        }


def create_default_registry() -> ExecutorRegistry:
    """Create ExecutorRegistry with all built-in executors registered.

    This factory function explicitly registers all built-in executor types.
    Use this to create a registry instance with standard workflow capabilities.

    This replaces the global EXECUTOR_REGISTRY singleton to enable:
    - Test isolation (each test can have its own registry)
    - Parallel test execution (no shared global state)
    - Clear dependency injection (explicit rather than implicit)
    - Better architecture (no hidden global dependencies)

    Returns:
        ExecutorRegistry instance with all built-in executors registered

    Example:
        # In application startup
        registry = create_default_registry()
        executor = WorkflowExecutor(registry=registry)

        # In tests
        def test_workflow():
            registry = create_default_registry()
            executor = WorkflowExecutor(registry=registry)
            # Test with isolated registry
    """
    from .executors_core import ShellExecutor
    from .executors_file import (
        CreateFileExecutor,
        EditFileExecutor,
        ReadFilesExecutor,
    )
    from .executors_http import HttpCallExecutor
    from .executors_interactive import PromptExecutor
    from .executors_llm import LLMCallExecutor
    from .executors_state import (
        MergeJSONStateExecutor,
        ReadJSONStateExecutor,
        WriteJSONStateExecutor,
    )
    from .executors_workflow import WorkflowExecutor

    registry = ExecutorRegistry()

    # Register core executors
    registry.register(ShellExecutor())
    registry.register(WorkflowExecutor())  # ADR-006: Returns Execution, not BlockOutput

    # Register file executors
    registry.register(CreateFileExecutor())
    registry.register(EditFileExecutor())
    registry.register(ReadFilesExecutor())

    # Register HTTP executor
    registry.register(HttpCallExecutor())

    # Register LLM executor
    registry.register(LLMCallExecutor())

    # Register interactive executor
    registry.register(PromptExecutor())

    # Register state executors
    registry.register(ReadJSONStateExecutor())
    registry.register(WriteJSONStateExecutor())
    registry.register(MergeJSONStateExecutor())

    return registry
