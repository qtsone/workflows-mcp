"""
YAML workflow schema with Pydantic v2 models for Phase 1.

This module defines the complete schema for YAML workflow definitions, including:
- Workflow metadata (name, description, tags)
- Input declarations with types and defaults
- Block definitions with dependencies
- Output mappings with variable substitution
- Comprehensive validation logic

The schema validates:
- YAML syntax and structure
- Required fields and types
- Block type existence in registry
- Dependency validity (no cycles, valid references)
- Variable substitution syntax (${block_id.field})
"""

import re
from enum import Enum
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Block type validation moved to Block.__init__ (requires ExecutorRegistry instance)
# Schema validation only checks structural validity (YAML structure, dependencies, etc.)
from .dag import DAGResolver
from .load_result import LoadResult


class ValueType(str, Enum):
    """
    Unified Python type system for workflow values.

    Single source of truth for all type declarations across:
    - Workflow inputs
    - Block outputs (file-based)
    - Workflow outputs
    - Output parsing logic

    These match Python's built-in types for consistency with:
    - isinstance() checks in conditions: isinstance(${inputs.name}, str)
    - Type annotations in executor models: Field(default="", description="...")
    - Variable resolution return types

    Type mappings:
    - str: Text values (Python str)
    - int: Integer values (Python int)
    - float: Floating-point values (Python float)
    - bool: Boolean values (Python bool)
    - list: List/array values (Python list)
    - dict: Dictionary/object values (Python dict)
    - json: Special - parse as JSON (returns dict/list/str/int/float/bool/None)
    """

    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    JSON = "json"  # Special: JSON parsing

    @classmethod
    def input_types(cls) -> tuple[str, ...]:
        """Get tuple of input type values (all types)."""
        return (
            cls.STR.value,
            cls.INT.value,
            cls.FLOAT.value,
            cls.BOOL.value,
            cls.LIST.value,
            cls.DICT.value,
        )

    @classmethod
    def output_types(cls) -> tuple[str, ...]:
        """Get tuple of output type values (excludes list/dict, adds json)."""
        return (cls.STR.value, cls.INT.value, cls.FLOAT.value, cls.BOOL.value, cls.JSON.value)


# Input types - uses ValueType for consistency
class InputType(str, Enum):
    """
    Python native types for workflow input parameters.

    Reuses ValueType for consistency across the type system.
    See ValueType for full documentation.
    """

    STR = ValueType.STR.value
    INT = ValueType.INT.value
    FLOAT = ValueType.FLOAT.value
    BOOL = ValueType.BOOL.value
    LIST = ValueType.LIST.value
    DICT = ValueType.DICT.value


class WorkflowMetadata(BaseModel):
    """
    Workflow metadata for identification and documentation.

    Attributes:
        name: Unique workflow identifier (kebab-case recommended)
        description: Human-readable workflow description
        version: Semantic version string (default: "1.0")
        author: Optional workflow author
        tags: List of searchable tags for organization and discovery
    """

    name: str = Field(
        description="Unique workflow identifier (kebab-case)",
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
        min_length=1,
        max_length=100,
    )
    description: str = Field(description="Human-readable workflow description", min_length=1)
    version: str = Field(
        default="1.0", description="Semantic version", pattern=r"^\d+\.\d+(\.\d+)?$"
    )
    author: str | None = Field(default=None, description="Workflow author")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")

    model_config = {"extra": "forbid"}


class OutputSchema(BaseModel):
    """
    Schema for block output declaration.

    Defines file-based outputs that blocks can declare. The workflow engine
    validates paths and reads files after block execution.

    Attributes:
        type: Output type (str, int, float, bool, json)
        path: Relative or absolute file path
        description: Optional human-readable output description
        validation: Optional Python expression for validation
        unsafe: Allow absolute paths (default: False for security)
        required: Whether output is required (default: True)

    Example:
        outputs:
          test_results:
            type: json
            path: "$SCRATCH/test-results.json"
            description: "Test execution results"
            required: true
          coverage_percent:
            type: float
            path: ".scratch/coverage.txt"
            description: "Code coverage percentage"
    """

    type: ValueType = Field(
        description="Output type: Python native types or 'json' for JSON parsing"
    )
    path: str = Field(description="Relative or absolute file path", min_length=1)
    description: str | None = Field(default=None, description="Human-readable description")
    validation: str | None = Field(
        default=None, description="Optional Python expression for validation"
    )
    unsafe: bool = Field(
        default=False, description="Allow absolute paths (security risk if enabled)"
    )
    required: bool = Field(default=True, description="Whether output is required")

    model_config = {"extra": "forbid"}


class DependencySpec(BaseModel):
    """
    Dependency specification with skip propagation control.

    Defines a dependency on another block with control over whether the
    dependent block (child) skips if the dependency (parent) fails/skips.

    Attributes:
        block: Block ID of the parent dependency
        required: Whether to skip THIS block if parent fails/skips (default: True)
            - True (default): Skip this block unless parent completes successfully
            - False (optional): Run this block even if parent fails/skips (ordering only)

    Example:
        depends_on:
          - block: build
            required: true   # Skip THIS block if build fails/skips (default)
          - block: lint
            required: false  # Run THIS block even if lint fails/skips (optional)
    """

    block: str = Field(
        description="Block ID of the parent dependency",
        pattern=r"^[a-z_][a-z0-9_]*$",
        min_length=1,
        max_length=100,
    )
    required: bool = Field(
        default=True,
        description=(
            "Whether THIS block skips if parent dependency fails/skips "
            "(default: true - skip unless parent succeeds)"
        ),
    )

    model_config = {"extra": "forbid"}


class WorkflowInputDeclaration(BaseModel):
    """
    Workflow input parameter declaration.

    Defines expected runtime inputs with types, descriptions, and defaults.

    Attributes:
        type: Input type (string, integer, boolean, array, object)
        description: Human-readable input description
        default: Optional default value (must match type)
        required: Whether input is required (default: True if no default)

    Example:
        inputs:
          branch_name:
            type: string
            description: "Git branch name"
            default: "main"
          issue_number:
            type: integer
            description: "GitHub issue number"
            required: true
    """

    type: InputType = Field(description="Input value type")
    description: str = Field(description="Human-readable description", min_length=1)
    default: Any | None = Field(default=None, description="Default value (must match type)")
    required: bool = Field(default=True, description="Whether input is required")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_default_type(self) -> "WorkflowInputDeclaration":
        """Validate that default value matches declared type."""
        if self.default is None:
            return self

        # Type validation mapping (Python types)
        type_validators = {
            InputType.STR: lambda v: isinstance(v, str),
            InputType.INT: lambda v: isinstance(v, int) and not isinstance(v, bool),
            InputType.FLOAT: lambda v: isinstance(v, float) and not isinstance(v, bool),
            InputType.BOOL: lambda v: isinstance(v, bool),
            InputType.LIST: lambda v: isinstance(v, list),
            InputType.DICT: lambda v: isinstance(v, dict),
        }

        validator = type_validators.get(self.type)
        if validator and not validator(self.default):
            raise ValueError(
                f"Default value {self.default!r} does not match declared type '{self.type.value}'"
            )

        return self

    @model_validator(mode="after")
    def validate_required_with_default(self) -> "WorkflowInputDeclaration":
        """If default is provided, required should be False."""
        if self.default is not None and self.required:
            # Auto-correct: if default exists, input is not required
            self.required = False

        return self


class BlockDefinition(BaseModel):
    """
    Workflow block definition.

    Defines a single block in the workflow with its inputs and dependencies.

    Attributes:
        id: Unique block identifier within workflow
        type: Block type name (must exist in EXECUTOR_REGISTRY)
        description: Optional human-readable block description for documentation
        inputs: Block input parameters (dict with variable substitution support)
        depends_on: List of block IDs this block depends on
        condition: Optional boolean expression for conditional execution
        outputs: Optional custom file-based outputs (for Shell blocks)

    Example:
        blocks:
          - id: create_worktree
            type: CreateWorktree
            description: "Create isolated git worktree for feature development"
            inputs:
              branch: "feature/${issue_number}"
              base_branch: "main"

          - id: create_file
            type: CreateFile
            description: "Create initial README file in worktree"
            inputs:
              path: "${create_worktree.worktree_path}/README.md"
              content: "# Feature"
            depends_on:
              - create_worktree

          - id: run_tests
            type: Shell
            description: "Run test suite and capture results as JSON"
            inputs:
              command: "pytest --json-report --json-report-file=$SCRATCH/results.json"
            outputs:
              test_results:
                type: json
                path: "$SCRATCH/results.json"
                description: "Test execution results"

          - id: deploy
            type: Shell
            description: "Deploy to production if tests pass"
            inputs:
              command: "echo 'Deploying...'"
            condition: "${run_tests.exit_code} == 0"
            depends_on:
              - run_tests
    """

    id: str = Field(
        description="Unique block identifier",
        pattern=r"^[a-z_][a-z0-9_]*$",
        min_length=1,
        max_length=100,
    )
    type: str = Field(description="Block type name from EXECUTOR_REGISTRY", min_length=1)
    description: str = Field(
        default="",
        description="Optional human-readable block description for documentation",
    )
    inputs: dict[str, Any] = Field(default_factory=dict, description="Block input parameters")
    depends_on: list[DependencySpec] = Field(
        default_factory=list,
        description="Dependencies with optional 'required' flag (normalized from strings or dicts)",
    )
    condition: str | None = Field(
        default=None,
        description="Optional condition expression for conditional execution",
    )
    outputs: dict[str, OutputSchema] | None = Field(
        default=None, description="Custom file-based outputs"
    )

    model_config = {"extra": "forbid"}

    @field_validator("depends_on", mode="before")
    @classmethod
    def normalize_depends_on(cls, v: Any) -> list[dict[str, Any]]:
        """
        Normalize depends_on to uniform dict format and validate for duplicates.

        Accepts:
        - List of strings: ["block1", "block2"] → [{"block": "block1", "required": true}, ...]
        - List of dicts: [{"block": "block1", "required": true}] → unchanged
        - Mixed: ["block1", {"block": "block2", "required": false}] → all converted to dicts

        Returns:
            List of dicts with 'block' and 'required' keys (normalized format)
        """
        if not isinstance(v, list):
            raise ValueError("depends_on must be a list")

        # Track block IDs to check for duplicates
        seen_blocks = []
        normalized = []

        for item in v:
            if isinstance(item, str):
                # String format - convert to dict with required=True (default)
                block_id = item
                normalized.append({"block": block_id, "required": True})
            elif isinstance(item, dict):
                # Dict format - validate and normalize
                if "block" not in item:
                    raise ValueError(f"Dependency dict must have 'block' key: {item}")
                block_id = item["block"]
                # Ensure 'required' field exists (default to True)
                required = item.get("required", True)
                normalized.append({"block": block_id, "required": required})
            else:
                raise ValueError(
                    f"Invalid dependency format: {item}. Must be string or dict with 'block' key"
                )

            # Check for duplicates
            if block_id in seen_blocks:
                raise ValueError(f"Duplicate dependency on block '{block_id}'")
            seen_blocks.append(block_id)

        return normalized


class WorkflowOutputSchema(BaseModel):
    """
    Schema for workflow-level output.

    Defines outputs that the workflow exposes to callers. These are typically
    expressions that reference block outputs.

    Attributes:
        value: Expression (e.g., "${block.outputs.field}" or "${block.exit_code}")
        type: Output type (str, int, float, bool, json)
        description: Optional human-readable output description

    Example:
        outputs:
          test_results:
            value: "${run_tests.outputs.test_results}"
            type: json
            description: "Test execution results"
          success:
            value: "${run_tests.exit_code}"
            type: int
            description: "Test exit code"
    """

    value: str = Field(description="Expression referencing block outputs", min_length=1)
    type: ValueType = Field(
        description="Output type: Python native types or 'json' for JSON parsing"
    )
    description: str | None = Field(default=None, description="Human-readable description")

    model_config = {"extra": "forbid"}


class WorkflowSchema(BaseModel):
    """
    Complete YAML workflow schema.

    This is the root model for workflow definitions loaded from YAML files.
    It validates the entire workflow structure and provides conversion to
    the executor's WorkflowDefinition format.

    Attributes:
        name: Workflow name (from metadata)
        description: Workflow description (from metadata)
        version: Workflow version (from metadata)
        author: Optional workflow author (from metadata)
        tags: Searchable tags for organization and discovery
        inputs: Input parameter declarations
        blocks: Block definitions
        outputs: Output mappings with variable substitution

    Example YAML:
        name: example-workflow
        description: Example workflow with validation
        version: "1.0"
        tags: [test, example]

        inputs:
          input_name:
            type: string
            description: Input parameter
            default: "default_value"

        blocks:
          - id: block1
            type: Shell
            inputs:
              command: 'printf "Hello ${input_name}"'

          - id: block2
            type: Shell
            inputs:
              command: 'printf "Output: ${blocks.block1.outputs.stdout}"'
            depends_on:
              - block1

        outputs:
          final_message: "${blocks.block2.outputs.stdout}"
    """

    # Metadata fields (flattened for YAML convenience)
    name: str = Field(
        description="Unique workflow identifier",
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
        min_length=1,
        max_length=100,
    )
    description: str = Field(description="Workflow description", min_length=1)
    version: str = Field(default="1.0", pattern=r"^\d+\.\d+(\.\d+)?$")
    author: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)

    # Workflow structure
    inputs: dict[str, WorkflowInputDeclaration] = Field(
        default_factory=dict, description="Input parameter declarations"
    )
    blocks: list[BlockDefinition] = Field(description="Workflow block definitions", min_length=1)
    outputs: dict[str, str | WorkflowOutputSchema] = Field(
        default_factory=dict, description="Output mappings with variable substitution"
    )

    model_config = {"extra": "forbid"}

    @property
    def metadata(self) -> WorkflowMetadata:
        """Extract metadata as separate model."""
        return WorkflowMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            author=self.author,
            tags=self.tags,
        )

    @cached_property
    def execution_waves(self) -> list[list[str]]:
        """
        Compute execution waves (DAG resolution).

        Cached for performance - computed once at load time.

        Returns:
            List of waves, each wave is a list of block IDs
            that can execute in parallel.

        Raises:
            ValueError: If DAG resolution fails (cyclic dependencies)
        """
        # Extract block IDs and dependencies
        block_ids = [block.id for block in self.blocks]
        dependencies = {
            block.id: [self._get_block_id_from_dependency(dep) for dep in block.depends_on]
            for block in self.blocks
        }

        # Create DAGResolver and compute waves
        resolver = DAGResolver(block_ids, dependencies)
        result = resolver.get_execution_waves()

        if not result.is_success:
            raise ValueError(f"DAG resolution failed: {result.error}")

        # Type narrowing: result.is_success guarantees value is not None
        assert result.value is not None
        return result.value

    @field_validator("blocks")
    @classmethod
    def validate_unique_block_ids(cls, v: list[BlockDefinition]) -> list[BlockDefinition]:
        """Ensure all block IDs are unique."""
        block_ids = [block.id for block in v]
        if len(block_ids) != len(set(block_ids)):
            duplicates = [bid for bid in block_ids if block_ids.count(bid) > 1]
            raise ValueError(f"Duplicate block IDs found: {duplicates}")
        return v

    # Block type validation removed - now done at Block instantiation time
    # Block.__init__ validates types against the injected ExecutorRegistry
    # This allows for isolated registries per test and proper dependency injection

    @staticmethod
    def _get_block_id_from_dependency(dep: DependencySpec) -> str:
        """Extract block ID from DependencySpec."""
        return dep.block

    @model_validator(mode="after")
    def validate_dependencies_exist(self) -> "WorkflowSchema":
        """Validate that all dependencies reference existing blocks."""
        block_ids = {block.id for block in self.blocks}

        for block in self.blocks:
            for dep in block.depends_on:
                dep_block_id = self._get_block_id_from_dependency(dep)
                if dep_block_id not in block_ids:
                    raise ValueError(
                        f"Block '{block.id}' depends on non-existent block '{dep_block_id}'. "
                        f"Available blocks: {sorted(block_ids)}"
                    )

        return self

    @model_validator(mode="after")
    def validate_no_cyclic_dependencies(self) -> "WorkflowSchema":
        """Validate that dependencies form a valid DAG (no cycles)."""
        block_ids = [block.id for block in self.blocks]

        # Extract just block IDs from dependencies (string or DependencySpec)
        dependencies = {
            block.id: [self._get_block_id_from_dependency(dep) for dep in block.depends_on]
            for block in self.blocks
        }

        resolver = DAGResolver(block_ids, dependencies)
        result = resolver.topological_sort()

        if not result.is_success:
            raise ValueError(f"Invalid workflow dependencies: {result.error}")

        return self

    @model_validator(mode="after")
    def validate_variable_substitution_syntax(self) -> "WorkflowSchema":
        """Validate variable substitution syntax in all string values."""
        # Pattern: ${namespace.path.to.field} - captures full dotted path
        var_pattern = re.compile(r"\$\{([a-z_][a-z0-9_.]*)\}")

        block_ids = {block.id for block in self.blocks}
        input_names = set(self.inputs.keys())

        def validate_string_value(value: str, context: str) -> None:
            """Validate variable references in a string value."""
            matches = var_pattern.findall(value)
            for var_path in matches:
                parts = var_path.split(".")

                # ${inputs.field} - workflow input
                if parts[0] == "inputs":
                    if len(parts) < 2:
                        raise ValueError(
                            f"{context}: Invalid variable reference '${{{var_path}}}'. "
                            f"Input reference must include field name: ${{inputs.field_name}}"
                        )
                    field_name = parts[1]
                    if field_name not in input_names:
                        raise ValueError(
                            f"{context}: Invalid variable reference '${{{var_path}}}'. "
                            f"Input '{field_name}' does not exist. "
                            f"Available inputs: {sorted(input_names)}"
                        )

                # ${blocks.block_id.namespace.field} - block outputs/metadata
                # Also supports shortcut: ${blocks.block_id.field} -> outputs.field
                elif parts[0] == "blocks":
                    if len(parts) < 3:
                        raise ValueError(
                            f"{context}: Invalid variable reference '${{{var_path}}}'. "
                            f"Block reference must be: "
                            f"${{blocks.block_id.outputs.field}}, "
                            f"${{blocks.block_id.metadata.field}}, or "
                            f"${{blocks.block_id.field}} (shortcut for outputs)"
                        )
                    block_id = parts[1]

                    # Check if block exists
                    if block_id not in block_ids:
                        raise ValueError(
                            f"{context}: Invalid variable reference '${{{var_path}}}'. "
                            f"Block '{block_id}' does not exist. "
                            f"Available blocks: {sorted(block_ids)}"
                        )

                    # If 4+ parts, second level can be:
                    # - Standard namespaces: outputs, metadata, inputs
                    # - Custom output fields: any valid identifier
                    # Both are valid, so no validation needed beyond block existence
                    # If 3 parts, it's shortcut form ${blocks.block_id.field}
                    # This is valid and will be auto-expanded to outputs.field

                # ${metadata.field} - workflow metadata
                elif parts[0] == "metadata":
                    # Metadata references are valid (read-only workflow metadata)
                    pass

                # Unknown namespace
                else:
                    raise ValueError(
                        f"{context}: Invalid variable reference '${{{var_path}}}'. "
                        f"Unknown namespace '{parts[0]}'. "
                        f"Valid namespaces: 'inputs', 'blocks', 'metadata'"
                    )

        def check_dict_values(obj: Any, path: str) -> None:
            """Recursively check all string values in nested structures."""
            if isinstance(obj, str):
                validate_string_value(obj, path)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    check_dict_values(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    check_dict_values(item, f"{path}[{idx}]")

        # Validate block inputs
        for block in self.blocks:
            check_dict_values(block.inputs, f"Block '{block.id}' inputs")

        # Validate outputs
        for output_name, output_value in self.outputs.items():
            if isinstance(output_value, str):
                validate_string_value(output_value, f"Output '{output_name}'")

        return self

    @staticmethod
    def validate_yaml_dict(data: dict[str, Any]) -> LoadResult["WorkflowSchema"]:
        """
        Validate YAML dictionary against schema with detailed error messages.

        This is the primary validation entry point for loaded YAML data.

        Args:
            data: Dictionary loaded from YAML file

        Returns:
            LoadResult.success(WorkflowSchema) if valid
            LoadResult.failure(error_message) with clear validation errors

        Example:
            import yaml

            with open("workflow.yaml") as f:
                data = yaml.safe_load(f)

            result = WorkflowSchema.validate_yaml_dict(data)
            if result.is_success:
                workflow_schema = result.value
                # Use workflow_schema directly with executor
                executor.load_workflow(workflow_schema)
            else:
                print(f"Validation failed: {result.error}")
        """
        try:
            schema = WorkflowSchema(**data)
            return LoadResult.success(schema)
        except Exception as e:
            # Extract meaningful error from Pydantic ValidationError
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                # Pydantic v2 provides detailed error messages
                return LoadResult.failure(f"Workflow validation failed:\n{error_msg}")
            else:
                return LoadResult.failure(f"Workflow validation failed: {error_msg}")
