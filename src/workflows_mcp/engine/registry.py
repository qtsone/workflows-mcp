"""
Workflow registry for managing loaded workflow definitions.

This module provides the WorkflowRegistry class, which acts as a central registry
for workflow definitions loaded from YAML files. It provides discovery, filtering,
and retrieval capabilities for MCP tools.

Features:
- Register workflows with duplicate detection
- Retrieve workflows by name
- List all workflows or filter by tags
- Load workflows from directories (recursive)
- Load workflows from multiple directories with priority ordering
- Track source directories for each workflow
- Clear registry for testing
- Thread-safe read operations (dict reads are atomic in CPython)
"""

import logging
import sys
from pathlib import Path
from typing import Any, Literal

from .load_result import LoadResult
from .loader import load_workflow_from_file
from .schema import WorkflowSchema

# Configure logging to stderr (MCP requirement)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class WorkflowRegistry:
    """
    Central registry for managing loaded workflow definitions.

    The registry maintains WorkflowSchema instances with:
    1. Complete workflow validation via Pydantic models
    2. Mapping between workflow names and schemas
    3. Source directory tracking for each workflow

    This enables MCP tools to:
    - List available workflows with metadata
    - Get detailed workflow information
    - Execute workflows by name

    Example:
        registry = WorkflowRegistry()
        registry.load_from_directory("templates/")

        # List all workflows
        workflows = registry.list_all()

        # Get specific workflow
        workflow = registry.get("create-python-project")

        # Execute via executor
        executor.load_workflow(workflow)
        result = await executor.execute_workflow("create-python-project", inputs)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        # Store WorkflowSchema instances
        self._workflows: dict[str, WorkflowSchema] = {}

        # Track source directory for each workflow
        self._workflow_sources: dict[str, Path] = {}

    def register(
        self,
        workflow: WorkflowSchema,
        source_dir: Path | None = None,
    ) -> None:
        """
        Register a workflow schema.

        Args:
            workflow: WorkflowSchema instance to register
            source_dir: Optional source directory path for tracking

        Raises:
            ValueError: If workflow with same name already exists
        """
        if workflow.name in self._workflows:
            raise ValueError(
                f"Workflow '{workflow.name}' already registered. Use clear() or unregister() first."
            )

        self._workflows[workflow.name] = workflow

        if source_dir is not None:
            self._workflow_sources[workflow.name] = source_dir

        logger.info(f"Registered workflow: {workflow.name}")

    def unregister(self, name: str) -> None:
        """
        Unregister a workflow by name.

        Args:
            name: Workflow name to unregister

        Raises:
            KeyError: If workflow not found
        """
        if name not in self._workflows:
            raise KeyError(f"Workflow '{name}' not found in registry")

        del self._workflows[name]
        if name in self._workflow_sources:
            del self._workflow_sources[name]

        logger.info(f"Unregistered workflow: {name}")

    def get(self, name: str) -> WorkflowSchema:
        """
        Get workflow by name.

        Args:
            name: Workflow name

        Returns:
            WorkflowSchema instance

        Raises:
            KeyError: If workflow not found
        """
        if name not in self._workflows:
            available = sorted(self._workflows.keys())
            raise KeyError(f"Workflow '{name}' not found. Available workflows: {available}")

        return self._workflows[name]

    def exists(self, name: str) -> bool:
        """
        Check if workflow exists in registry.

        Args:
            name: Workflow name

        Returns:
            True if workflow exists, False otherwise
        """
        return name in self._workflows

    def list_all(self) -> list[WorkflowSchema]:
        """
        List all registered workflows.

        Returns:
            List of all WorkflowSchema instances
        """
        return list(self._workflows.values())

    def list_names(self, tags: list[str] = []) -> list[str]:
        """
        List workflow names, optionally filtered by tags.

        Args:
            tags: Optional list of tags to filter workflows.
                  Workflows matching ALL tags are included (AND logic).

        Returns:
            Sorted list of workflow names
        """
        # If no tags filter, return all workflows
        if not tags:
            return sorted(self._workflows.keys())

        # Convert tags to set for efficient subset checking
        required_tags = set(tags)

        # Filter workflows by tags (must have ALL requested tags)
        filtered_names = []
        for name, workflow in self._workflows.items():
            workflow_tags = set(workflow.tags or [])
            # Include workflow only if it has ALL of the requested tags
            if required_tags.issubset(workflow_tags):
                filtered_names.append(name)

        return sorted(filtered_names)

    def get_workflow_metadata(self, name: str, detailed: bool = False) -> dict[str, Any]:
        """
        Get workflow metadata as dictionary (for MCP tools).

        Args:
            name: Workflow name
            detailed: If True, include version, author, and outputs (default: False)

        Returns:
            Dictionary with metadata fields.

            Default mode (detailed=False):
            - name: Workflow name
            - description: Workflow description
            - tags: List of tags
            - inputs: Input schema (type, description, required, default)

            Detailed mode (detailed=True):
            Additionally includes:
            - version: Workflow version
            - author: Workflow author (if available)
            - outputs: Output mappings

        Raises:
            KeyError: If workflow not found
        """
        workflow = self.get(name)

        # Build default metadata (name, description, tags, inputs)
        metadata: dict[str, Any] = {
            "name": workflow.name,
            "description": workflow.description,
            "tags": workflow.tags or [],
        }

        # Add inputs (always included)
        if workflow.inputs:
            metadata["inputs"] = {
                input_name: {
                    "type": decl.type.value,
                    "description": decl.description,
                    "required": decl.required,
                    "default": decl.default,
                }
                for input_name, decl in workflow.inputs.items()
            }
        else:
            metadata["inputs"] = {}

        # Add detailed fields if requested
        if detailed:
            metadata["version"] = workflow.version
            if workflow.author:
                metadata["author"] = workflow.author
            if workflow.outputs:
                metadata["outputs"] = workflow.outputs

        return metadata

    def list_all_metadata(self, detailed: bool = False) -> list[dict[str, Any]]:
        """
        List metadata for all workflows (for MCP tools).

        Args:
            detailed: If True, include version, author, and outputs (default: False)

        Returns:
            List of metadata dictionaries for all workflows
        """
        return [
            self.get_workflow_metadata(name, detailed=detailed)
            for name in sorted(self._workflows.keys())
        ]

    def list_metadata_by_tags(
        self, tags: list[str], match_all: bool = True, detailed: bool = False
    ) -> list[dict[str, Any]]:
        """
        List metadata for workflows filtered by tags (for MCP tools).

        Args:
            tags: List of tags to filter by
            match_all: If True, workflow must have ALL specified tags (AND semantics).
                      If False, workflow must have ANY specified tag (OR semantics).
                      Default: True (AND semantics)
            detailed: If True, include version, author, and outputs (default: False)

        Returns:
            List of metadata dictionaries for workflows matching the tag criteria

        Example:
            # Get all Python linting workflows
            registry.list_metadata_by_tags(["python", "linting"])
            # Returns workflows with BOTH "python" AND "linting" tags

            # Get all CI or testing workflows
            registry.list_metadata_by_tags(["ci", "testing"], match_all=False)
            # Returns workflows with EITHER "ci" OR "testing" tag
        """
        if not tags:
            return []

        result: list[dict[str, Any]] = []

        for name, workflow in self._workflows.items():
            if not workflow.tags:
                continue

            # Convert workflow tags to set for efficient lookup
            workflow_tags = set(workflow.tags)

            # Apply matching logic
            if match_all:
                # AND semantics: workflow must have ALL specified tags
                if all(tag in workflow_tags for tag in tags):
                    result.append(self.get_workflow_metadata(name, detailed=detailed))
            else:
                # OR semantics: workflow must have ANY specified tag
                if any(tag in workflow_tags for tag in tags):
                    result.append(self.get_workflow_metadata(name, detailed=detailed))

        return result

    def get_workflow_source(self, workflow_name: str) -> Path | None:
        """
        Get the source directory for a workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Path to source directory, or None if not tracked

        Example:
            source = registry.get_workflow_source("setup-python-env")
            print(f"Source: {source}")  # library/python/
        """
        return self._workflow_sources.get(workflow_name)

    def list_by_source(self, source_dir: Path) -> list[str]:
        """
        List all workflow names from a specific source directory.

        Args:
            source_dir: Source directory path to filter by

        Returns:
            List of workflow names from the specified source directory

        Example:
            lib_workflows = registry.list_by_source(Path("library/"))
            print(f"Library workflows: {lib_workflows}")
        """
        return [name for name, src in self._workflow_sources.items() if src == source_dir]

    def load_from_directory(self, directory: str | Path) -> LoadResult[int]:
        """
        Load all workflows from a directory (recursive).

        This method:
        1. Finds and loads all YAML files
        2. Registers each successfully loaded workflow
        3. Logs warnings for invalid workflows (but continues loading)
        4. Returns count of successfully loaded workflows

        Args:
            directory: Directory path to search for workflow YAML files

        Returns:
            LoadResult.success(count) with number of workflows loaded
            LoadResult.failure(error_message) if directory doesn't exist

        Example:
            result = registry.load_from_directory("templates/")
            if result.is_success:
                print(f"Loaded {result.value} workflows")
        """
        dir_path = Path(directory)

        logger.info(f"Loading workflows from directory: {dir_path}")

        if not dir_path.exists():
            error_msg = f"Directory not found: {dir_path}"
            logger.error(error_msg)
            return LoadResult.failure(error_msg)

        if not dir_path.is_dir():
            error_msg = f"Not a directory: {dir_path}"
            logger.error(error_msg)
            return LoadResult.failure(error_msg)

        # Find all YAML files
        yaml_files = list(dir_path.glob("**/*.yaml")) + list(dir_path.glob("**/*.yml"))

        loaded_count = 0
        for yaml_file in yaml_files:
            # Load workflow schema
            workflow_result = load_workflow_from_file(yaml_file)

            if not workflow_result.is_success:
                logger.warning(
                    f"Failed to load workflow from {yaml_file.name}: {workflow_result.error}"
                )
                continue

            workflow = workflow_result.value
            if workflow is None:
                logger.warning(f"Workflow from {yaml_file.name} is None")
                continue

            # Register workflow (skip duplicates)
            try:
                self.register(workflow, source_dir=dir_path)
                loaded_count += 1
            except ValueError as e:
                logger.warning(f"Skipping duplicate workflow: {e}")
                continue

        logger.info(
            f"Successfully loaded {loaded_count} workflows from {dir_path} "
            f"({len(yaml_files)} YAML files found)"
        )

        return LoadResult.success(loaded_count)

    def load_from_directories(
        self,
        directories: list[str | Path],
        on_duplicate: Literal["skip", "overwrite", "error"] = "skip",
    ) -> LoadResult[dict[str, int]]:
        """
        Load workflows from multiple directories in priority order.

        Directories are processed in order. The on_duplicate parameter controls
        how duplicate workflow names are handled:
        - "skip": Keep first loaded workflow (default)
        - "overwrite": Replace with later version
        - "error": Raise error on duplicate

        Args:
            directories: List of directory paths (ordered by priority)
            on_duplicate: How to handle duplicate workflow names

        Returns:
            LoadResult.success(dict) with workflows loaded per directory
            LoadResult.failure(error_message) on error

        Example:
            result = registry.load_from_directories([
                "templates/",      # Built-in workflows (priority 1)
                "library/",        # Reusable library (priority 2)
                "~/.workflows/"    # User workflows (priority 3)
            ], on_duplicate="skip")

            if result.is_success:
                print(f"Loaded: {result.value}")
                # Output: {'templates/': 11, 'library/': 7, '~/.workflows/': 3}
        """
        if not directories:
            return LoadResult.failure("No directories provided")

        # Convert all paths to absolute Path objects
        dir_paths = [Path(d).resolve() for d in directories]

        # Track results per directory
        results: dict[str, int] = {}
        all_errors: list[str] = []

        logger.info(
            f"Loading workflows from {len(dir_paths)} directories (on_duplicate={on_duplicate})"
        )

        for dir_path in dir_paths:
            logger.info(f"Processing directory: {dir_path}")

            # Check if directory exists
            if not dir_path.exists():
                error_msg = f"Directory not found: {dir_path}"
                logger.warning(error_msg)
                all_errors.append(error_msg)
                results[str(dir_path)] = 0
                continue

            if not dir_path.is_dir():
                error_msg = f"Not a directory: {dir_path}"
                logger.warning(error_msg)
                all_errors.append(error_msg)
                results[str(dir_path)] = 0
                continue

            # Find all YAML files
            yaml_files = list(dir_path.glob("**/*.yaml")) + list(dir_path.glob("**/*.yml"))

            loaded_count = 0
            for yaml_file in yaml_files:
                # Load workflow schema
                workflow_result = load_workflow_from_file(yaml_file)

                if not workflow_result.is_success:
                    logger.warning(
                        f"Failed to load workflow from {yaml_file.name}: {workflow_result.error}"
                    )
                    continue

                workflow = workflow_result.value
                if workflow is None:
                    logger.warning(f"Workflow from {yaml_file.name} is None")
                    continue

                # Handle duplicates based on policy
                if workflow.name in self._workflows:
                    if on_duplicate == "skip":
                        logger.info(
                            f"Skipping duplicate workflow '{workflow.name}' from "
                            f"{dir_path} (keeping existing from "
                            f"{self._workflow_sources.get(workflow.name, 'unknown')})"
                        )
                        continue
                    elif on_duplicate == "overwrite":
                        logger.info(
                            f"Overwriting workflow '{workflow.name}' from "
                            f"{self._workflow_sources.get(workflow.name, 'unknown')} "
                            f"with version from {dir_path}"
                        )
                        self.unregister(workflow.name)
                    elif on_duplicate == "error":
                        error_msg = (
                            f"Duplicate workflow '{workflow.name}' found in "
                            f"{dir_path} (already exists from "
                            f"{self._workflow_sources.get(workflow.name, 'unknown')})"
                        )
                        logger.error(error_msg)
                        return LoadResult.failure(error_msg)

                # Register workflow
                try:
                    self.register(workflow, source_dir=dir_path)
                    loaded_count += 1
                except ValueError as e:
                    # This should not happen with our duplicate handling above,
                    # but handle it just in case
                    logger.warning(f"Failed to register workflow: {e}")
                    continue

            results[str(dir_path)] = loaded_count
            logger.info(
                f"Loaded {loaded_count} workflows from {dir_path} "
                f"({len(yaml_files)} YAML files found)"
            )

        total_loaded = sum(results.values())
        logger.info(
            f"Successfully loaded {total_loaded} total workflows from {len(dir_paths)} directories"
        )

        return LoadResult.success(results)

    def clear(self) -> None:
        """
        Clear all registered workflows.

        This is primarily useful for testing to reset the registry state.
        """
        count = len(self._workflows)
        self._workflows.clear()
        self._workflow_sources.clear()
        logger.info(f"Cleared {count} workflows from registry")

    def __len__(self) -> int:
        """Return number of registered workflows."""
        return len(self._workflows)

    def __contains__(self, name: str) -> bool:
        """Check if workflow exists using 'in' operator."""
        return name in self._workflows

    def __repr__(self) -> str:
        """String representation of registry."""
        return f"<WorkflowRegistry: {len(self._workflows)} workflows>"
