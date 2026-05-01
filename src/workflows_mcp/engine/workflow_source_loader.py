from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .loader import load_workflow_from_file
from .registry import WorkflowRegistry


@dataclass(frozen=True)
class WorkflowSourceReloadSummary:
    source_count: int
    workflow_count: int
    workflow_names: list[str]


class WorkflowSourceReloadError(Exception):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def reload_registry_from_source_paths(
    registry: WorkflowRegistry, source_paths: Sequence[str | Path]
) -> WorkflowSourceReloadSummary:
    if not source_paths:
        registry.clear()
        return WorkflowSourceReloadSummary(source_count=0, workflow_count=0, workflow_names=[])

    temp_registry = WorkflowRegistry()
    normalized_sources = [_normalize_source_path(path) for path in source_paths]

    for source_path in normalized_sources:
        yaml_files = sorted([*source_path.glob("**/*.yaml"), *source_path.glob("**/*.yml")])
        for yaml_file in yaml_files:
            result = load_workflow_from_file(yaml_file)
            if not result.is_success or result.value is None:
                raise WorkflowSourceReloadError(
                    code="workflow_invalid_definition",
                    message=(
                        "Invalid workflow definition "
                        f"at '{yaml_file}': {result.error or 'schema validation returned None'}"
                    ),
                )

            workflow = result.value
            if temp_registry.exists(workflow.name):
                prior_source = temp_registry.get_workflow_source(workflow.name)
                raise WorkflowSourceReloadError(
                    code="workflow_duplicate_name",
                    message=(
                        f"Duplicate workflow name '{workflow.name}' found in '{yaml_file}'"
                        f" (already loaded from '{prior_source}')"
                    ),
                )

            temp_registry.register(workflow, source_dir=source_path)

    registry.replace_with(temp_registry)
    names = registry.list_names()
    return WorkflowSourceReloadSummary(
        source_count=len(normalized_sources),
        workflow_count=len(names),
        workflow_names=names,
    )


def _normalize_source_path(path: str | Path) -> Path:
    normalized = Path(path).expanduser().resolve(strict=False)
    if not normalized.exists() or not normalized.is_dir():
        raise WorkflowSourceReloadError(
            code="workflow_source_invalid_path",
            message=f"Workflow source path is missing or not a directory: '{normalized}'",
        )
    return normalized
