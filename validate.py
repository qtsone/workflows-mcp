#!/usr/bin/env python3
"""Validate workflow files using the same loading mechanism as MCP server.

This script replicates the exact workflow loading process from the MCP server
to help debug workflow validation issues.

Usage:
    python validate_workflow.py <workflow_file_or_directory>
    python validate_workflow.py local/simple-loop-counter.yaml
    python validate_workflow.py local/
"""

import sys
from pathlib import Path

from workflows_mcp.engine import WorkflowRegistry
from workflows_mcp.engine.loader import load_workflow_from_file


def validate_workflow_file(file_path: Path) -> tuple[bool, str]:
    """Validate a single workflow file.

    Returns:
        Tuple of (success, message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if not file_path.is_file():
        return False, f"Not a file: {file_path}"

    if file_path.suffix not in ['.yaml', '.yml']:
        return False, f"Not a YAML file: {file_path}"

    # Load using the same mechanism as MCP server
    result = load_workflow_from_file(file_path)

    if not result.is_success:
        return False, f"Load failed: {result.error}"

    # Successfully loaded
    workflow = result.value
    return True, f"✓ Valid workflow: {workflow.name}"


def validate_directory(dir_path: Path) -> dict[str, tuple[bool, str]]:
    """Validate all workflow files in a directory.

    Returns:
        Dictionary mapping file paths to (success, message) tuples
    """
    results = {}

    if not dir_path.exists():
        return {"error": (False, f"Directory not found: {dir_path}")}

    if not dir_path.is_dir():
        return {"error": (False, f"Not a directory: {dir_path}")}

    # Find all YAML files
    yaml_files = list(dir_path.glob("**/*.yaml")) + list(dir_path.glob("**/*.yml"))

    if not yaml_files:
        return {"error": (False, f"No YAML files found in: {dir_path}")}

    for yaml_file in sorted(yaml_files):
        success, message = validate_workflow_file(yaml_file)
        results[str(yaml_file)] = (success, message)

    return results


def test_registry_loading(paths: list[Path]) -> None:
    """Test loading workflows into registry like MCP server does.

    This replicates the exact loading process from server.py
    """
    print("\n" + "="*80)
    print("Testing Registry Loading (MCP Server Pattern)")
    print("="*80)

    registry = WorkflowRegistry()

    for path in paths:
        print(f"\nLoading from: {path}")

        if path.is_file():
            # Load single file
            result = load_workflow_from_file(path)
            if result.is_success:
                workflow = result.value
                try:
                    registry.register(workflow, source_dir=path.parent)
                    print(f"  ✓ Registered: {workflow.name}")
                except Exception as e:
                    print(f"  ✗ Registration failed: {e}")
            else:
                print(f"  ✗ Load failed: {result.error}")

        elif path.is_dir():
            # Load directory
            try:
                stats = registry.load_from_directory(
                    path,
                    on_duplicate="overwrite"
                )
                print(f"  ✓ Loaded: {stats['loaded']} workflows")
                print(f"  ✗ Failed: {stats['failed']} workflows")

                if stats['failed'] > 0:
                    print("\n  Failed files:")
                    for error in stats.get('errors', []):
                        print(f"    - {error}")

            except Exception as e:
                print(f"  ✗ Directory load failed: {e}")

    # Show final registry state
    print(f"\n{'='*80}")
    print("Registry Summary")
    print(f"{'='*80}")
    print(f"Total workflows: {len(registry.list_names())}")
    print(f"Workflows: {', '.join(sorted(registry.list_names()))}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target_path = Path(sys.argv[1])

    print("Workflow Validation Tool")
    print("="*80)

    if target_path.is_file():
        # Validate single file
        print(f"Validating file: {target_path}")
        print("-"*80)

        success, message = validate_workflow_file(target_path)
        print(message)

        if not success:
            sys.exit(1)

        # Also test registry loading
        test_registry_loading([target_path])

    elif target_path.is_dir():
        # Validate directory
        print(f"Validating directory: {target_path}")
        print("-"*80)

        results = validate_directory(target_path)

        success_count = 0
        failure_count = 0

        for file_path, (success, message) in results.items():
            print(f"\n{file_path}")
            print(f"  {message}")

            if success:
                success_count += 1
            else:
                failure_count += 1

        print(f"\n{'='*80}")
        print(f"Summary: {success_count} valid, {failure_count} failed")

        if failure_count > 0:
            sys.exit(1)

        # Also test registry loading
        test_registry_loading([target_path])

    else:
        print(f"Error: Path not found: {target_path}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("All validations passed!")


if __name__ == "__main__":
    main()
