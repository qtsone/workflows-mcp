#!/usr/bin/env python3
"""Generate schema.json from Pydantic models.

This script regenerates the workflow schema file:
- schema.json - Complete schema for validation and documentation

Usage:
    uv run python scripts/generate_schema.py
"""

import json
from pathlib import Path

from workflows_mcp.engine.executor_base import create_default_registry


def main() -> None:
    """Generate and save schema files."""
    print("Generating workflow schemas...")

    # Create registry with all built-in executors
    registry = create_default_registry()

    # Add the subsystem blocks — registered through their seams at runtime, but
    # always present in the generated schema.
    from workflows_mcp.code_intelligence import register_code_intelligence_executors
    from workflows_mcp.memory_runtime import register_memory_executors

    register_code_intelligence_executors(registry)
    register_memory_executors(registry)

    # Generate complete schema
    schema = registry.generate_workflow_schema()

    # Save complete schema
    schema_path = Path(__file__).parent.parent / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")

    print(f"✓ Complete schema: {schema_path}")
    print(f"  Executors: {len([k for k in schema.get('$defs', {}).keys() if k.endswith('Input')])}")
    print(f"  Size: {schema_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
