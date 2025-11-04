#!/usr/bin/env python3
"""Generate schema.json from Pydantic models.

This script regenerates the workflow schema JSON file that provides:
- VS Code YAML autocomplete
- Workflow validation
- Documentation generation

Usage:
    python generate_schema.py
"""

import json
from pathlib import Path

from workflows_mcp.engine.executor_base import create_default_registry


def main():
    """Generate and save schema.json."""
    print("Generating workflow schema from executor registry...")

    # Create registry with all built-in executors
    registry = create_default_registry()

    # Generate complete schema
    schema = registry.generate_workflow_schema()

    # Save to schema.json in project root
    schema_path = Path(__file__).parent / "schema.json"

    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")  # Add trailing newline

    print(f"✓ Schema generated: {schema_path}")
    print(f"  Executors: {len([k for k in schema.get('$defs', {}).keys() if k.endswith('Input')])}")
    print(f"  Schema size: {schema_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
