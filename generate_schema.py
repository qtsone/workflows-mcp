#!/usr/bin/env python3
"""Generate schema.json and llm_schema.json from Pydantic models.

This script regenerates two workflow schema files:
1. schema.json - Complete schema for validation and documentation
2. llm_schema.json - Simplified schema compatible with OpenAI strict mode

Usage:
    python generate_schema.py
"""

import json
from pathlib import Path
from typing import Any

from workflows_mcp.engine.executor_base import create_default_registry


def generate_llm_schema() -> dict[str, Any]:
    """
    Generate simplified workflow schema for LLM structured outputs.

    This schema is compatible with OpenAI's strict mode requirements:
    - All properties are required
    - additionalProperties: false everywhere
    - No patternProperties or complex constraints
    - Simplified block structure with generic inputs
    """
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Workflow name",
            },
            "description": {
                "type": "string",
                "description": "Workflow description",
            },
            "tags": {
                "type": "array",
                "description": "Workflow tags (can be empty array)",
                "items": {
                    "type": "string",
                },
            },
            "inputs": {
                "type": "object",
                "description": "Workflow input parameters (can be empty object)",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["str", "num", "bool", "list", "dict"],
                            "description": "Input type",
                        },
                        "description": {
                            "type": "string",
                            "description": "Input description",
                        },
                        "required": {
                            "type": "boolean",
                            "description": "Whether input is required",
                        },
                    },
                    "required": ["type", "description", "required"],
                    "additionalProperties": False,
                },
            },
            "outputs": {
                "type": "object",
                "description": "Workflow output definitions (optional)",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "description": "Output value expression",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["str", "int", "float", "bool", "json", "list", "dict"],
                            "description": "Output type",
                        },
                        "description": {
                            "type": "string",
                            "description": "Output description",
                        },
                    },
                    "required": ["value", "type", "description"],
                    "additionalProperties": False,
                },
            },
            "blocks": {
                "type": "array",
                "description": "Workflow execution blocks",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique block identifier",
                        },
                        "type": {
                            "type": "string",
                            "description": "Block type (e.g., Shell, LLMCall, CreateFile)",
                        },
                        "condition": {
                            "type": "string",
                            "description": "Optional condition for conditional execution",
                        },
                        "depends_on": {
                            "type": "array",
                            "description": "Block dependencies",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "block": {
                                        "type": "string",
                                        "description": "ID of the block this depends on",
                                    },
                                    "required": {
                                        "type": "boolean",
                                        "description": "Whether dependency must succeed",
                                    },
                                },
                                "required": ["block", "required"],
                                "additionalProperties": False,
                            },
                        },
                        "inputs": {
                            "type": "object",
                            "description": "Block-specific inputs",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "Command or primary input for the block",
                                }
                            },
                            "required": ["command"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["id", "type", "condition", "depends_on", "inputs"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["name", "description", "tags", "blocks"],
        "additionalProperties": False,
    }


def main() -> None:
    """Generate and save schema files."""
    print("Generating workflow schemas...")

    # Create registry with all built-in executors
    registry = create_default_registry()

    # Generate complete schema
    schema = registry.generate_workflow_schema()

    # Save complete schema
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")

    print(f"✓ Complete schema: {schema_path}")
    print(f"  Executors: {len([k for k in schema.get('$defs', {}).keys() if k.endswith('Input')])}")
    print(f"  Size: {schema_path.stat().st_size:,} bytes")

    # Generate simplified LLM schema
    llm_schema = generate_llm_schema()

    # Save LLM schema
    llm_schema_path = Path(__file__).parent / "llm_schema.json"
    with open(llm_schema_path, "w") as f:
        json.dump(llm_schema, f, indent=2)
        f.write("\n")

    print(f"✓ LLM schema: {llm_schema_path}")
    print(f"  Size: {llm_schema_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
