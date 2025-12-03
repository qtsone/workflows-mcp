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


def collect_all_input_fields() -> dict[str, dict[str, Any]]:
    """
    Collect all unique input field names from all block executors.

    Returns a dict mapping field names to their schema definitions.
    Fields that appear in multiple block types are merged (using string type
    since most fields support interpolation).

    All schemas are OpenAI strict mode compliant:
    - Objects use additionalProperties: {"type": "string"} (not true)
    - Arrays use items: {"type": "string"} (not empty {})
    """
    registry = create_default_registry()
    all_fields: dict[str, dict[str, Any]] = {}

    # Fields that are objects with string-valued properties (key-value maps)
    # These use additionalProperties: {"type": "string"} for OpenAI compliance
    string_map_fields = {"env", "headers"}

    # Fields that are objects with potentially complex nested structures
    # We use additionalProperties: {"type": "string"} as a compromise
    # (LLM can generate JSON strings for complex values)
    complex_object_fields = {"json", "inputs", "data", "updates", "response_schema"}

    # Fields that are arrays of strings
    string_array_fields = {"patterns", "exclude_patterns"}

    # Fields that are arrays of objects (edit operations)
    # Use items with string additionalProperties for flexibility
    object_array_fields = {"operations"}

    for type_name in registry.list_types():
        executor = registry.get(type_name)
        input_schema = executor.input_type.model_json_schema()
        properties = input_schema.get("properties", {})

        for field_name, field_info in properties.items():
            if field_name in all_fields:
                # Already have this field, skip
                continue

            desc = field_info.get("description", f"Input field for {type_name}")

            # Determine the appropriate schema for OpenAI strict mode compliance
            if field_name in string_map_fields:
                # Simple string-valued key-value maps
                all_fields[field_name] = {
                    "type": "object",
                    "description": desc,
                    "additionalProperties": {"type": "string"},
                }
            elif field_name in complex_object_fields:
                # Complex objects - use string additionalProperties as compromise
                all_fields[field_name] = {
                    "type": "object",
                    "description": desc,
                    "additionalProperties": {"type": "string"},
                }
            elif field_name in string_array_fields:
                # Arrays of strings (glob patterns, etc.)
                all_fields[field_name] = {
                    "type": "array",
                    "description": desc,
                    "items": {"type": "string"},
                }
            elif field_name in object_array_fields:
                # Arrays of objects (edit operations)
                all_fields[field_name] = {
                    "type": "array",
                    "description": desc,
                    "items": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                }
            else:
                # Default to string (supports interpolation like "{{inputs.value}}")
                all_fields[field_name] = {
                    "type": "string",
                    "description": desc,
                }

    return all_fields


def generate_llm_schema() -> dict[str, Any]:
    """
    Generate simplified workflow schema for LLM structured outputs.

    This schema is compatible with OpenAI's strict mode requirements:
    - additionalProperties: false everywhere (except where nested objects need flexibility)
    - All possible block input fields listed as optional properties
    - No patternProperties or complex constraints
    """
    # Collect all possible input fields from all block types
    all_input_fields = collect_all_input_fields()

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
                        # LLM compatibility: use simple types with descriptions
                        # indicating optionality. Many providers (LMStudio, etc.)
                        # don't support ["type", "null"] unions.
                        "description": {
                            "type": "string",
                            "description": (
                                "Optional block description (empty string if not needed)"
                            ),
                        },
                        "condition": {
                            "type": "string",
                            "description": (
                                "Optional condition for execution (empty string if not needed)"
                            ),
                        },
                        "continue_on_error": {
                            "type": "boolean",
                            "description": (
                                "Continue workflow if block fails (false if not needed)"
                            ),
                        },
                        "for_each": {
                            "type": "string",
                            "description": (
                                "Expression for iterating over a list (empty string if none)"
                            ),
                        },
                        "for_each_mode": {
                            "type": "string",
                            "description": (
                                "Iteration mode: parallel or sequential (empty string if none)"
                            ),
                        },
                        "depends_on": {
                            "type": "array",
                            "description": (
                                "Block dependencies as block IDs (empty array [] if none)"
                            ),
                            "items": {"type": "string"},
                        },
                        "inputs": {
                            "type": "object",
                            "description": (
                                "Block-specific inputs. Fields vary by block type: "
                                "Shell uses 'command', LLMCall/Prompt use 'prompt', "
                                "CreateFile uses 'path'+'content', etc."
                            ),
                            "properties": all_input_fields,
                            # All fields optional - different blocks need different fields
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                    # OpenAI strict mode requires ALL properties in required array
                    "required": [
                        "id",
                        "type",
                        "description",
                        "condition",
                        "continue_on_error",
                        "for_each",
                        "for_each_mode",
                        "depends_on",
                        "inputs",
                    ],
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
