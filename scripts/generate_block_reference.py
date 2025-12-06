#!/usr/bin/env python3
"""Generate block-reference.md from actual Pydantic executor schemas.

This script extracts the authoritative block type schemas from the executor
registry and generates comprehensive documentation for LLM workflow generation.

Usage:
    python scripts/generate_block_reference.py

The output is written to docs/llm/block-reference.md
"""

import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflows_mcp.engine import create_default_registry

# Block type descriptions (canonical)
# REMOVED: BLOCK_DESCRIPTIONS and BLOCK_EXAMPLES are now extracted dynamically from executors.

# Configuration requirements that span multiple fields (not captured by Pydantic 'required')
# These are runtime requirements enforced by executors
CONFIGURATION_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "LLMCall": {
        "requirement": "profile OR provider",
        "description": (
            "LLMCall blocks REQUIRE exactly ONE of:\n"
            "  - `profile`: Load config from ~/.workflows/llm-config.yml\n"
            "  - `provider` + optionally `model`: Specify inline\n"
            "Without this, execution fails with 'LLM configuration required' error."
        ),
        "examples": {
            "valid_profile": "profile: default",
            "valid_provider": "provider: openai\nmodel: gpt-4o",
            "invalid": "# Missing both profile and provider - WILL FAIL",
        },
    },
    "ImageGen": {
        "requirement": "profile OR provider",
        "description": (
            "ImageGen blocks REQUIRE exactly ONE of:\n"
            "  - `profile`: Load config from ~/.workflows/llm-config.yml\n"
            "  - `provider`: Specify inline (openai, openai_compatible)\n"
            "Without this, execution fails with configuration error."
        ),
        "examples": {
            "valid_profile": "profile: default",
            "valid_provider": "provider: openai",
        },
    },
}


def extract_schemas() -> dict[str, dict[str, Any]]:
    """Extract schemas and metadata from all registered executors."""
    registry = create_default_registry()
    result = {}

    for type_name in registry.list_types():
        executor = registry.get(type_name)
        input_schema = executor.input_type.model_json_schema()

        # Handle executors without output types (e.g., Workflow uses NoneType)
        output_schema: dict[str, Any]
        has_no_output = (
            executor.output_type is None or str(executor.output_type) == "<class 'NoneType'>"
        )
        if has_no_output:
            output_schema = {"properties": {}}
        else:
            output_schema = executor.output_type.model_json_schema()

        required = input_schema.get("required", [])
        properties = input_schema.get("properties", {})

        required_fields = []
        optional_fields = []

        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "any")
            if isinstance(field_type, list):
                types = [t.get("type", "any") if isinstance(t, dict) else t for t in field_type]
                field_type = "|".join(str(t) for t in types if t != "null")

            desc = field_info.get("description", "")
            default = field_info.get("default")

            entry = {"name": field_name, "type": str(field_type), "description": desc}
            if default is not None and field_name not in required:
                entry["default"] = default

            if field_name in required:
                required_fields.append(entry)
            else:
                optional_fields.append(entry)

        # Get output fields
        out_props = output_schema.get("properties", {})
        output_fields = [
            {
                "name": k,
                "type": str(v.get("type", "any")),
                "description": v.get("description", ""),
            }
            for k, v in out_props.items()
        ]

        # Extract description from docstring (first line)
        description = "Execute block operations"
        if executor.__doc__:
            description = executor.__doc__.strip().split("\n")[0]

        # Extract example from ClassVar
        example = getattr(executor, "examples", "")

        result[type_name] = {
            "description": description,
            "example": example,
            "required": required_fields,
            "optional": optional_fields,
            "outputs": output_fields,
        }

    return result


def generate_markdown(schemas: dict[str, dict[str, Any]]) -> str:
    """Generate markdown documentation from schemas."""
    lines = []

    # Header
    lines.append("# Block Executor Reference")
    lines.append("")
    lines.append("> **This file is auto-generated.** Do not edit manually.")
    lines.append("> Run `python scripts/generate_block_reference.py` to regenerate.")
    lines.append("")
    lines.append("This reference is programmatically extracted from Pydantic executor models.")
    lines.append("Field names are **exact** - use them precisely in your workflows.")
    lines.append("")

    # Critical field names summary table
    lines.append("## CRITICAL: Required Input Field Names")
    lines.append("")
    lines.append("| Block Type | Required Fields |")
    lines.append("|------------|-----------------|")
    for block_type, schema in schemas.items():
        req_fields = [f["name"] for f in schema["required"]]
        # Add configuration requirements if they exist
        if block_type in CONFIGURATION_REQUIREMENTS:
            config_req = CONFIGURATION_REQUIREMENTS[block_type]["requirement"]
            if req_fields:
                req_str = f"{', '.join(req_fields)}; **{config_req}**"
            else:
                req_str = f"**{config_req}**"
        else:
            req_str = ", ".join(req_fields) if req_fields else "(none)"
        lines.append(f"| {block_type} | {req_str} |")
    lines.append("")

    # Common mistakes warning
    lines.append("### Common Field Name Mistakes")
    lines.append("")
    lines.append("| Wrong | Correct |")
    lines.append("|-------|---------|")
    lines.append("| LLMCall with `command` | LLMCall with `prompt` |")
    lines.append("| Prompt with `message` | Prompt with `prompt` |")
    lines.append("| Prompt with `command` | Prompt with `prompt` |")
    lines.append("| CreateFile with `command` | CreateFile with `path` and `content` |")
    lines.append("")

    # Detailed block documentation
    lines.append("---")
    lines.append("")

    for block_type, schema in schemas.items():
        lines.append(f"## {block_type}")
        lines.append("")
        lines.append(f"**Description**: {schema['description']}")
        lines.append("")

        # Add configuration requirement warning if applicable
        if block_type in CONFIGURATION_REQUIREMENTS:
            config = CONFIGURATION_REQUIREMENTS[block_type]
            lines.append("### ⚠️ Configuration Requirement")
            lines.append("")
            for line in config["description"].split("\n"):
                lines.append(line)
            lines.append("")

        if schema["required"]:
            lines.append("### Required Inputs")
            lines.append("")
            for field in schema["required"]:
                lines.append(f"- **`{field['name']}`** ({field['type']}): {field['description']}")
            lines.append("")

        if schema["optional"]:
            lines.append("### Optional Inputs")
            lines.append("")
            for field in schema["optional"]:
                default_str = ""
                if "default" in field:
                    default_str = f" *(default: `{field['default']}`)*"
                desc = field["description"]
                lines.append(f"- **`{field['name']}`** ({field['type']}){default_str}: {desc}")
            lines.append("")

        if schema["outputs"]:
            lines.append("### Outputs")
            lines.append("")
            for field in schema["outputs"]:
                lines.append(f"- **`{field['name']}`** ({field['type']}): {field['description']}")
            lines.append("")

        if schema["example"]:
            lines.append("### Example")
            lines.append("")
            lines.append(schema["example"])
            lines.append("")

        lines.append("---")
        lines.append("")

    # Variable access patterns
    lines.append("## Variable Access Patterns")
    lines.append("")
    lines.append("Use double braces `{{ }}` for variable interpolation in YAML:")
    lines.append("")
    lines.append("| Pattern | Description |")
    lines.append("|---------|-------------|")
    lines.append("| `{{inputs.param}}` | Access workflow input parameters |")
    lines.append("| `{{blocks.id.outputs.field}}` | Access another block's output |")
    lines.append("| `{{blocks.id.succeeded}}` | Check if block succeeded (boolean) |")
    lines.append("| `{{blocks.id.failed}}` | Check if block failed (boolean) |")
    lines.append("| `{{blocks.id.skipped}}` | Check if block was skipped (boolean) |")
    lines.append("| `{{secrets.NAME}}` | Access secrets (e.g., API keys) |")
    lines.append("| `{{tmp}}` | Temporary directory path |")
    lines.append("| `{{now()}}` | Current timestamp |")
    lines.append("")

    # Control flow
    lines.append("## Control Flow")
    lines.append("")
    lines.append("| Field | Description |")
    lines.append("|-------|-------------|")
    lines.append("| `depends_on: [block_id]` | Generate DAG. Expects parent blocks to succeed |")
    lines.append('| `condition: "{{expression}}"` | Conditional execution |')
    lines.append("| `continue_on_error: true` | Only used in sequential for_each |")
    lines.append('| `for_each: "{{list}}"` | Iterate over list items |')
    lines.append("| `for_each_mode: parallel` | Run iterations in parallel (default) |")
    lines.append("| `for_each_mode: sequential` | Run iterations sequentially |")
    lines.append("")

    # Jinja filters
    lines.append("## Jinja2 Filters")
    lines.append("")
    lines.append("Common filters for transforming values:")
    lines.append("")
    lines.append("| Filter | Description |")
    lines.append("|--------|-------------|")
    lines.append("| `trim` | Remove leading/trailing whitespace |")
    lines.append("| `lower` | Convert to lowercase |")
    lines.append("| `upper` | Convert to uppercase |")
    lines.append("| `replace(old, new)` | Replace text |")
    lines.append("| `length` | Get length of string or list |")
    lines.append("| `tojson` | Convert to JSON string |")
    lines.append("| `fromjson` | Parse JSON string |")
    lines.append("| `toyaml` | Convert to YAML string |")
    lines.append("")

    # Workflow Composition section
    lines.append("## Workflow Composition")
    lines.append("")
    lines.append("Use the `Workflow` block type to call other workflows (composition pattern):")
    lines.append("")
    lines.append("```yaml")
    lines.append("- id: process_data")
    lines.append("  type: Workflow")
    lines.append("  inputs:")
    lines.append('    workflow: "child-workflow-name"')
    lines.append("    inputs:")
    lines.append('      param1: "{{inputs.value}}"')
    lines.append('      param2: "{{blocks.previous.outputs.result}}"')
    lines.append("```")
    lines.append("")
    lines.append("### Accessing Child Workflow Outputs")
    lines.append("")
    lines.append("```yaml")
    lines.append("{{blocks.process_data.outputs.result}}  # Access child output")
    lines.append("{{blocks.process_data.succeeded}}       # Check success")
    lines.append("```")
    lines.append("")

    # Recursive Workflows section
    lines.append("## Recursive Workflows")
    lines.append("")
    lines.append("A workflow can call itself for iterative processing:")
    lines.append("")
    lines.append("```yaml")
    lines.append("name: recursive-processor")
    lines.append("inputs:")
    lines.append("  count: {type: num, default: 0}")
    lines.append("  max: {type: num, default: 5}")
    lines.append("")
    lines.append("blocks:")
    lines.append("  - id: process")
    lines.append("    type: Shell")
    lines.append("    inputs:")
    lines.append('      command: echo "Processing {{inputs.count}}"')
    lines.append("")
    lines.append("  - id: recurse")
    lines.append("    type: Workflow")
    lines.append("    depends_on: [process]")
    lines.append('    condition: "{{inputs.count < inputs.max}}"  # Termination condition')
    lines.append("    inputs:")
    lines.append("      workflow: recursive-processor  # Self-reference")
    lines.append("      inputs:")
    lines.append('        count: "{{inputs.count + 1}}"')
    lines.append('        max: "{{inputs.max}}"')
    lines.append("```")
    lines.append("")
    lines.append("**Important**: Always include a termination condition to prevent infinite loops.")
    lines.append("")

    # for_each Iteration section
    lines.append("## for_each Iteration")
    lines.append("")
    lines.append("Any block can iterate over collections using `for_each`:")
    lines.append("")
    lines.append("```yaml")
    lines.append("- id: process_items")
    lines.append("  type: Shell")
    lines.append('  for_each: "{{inputs.items}}"  # List or dict')
    lines.append("  for_each_mode: parallel       # parallel (default) or sequential")
    lines.append("  inputs:")
    lines.append('    command: "process {{each.value}}"')
    lines.append("```")
    lines.append("")
    lines.append("### Iteration Variables")
    lines.append("")
    lines.append("| Variable | Description |")
    lines.append("|----------|-------------|")
    lines.append("| `{{each.key}}` | Current key (index for lists) |")
    lines.append("| `{{each.value}}` | Current value |")
    lines.append("| `{{each.index}}` | Zero-based position |")
    lines.append("| `{{each.count}}` | Total iterations |")
    lines.append("")
    lines.append("### Accessing Iteration Results (Bracket Notation)")
    lines.append("")
    lines.append("```yaml")
    lines.append('{{blocks.process_items["0"].outputs.stdout}}       # By index')
    lines.append('{{blocks.process_items["key1"].outputs.result}}    # By key')
    lines.append("{{blocks.process_items.metadata.count}}            # Total count")
    lines.append("{{blocks.process_items.metadata.count_failed}}     # Failed count")
    lines.append("```")
    lines.append("")
    lines.append("### Dynamic Iteration (Generating Lists)")
    lines.append("")
    lines.append("**IMPORTANT**: `range()` is NOT valid in for_each expressions.")
    lines.append("To iterate a dynamic number of times, generate a list first:")
    lines.append("")
    lines.append("```yaml")
    lines.append("# Step 1: Generate list of indices")
    lines.append("- id: generate_indices")
    lines.append("  type: Shell")
    lines.append("  inputs:")
    lines.append("    command: |")
    lines.append('      python3 -c "import json; print(json.dumps(list(range({{inputs.count}}))))"')
    lines.append("")
    lines.append("# Step 2: Use the generated list in for_each")
    lines.append("- id: process_each")
    lines.append("  type: Shell")
    lines.append("  depends_on: [generate_indices]")
    lines.append('  for_each: "{{blocks.generate_indices.outputs.stdout | fromjson}}"')
    lines.append("  inputs:")
    lines.append('    command: "echo Processing item {{each.value}}"')
    lines.append("```")
    lines.append("")
    lines.append("### Common for_each Mistakes")
    lines.append("")
    lines.append("| Wrong | Correct |")
    lines.append("|-------|---------|")
    lines.append('| `for_each: "{{range(inputs.n)}}"` | Use Shell block to generate list |')
    lines.append("| `{{loop.index}}` | `{{each.index}}` |")
    lines.append("| `{{item}}` | `{{each.value}}` |")
    lines.append(
        '| `blocks.id.outputs.result` (for_each block) | `blocks.id["0"].outputs.result` |'
    )
    lines.append("")
    lines.append("### Nested Iteration (via Composition)")
    lines.append("")
    lines.append("For nested loops, use workflow composition:")
    lines.append("")
    lines.append("```yaml")
    lines.append("# Parent: iterate regions")
    lines.append("- id: process_regions")
    lines.append("  type: Workflow")
    lines.append('  for_each: "{{inputs.regions}}"')
    lines.append("  inputs:")
    lines.append("    workflow: process-servers")
    lines.append("    inputs:")
    lines.append('      region: "{{each.key}}"')
    lines.append('      servers: "{{each.value.servers}}"')
    lines.append("")
    lines.append("# Child workflow: process-servers.yaml")
    lines.append("# Iterates over servers within each region")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    # Determine output paths
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / "docs" / "llm"
    docs_dir.mkdir(parents=True, exist_ok=True)

    md_path = docs_dir / "block-reference.md"

    print("Extracting schemas from executor registry...")
    schemas = extract_schemas()

    print(f"Found {len(schemas)} block types:")
    for block_type in schemas:
        req = [f["name"] for f in schemas[block_type]["required"]]
        print(f"  - {block_type}: {', '.join(req) if req else '(no required fields)'}")

    print(f"\nWriting markdown to {md_path}...")
    md_content = generate_markdown(schemas)
    md_path.write_text(md_content)

    print("\nDone! Generated file:")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
