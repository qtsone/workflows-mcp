"""Tests for block reference schema extraction and docs metadata quality."""

from importlib import util
from pathlib import Path


def _load_generator_module() -> object:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_block_reference.py"
    spec = util.spec_from_file_location("generate_block_reference", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load_generator_module()
extract_schemas = _module.extract_schemas
extract_type_and_description = _module.extract_type_and_description


def test_extract_type_and_description_handles_unions_refs_and_fallbacks() -> None:
    """Type/description extraction supports unions, refs, and title fallback."""
    schema = {
        "$defs": {
            "Scope": {"type": "object", "description": "Scope payload"},
            "QueryText": {"type": "string", "title": "Query text"},
        },
        "properties": {
            "scope": {"$ref": "#/$defs/Scope"},
            "query": {
                "anyOf": [{"$ref": "#/$defs/QueryText"}, {"type": "array"}, {"type": "null"}],
                "title": "Query envelope",
            },
            "mode": {"type": ["string", "null"], "title": "Mode selection"},
            "graph": {"oneOf": [{"type": "object", "title": "Graph payload"}, {"type": "string"}]},
            "maintenance": {
                "allOf": [{"$ref": "#/$defs/Scope"}],
                "title": "Maintenance envelope",
            },
        },
    }

    scope_type, scope_desc = extract_type_and_description(schema["properties"]["scope"], schema)
    query_type, query_desc = extract_type_and_description(schema["properties"]["query"], schema)
    mode_type, mode_desc = extract_type_and_description(schema["properties"]["mode"], schema)
    graph_type, graph_desc = extract_type_and_description(schema["properties"]["graph"], schema)
    maintenance_type, maintenance_desc = extract_type_and_description(
        schema["properties"]["maintenance"], schema
    )

    assert scope_type == "object"
    assert scope_desc == "Scope payload"

    assert query_type == "string|array"
    assert query_desc == "Query envelope"

    assert mode_type == "string"
    assert mode_desc == "Mode selection"

    assert graph_type == "object|string"
    assert graph_desc == "Graph payload"

    assert maintenance_type == "object"
    assert maintenance_desc == "Maintenance envelope"


def test_extract_schemas_includes_memory_field_descriptions() -> None:
    """Memory fields expose explicit descriptions for docs generation."""
    schemas = extract_schemas()
    memory = schemas["Memory"]

    optional_by_name = {field["name"]: field for field in memory["optional"]}
    output_by_name = {field["name"]: field for field in memory["outputs"]}

    for name in ["host", "port", "database", "username", "password"]:
        assert optional_by_name[name]["description"]

    for name in ["scope", "query", "record", "graph", "maintenance", "response"]:
        assert optional_by_name[name]["description"]

    for name in ["result", "error"]:
        assert output_by_name[name]["description"]
