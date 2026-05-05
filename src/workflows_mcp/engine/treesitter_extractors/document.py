"""Document-language extractor for Markdown, YAML, and JSON.

Document languages are file-only: they produce exactly one File entity and no
Module, Class, Function, or Method entities. No relations or unresolved_imports
are emitted.

Design rationale:
- Markdown/YAML/JSON files contain structured prose or configuration data, not
  code symbols. Emitting Module/Class/Function would produce meaningless graph
  nodes with no semantic value for these formats.
- `module_qualified_name` in TreeSitterOutput is non-empty for output model
  compatibility and traceability, but NO Module entity is emitted for document
  languages. The caller (executors_treesitter) sets module_qualified_name to the
  repo-relative qualified name (derived by _module_qname) before delegating here.

Grammar compatibility:
- The executor optionally parses the file via get_parser(language) to exercise
  the grammar and surface parse errors early (per ADR-006). This extractor does
  NOT use parse results to emit any symbol entities or relations.
"""

from __future__ import annotations

from typing import Any


def extract_document(
    *,
    file_entity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract entities and relations for a document-language file.

    Document languages (Markdown, YAML, JSON) are file-only: only a single
    File entity is emitted, with no Module, Class, Function, or Method entities
    and no relations of any kind.

    Args:
        file_entity: Pre-built File entity dict from the executor (must have
            entity_type='File', name, qualified_name, stable_id, metadata,
            confidence keys).

    Returns:
        A tuple (entities, relations) where:
        - entities contains exactly one item: the provided file_entity.
        - relations is always an empty list.

    Note:
        module_qualified_name is set by the executor on TreeSitterOutput for
        output model compatibility, but this function does NOT emit a Module
        entity. The field's value is the repo-relative qualified name for
        traceability purposes only.
    """
    # Return only the File entity; no Module, no relations.
    entities: list[dict[str, Any]] = [file_entity]
    relations: list[dict[str, Any]] = []
    return entities, relations
