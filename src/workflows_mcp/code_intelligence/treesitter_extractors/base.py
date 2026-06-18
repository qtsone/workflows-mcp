"""Shared base for tree-sitter structural extractors.

Every language extractor (python / go / javascript / typescript / rust) walks a
parsed AST through a small multi-pass pipeline (collect symbols → structural
entities → imports → calls) and emits the same entity/relation dict shapes. This
module factors the parts that are byte-for-byte identical across all of them:

- ``_node_text`` — decode a node's source text.
- ``_make_relation`` — build a relation dict with all required fields.
- the constructor that stores the executor-supplied context and initializes the
  output accumulators.
- ``extract`` — run the language passes, then assemble entities and relations in
  the canonical order.

A concrete extractor implements ``_run_passes`` (its own ordered passes,
populating the accumulators) and ``_relation_buckets`` (the relation emission
order, which differs by language — e.g. Rust emits IMPORTS before INHERITS_FROM
while Python emits INHERITS_FROM first). Entity order is identical everywhere, so
the base owns it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tree_sitter import Node


def node_text(node: Node) -> str | None:
    """Return decoded text for a node."""
    text = node.text
    if text is None:
        return None
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return str(text)


def make_relation(
    *,
    source_qname: str,
    source_entity_type: str,
    target_qname: str,
    target_entity_type: str,
    relation_type: str,
    confidence: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a relation dict with all required fields."""
    return {
        "source_qname": source_qname,
        "source_entity_type": source_entity_type,
        "target_qname": target_qname,
        "target_entity_type": target_entity_type,
        "relation_type": relation_type,
        "confidence": confidence,
        "metadata": metadata or {},
    }


class BaseExtractor(ABC):
    """Stateless per-invocation structural extractor for one language.

    Subclasses populate the entity/relation accumulators in ``_run_passes`` and
    declare their relation emission order in ``_relation_buckets``. The base
    assembles the final ``(entities, relations)`` tuple.
    """

    def __init__(
        self,
        root: Node,
        file_qname: str,
        module_qname: str,
        file_entity: dict[str, Any],
        module_entity: dict[str, Any],
        contains_file_module: dict[str, Any],
        stable_id_fn: Any,
    ) -> None:
        self._root = root
        self._file_qname = file_qname
        self._module_qname = module_qname
        self._file_entity = file_entity
        self._module_entity = module_entity
        self._contains_file_module = contains_file_module
        self._stable_id_fn = stable_id_fn

        # Accumulated output — populated during _run_passes()
        self._class_entities: list[dict[str, Any]] = []
        self._method_entities: list[dict[str, Any]] = []
        self._function_entities: list[dict[str, Any]] = []

        self._contains_relations: list[dict[str, Any]] = []
        self._inherits_relations: list[dict[str, Any]] = []
        self._import_relations: list[dict[str, Any]] = []
        self._call_relations: list[dict[str, Any]] = []

        # Known same-module symbols for CALLS resolution: simple_name -> qname.
        # Populated in a first pass before CALLS extraction.
        self._known_symbols: dict[str, str] = {}

    def extract(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run the language passes and return (entities, relations)."""
        self._run_passes()

        entities = (
            [self._file_entity, self._module_entity]
            + self._class_entities
            + self._method_entities
            + self._function_entities
        )
        relations = [self._contains_file_module]
        for bucket in self._relation_buckets():
            relations.extend(bucket)
        return entities, relations

    @abstractmethod
    def _run_passes(self) -> None:
        """Walk the AST, populating the entity and relation accumulators."""

    @abstractmethod
    def _relation_buckets(self) -> list[list[dict[str, Any]]]:
        """Return the language's relation accumulators in emission order.

        The base prepends the File->Module CONTAINS relation; this returns the
        remaining buckets (CONTAINS, INHERITS_FROM, IMPORTS, CALLS) in the order
        the language emits them.
        """
