"""TreeSitter block executor for static code analysis.

Parses source files using tree-sitter grammars and emits a structured graph
of entities (File, Module, Class, Function, Method) and relations (CONTAINS,
IMPORTS, INHERITS_FROM, CALLS) suitable for memory graph ingestion.

Architecture (ADR-006):
- Returns TreeSitterOutput directly on success
- Raises exceptions on failure
- Stateless executor (singleton-safe)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, ClassVar, TypedDict, cast

from pydantic import Field

from .block import BlockInput, BlockOutput
from .execution import Execution
from .executor_base import (
    BlockExecutor,
    ExecutorCapabilities,
    ExecutorSecurityLevel,
)
from .treesitter_extractors import (
    extract_code,
    extract_document,
    extract_package_name,
    has_extractor,
)
from .treesitter_languages import (
    SupportedLanguage,
    content_hash,
    detect_language,
    get_parser,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------


class TreeSitterInput(BlockInput):
    """Input for the TreeSitter executor.

    Security note: this is a TRUSTED executor that reads the provided path
    directly with no boundary enforcement. The caller is responsible for
    ensuring the path is within the intended scope before invoking this block.
    """

    path: str = Field(
        description=(
            "Path to the source file to parse. May be absolute or workspace-relative. "
            "No path sandboxing is applied — the caller is responsible for scoping."
        )
    )
    language: SupportedLanguage | None = Field(
        default=None,
        description=(
            "Override language detection. If omitted, language is inferred from extension."
        ),
    )
    repo_relative_path: str | None = Field(
        default=None,
        description="Path of the file relative to the repository root (used for qualified names).",
    )
    palace: str | None = Field(
        default=None,
        description="Memory palace name (used for stable ID generation).",
    )
    item_id: str | None = Field(
        default=None,
        description="Source item ID (used for stable ID generation).",
    )


class TreeSitterOutput(BlockOutput):
    """Output for the TreeSitter executor."""

    language: str = Field(description="Detected or overridden language.")
    content_hash: str = Field(description="SHA-256 hex digest of the file content.")
    size_bytes: int = Field(description="File size in bytes.")
    mtime_ns: int = Field(description="File modification time in nanoseconds.")
    module_qualified_name: str = Field(description="Qualified name of the top-level module entity.")
    entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted graph entities (File, Module, …).",
    )
    relations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted graph relations (CONTAINS, IMPORTS, …).",
    )
    unresolved_imports: list[str] = Field(
        default_factory=list,
        description="Import strings that could not be resolved to known entities.",
    )
    structural_evidence_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "ADR-013: entities pre-mapped to StructuralEvidenceItem shape "
            "(entity_stable_id, entity_type, evidence_category, evidence_data) "
            "ready for store_system1_structural_evidence."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_id(palace: str, item_id: str, qualified_name: str, entity_type: str) -> str:
    """Compute a 32-character stable ID for an entity.

    Formula: sha256(palace + ':' + item_id + ':' + qualified_name + ':' + entity_type)[:32]
    Empty strings are used for missing palace / item_id components.
    """
    raw = f"{palace}:{item_id}:{qualified_name}:{entity_type}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _module_qname(path: str, repo_relative_path: str | None) -> str:
    """Derive a module qualified name from the file path.

    Uses repo_relative_path when available; falls back to the basename stem.
    Converts path separators to dots and strips common source-root prefixes.

    Special case: pkg/__init__.py -> pkg (not pkg.__init__).
    Top-level __init__.py with no parent package falls back to the stem
    "__init__" so the qualified name is never empty.
    """
    if repo_relative_path:
        rel = repo_relative_path
    else:
        rel = Path(path).name

    rel_path = Path(rel)

    # __init__.py -> parent package name
    if rel_path.name == "__init__.py":
        parent = rel_path.parent
        if str(parent) == ".":
            # top-level __init__.py — use "__init__" to avoid an empty qname
            qname = "__init__"
        else:
            qname = str(parent).replace("/", ".").replace("\\", ".")
    else:
        # Strip extension
        stem = rel_path.with_suffix("")
        # Convert separators to dots
        qname = str(stem).replace("/", ".").replace("\\", ".")

    # Strip leading src. prefix (common convention)
    if qname.startswith("src."):
        qname = qname[4:]
    return qname


# ---------------------------------------------------------------------------
# ADR-013: map TreeSitter entities to StructuralEvidenceItem shape
# ---------------------------------------------------------------------------

_ENTITY_TYPE_TO_EVIDENCE_CATEGORY: dict[str, str] = {
    "file": "structural_module",
    "module": "structural_module",
    "class": "structural_class",
    "function": "structural_function",
    "method": "structural_function",
}

SYSTEM1_REQUIRED_ENTITY_KEYS: frozenset[str] = frozenset(
    {
        "qualified_name",
        "stable_id",
        "entity_type",
        "name",
        "metadata",
        "confidence",
    }
)

SYSTEM1_REQUIRED_RELATION_KEYS: frozenset[str] = frozenset(
    {
        "source_qname",
        "source_entity_type",
        "target_qname",
        "target_entity_type",
        "relation_type",
        "confidence",
        "metadata",
    }
)

SYSTEM1_ALLOWED_RELATION_TYPES: frozenset[str] = frozenset(
    {
        "CONTAINS",
        "IMPORTS",
        "CALLS",
        "INHERITS_FROM",
    }
)

SYSTEM1_ERROR_MESSAGE_MAX_ITEMS: int = 10


class System1ExtractionDiagnostics(TypedDict):
    """Validation diagnostics for System1 extraction payloads."""

    valid: bool
    errors: list[str]
    entity_count: int
    relation_count: int
    relation_types: list[str]


def validate_system1_extraction_payload(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> System1ExtractionDiagnostics:
    """Validate TreeSitter extraction payload contract for System1 structural graph.

    Empty payloads are valid to support unsupported/no-op extraction paths.
    """
    errors: list[str] = []
    relation_types: set[str] = set()

    if not entities and not relations:
        return {
            "valid": True,
            "errors": [],
            "entity_count": 0,
            "relation_count": 0,
            "relation_types": [],
        }

    for index, entity in enumerate(entities):
        missing = SYSTEM1_REQUIRED_ENTITY_KEYS.difference(entity.keys())
        if missing:
            errors.append(f"entity[{index}] missing required keys: {sorted(missing)}")

    for index, relation in enumerate(relations):
        missing = SYSTEM1_REQUIRED_RELATION_KEYS.difference(relation.keys())
        if missing:
            errors.append(f"relation[{index}] missing required keys: {sorted(missing)}")
        relation_type_raw = relation.get("relation_type")
        if isinstance(relation_type_raw, str):
            relation_types.add(relation_type_raw)
            if relation_type_raw not in SYSTEM1_ALLOWED_RELATION_TYPES:
                errors.append(
                    "relation["
                    f"{index}] invalid relation_type={relation_type_raw!r}; "
                    f"allowed={sorted(SYSTEM1_ALLOWED_RELATION_TYPES)}"
                )
        else:
            relation_type_name = type(relation_type_raw).__name__
            errors.append(f"relation[{index}] relation_type must be str, got {relation_type_name}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "entity_count": len(entities),
        "relation_count": len(relations),
        "relation_types": sorted(relation_types),
    }


def _raise_if_invalid_system1_extraction_payload(
    entities: list[dict[str, Any]], relations: list[dict[str, Any]]
) -> System1ExtractionDiagnostics:
    diagnostics = validate_system1_extraction_payload(entities, relations)
    if diagnostics["valid"]:
        return diagnostics
    errors = diagnostics["errors"]
    total_error_count = len(errors)
    shown = errors[:SYSTEM1_ERROR_MESSAGE_MAX_ITEMS]
    remaining = total_error_count - len(shown)
    relation_types = diagnostics["relation_types"]
    relation_types_summary = relation_types if relation_types else ["none"]

    summary = (
        f"SYSTEM1_EXTRACTION_SCHEMA_INVALID: {total_error_count} validation errors "
        f"(showing first {len(shown)}); entity_count={diagnostics['entity_count']}; "
        f"relation_count={diagnostics['relation_count']}; "
        f"relation_types={relation_types_summary}."
    )

    details = "; ".join(shown)
    tail = f"; and {remaining} more" if remaining > 0 else ""
    raise ValueError(f"{summary} Details: {details}{tail}")


def _build_structural_evidence_items(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map parsed entities to StructuralEvidenceItem-compatible dicts.

    Each item has: entity_stable_id, entity_type, evidence_category, evidence_data.
    Corridor is never emitted as a topology level (it is represented only as
    directed typed edges — relations — not as a structural evidence entity type).

    Import relations are emitted as separate structural_import evidence items
    so that the evidence category set reflects actual import topology.
    Relations carry source_qname (not source_stable_id), so the qname→entity
    map is built first to resolve stable IDs reliably.
    """
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    # Build qname → entity map for reliable stable-ID resolution from relations.
    qname_to_entity: dict[str, dict[str, Any]] = {}
    for entity in entities:
        qname: str = entity.get("qualified_name", "")
        if qname:
            qname_to_entity[qname] = entity

    for entity in entities:
        stable_id: str = entity.get("stable_id", "")
        raw_type: str = entity.get("entity_type", "unknown")
        category = _ENTITY_TYPE_TO_EVIDENCE_CATEGORY.get(raw_type.lower(), "structural_module")
        key = (stable_id, category)
        if key in seen or not stable_id:
            continue
        seen.add(key)
        items.append(
            {
                "entity_stable_id": stable_id,
                "entity_type": raw_type.lower(),
                "evidence_category": category,
                "evidence_data": {
                    "qualified_name": entity.get("qualified_name", ""),
                    "name": entity.get("name", ""),
                },
            }
        )

    # Emit structural_import evidence for each unique import-relation source entity.
    # Relations use source_qname (not source_stable_id); resolve via qname_to_entity.
    import_source_qnames: set[str] = set()
    for rel in relations:
        if rel.get("relation_type") != "IMPORTS":
            continue
        src_qname: str = rel.get("source_qname", "")
        if not src_qname or src_qname in import_source_qnames:
            continue
        src_entity = qname_to_entity.get(src_qname)
        if src_entity is None:
            continue
        src_id: str = src_entity.get("stable_id", "")
        if not src_id:
            continue
        import_source_qnames.add(src_qname)
        key = (src_id, "structural_import")
        if key in seen:
            continue
        seen.add(key)
        import_count = sum(
            1
            for r in relations
            if r.get("relation_type") == "IMPORTS" and r.get("source_qname") == src_qname
        )
        import_relations = [
            r
            for r in relations
            if r.get("relation_type") == "IMPORTS" and r.get("source_qname") == src_qname
        ]
        imports_payload: list[dict[str, Any]] = []
        for import_relation in import_relations:
            relation_metadata = import_relation.get("metadata", {})
            if not isinstance(relation_metadata, dict):
                relation_metadata = {}
            resolution = relation_metadata.get("resolution")
            normalized_resolution = "resolved" if resolution == "resolved" else "unresolved"
            imports_payload.append(
                {
                    "target_qname": str(import_relation.get("target_qname", "")),
                    "target_entity_type": str(import_relation.get("target_entity_type", "Unknown")),
                    "resolution": normalized_resolution,
                    "confidence": float(import_relation.get("confidence", 1.0)),
                    "source_range": relation_metadata.get("source_range", {}),
                }
            )
        items.append(
            {
                "entity_stable_id": src_id,
                "entity_type": src_entity.get("entity_type", "module").lower(),
                "evidence_category": "structural_import",
                "evidence_data": {
                    "import_count": import_count,
                    "imports": imports_payload,
                },
            }
        )

    return items


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TreeSitterExecutor(BlockExecutor):
    """Parse source files and emit a static-analysis entity graph.

    Security note: this executor is classified TRUSTED and reads caller-supplied
    paths without boundary enforcement. Path scoping is the caller's
    responsibility — do not expose this block to untrusted input without
    a prior path-validation step in the workflow.

    Code languages (Python, TypeScript, TSX, JavaScript, Go, Rust) emit File
    and Module entities plus language-specific symbols (Class, Function, Method)
    and relations (CONTAINS, IMPORTS, INHERITS_FROM, CALLS).

    Document languages (Markdown, YAML, JSON) are file-only: they emit exactly
    one File entity and no Module, Class, Function, or Method entities, and no
    relations. `module_qualified_name` on the output is non-empty for output
    model compatibility but no Module entity is emitted.

    Returns language="unsupported" with empty entities for unknown file types.
    """

    type_name: ClassVar[str] = "TreeSitter"
    input_type: ClassVar[type[BlockInput]] = TreeSitterInput
    output_type: ClassVar[type[BlockOutput]] = TreeSitterOutput

    security_level: ClassVar[ExecutorSecurityLevel] = ExecutorSecurityLevel.TRUSTED
    capabilities: ClassVar[ExecutorCapabilities] = ExecutorCapabilities(can_read_files=True)

    async def execute(  # type: ignore[override]
        self, inputs: TreeSitterInput, context: Execution
    ) -> TreeSitterOutput:
        """Parse a source file and return a graph of entities and relations.

        Reads the file at inputs.path directly. No path boundary checks are
        performed — the caller is responsible for path scoping before invoking
        this executor.

        For unsupported extensions, returns language="unsupported" with empty
        entities and relations rather than raising an exception.

        Raises:
            FileNotFoundError: If the path does not exist (ADR-006: propagated as-is).
            OSError: On other I/O failures (ADR-006: propagated as-is).
        """
        file_path = Path(inputs.path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        file_hash = content_hash(text)
        stat = file_path.stat()
        size_bytes = stat.st_size
        mtime_ns = stat.st_mtime_ns

        # Resolve language
        language: str = (
            inputs.language if inputs.language is not None else detect_language(inputs.path)
        )

        if language == "unsupported":
            logger.info("TreeSitter: unsupported extension for %s", inputs.path)
            return TreeSitterOutput(
                language="unsupported",
                content_hash=file_hash,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
                module_qualified_name="",
                entities=[],
                relations=[],
                unresolved_imports=[],
            )

        # Parse the file — language is a supported non-"unsupported" value at this point
        parser = get_parser(cast(SupportedLanguage, language))
        tree = parser.parse(text.encode("utf-8", errors="replace"))

        # Stable-ID components
        palace = inputs.palace or ""
        item_id = inputs.item_id or ""

        def _sid(qname: str, entity_type: str) -> str:
            return _stable_id(palace, item_id, qname, entity_type)

        # Derive qualified names
        file_qname = str(file_path)
        module_qname = _module_qname(inputs.path, inputs.repo_relative_path)

        # For Go, override module_qname with the package name from the source
        if language == "go":
            pkg_name = extract_package_name(tree.root_node)
            if pkg_name:
                module_qname = pkg_name

        # Build File entity
        file_entity: dict[str, Any] = {
            "qualified_name": file_qname,
            "stable_id": _stable_id(palace, item_id, file_qname, "File"),
            "entity_type": "File",
            "name": file_path.name,
            "metadata": {
                "path": str(file_path),
                "language": language,
                "syntax_errors": tree.root_node.has_error,
            },
            "confidence": 1.0,
        }

        logger.info(
            "TreeSitter: parsed %s as %s (errors=%s)",
            inputs.path,
            language,
            tree.root_node.has_error,
        )

        source_range_end_line = text.count("\n") + 1

        def _new_source_range_default() -> dict[str, int]:
            return {
                "start_line": 1,
                "start_column": 0,
                "end_line": source_range_end_line,
                "end_column": 0,
            }

        base_metadata: dict[str, Any] = {
            "source_file": str(file_path),
            "repo_relative_path": inputs.repo_relative_path,
            "content_hash": file_hash,
            "language": language,
        }

        def _enrich_entities_and_relations(
            entities_in: list[dict[str, Any]], relations_in: list[dict[str, Any]]
        ) -> None:
            for entity in entities_in:
                metadata = entity.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                source_range = metadata.get("source_range")
                if source_range is None:
                    source_range = _new_source_range_default()
                entity["metadata"] = {
                    **metadata,
                    **base_metadata,
                    "source_range": source_range,
                }

            for relation in relations_in:
                metadata = relation.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                source_range = metadata.get("source_range")
                if source_range is None:
                    source_range = _new_source_range_default()
                relation["metadata"] = {
                    **metadata,
                    "source_file": str(file_path),
                    "repo_relative_path": inputs.repo_relative_path,
                    "content_hash": file_hash,
                    "provenance": "treesitter",
                    "source_range": source_range,
                }

        # Document languages (Markdown, YAML, JSON) are file-only: emit exactly
        # one File entity, no Module/Class/Function/Method entities, and no
        # relations. module_qualified_name is set for output model compatibility
        # and traceability, but no Module entity is emitted.
        if language in ("markdown", "yaml", "json"):
            entities, relations = extract_document(file_entity=file_entity)
            _enrich_entities_and_relations(entities, relations)
            _raise_if_invalid_system1_extraction_payload(entities, relations)
            return TreeSitterOutput(
                language=language,
                content_hash=file_hash,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
                module_qualified_name=module_qname,
                entities=entities,
                relations=relations,
                unresolved_imports=[],
                structural_evidence_items=_build_structural_evidence_items(entities, relations),
            )

        # Code languages: build Module entity and File->Module CONTAINS relation,
        # then delegate to language-specific extractor for symbols and relations.

        # Build Module entity — name is the last dotted component of qname
        module_name = module_qname.split(".")[-1] if module_qname else module_qname
        module_entity: dict[str, Any] = {
            "qualified_name": module_qname,
            "stable_id": _stable_id(palace, item_id, module_qname, "Module"),
            "entity_type": "Module",
            "name": module_name,
            "metadata": {
                "source_path": str(file_path),
                "language": language,
            },
            "confidence": 1.0,
        }

        # Build CONTAINS relation File -> Module
        contains_relation: dict[str, Any] = {
            "source_qname": file_qname,
            "source_entity_type": "File",
            "target_qname": module_qname,
            "target_entity_type": "Module",
            "relation_type": "CONTAINS",
            "confidence": 1.0,
            "metadata": {},
        }

        # Delegate to the registered language extractor for additional
        # entities/relations. Adding a language is a registry edit, not a change here.
        if has_extractor(language):
            entities, relations = extract_code(
                language,
                root=tree.root_node,
                file_qname=file_qname,
                module_qname=module_qname,
                file_entity=file_entity,
                module_entity=module_entity,
                contains_file_module=contains_relation,
                stable_id_fn=_sid,
            )
        else:
            entities = [file_entity, module_entity]
            relations = [contains_relation]

        _enrich_entities_and_relations(entities, relations)

        # Collect unresolved import targets (deduplicated, sorted)
        _raise_if_invalid_system1_extraction_payload(entities, relations)
        unresolved_imports = sorted(
            {
                r["target_qname"]
                for r in relations
                if r["relation_type"] == "IMPORTS"
                and r.get("metadata", {}).get("resolution") == "unresolved"
            }
        )

        return TreeSitterOutput(
            language=language,
            content_hash=file_hash,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            module_qualified_name=module_qname,
            entities=entities,
            relations=relations,
            unresolved_imports=unresolved_imports,
            structural_evidence_items=_build_structural_evidence_items(entities, relations),
        )
