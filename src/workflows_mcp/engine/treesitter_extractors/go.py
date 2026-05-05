"""Go structural extraction using tree-sitter.

Traverses a parsed Go AST (source_file root) and emits:
- Entities: File, Module, Class (struct/interface), Method, Function
- Relations: CONTAINS, IMPORTS, CALLS

Design constraints:
- Stateless: no DB, no cross-file resolution, no shared state
- No stdout/print: use logger only
- Returns data suitable for TreeSitterOutput entities/relations lists

Entity ordering: File, Module, Class (declaration order), Method (receiver-class
then declaration order), Function (top-level declaration order).

Relation ordering: CONTAINS (File->Module first, Module->Class/Function,
Class->Method), IMPORTS, CALLS.

Note: Go has no class inheritance; INHERITS_FROM is not emitted in v1.
"""

from __future__ import annotations

import logging
from typing import Any

from tree_sitter import Node

logger = logging.getLogger(__name__)

# Sentinel prefix for parent class hint in method metadata.
_PARENT_CLASS_PREFIX = "__class_qname__:"


def extract_go(
    root: Node,
    file_qname: str,
    module_qname: str,
    file_entity: dict[str, Any],
    module_entity: dict[str, Any],
    contains_file_module: dict[str, Any],
    stable_id_fn: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract Go entities and relations from a parsed tree-sitter tree.

    Args:
        root: The root node of the parsed tree (node.type == "source_file").
        file_qname: Qualified name of the File entity.
        module_qname: Qualified name of the Module entity (package name).
        file_entity: Pre-built File entity dict from the executor.
        module_entity: Pre-built Module entity dict from the executor.
        contains_file_module: Pre-built CONTAINS File->Module relation.
        stable_id_fn: Callable(qualified_name, entity_type) -> str.

    Returns:
        (entities, relations): Lists of entity/relation dicts in deterministic
        order as defined in the module docstring.
    """
    extractor = _GoExtractor(
        root=root,
        file_qname=file_qname,
        module_qname=module_qname,
        file_entity=file_entity,
        module_entity=module_entity,
        contains_file_module=contains_file_module,
        stable_id_fn=stable_id_fn,
    )
    return extractor.extract()


def extract_package_name(root: Node) -> str | None:
    """Return the package identifier from the source_file root node, or None."""
    for child in root.children:
        if child.type == "package_clause":
            for sub in child.children:
                if sub.type == "package_identifier":
                    return _node_text(sub)
    return None


class _GoExtractor:
    """Stateless per-invocation Go extractor."""

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

        # Accumulated output — populated during extract()
        self._class_entities: list[dict[str, Any]] = []
        self._method_entities: list[dict[str, Any]] = []
        self._function_entities: list[dict[str, Any]] = []

        self._contains_relations: list[dict[str, Any]] = []
        self._import_relations: list[dict[str, Any]] = []
        self._call_relations: list[dict[str, Any]] = []

        # Known same-package symbols for CALLS resolution: simple_name -> qname
        self._known_symbols: dict[str, str] = {}

        # Track seen import paths for deduplication
        self._seen_import_paths: set[str] = set()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run extraction and return (entities, relations)."""
        # Pass 1: collect symbol names for resolution
        self._collect_known_symbols()

        # Pass 2: extract structural entities
        self._extract_source_file_children()

        # Pass 3: extract imports
        self._extract_imports()

        # Pass 4: extract calls
        self._extract_calls()

        entities = (
            [self._file_entity, self._module_entity]
            + self._class_entities
            + self._method_entities
            + self._function_entities
        )
        relations = (
            [self._contains_file_module]
            + self._contains_relations
            + self._import_relations
            + self._call_relations
        )
        return entities, relations

    # ------------------------------------------------------------------
    # Pass 1: symbol index
    # ------------------------------------------------------------------

    def _collect_known_symbols(self) -> None:
        """Index top-level functions, structs, interfaces for CALLS resolution."""
        for child in self._root.children:
            if child.type == "function_declaration":
                name = _function_decl_name(child)
                if name:
                    self._known_symbols[name] = f"{self._module_qname}.{name}"
            elif child.type == "type_declaration":
                for spec in child.children:
                    if spec.type == "type_spec":
                        name = _type_spec_name(spec)
                        if name:
                            self._known_symbols[name] = f"{self._module_qname}.{name}"

    # ------------------------------------------------------------------
    # Pass 2: structural extraction
    # ------------------------------------------------------------------

    def _extract_source_file_children(self) -> None:
        """Walk direct children of source_file."""
        for child in self._root.children:
            if child.type == "type_declaration":
                self._extract_type_declaration(child)
            elif child.type == "method_declaration":
                self._extract_method_declaration(child)
            elif child.type == "function_declaration":
                self._extract_function_declaration(child)

    def _extract_type_declaration(self, node: Node) -> None:
        """Handle type_declaration nodes (struct and interface types)."""
        for spec in node.children:
            if spec.type == "type_spec":
                self._extract_type_spec(spec)

    def _extract_type_spec(self, spec_node: Node) -> None:
        """Extract a single type_spec as a Class entity."""
        name = _type_spec_name(spec_node)
        if not name:
            return

        # Determine if struct or interface
        is_interface = False
        for child in spec_node.children:
            if child.type == "interface_type":
                is_interface = True
                break

        class_qname = f"{self._module_qname}.{name}"
        start = spec_node.start_point

        meta: dict[str, Any] = {
            "start_line": start[0] + 1,
            "start_column": start[1],
        }
        if is_interface:
            meta["is_interface"] = True

        class_entity: dict[str, Any] = {
            "entity_type": "Class",
            "name": name,
            "qualified_name": class_qname,
            "stable_id": self._stable_id_fn(class_qname, "Class"),
            "metadata": meta,
            "confidence": 1.0,
        }
        self._class_entities.append(class_entity)

        # CONTAINS Module -> Class
        self._contains_relations.append(
            _make_relation(
                source_qname=self._module_qname,
                source_entity_type="Module",
                target_qname=class_qname,
                target_entity_type="Class",
                relation_type="CONTAINS",
            )
        )

    def _extract_method_declaration(self, node: Node) -> None:
        """Handle method_declaration: func (recv Type) Name() -> Method entity."""
        # Receiver is first parameter_list child; method name is field_identifier
        receiver_type = _method_receiver_type(node)
        method_name = _method_decl_name(node)

        if not receiver_type or not method_name:
            return

        class_qname = f"{self._module_qname}.{receiver_type}"
        method_qname = f"{class_qname}.{method_name}"
        start = node.start_point

        method_entity: dict[str, Any] = {
            "entity_type": "Method",
            "name": method_name,
            "qualified_name": method_qname,
            "stable_id": self._stable_id_fn(method_qname, "Method"),
            "metadata": {
                "start_line": start[0] + 1,
                "start_column": start[1],
                "parent_class_id": f"{_PARENT_CLASS_PREFIX}{class_qname}",
            },
            "confidence": 1.0,
        }
        self._method_entities.append(method_entity)

        # CONTAINS Class -> Method
        self._contains_relations.append(
            _make_relation(
                source_qname=class_qname,
                source_entity_type="Class",
                target_qname=method_qname,
                target_entity_type="Method",
                relation_type="CONTAINS",
            )
        )

    def _extract_function_declaration(self, node: Node) -> None:
        """Handle function_declaration (top-level, not methods)."""
        name = _function_decl_name(node)
        if not name:
            return

        func_qname = f"{self._module_qname}.{name}"
        start = node.start_point

        func_entity: dict[str, Any] = {
            "entity_type": "Function",
            "name": name,
            "qualified_name": func_qname,
            "stable_id": self._stable_id_fn(func_qname, "Function"),
            "metadata": {
                "start_line": start[0] + 1,
                "start_column": start[1],
            },
            "confidence": 1.0,
        }
        self._function_entities.append(func_entity)

        # CONTAINS Module -> Function
        self._contains_relations.append(
            _make_relation(
                source_qname=self._module_qname,
                source_entity_type="Module",
                target_qname=func_qname,
                target_entity_type="Function",
                relation_type="CONTAINS",
            )
        )

    # ------------------------------------------------------------------
    # Pass 3: imports
    # ------------------------------------------------------------------

    def _extract_imports(self) -> None:
        """Emit IMPORTS relations for import_declaration nodes."""
        for child in self._root.children:
            if child.type == "import_declaration":
                self._handle_import_declaration(child)

    def _handle_import_declaration(self, node: Node) -> None:
        """Handle both single and grouped import declarations."""
        for child in node.children:
            if child.type == "import_spec_list":
                # Grouped: import ( "pkg1" "pkg2" )
                for spec in child.children:
                    if spec.type == "import_spec":
                        self._emit_import_from_spec(spec)
            elif child.type == "import_spec":
                # Single: import "pkg"
                self._emit_import_from_spec(child)

    def _emit_import_from_spec(self, spec_node: Node) -> None:
        """Extract path from import_spec and emit IMPORTS relation."""
        path = _import_path_from_spec(spec_node)
        if not path:
            return
        # Deduplicate
        if path in self._seen_import_paths:
            return
        self._seen_import_paths.add(path)

        self._import_relations.append(
            _make_relation(
                source_qname=self._module_qname,
                source_entity_type="Module",
                target_qname=path,
                target_entity_type="Module",
                relation_type="IMPORTS",
                confidence=0.5,
                metadata={"resolution": "unresolved"},
            )
        )

    # ------------------------------------------------------------------
    # Pass 4: CALLS extraction
    # ------------------------------------------------------------------

    def _extract_calls(self) -> None:
        """Extract CALLS from all top-level functions and methods."""
        for child in self._root.children:
            if child.type == "function_declaration":
                name = _function_decl_name(child)
                if name:
                    func_qname = f"{self._module_qname}.{name}"
                    self._collect_calls_in_body(child, func_qname, "Function")
            elif child.type == "method_declaration":
                receiver_type = _method_receiver_type(child)
                method_name = _method_decl_name(child)
                if receiver_type and method_name:
                    class_qname = f"{self._module_qname}.{receiver_type}"
                    method_qname = f"{class_qname}.{method_name}"
                    self._collect_calls_in_body(child, method_qname, "Method")

    def _collect_calls_in_body(
        self, func_node: Node, scope_qname: str, scope_entity_type: str
    ) -> None:
        """Find the block child and recursively walk for call_expression nodes."""
        for child in func_node.children:
            if child.type == "block":
                self._walk_for_calls(child, scope_qname, scope_entity_type)
                return

    def _walk_for_calls(
        self, node: Node, scope_qname: str, scope_entity_type: str
    ) -> None:
        """Recursively walk AST collecting call_expression nodes."""
        for child in node.children:
            if child.type == "call_expression":
                self._emit_call(child, scope_qname, scope_entity_type)
                # Recurse to find nested calls in arguments
                self._walk_for_calls(child, scope_qname, scope_entity_type)
            else:
                self._walk_for_calls(child, scope_qname, scope_entity_type)

    def _emit_call(
        self, call_node: Node, scope_qname: str, scope_entity_type: str
    ) -> None:
        """Emit a CALLS relation for a call_expression node."""
        # First child is the function expression (identifier or selector_expression)
        func_node = call_node.children[0] if call_node.children else None
        if func_node is None:
            return

        call_site = call_node.start_point

        if func_node.type == "identifier":
            callee_text = _node_text(func_node)
            if not callee_text:
                return
            resolved = self._known_symbols.get(callee_text)
            if resolved is not None:
                target_qname = resolved
                confidence = 1.0
                meta: dict[str, Any] = {
                    "call_line": call_site[0] + 1,
                    "call_column": call_site[1],
                }
            else:
                target_qname = f"{self._module_qname}.{callee_text}"
                confidence = 0.5
                meta = {
                    "resolution": "unresolved",
                    "call_line": call_site[0] + 1,
                    "call_column": call_site[1],
                }
        elif func_node.type == "selector_expression":
            # e.g. fmt.Println, g.Method
            callee_text = _node_text(func_node) or ""
            target_qname = callee_text
            confidence = 0.5
            meta = {
                "resolution": "unresolved",
                "call_line": call_site[0] + 1,
                "call_column": call_site[1],
            }
        else:
            # Other call forms (e.g. function literal calls) — skip
            return

        # Determine target entity type
        target_entity_type = "Function"
        if func_node.type == "identifier":
            class_qnames = {e["qualified_name"] for e in self._class_entities}
            if target_qname in class_qnames:
                target_entity_type = "Class"

        self._call_relations.append(
            _make_relation(
                source_qname=scope_qname,
                source_entity_type=scope_entity_type,
                target_qname=target_qname,
                target_entity_type=target_entity_type,
                relation_type="CALLS",
                confidence=confidence,
                metadata=meta,
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_text(node: Node) -> str | None:
    """Return decoded text for a node."""
    text = node.text
    if text is None:
        return None
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return str(text)


def _type_spec_name(spec_node: Node) -> str | None:
    """Return the type_identifier name from a type_spec node."""
    for child in spec_node.children:
        if child.type == "type_identifier":
            return _node_text(child)
    return None


def _function_decl_name(node: Node) -> str | None:
    """Return the identifier name from a function_declaration node."""
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child)
    return None


def _method_decl_name(node: Node) -> str | None:
    """Return the field_identifier (method name) from a method_declaration node."""
    for child in node.children:
        if child.type == "field_identifier":
            return _node_text(child)
    return None


def _method_receiver_type(node: Node) -> str | None:
    """Extract the receiver type name from a method_declaration node.

    Go grammar: method_declaration -> parameter_list (receiver) field_identifier ...
    The first parameter_list child is the receiver list, which contains a
    parameter_declaration with an optional name and a type (pointer_type or
    type_identifier).
    """
    # First parameter_list is the receiver
    receiver_list = None
    for child in node.children:
        if child.type == "parameter_list":
            receiver_list = child
            break

    if receiver_list is None:
        return None

    # Extract type from parameter_declaration inside receiver_list
    for child in receiver_list.children:
        if child.type == "parameter_declaration":
            return _extract_type_name_from_param(child)

    return None


def _extract_type_name_from_param(param_node: Node) -> str | None:
    """Extract the base type name from a parameter_declaration node.

    Handles both value receivers (Type) and pointer receivers (*Type).
    """
    for child in param_node.children:
        if child.type == "type_identifier":
            return _node_text(child)
        elif child.type == "pointer_type":
            # pointer_type -> * type_identifier
            for sub in child.children:
                if sub.type == "type_identifier":
                    return _node_text(sub)
    return None


def _import_path_from_spec(spec_node: Node) -> str | None:
    """Extract the import path string (without quotes) from import_spec node.

    The import path is stored in an interpreted_string_literal child, which
    contains an interpreted_string_literal_content grandchild.
    """
    for child in spec_node.children:
        if child.type == "interpreted_string_literal":
            for sub in child.children:
                if sub.type == "interpreted_string_literal_content":
                    return _node_text(sub)
            # Fallback: raw text without surrounding quotes
            raw = _node_text(child)
            if raw:
                return raw.strip('"')
    return None


def _make_relation(
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
