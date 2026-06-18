"""JavaScript/JSX structural extraction using tree-sitter.

Traverses a parsed JavaScript or JSX AST and emits:
- Entities: File, Module, Class, Method, Function
- Relations: CONTAINS, INHERITS_FROM, IMPORTS, CALLS

Design constraints:
- Stateless: no DB, no cross-file resolution, no shared state
- No stdout/print: use logger only
- Returns data suitable for TreeSitterOutput entities/relations lists

Grammar notes (tree-sitter-javascript 0.25):
- Root node type: "program"
- class_declaration: identifier (name field), class_heritage?, class_body
  - NOTE: name is 'identifier', NOT 'type_identifier' as in TypeScript
- class_heritage: direct 'identifier' or 'member_expression' child
  - NOTE: no 'extends_clause' wrapper — base is a direct child of class_heritage
- method_definition: property_identifier (name field), statement_block (body field)
- function_declaration: identifier (name field), statement_block (body field)
- import_statement: same shape as TypeScript grammar (import_clause, string source)
- call_expression: (identifier|member_expression) (function field), arguments
- new_expression: identifier (constructor), arguments

Entity ordering: File, Module, Class (declaration order), Method (class then
declaration order), Function (top-level declaration order).

Relation ordering: CONTAINS (File->Module first, Module->Class/Function,
Class->Method), INHERITS_FROM, IMPORTS, CALLS.

Concerns:
- Arrow functions (const arrowFn = () => ...): NOT extracted as Function
  entities; only function_declaration nodes are extracted.
- Member-expression base class in class_heritage (e.g., React.Component):
  treated as external/unresolved with confidence 0.5.
- JSX nodes in .jsx files: ignored during traversal; grammar handles them
  without special treatment needed from the extractor.
"""

from __future__ import annotations

import logging
from typing import Any

from tree_sitter import Node

logger = logging.getLogger(__name__)


def extract_javascript(
    root: Node,
    file_qname: str,
    module_qname: str,
    file_entity: dict[str, Any],
    module_entity: dict[str, Any],
    contains_file_module: dict[str, Any],
    stable_id_fn: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract JavaScript/JSX entities and relations from a parsed tree-sitter tree.

    Args:
        root: The root node of the parsed tree (node.type == "program").
        file_qname: Qualified name of the File entity.
        module_qname: Qualified name of the Module entity (dotted, no extension).
        file_entity: Pre-built File entity dict from the executor.
        module_entity: Pre-built Module entity dict from the executor.
        contains_file_module: Pre-built CONTAINS File->Module relation.
        stable_id_fn: Callable(qualified_name, entity_type) -> str.

    Returns:
        (entities, relations): Lists of entity/relation dicts in deterministic
        order as defined in the module docstring.
    """
    extractor = _JavaScriptExtractor(
        root=root,
        file_qname=file_qname,
        module_qname=module_qname,
        file_entity=file_entity,
        module_entity=module_entity,
        contains_file_module=contains_file_module,
        stable_id_fn=stable_id_fn,
    )
    return extractor.extract()


class _JavaScriptExtractor:
    """Stateless per-invocation JavaScript/JSX extractor."""

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
        self._inherits_relations: list[dict[str, Any]] = []
        self._import_relations: list[dict[str, Any]] = []
        self._call_relations: list[dict[str, Any]] = []

        # Known same-module symbols for CALLS resolution.
        # Populated in a first pass before CALLS extraction.
        self._known_symbols: dict[str, str] = {}  # simple_name -> qname

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run extraction and return (entities, relations)."""
        # Pass 1: collect class/function names for symbol resolution
        self._collect_known_symbols()

        # Pass 2: extract structural entities
        self._extract_program_children(self._root)

        # Pass 3: extract imports
        self._extract_imports(self._root)

        # Pass 4: extract calls (needs known symbols)
        self._extract_calls_from_program(self._root)

        entities = (
            [self._file_entity, self._module_entity]
            + self._class_entities
            + self._method_entities
            + self._function_entities
        )
        relations = (
            [self._contains_file_module]
            + self._contains_relations
            + self._inherits_relations
            + self._import_relations
            + self._call_relations
        )
        return entities, relations

    # ------------------------------------------------------------------
    # Pass 1: symbol index
    # ------------------------------------------------------------------

    def _collect_known_symbols(self) -> None:
        """Index top-level classes, functions, and methods for CALLS resolution."""
        for child in self._root.named_children:
            if child.type == "class_declaration":
                class_name = _identifier_name(child)
                if not class_name:
                    continue
                class_qname = f"{self._module_qname}.{class_name}"
                self._known_symbols[class_name] = class_qname

                body = child.child_by_field_name("body")
                if body:
                    for item in body.named_children:
                        if item.type == "method_definition":
                            method_name = _property_identifier_name(item)
                            if method_name:
                                method_qname = f"{class_qname}.{method_name}"
                                self._known_symbols[f"{class_name}.{method_name}"] = method_qname

            elif child.type == "function_declaration":
                func_name = _identifier_name(child)
                if func_name:
                    func_qname = f"{self._module_qname}.{func_name}"
                    self._known_symbols[func_name] = func_qname

    # ------------------------------------------------------------------
    # Pass 2: structural extraction
    # ------------------------------------------------------------------

    def _extract_program_children(self, program_node: Node) -> None:
        """Walk direct children of the program node."""
        for child in program_node.named_children:
            if child.type == "class_declaration":
                self._extract_class(child)
            elif child.type == "function_declaration":
                self._extract_top_level_function(child)
            # Arrow functions (lexical_declaration with arrow_function): not extracted
            # JSX nodes: ignored (grammar handles them transparently)

    def _extract_class(self, node: Node) -> None:
        class_name = _identifier_name(node)
        if not class_name:
            return

        class_qname = f"{self._module_qname}.{class_name}"
        start = node.start_point

        class_entity: dict[str, Any] = {
            "entity_type": "Class",
            "name": class_name,
            "qualified_name": class_qname,
            "stable_id": self._stable_id_fn(class_qname, "Class"),
            "metadata": {
                "start_line": start[0] + 1,
                "start_column": start[1],
            },
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

        # INHERITS_FROM
        self._extract_inheritance(node, class_qname)

        # Methods
        body = node.child_by_field_name("body")
        if body:
            for item in body.named_children:
                if item.type == "method_definition":
                    self._extract_method(item, class_qname)

    def _extract_inheritance(self, class_node: Node, class_qname: str) -> None:
        """Emit INHERITS_FROM for the base class in class_heritage.

        JavaScript grammar differs from TypeScript:
        - class_heritage is a direct child of class_declaration (not wrapped in
          class_heritage -> extends_clause like TypeScript)
        - The base class is a direct identifier or member_expression child of
          class_heritage (no extends_clause wrapper)
        """
        heritage = None
        for child in class_node.named_children:
            if child.type == "class_heritage":
                heritage = child
                break

        if heritage is None:
            return

        # Collect same-module class names for resolution
        same_module_classes: set[str] = set()
        for sibling in self._root.named_children:
            if sibling.type == "class_declaration":
                name = _identifier_name(sibling)
                if name:
                    same_module_classes.add(name)

        # In JS grammar: class_heritage children are: 'extends' keyword + base expression
        # The base is the first named child that is identifier or member_expression
        for base_node in heritage.named_children:
            if base_node.type in ("identifier", "member_expression"):
                base_text = _node_text(base_node)
                if not base_text:
                    continue

                simple = base_text.split(".")[0]
                if base_node.type == "identifier" and simple in same_module_classes:
                    target_qname = f"{self._module_qname}.{base_text}"
                    confidence = 1.0
                    meta: dict[str, Any] = {}
                else:
                    # External/unresolved
                    target_qname = base_text
                    confidence = 0.5
                    meta = {"resolution": "unresolved"}

                self._inherits_relations.append(
                    _make_relation(
                        source_qname=class_qname,
                        source_entity_type="Class",
                        target_qname=target_qname,
                        target_entity_type="Class",
                        relation_type="INHERITS_FROM",
                        confidence=confidence,
                        metadata=meta,
                    )
                )
                # Only first base (JS supports single class extension)
                break

    def _extract_method(self, node: Node, class_qname: str) -> None:
        method_name = _property_identifier_name(node)
        if not method_name:
            return

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
                "parent_class_qname": class_qname,
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

    def _extract_top_level_function(self, node: Node) -> None:
        func_name = _identifier_name(node)
        if not func_name:
            return

        func_qname = f"{self._module_qname}.{func_name}"
        start = node.start_point

        func_entity: dict[str, Any] = {
            "entity_type": "Function",
            "name": func_name,
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

    def _extract_imports(self, program_node: Node) -> None:
        """Emit IMPORTS relations for all import statements at program level.

        Deduplicates by target module: multiple import statements from the same
        source (e.g. default + named imports from 'react') emit exactly one
        IMPORTS relation per unique target_qname.
        """
        seen: set[str] = set()
        for child in program_node.named_children:
            if child.type == "import_statement":
                self._handle_import_statement(child, seen)

    def _handle_import_statement(self, node: Node, seen: set[str]) -> None:
        """Handle ES import statements of all forms.

        Forms handled:
        - import X from 'pkg'             -> default import
        - import { x, y } from 'pkg'      -> named imports
        - import * as ns from 'pkg'       -> namespace import
        - import 'pkg'                    -> side-effect import
        """
        source = _import_source(node)
        if source and source not in seen:
            seen.add(source)
            self._emit_import(source)

    def _emit_import(self, target_module: str) -> None:
        self._import_relations.append(
            _make_relation(
                source_qname=self._module_qname,
                source_entity_type="Module",
                target_qname=target_module,
                target_entity_type="Module",
                relation_type="IMPORTS",
                confidence=0.5,
                metadata={"resolution": "unresolved"},
            )
        )

    # ------------------------------------------------------------------
    # Pass 4: CALLS extraction
    # ------------------------------------------------------------------

    def _extract_calls_from_program(self, program_node: Node) -> None:
        """Extract CALLS from all top-level functions and methods."""
        for child in program_node.named_children:
            if child.type == "class_declaration":
                class_name = _identifier_name(child)
                if not class_name:
                    continue
                class_qname = f"{self._module_qname}.{class_name}"
                body = child.child_by_field_name("body")
                if body:
                    for item in body.named_children:
                        if item.type == "method_definition":
                            method_name = _property_identifier_name(item)
                            if method_name:
                                method_qname = f"{class_qname}.{method_name}"
                                body_node = item.child_by_field_name("body")
                                if body_node:
                                    self._walk_for_calls(body_node, method_qname, "Method")
            elif child.type == "function_declaration":
                func_name = _identifier_name(child)
                if func_name:
                    func_qname = f"{self._module_qname}.{func_name}"
                    body_node = child.child_by_field_name("body")
                    if body_node:
                        self._walk_for_calls(body_node, func_qname, "Function")

    def _walk_for_calls(self, node: Node, scope_qname: str, scope_entity_type: str) -> None:
        """Walk AST recursively collecting call and new_expression nodes."""
        for child in node.named_children:
            if child.type == "call_expression":
                self._emit_call(child, scope_qname, scope_entity_type)
                self._walk_for_calls(child, scope_qname, scope_entity_type)
            elif child.type == "new_expression":
                self._emit_new(child, scope_qname, scope_entity_type)
                self._walk_for_calls(child, scope_qname, scope_entity_type)
            else:
                self._walk_for_calls(child, scope_qname, scope_entity_type)

    def _emit_call(self, call_node: Node, scope_qname: str, scope_entity_type: str) -> None:
        """Emit a CALLS relation for a call_expression node."""
        # The function being called: first named child (identifier or member_expression)
        func_node = None
        for child in call_node.named_children:
            if child.type in ("identifier", "member_expression"):
                func_node = child
                break

        if func_node is None:
            return

        call_site = call_node.start_point
        callee_text = _node_text(func_node)
        if not callee_text:
            return

        if func_node.type == "member_expression":
            # Attribute-style call: unresolved
            target_qname = callee_text
            confidence = 0.5
            meta: dict[str, Any] = {
                "resolution": "unresolved",
                "call_line": call_site[0] + 1,
                "call_column": call_site[1],
            }
        else:
            # Simple identifier call — look up in known symbols
            resolved = self._known_symbols.get(callee_text)
            if resolved is not None:
                target_qname = resolved
                confidence = 1.0
                meta = {
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

        # Determine target entity type: Class if known as class, else Function
        target_entity_type = "Function"
        if confidence == 1.0:
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

    def _emit_new(self, new_node: Node, scope_qname: str, scope_entity_type: str) -> None:
        """Emit a CALLS relation for a new_expression node (constructor call)."""
        ctor_node = None
        for child in new_node.named_children:
            if child.type in ("identifier", "member_expression"):
                ctor_node = child
                break

        if ctor_node is None:
            return

        call_site = new_node.start_point
        callee_text = _node_text(ctor_node)
        if not callee_text:
            return

        if ctor_node.type == "member_expression":
            target_qname = callee_text
            confidence = 0.5
            meta: dict[str, Any] = {
                "resolution": "unresolved",
                "call_line": call_site[0] + 1,
                "call_column": call_site[1],
                "is_constructor": True,
            }
            target_entity_type = "Class"
        else:
            resolved = self._known_symbols.get(callee_text)
            class_qnames = {e["qualified_name"] for e in self._class_entities}
            if resolved is not None:
                target_qname = resolved
                # confidence 1.0 only when the resolved symbol is a known Class.
                # new() applied to a Function is semantically unresolved.
                if resolved in class_qnames:
                    confidence = 1.0
                    meta = {
                        "call_line": call_site[0] + 1,
                        "call_column": call_site[1],
                        "is_constructor": True,
                    }
                else:
                    confidence = 0.5
                    meta = {
                        "resolution": "unresolved",
                        "call_line": call_site[0] + 1,
                        "call_column": call_site[1],
                        "is_constructor": True,
                    }
            else:
                target_qname = f"{self._module_qname}.{callee_text}"
                confidence = 0.5
                meta = {
                    "resolution": "unresolved",
                    "call_line": call_site[0] + 1,
                    "call_column": call_site[1],
                    "is_constructor": True,
                }
            # new_expression always targets a Class entity type
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


def _identifier_name(node: Node) -> str | None:
    """Return the identifier name for a class_declaration or function_declaration node.

    JavaScript uses 'identifier' for both class names and function names
    (unlike TypeScript which uses 'type_identifier' for class names).
    Uses field_name 'name' when available, falls back to scanning named children.
    """
    # Try field first (most efficient)
    name_node = node.child_by_field_name("name")
    if name_node is not None and name_node.type == "identifier":
        return _node_text(name_node)
    # Fallback: scan named children
    for child in node.named_children:
        if child.type == "identifier":
            return _node_text(child)
    return None


def _property_identifier_name(node: Node) -> str | None:
    """Return the property_identifier name for a method_definition node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None and name_node.type == "property_identifier":
        return _node_text(name_node)
    for child in node.named_children:
        if child.type == "property_identifier":
            return _node_text(child)
    return None


def _import_source(import_node: Node) -> str | None:
    """Extract the module specifier string from an import_statement node.

    The source string is the last string node child. We extract the
    string_fragment content (without quotes).
    """
    for child in reversed(import_node.named_children):
        if child.type == "string":
            for frag in child.named_children:
                if frag.type == "string_fragment":
                    return _node_text(frag)
            raw = _node_text(child)
            if raw:
                return raw.strip("'\"")
    return None


def _node_text(node: Node) -> str | None:
    """Return decoded text for a node."""
    text = node.text
    if text is None:
        return None
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return str(text)


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
