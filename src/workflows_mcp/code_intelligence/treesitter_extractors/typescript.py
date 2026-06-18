"""TypeScript/TSX structural extraction using tree-sitter.

Traverses a parsed TypeScript or TSX AST and emits:
- Entities: File, Module, Class, Method, Function
- Relations: CONTAINS, INHERITS_FROM, IMPORTS, CALLS

Design constraints:
- Stateless: no DB, no cross-file resolution, no shared state
- No stdout/print: use logger only
- Returns data suitable for TreeSitterOutput entities/relations lists

Grammar notes (tree-sitter-typescript 0.25):
- Root node type: "program"
- class_declaration: type_identifier (name), class_heritage?, class_body
- class_heritage: extends_clause (identifier or member_expression)
- method_definition: property_identifier (name), statement_block (body)
- function_declaration: identifier (name), statement_block (body)
- import_statement: import_clause (default/named/namespace), string (source)
- call_expression: (identifier|member_expression) (function), arguments
- new_expression: identifier (constructor), arguments
- interface_declaration: emitted as concern only; not mapped to a new entity type

Entity ordering: File, Module, Class (declaration order), Method (class then
declaration order), Function (top-level declaration order).

Relation ordering: CONTAINS (File->Module first, Module->Class/Function,
Class->Method), INHERITS_FROM, IMPORTS, CALLS.

Concerns:
- interface_declaration: TypeScript interfaces are structurally similar to
  abstract classes. They are NOT extracted as entities to avoid inventing
  entity types outside the allowed set (Class/Method/Function). If the caller
  needs interface extraction, it must be added as a future extension under
  the same entity types (e.g., Class with metadata.is_interface=True).
- Arrow functions assigned to const/let/var: NOT extracted as Function
  entities; only function_declaration nodes are extracted.
- Member-expression base class in extends_clause (e.g., React.Component):
  treated as external/unresolved with confidence 0.5.
"""

from __future__ import annotations

import logging
from typing import Any

from tree_sitter import Node

from .base import BaseExtractor, make_relation, node_text

logger = logging.getLogger(__name__)


class TypeScriptExtractor(BaseExtractor):
    """Stateless per-invocation TypeScript/TSX extractor."""

    def _run_passes(self) -> None:
        # Pass 1: collect class/function names for symbol resolution
        self._collect_known_symbols()

        # Pass 2: extract structural entities
        self._extract_program_children(self._root)

        # Pass 3: extract imports
        self._extract_imports(self._root)

        # Pass 4: extract calls (needs known symbols)
        self._extract_calls_from_program(self._root)

    def _relation_buckets(self) -> list[list[dict[str, Any]]]:
        return [
            self._contains_relations,
            self._inherits_relations,
            self._import_relations,
            self._call_relations,
        ]

    # ------------------------------------------------------------------
    # Pass 1: symbol index
    # ------------------------------------------------------------------

    def _collect_known_symbols(self) -> None:
        """Index top-level classes, functions, and methods for CALLS resolution."""
        for child in self._root.named_children:
            if child.type == "class_declaration":
                class_name = _type_identifier_name(child)
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
            # interface_declaration: not extracted (see module docstring concerns)

    def _extract_class(self, node: Node) -> None:
        class_name = _type_identifier_name(node)
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
            make_relation(
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
        """Emit INHERITS_FROM for each base class in extends_clause."""
        # In tree-sitter TS grammar, inheritance is in class_heritage -> extends_clause
        heritage = None
        for child in class_node.named_children:
            if child.type == "class_heritage":
                heritage = child
                break

        if heritage is None:
            return

        extends_clause = None
        for child in heritage.named_children:
            if child.type == "extends_clause":
                extends_clause = child
                break

        if extends_clause is None:
            return

        # Collect same-module class names for resolution
        same_module_classes: set[str] = set()
        for sibling in self._root.named_children:
            if sibling.type == "class_declaration":
                name = _type_identifier_name(sibling)
                if name:
                    same_module_classes.add(name)

        # The base is the first significant named child of extends_clause
        # Could be identifier (simple name) or member_expression (e.g., React.Component)
        for base_node in extends_clause.named_children:
            if base_node.type in ("identifier", "member_expression", "type_identifier"):
                base_text = node_text(base_node)
                if not base_text:
                    continue

                # Simple name resolution: only simple identifiers can be same-module
                simple = base_text.split(".")[0]
                if base_node.type == "identifier" and simple in same_module_classes:
                    target_qname = f"{self._module_qname}.{base_text}"
                    confidence = 1.0
                    meta: dict[str, Any] = {}
                else:
                    # External/unresolved (member_expression or unknown identifier)
                    target_qname = base_text
                    confidence = 0.5
                    meta = {"resolution": "unresolved"}

                self._inherits_relations.append(
                    make_relation(
                        source_qname=class_qname,
                        source_entity_type="Class",
                        target_qname=target_qname,
                        target_entity_type="Class",
                        relation_type="INHERITS_FROM",
                        confidence=confidence,
                        metadata=meta,
                    )
                )
                # Only first base class (TS supports single class extension)
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
            make_relation(
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
            make_relation(
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
        - import type { T } from 'pkg'    -> type-only import (same handling)

        The module source is always a string literal — the last named child
        of type "string" containing a "string_fragment".
        """
        source = _import_source(node)
        if source and source not in seen:
            seen.add(source)
            self._emit_import(source)

    def _emit_import(self, target_module: str) -> None:
        self._import_relations.append(
            make_relation(
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
                class_name = _type_identifier_name(child)
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
        # The function being called is the first named child
        # (identifier or member_expression in TS grammar)
        func_node = None
        for child in call_node.named_children:
            if child.type in ("identifier", "member_expression"):
                func_node = child
                break

        if func_node is None:
            return

        call_site = call_node.start_point
        callee_text = node_text(func_node)
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
            make_relation(
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
        # new_expression: first named child is typically identifier (class name)
        ctor_node = None
        for child in new_node.named_children:
            if child.type in ("identifier", "member_expression"):
                ctor_node = child
                break

        if ctor_node is None:
            return

        call_site = new_node.start_point
        callee_text = node_text(ctor_node)
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
                # A `new` on a known Function is semantically unresolved.
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
            make_relation(
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


def _type_identifier_name(node: Node) -> str | None:
    """Return the type_identifier name for a class_declaration node."""
    for child in node.named_children:
        if child.type == "type_identifier":
            return node_text(child)
    return None


def _identifier_name(node: Node) -> str | None:
    """Return the identifier name for a function_declaration node."""
    for child in node.named_children:
        if child.type == "identifier":
            return node_text(child)
    return None


def _property_identifier_name(node: Node) -> str | None:
    """Return the property_identifier name for a method_definition node."""
    for child in node.named_children:
        if child.type == "property_identifier":
            return node_text(child)
    return None


def _import_source(import_node: Node) -> str | None:
    """Extract the module specifier string from an import_statement node.

    The source string is the last string node child. We extract the
    string_fragment content (without quotes).
    """
    # Walk named children in reverse to find the string node (the source)
    for child in reversed(import_node.named_children):
        if child.type == "string":
            for frag in child.named_children:
                if frag.type == "string_fragment":
                    return node_text(frag)
            # Fallback: decode the whole string and strip quotes
            raw = node_text(child)
            if raw:
                return raw.strip("'\"")
    return None
