"""Rust structural extraction using tree-sitter.

Traverses a parsed Rust AST (source_file root) and emits:
- Entities: File, Module, Class (struct/trait), Method, Function
- Relations: CONTAINS, IMPORTS, INHERITS_FROM, CALLS

Design constraints:
- Stateless: no DB, no cross-file resolution, no shared state
- No stdout/print: use logger only
- Returns data suitable for TreeSitterOutput entities/relations lists

Entity ordering: File, Module, Classes (declaration order), Methods (impl
order then declaration order), Functions (top-level declaration order).

Relation ordering: CONTAINS (File->Module first, Module->Class/Function,
Class->Method), IMPORTS, INHERITS_FROM, CALLS.

Rust specifics:
- struct_item -> Class entity (metadata.is_trait omitted / False)
- trait_item -> Class entity (metadata.is_trait=True)
- function_item inside impl_item -> Method entity; class qname inferred from
  impl target type (the struct being implemented, not the trait)
- top-level function_item -> Function entity
- use_declaration -> IMPORTS relation (dotted qname, confidence=0.5 unresolved)
- impl Trait for Struct -> INHERITS_FROM Class->Class, confidence=0.8 if
  trait known in same module (resolution='resolved'), else 0.5 unresolved
- call_expression with identifier callee -> CALLS resolved if in known_symbols
  else unresolved; field_expression callee -> unresolved CALLS
"""

from __future__ import annotations

import logging
from typing import Any

from tree_sitter import Node

from .base import BaseExtractor, make_relation, node_text

logger = logging.getLogger(__name__)

# Sentinel prefix for parent class hint in method metadata.
_PARENT_CLASS_PREFIX = "__class_qname__:"


class RustExtractor(BaseExtractor):
    """Stateless per-invocation Rust extractor."""

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
        super().__init__(
            root=root,
            file_qname=file_qname,
            module_qname=module_qname,
            file_entity=file_entity,
            module_entity=module_entity,
            contains_file_module=contains_file_module,
            stable_id_fn=stable_id_fn,
        )
        # Known trait qnames in this module (for INHERITS_FROM resolution)
        self._local_trait_qnames: set[str] = set()

        # Track seen import paths for deduplication
        self._seen_import_paths: set[str] = set()

    def _run_passes(self) -> None:
        # Pass 1: collect symbol names for resolution
        self._collect_known_symbols()

        # Pass 2: extract structural entities (structs, traits, top-level functions)
        self._extract_source_file_children()

        # Pass 3: extract impl blocks (methods + INHERITS_FROM)
        self._extract_impl_blocks()

        # Pass 4: extract imports
        self._extract_imports()

        # Pass 5: extract calls
        self._extract_calls()

    def _relation_buckets(self) -> list[list[dict[str, Any]]]:
        return [
            self._contains_relations,
            self._import_relations,
            self._inherits_relations,
            self._call_relations,
        ]

    # ------------------------------------------------------------------
    # Pass 1: symbol index
    # ------------------------------------------------------------------

    def _collect_known_symbols(self) -> None:
        """Index top-level structs, traits, and functions for CALLS/INHERITS resolution."""
        for child in self._root.children:
            if child.type == "struct_item":
                name = _item_name(child)
                if name:
                    qname = f"{self._module_qname}.{name}"
                    self._known_symbols[name] = qname
            elif child.type == "trait_item":
                name = _item_name(child)
                if name:
                    qname = f"{self._module_qname}.{name}"
                    self._known_symbols[name] = qname
                    self._local_trait_qnames.add(qname)
            elif child.type == "function_item":
                name = _item_name(child)
                if name:
                    self._known_symbols[name] = f"{self._module_qname}.{name}"

    # ------------------------------------------------------------------
    # Pass 2: structural extraction (structs, traits, top-level functions)
    # ------------------------------------------------------------------

    def _extract_source_file_children(self) -> None:
        """Walk direct children of source_file for struct/trait/function items."""
        for child in self._root.children:
            if child.type == "struct_item":
                self._extract_struct(child)
            elif child.type == "trait_item":
                self._extract_trait(child)
            elif child.type == "function_item":
                self._extract_top_level_function(child)

    def _extract_struct(self, node: Node) -> None:
        """Handle struct_item -> Class entity."""
        name = _item_name(node)
        if not name:
            return

        class_qname = f"{self._module_qname}.{name}"
        start = node.start_point

        class_entity: dict[str, Any] = {
            "entity_type": "Class",
            "name": name,
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

    def _extract_trait(self, node: Node) -> None:
        """Handle trait_item -> Class entity with metadata.is_trait=True."""
        name = _item_name(node)
        if not name:
            return

        class_qname = f"{self._module_qname}.{name}"
        start = node.start_point

        class_entity: dict[str, Any] = {
            "entity_type": "Class",
            "name": name,
            "qualified_name": class_qname,
            "stable_id": self._stable_id_fn(class_qname, "Class"),
            "metadata": {
                "start_line": start[0] + 1,
                "start_column": start[1],
                "is_trait": True,
            },
            "confidence": 1.0,
        }
        self._class_entities.append(class_entity)

        # CONTAINS Module -> Class (trait)
        self._contains_relations.append(
            make_relation(
                source_qname=self._module_qname,
                source_entity_type="Module",
                target_qname=class_qname,
                target_entity_type="Class",
                relation_type="CONTAINS",
            )
        )

    def _extract_top_level_function(self, node: Node) -> None:
        """Handle top-level function_item -> Function entity."""
        name = _item_name(node)
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
            make_relation(
                source_qname=self._module_qname,
                source_entity_type="Module",
                target_qname=func_qname,
                target_entity_type="Function",
                relation_type="CONTAINS",
            )
        )

    # ------------------------------------------------------------------
    # Pass 3: impl blocks
    # ------------------------------------------------------------------

    def _extract_impl_blocks(self) -> None:
        """Process all impl_item nodes at the top level."""
        for child in self._root.children:
            if child.type == "impl_item":
                self._extract_impl(child)

    def _extract_impl(self, node: Node) -> None:
        """Handle an impl_item node.

        Rust grammar for impl_item children (sequential scan):
          - impl <type_identifier|scoped_type_identifier> for <type_identifier> <declaration_list>
            (trait impl — trait may be scoped, struct is always type_identifier after 'for')
          - impl <type_identifier> <declaration_list>
            (plain impl — no 'for' keyword)

        We scan children sequentially to correctly distinguish the trait type
        (before 'for') from the implementing struct type (after 'for'), handling
        both plain `type_identifier` and `scoped_type_identifier` for the trait.
        """
        struct_name: str | None = None
        trait_dotted: str | None = None  # dotted qname for the trait (may be scoped)

        after_for = False
        for child in node.children:
            if child.type == "for":
                after_for = True
                continue
            if child.type == "type_identifier":
                text = node_text(child)
                if not text:
                    continue
                if after_for:
                    # This is the implementing struct type
                    struct_name = text
                else:
                    # This is the trait type (simple name)
                    trait_dotted = text
            elif child.type == "scoped_type_identifier":
                # e.g. std::fmt::Display — convert to dotted path
                dotted = _scoped_type_identifier_to_dotted(child)
                if dotted and not after_for:
                    trait_dotted = dotted
                # scoped_type_identifier after 'for' would be a generic struct —
                # extremely rare; skip conservatively (struct_name stays None)

        # Plain impl: if no 'for' was seen, the first type_identifier is the struct
        if not after_for:
            # Re-scan: first type_identifier is the struct, trait_dotted stays None
            struct_name = None
            trait_dotted = None
            for child in node.children:
                if child.type == "type_identifier":
                    struct_name = node_text(child)
                    break

        if not struct_name:
            logger.debug(
                "Rust extractor: skipping impl_item with no resolvable struct type in %s",
                self._file_qname,
            )
            return

        class_qname = f"{self._module_qname}.{struct_name}"

        # Extract methods from declaration_list
        for child in node.children:
            if child.type == "declaration_list":
                for item in child.children:
                    if item.type == "function_item":
                        self._extract_method(item, class_qname)

        # Emit INHERITS_FROM for trait impl
        if trait_dotted:
            # Determine if trait is local: simple name that matches a known local trait
            local_trait_qname = f"{self._module_qname}.{trait_dotted}"
            if local_trait_qname in self._local_trait_qnames:
                target_qname = local_trait_qname
                confidence = 0.8
                meta: dict[str, Any] = {
                    "rust_trait": True,
                    "resolution": "resolved",
                }
            else:
                # Scoped or external trait — use the dotted path as-is
                target_qname = trait_dotted
                confidence = 0.5
                meta = {
                    "rust_trait": True,
                    "resolution": "unresolved",
                }

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

    def _extract_method(self, node: Node, class_qname: str) -> None:
        """Handle a function_item inside an impl block -> Method entity."""
        name = _item_name(node)
        if not name:
            return

        method_qname = f"{class_qname}.{name}"
        start = node.start_point

        method_entity: dict[str, Any] = {
            "entity_type": "Method",
            "name": name,
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
            make_relation(
                source_qname=class_qname,
                source_entity_type="Class",
                target_qname=method_qname,
                target_entity_type="Method",
                relation_type="CONTAINS",
            )
        )

    # ------------------------------------------------------------------
    # Pass 4: imports (use_declaration)
    # ------------------------------------------------------------------

    def _extract_imports(self) -> None:
        """Emit IMPORTS relations for use_declaration nodes."""
        for child in self._root.children:
            if child.type == "use_declaration":
                self._handle_use_declaration(child)

    def _handle_use_declaration(self, node: Node) -> None:
        """Extract import path(s) from a use_declaration."""
        for child in node.children:
            if child.type == "scoped_identifier":
                path = _scoped_identifier_to_dotted(child)
                if path:
                    self._emit_import(path)
            elif child.type == "scoped_use_list":
                # e.g. use std::io::{self, Write}
                # prefix is the scoped_identifier before '::'
                prefix = None
                for sub in child.children:
                    if sub.type == "scoped_identifier":
                        prefix = _scoped_identifier_to_dotted(sub)
                    elif sub.type == "use_list" and prefix:
                        # Emit import for each named item (skip 'self')
                        for item in sub.children:
                            if item.type == "identifier":
                                name = node_text(item)
                                if name:
                                    self._emit_import(f"{prefix}.{name}")
                            elif item.type == "scoped_identifier":
                                item_path = _scoped_identifier_to_dotted(item)
                                if item_path:
                                    self._emit_import(f"{prefix}.{item_path}")
            elif child.type == "identifier":
                # bare `use foo;`
                name = node_text(child)
                if name:
                    self._emit_import(name)

    def _emit_import(self, path: str) -> None:
        """Emit one IMPORTS relation if not already seen."""
        if not path or path in self._seen_import_paths:
            return
        self._seen_import_paths.add(path)

        self._import_relations.append(
            make_relation(
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
    # Pass 5: CALLS extraction
    # ------------------------------------------------------------------

    def _extract_calls(self) -> None:
        """Extract CALLS from all top-level functions and impl methods."""
        for child in self._root.children:
            if child.type == "function_item":
                name = _item_name(child)
                if name:
                    func_qname = f"{self._module_qname}.{name}"
                    self._collect_calls_in_body(child, func_qname, "Function")
            elif child.type == "impl_item":
                self._extract_calls_from_impl(child)

    def _extract_calls_from_impl(self, node: Node) -> None:
        """Extract CALLS from function_items inside an impl block.

        Uses sequential child scan to correctly handle both plain and scoped
        trait impls:
          - impl Trait for Struct { ... }         (type_identifier for both)
          - impl std::fmt::Display for Foo { ... } (scoped_type_identifier + type_identifier)

        The struct type is always the type_identifier after 'for' (trait impl)
        or the first type_identifier (plain impl). The trait type, whether
        simple or scoped, does not affect struct resolution.
        """
        struct_name: str | None = None
        after_for = False

        for child in node.children:
            if child.type == "for":
                after_for = True
                continue
            if child.type == "type_identifier":
                text = node_text(child)
                if not text:
                    continue
                if after_for:
                    struct_name = text
                    break
                else:
                    # Plain impl: first type_identifier is the struct
                    struct_name = text

        if not struct_name:
            logger.debug(
                "Rust extractor: skipping impl_item CALLS (no resolvable struct type) in %s",
                self._file_qname,
            )
            return

        class_qname = f"{self._module_qname}.{struct_name}"

        for child in node.children:
            if child.type == "declaration_list":
                for item in child.children:
                    if item.type == "function_item":
                        name = _item_name(item)
                        if name:
                            method_qname = f"{class_qname}.{name}"
                            self._collect_calls_in_body(item, method_qname, "Method")

    def _collect_calls_in_body(
        self, func_node: Node, scope_qname: str, scope_entity_type: str
    ) -> None:
        """Find the block child and recursively walk for call_expression nodes."""
        for child in func_node.children:
            if child.type == "block":
                self._walk_for_calls(child, scope_qname, scope_entity_type)
                return

    def _walk_for_calls(self, node: Node, scope_qname: str, scope_entity_type: str) -> None:
        """Recursively walk AST collecting call_expression nodes."""
        for child in node.children:
            if child.type == "call_expression":
                self._emit_call(child, scope_qname, scope_entity_type)
                # Recurse to find nested calls in arguments
                self._walk_for_calls(child, scope_qname, scope_entity_type)
            else:
                self._walk_for_calls(child, scope_qname, scope_entity_type)

    def _emit_call(self, call_node: Node, scope_qname: str, scope_entity_type: str) -> None:
        """Emit a CALLS relation for a call_expression node."""
        # First child is the function expression (identifier or field_expression)
        func_node = call_node.children[0] if call_node.children else None
        if func_node is None:
            return

        call_site = call_node.start_point

        if func_node.type == "identifier":
            callee_text = node_text(func_node)
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
        elif func_node.type == "field_expression":
            # e.g. self.method(), bar.baz()
            callee_text = node_text(func_node) or ""
            target_qname = callee_text
            confidence = 0.5
            meta = {
                "resolution": "unresolved",
                "call_line": call_site[0] + 1,
                "call_column": call_site[1],
            }
        else:
            # Scoped paths (std::fmt::write), macro-like calls, etc. — skip
            return

        # Determine target entity type
        target_entity_type = "Function"
        if func_node.type == "identifier":
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item_name(node: Node) -> str | None:
    """Return the identifier or type_identifier name from a top-level item node.

    Works for struct_item, trait_item, function_item (identifier child).
    """
    for child in node.children:
        if child.type == "type_identifier":
            return node_text(child)
        if child.type == "identifier":
            return node_text(child)
    return None


def _scoped_identifier_to_dotted(node: Node) -> str | None:
    """Convert a scoped_identifier node to a dotted path string.

    e.g. std::fmt -> 'std.fmt'
         std::collections::HashMap -> 'std.collections.HashMap'
    """
    # Collect all identifiers (and crate/self/super keywords) from the node
    parts: list[str] = []

    def _collect(n: Node) -> None:
        if n.type in ("identifier", "crate", "super"):
            text = node_text(n)
            if text:
                parts.append(text)
        elif n.type == "scoped_identifier":
            for c in n.children:
                _collect(c)
        # Skip "::" and other tokens

    _collect(node)
    if not parts:
        return None
    return ".".join(parts)


def _scoped_type_identifier_to_dotted(node: Node) -> str | None:
    """Convert a scoped_type_identifier node to a dotted path string.

    e.g. std::fmt::Display -> 'std.fmt.Display'

    scoped_type_identifier grammar:
      scoped_type_identifier: (path '::')? type_identifier
    where path may itself be a scoped_identifier or scoped_type_identifier.
    We collect all identifier/type_identifier leaf text in order.
    """
    parts: list[str] = []

    def _collect(n: Node) -> None:
        if n.type in ("identifier", "type_identifier", "crate", "super"):
            text = node_text(n)
            if text:
                parts.append(text)
        elif n.type in ("scoped_identifier", "scoped_type_identifier"):
            for c in n.children:
                _collect(c)
        # Skip "::" and other punctuation tokens

    _collect(node)
    if not parts:
        return None
    return ".".join(parts)
