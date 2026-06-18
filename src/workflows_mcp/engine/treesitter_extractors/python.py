"""Python structural extraction using tree-sitter.

Traverses a parsed Python AST and emits:
- Entities: File, Module, Class, Method, Function
- Relations: CONTAINS, INHERITS_FROM, IMPORTS, CALLS

Design constraints:
- Stateless: no DB, no cross-file resolution, no shared state
- No stdout/print: use logger only
- Returns data suitable for TreeSitterOutput entities/relations lists

Entity ordering: File, Module, Class (declaration order), Method (class then
declaration order), Function (top-level declaration order).

Relation ordering: CONTAINS (File->Module first, Module->Class/Function,
Class->Method), INHERITS_FROM, IMPORTS, CALLS.
"""

from __future__ import annotations

import logging
from typing import Any

from tree_sitter import Node

from .base import BaseExtractor, make_relation, node_text

logger = logging.getLogger(__name__)

# Sentinel prefix for parent class hint in method metadata.
_PARENT_CLASS_PREFIX = "__class_qname__:"


class PythonExtractor(BaseExtractor):
    """Stateless per-invocation Python extractor."""

    def _run_passes(self) -> None:
        # Pass 1: collect class/function names for symbol resolution
        self._collect_known_symbols()

        # Pass 2: extract structural entities
        self._extract_module_children(self._root)

        # Pass 3: extract imports
        self._extract_imports(self._root)

        # Pass 4: extract calls (needs known symbols)
        self._extract_calls_from_module(self._root)

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
            if child.type == "class_definition":
                class_name = _node_name(child)
                if not class_name:
                    continue
                class_qname = f"{self._module_qname}.{class_name}"
                self._known_symbols[class_name] = class_qname

                body = child.child_by_field_name("body")
                if body:
                    for item in body.named_children:
                        if item.type == "function_definition":
                            method_name = _node_name(item)
                            if method_name:
                                method_qname = f"{class_qname}.{method_name}"
                                # Register both dotted and simple for method lookup
                                self._known_symbols[f"{class_name}.{method_name}"] = method_qname

            elif child.type == "function_definition":
                func_name = _node_name(child)
                if func_name:
                    func_qname = f"{self._module_qname}.{func_name}"
                    self._known_symbols[func_name] = func_qname

    # ------------------------------------------------------------------
    # Pass 2: structural extraction
    # ------------------------------------------------------------------

    def _extract_module_children(self, module_node: Node) -> None:
        """Walk direct children of the module node."""
        for child in module_node.named_children:
            if child.type == "class_definition":
                self._extract_class(child)
            elif child.type == "function_definition":
                self._extract_top_level_function(child)

    def _extract_class(self, node: Node) -> None:
        class_name = _node_name(node)
        if not class_name:
            return

        class_qname = f"{self._module_qname}.{class_name}"
        start = node.start_point  # (row, col) — 0-indexed

        class_entity: dict[str, Any] = {
            "entity_type": "Class",
            "name": class_name,
            "qualified_name": class_qname,
            "stable_id": self._stable_id_fn(class_qname, "Class"),
            "metadata": {
                "start_line": start[0] + 1,  # convert to 1-indexed
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
                if item.type == "function_definition":
                    self._extract_method(item, class_qname)

    def _extract_inheritance(self, class_node: Node, class_qname: str) -> None:
        """Emit INHERITS_FROM for each base class in argument_list."""
        # In tree-sitter Python grammar, base classes are in argument_list
        args_node = class_node.child_by_field_name("superclasses")
        if args_node is None:
            # Try direct child of type argument_list (grammar field may differ)
            for child in class_node.named_children:
                if child.type == "argument_list":
                    args_node = child
                    break

        if args_node is None:
            return

        # Collect same-module class names for resolution
        same_module_classes = {e["name"] for e in self._class_entities}
        # Also peek ahead at siblings not yet extracted:
        # re-scan root for class names to allow forward references
        for sibling in self._root.named_children:
            if sibling.type == "class_definition":
                name = _node_name(sibling)
                if name:
                    same_module_classes.add(name)

        for base_node in args_node.named_children:
            base_text = node_text(base_node)
            if not base_text:
                continue

            # Determine if base is in same module
            # base_text may be simple name or dotted
            simple = base_text.split(".")[0]
            if simple in same_module_classes:
                # Same-module base — resolve to module-qualified name
                target_qname = f"{self._module_qname}.{base_text}"
                confidence = 1.0
                meta: dict[str, Any] = {}
            else:
                # External/unresolved
                target_qname = base_text  # best-effort
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

    def _extract_method(self, node: Node, class_qname: str) -> None:
        method_name = _node_name(node)
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

    def _extract_top_level_function(self, node: Node) -> None:
        func_name = _node_name(node)
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

    def _extract_imports(self, module_node: Node) -> None:
        """Emit IMPORTS relations for all import statements at module level."""
        for child in module_node.named_children:
            if child.type == "import_statement":
                self._handle_import_statement(child)
            elif child.type == "import_from_statement":
                self._handle_from_import(child)

    def _handle_import_statement(self, node: Node) -> None:
        """Handle: import x, import x.y, import x as z."""
        for child in node.named_children:
            if child.type == "dotted_name":
                module_path = node_text(child)
                if module_path:
                    self._emit_import(module_path)
            elif child.type == "aliased_import":
                # aliased_import -> dotted_name + identifier(alias)
                for sub in child.named_children:
                    if sub.type == "dotted_name":
                        module_path = node_text(sub)
                        if module_path:
                            self._emit_import(module_path)
                        break

    def _handle_from_import(self, node: Node) -> None:
        """Handle: from x.y import z, from x import y, from .rel import z."""
        # The module being imported from is the first dotted_name or relative_import child
        module_node = None
        for child in node.named_children:
            if child.type in ("dotted_name", "relative_import"):
                module_node = child
                break

        if module_node is None:
            return

        raw = node_text(module_node)
        if not raw:
            return

        if module_node.type == "relative_import":
            target = self._resolve_relative_import(raw)
        else:
            target = raw

        self._emit_import(target)

    def _resolve_relative_import(self, raw: str) -> str:
        """Normalize a relative import string to an absolute dotted name.

        ``raw`` is the text of the ``relative_import`` node, e.g.:
        - ``.``          -> from . import foo  (bare current-package import)
        - ``.sibling``   -> from .sibling import bar
        - ``..``         -> from .. import baz
        - ``..parent``   -> from ..parent import x

        Resolution uses ``self._module_qname`` as context:
        - Count leading dots to determine how many package levels to ascend.
        - Strip those levels from the module qname.
        - Append any remaining dotted suffix from ``raw``.

        If the module qname has fewer levels than needed, return ``raw``
        unchanged (best-effort, unresolvable).
        """
        # Count leading dots
        dots = 0
        for ch in raw:
            if ch == ".":
                dots += 1
            else:
                break

        suffix = raw[dots:]  # dotted name after the leading dots, may be empty

        parts = self._module_qname.split(".") if self._module_qname else []

        # Ascend: dots=1 -> stay in same package (remove last component, the module name)
        #         dots=2 -> go up one more package level, etc.
        levels_to_remove = dots  # remove `dots` components from the right
        if levels_to_remove > len(parts):
            # Cannot resolve — return raw as best-effort
            return raw

        package_parts = parts[:-levels_to_remove] if levels_to_remove else parts

        if suffix:
            resolved = ".".join(package_parts + [suffix]) if package_parts else suffix
        else:
            resolved = ".".join(package_parts) if package_parts else raw

        return resolved if resolved else raw

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

    def _extract_calls_from_module(self, module_node: Node) -> None:
        """Extract CALLS from all top-level functions and methods."""
        for child in module_node.named_children:
            if child.type == "class_definition":
                class_name = _node_name(child)
                if not class_name:
                    continue
                class_qname = f"{self._module_qname}.{class_name}"
                body = child.child_by_field_name("body")
                if body:
                    for item in body.named_children:
                        if item.type == "function_definition":
                            method_name = _node_name(item)
                            if method_name:
                                method_qname = f"{class_qname}.{method_name}"
                                self._collect_calls_in_scope(item, method_qname, "Method")
            elif child.type == "function_definition":
                func_name = _node_name(child)
                if func_name:
                    func_qname = f"{self._module_qname}.{func_name}"
                    self._collect_calls_in_scope(child, func_qname, "Function")

    def _collect_calls_in_scope(
        self, scope_node: Node, scope_qname: str, scope_entity_type: str
    ) -> None:
        """Recursively find all call expressions within scope_node."""
        body = scope_node.child_by_field_name("body")
        if body is None:
            return
        self._walk_for_calls(body, scope_qname, scope_entity_type)

    def _walk_for_calls(self, node: Node, scope_qname: str, scope_entity_type: str) -> None:
        """Walk AST recursively collecting call nodes.

        When a ``call`` node is found, emit it and continue recursing into its
        children so that nested calls (e.g. ``bar(baz())``) are also captured.
        """
        for child in node.named_children:
            if child.type == "call":
                self._emit_call(child, scope_qname, scope_entity_type)
                # Recurse into the call's children to find nested calls
                self._walk_for_calls(child, scope_qname, scope_entity_type)
            else:
                self._walk_for_calls(child, scope_qname, scope_entity_type)

    def _emit_call(self, call_node: Node, scope_qname: str, scope_entity_type: str) -> None:
        """Emit a CALLS relation for a call expression node."""
        # Function part of the call is the first named child (identifier or attribute)
        func_node = None
        for child in call_node.named_children:
            if child.type in ("identifier", "attribute"):
                func_node = child
                break

        if func_node is None:
            return

        call_site = call_node.start_point
        callee_text = node_text(func_node)
        if not callee_text:
            return

        # For attribute calls like g.greet, use only simple function name part
        # for resolution lookup (last component of attribute)
        if func_node.type == "attribute":
            # Attribute calls are method invocations on objects — unresolved
            # unless we can track the receiver type (out of scope here)
            target_qname = callee_text  # best effort e.g. "g.greet"
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
        if confidence == 1.0 and callee_text in self._known_symbols:
            # Check if the resolved qname belongs to a class
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


def _node_name(node: Node) -> str | None:
    """Return the identifier name for a class/function definition node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        text = name_node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="replace")
        return str(text)
    return None
