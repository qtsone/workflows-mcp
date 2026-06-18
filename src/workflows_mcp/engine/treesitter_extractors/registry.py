"""Registry mapping source languages to their structural extractor.

The executor dispatches code extraction through this registry instead of an
if-elif over language names. Adding a language is a registration here — a new
``BaseExtractor`` subclass added to ``_EXTRACTOR_CLASSES`` — with no edit to the
executor's dispatch.

Document languages (Markdown, YAML, JSON) are file-only and handled separately
by ``extract_document``; they are not registered here.
"""

from __future__ import annotations

from typing import Any

from tree_sitter import Node

from .base import BaseExtractor
from .go import GoExtractor
from .javascript import JavaScriptExtractor
from .python import PythonExtractor
from .rust import RustExtractor
from .typescript import TypeScriptExtractor

# TypeScript and TSX share one grammar shape and one extractor; JavaScript covers
# .js/.jsx. Multiple language identifiers may map to the same extractor class.
_EXTRACTOR_CLASSES: dict[str, type[BaseExtractor]] = {
    "python": PythonExtractor,
    "go": GoExtractor,
    "typescript": TypeScriptExtractor,
    "tsx": TypeScriptExtractor,
    "javascript": JavaScriptExtractor,
    "rust": RustExtractor,
}


def has_extractor(language: str) -> bool:
    """Return whether a code extractor is registered for the language."""
    return language in _EXTRACTOR_CLASSES


def extract_code(
    language: str,
    *,
    root: Node,
    file_qname: str,
    module_qname: str,
    file_entity: dict[str, Any],
    module_entity: dict[str, Any],
    contains_file_module: dict[str, Any],
    stable_id_fn: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the registered extractor for ``language`` and return (entities, relations).

    Raises:
        KeyError: If no extractor is registered for the language. Callers should
            guard with ``has_extractor`` first.
    """
    extractor_cls = _EXTRACTOR_CLASSES[language]
    extractor = extractor_cls(
        root=root,
        file_qname=file_qname,
        module_qname=module_qname,
        file_entity=file_entity,
        module_entity=module_entity,
        contains_file_module=contains_file_module,
        stable_id_fn=stable_id_fn,
    )
    return extractor.extract()
