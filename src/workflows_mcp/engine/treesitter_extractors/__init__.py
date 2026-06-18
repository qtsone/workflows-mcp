"""Structural extractor package for tree-sitter languages.

Code languages (Python, Go, JavaScript, TypeScript, Rust) share a
``BaseExtractor`` and are dispatched through the registry:
- extract_code() runs the registered extractor for a language
- has_extractor() reports whether a language is registered

Document languages (Markdown, YAML, JSON) are file-only:
- extract_document() emits a single File entity, no relations

Go-specific:
- extract_package_name() reads the package clause for module naming
"""

from .document import extract_document
from .go import extract_package_name
from .registry import extract_code, has_extractor

__all__ = [
    "extract_code",
    "extract_document",
    "extract_package_name",
    "has_extractor",
]
