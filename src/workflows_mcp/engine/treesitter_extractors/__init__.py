"""Structural extractor package for tree-sitter languages.

Provides:
- extract_document() for document languages (Markdown, YAML, JSON) — file-only
- extract_go() for Go source files
- extract_javascript() for JavaScript and JSX source files
- extract_python() for Python source files
- extract_rust() for Rust source files
- extract_typescript() for TypeScript and TSX source files
"""

from .document import extract_document
from .go import extract_go, extract_package_name
from .javascript import extract_javascript
from .python import extract_python
from .rust import extract_rust
from .typescript import extract_typescript

__all__ = [
    "extract_document",
    "extract_go",
    "extract_javascript",
    "extract_package_name",
    "extract_python",
    "extract_rust",
    "extract_typescript",
]
