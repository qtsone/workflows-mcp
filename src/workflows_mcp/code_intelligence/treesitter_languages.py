"""Tree-sitter language detection and lazy parser loading.

Provides:
- detect_language: maps file path/extension to a supported language identifier
- get_parser: lazily loads and caches tree-sitter Parser instances per language
- content_hash: SHA-256 of UTF-8 encoded string (matches tools_memory._hash_content)
"""

from __future__ import annotations

import hashlib
import os
from typing import Literal

from tree_sitter import Language, Parser

SupportedLanguage = Literal[
    "python",
    "typescript",
    "tsx",
    "javascript",
    "go",
    "rust",
    "markdown",
    "yaml",
    "json",
    "unsupported",
]

_EXTENSION_MAP: dict[str, SupportedLanguage] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".md": "markdown",
    ".markdown": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}

# Module-level parser cache: language -> Parser instance
_parser_cache: dict[str, Parser] = {}


def detect_language(path: str) -> SupportedLanguage:
    """Detect language from file path extension.

    Args:
        path: File path (absolute, relative, or basename).

    Returns:
        Supported language identifier or "unsupported".
    """
    _root, ext = os.path.splitext(path)
    return _EXTENSION_MAP.get(ext.lower(), "unsupported")


def _load_language(language: str) -> Language:
    """Load a tree-sitter Language object for the given language name.

    Args:
        language: Supported language identifier (not "unsupported").

    Returns:
        tree-sitter Language instance.

    Raises:
        ValueError: If language is not supported.
    """
    if language == "python":
        import tree_sitter_python as _ts_python

        return Language(_ts_python.language())
    if language == "typescript":
        import tree_sitter_typescript as _ts_typescript

        return Language(_ts_typescript.language_typescript())
    if language == "tsx":
        import tree_sitter_typescript as _ts_tsx

        return Language(_ts_tsx.language_tsx())
    if language == "javascript":
        import tree_sitter_javascript as _ts_javascript

        return Language(_ts_javascript.language())
    if language == "go":
        import tree_sitter_go as _ts_go

        return Language(_ts_go.language())
    if language == "rust":
        import tree_sitter_rust as _ts_rust

        return Language(_ts_rust.language())
    if language == "markdown":
        import tree_sitter_markdown as _ts_markdown

        return Language(_ts_markdown.language())
    if language == "yaml":
        import tree_sitter_yaml as _ts_yaml

        return Language(_ts_yaml.language())
    if language == "json":
        import tree_sitter_json as _ts_json

        return Language(_ts_json.language())
    raise ValueError(f"No grammar available for language: {language!r}")


def get_parser(language: SupportedLanguage) -> Parser:
    """Return a cached Parser for the given language.

    Parsers are created once and reused. Thread safety is not guaranteed
    for concurrent writes, but in practice asyncio is single-threaded.

    Args:
        language: Supported language identifier (not "unsupported").

    Returns:
        Configured tree-sitter Parser instance.

    Raises:
        ValueError: If language is "unsupported" or has no grammar.
    """
    if language == "unsupported":
        raise ValueError("Cannot create a parser for 'unsupported' language")

    if language not in _parser_cache:
        lang = _load_language(language)
        parser = Parser(lang)
        _parser_cache[language] = parser

    return _parser_cache[language]


def content_hash(text: str) -> str:
    """SHA-256 hex digest of UTF-8 encoded text.

    Matches tools_memory._hash_content byte-for-byte.

    Args:
        text: String content to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
