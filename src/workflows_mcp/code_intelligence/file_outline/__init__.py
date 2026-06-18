"""File outline extraction for the ReadFiles executor.

Split along three seams, each independently unit-testable:

- ``filtering`` — gitignore/glob exclusion and binary detection (which files to
  read and how to encode them).
- ``extractors`` — the per-format outline interface; a single
  extension->extractor dispatch with one home for outline format knowledge.
  Source files route through the shared ``treesitter_languages`` language home.
- ``sections`` — Markdown section tree, token annotation, and document-level
  Markdown intelligence (frontmatter/references/code blocks).

``outline`` composes the format and section seams into the two entry points the
executor calls.
"""

from __future__ import annotations

from .extractors import OutlineMode, extract_outline
from .filtering import (
    BASE64_ENCODE_EXTENSIONS,
    DEFAULT_EXCLUDE_PATTERNS,
    create_gitignore_spec,
    is_binary,
    load_gitignore_patterns,
    matches_gitignore,
    matches_pattern,
)
from .outline import generate_file_outline, generate_file_outline_with_sections
from .sections import (
    OutlineDict,
    annotate_section_tokens,
    extract_markdown_code_blocks,
    extract_markdown_frontmatter,
    extract_markdown_references,
    extract_markdown_sections,
)

__all__ = [
    "BASE64_ENCODE_EXTENSIONS",
    "DEFAULT_EXCLUDE_PATTERNS",
    "OutlineDict",
    "OutlineMode",
    "annotate_section_tokens",
    "create_gitignore_spec",
    "extract_markdown_code_blocks",
    "extract_markdown_frontmatter",
    "extract_markdown_references",
    "extract_markdown_sections",
    "extract_outline",
    "generate_file_outline",
    "generate_file_outline_with_sections",
    "is_binary",
    "load_gitignore_patterns",
    "matches_gitignore",
    "matches_pattern",
]
