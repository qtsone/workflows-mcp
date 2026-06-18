"""Outline orchestration: compose the format and section seams for ReadFiles.

These are the two entry points the ReadFiles executor calls. They read line
counts and headers, then delegate format extraction and section building to the
respective seams.
"""

from __future__ import annotations

from pathlib import Path

from .extractors import OutlineMode, extract_outline
from .sections import OutlineDict, build_sections, count_tree

_MARKDOWN_EXTENSIONS = (".md", ".markdown")


def _count_lines(file_path: Path) -> int:
    """Count lines in a file, returning 0 when it cannot be read."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def generate_file_outline(file_path: Path, mode: OutlineMode) -> str:
    """Generate outline for a file based on its type and mode.

    Extracts structure/symbols with context reduction (90-97%).
    This function is only called for outline/summary modes.
    Full mode is handled separately by the executor (direct file read).

    Args:
        file_path: Path to file
        mode: Outline extraction mode
            - "outline": Structure/symbol extraction only (90-97% reduction)
            - "summary": Outline + docstrings/comments (85-95% reduction)

    Returns:
        Formatted file outline with header
    """
    header = f"--- {file_path.name} ({_count_lines(file_path)} lines) ---"
    return f"{header}\n{extract_outline(file_path, mode)}"


def generate_file_outline_with_sections(
    file_path: Path, mode: OutlineMode
) -> tuple[str, list[OutlineDict], int, int]:
    """Generate outline for a file AND return structured sections tree.

    Extends generate_file_outline() with structured section metadata
    suitable for recursive per-section processing.

    Args:
        file_path: Path to file
        mode: Outline extraction mode

    Returns:
        Tuple of (outline_string, sections_tree, max_depth, total_sections).
        For non-markdown files, sections contains a single implicit section
        covering the whole file (zero-section fallback).
    """
    outline_str = generate_file_outline(file_path, mode)

    is_markdown = file_path.suffix.lower() in _MARKDOWN_EXTENSIONS
    sections = build_sections(file_path, is_markdown)
    max_depth, total_sections = count_tree(sections)

    return outline_str, sections, max_depth, total_sections
