"""Section-annotation seam for ReadFiles.

Parses Markdown into a nested section tree (line ranges per section, ready for
recursive section-by-section processing), annotates it with token estimates, and
extracts the document-level Markdown intelligence the executor surfaces
alongside the outline (frontmatter, references, fenced code blocks). Independent
of file-filtering and format extraction, and unit-testable on its own.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

OutlineDict = dict[str, Any]


def extract_markdown_frontmatter(content: str) -> dict[str, Any] | None:
    """Extract YAML frontmatter from Markdown content.

    Parses the optional YAML block between ``---`` fences at the start
    of the document. Returns None if no frontmatter is present.

    Args:
        content: Raw markdown text

    Returns:
        Parsed frontmatter dict, or None if absent/invalid.
    """
    stripped = content.lstrip()
    if not stripped.startswith("---"):
        return None

    # Find closing fence
    end_idx = stripped.find("---", 3)
    if end_idx == -1:
        return None

    yaml_text = stripped[3:end_idx].strip()
    if not yaml_text:
        return None

    try:
        from datetime import date, datetime

        import yaml as _yaml

        result = _yaml.safe_load(yaml_text)
        if not isinstance(result, dict):
            return None

        # PyYAML safe_load converts date strings (e.g. "2026-03-06") to
        # datetime.date objects — sanitize to ISO strings for JSON safety.
        def _sanitize(obj: object) -> object:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, date):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj

        sanitized = _sanitize(result)
        if isinstance(sanitized, dict):
            return cast(dict[str, Any], sanitized)
        return None
    except Exception:
        return None


def extract_markdown_references(content: str) -> list[dict[str, Any]]:
    """Extract references from Markdown content.

    Finds wikilinks (``[[target]]``) and explicit file paths
    (lines containing filesystem-like paths).

    Args:
        content: Raw markdown text

    Returns:
        List of dicts with 'type' and 'target' keys.
    """
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Wikilinks: [[target]] or [[target|label]]
    for match in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", content):
        target = match.group(1).strip()
        if target and target not in seen:
            refs.append({"type": "wikilink", "target": target})
            seen.add(target)

    # File paths: common patterns like path/to/file.ext in backtick code spans
    for match in re.finditer(r"`([a-zA-Z0-9_\-./]+\.[a-z]{1,5})`", content):
        target = match.group(1)
        # Filter out things that are clearly not paths (e.g., version numbers)
        if "/" in target and target not in seen:
            refs.append({"type": "file_path", "target": target})
            seen.add(target)

    return refs


def extract_markdown_code_blocks(content: str) -> list[dict[str, Any]]:
    """Extract fenced code block locations from Markdown content.

    Args:
        content: Raw markdown text

    Returns:
        List of dicts with 'lang', 'start_line', 'end_line' keys.
    """
    blocks: list[dict[str, Any]] = []
    lines = content.split("\n")
    in_block = False
    block_start = 0
    block_lang = ""

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not in_block and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_block = True
            block_start = i
            # Extract language tag after the fence
            fence_char = stripped[0]
            block_lang = (
                stripped.lstrip(fence_char).strip().split()[0]
                if stripped.lstrip(fence_char).strip()
                else ""
            )
        elif in_block and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_block = False
            blocks.append(
                {
                    "lang": block_lang,
                    "start_line": block_start,
                    "end_line": i,
                }
            )

    return blocks


def annotate_section_tokens(
    sections: list[OutlineDict], total_lines: int, avg_chars_per_line: int = 10
) -> None:
    """Add token estimates to each section node in-place.

    Estimates tokens using: ``(line_count * avg_chars_per_line) / 4``.
    This is a rough approximation — conservative by design.

    Adds ``own_tokens`` and ``subtree_tokens`` fields to every node.

    Args:
        sections: Section tree (mutated in-place)
        total_lines: Total lines in document (for edge-case clamping)
        avg_chars_per_line: Average characters per line (default 10, conservative)
    """
    chars_per_token = 4

    for section in sections:
        own_lines = max(0, section["own_end"] - section["own_start"] + 1)
        section["own_tokens"] = (own_lines * avg_chars_per_line) // chars_per_token

        # Recurse into children
        annotate_section_tokens(section.get("children", []), total_lines, avg_chars_per_line)

        # Compute subtree tokens
        child_tokens = sum(c.get("subtree_tokens", 0) for c in section.get("children", []))
        section["subtree_tokens"] = section["own_tokens"] + child_tokens


def _implicit_document_section(total_lines: int) -> OutlineDict:
    """Single section covering the whole document (zero-header fallback)."""
    return {
        "id": "document",
        "heading": "",
        "path": "(full document)",
        "level": 0,
        "line_start": 1,
        "line_end": total_lines,
        "own_start": 1,
        "own_end": total_lines,
        "is_leaf": True,
        "children": [],
    }


def extract_markdown_sections(file_path: Path) -> list[OutlineDict]:
    """Extract structured section tree from Markdown headers.

    Parses heading hierarchy into a nested tree suitable for recursive
    section-by-section processing. Each section node carries line ranges
    for both the full section (line_start/line_end) and the section's own
    content excluding children (own_start/own_end).

    Args:
        file_path: Path to Markdown file

    Returns:
        List of top-level section dicts with nested children.
        Falls back to one implicit section covering the whole file
        if no headers are found.
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return []

    all_lines = content.split("\n")
    total_lines = len(all_lines)

    if total_lines == 0:
        return []

    # Parse all headers with their line numbers and levels
    headers: list[dict[str, Any]] = []
    in_code_block = False
    for i, line in enumerate(all_lines, 1):
        stripped = line.strip()
        # Track fenced code blocks (``` or ~~~)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith("#"):
            level = 0
            for char in stripped:
                if char == "#":
                    level += 1
                else:
                    break
            level = min(level, 6)
            heading = stripped.lstrip("#").strip()
            if heading:  # Skip lines with only # characters
                headers.append({"level": level, "heading": heading, "line": i})

    # Zero-section fallback: document has no headers
    if not headers:
        return [_implicit_document_section(total_lines)]

    def _make_id(heading: str) -> str:
        """Generate a URL-safe ID from heading text."""
        # Lowercase, replace non-alphanumeric with hyphens, collapse
        slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
        return slug or "section"

    # Build nested tree, passing section boundaries from parent context
    def _build_tree(
        headers: list[dict[str, Any]],
        parent_path: str = "",
        section_end: int = total_lines,
    ) -> list[OutlineDict]:
        """Build nested section tree from flat header list.

        Args:
            headers: Flat list of header dicts with 'line', 'level', 'heading'
            parent_path: Path string for parent section breadcrumb
            section_end: Line number where this group of sections ends
                         (parent's end or total_lines for top-level)
        """
        if not headers:
            return []

        result: list[OutlineDict] = []
        i = 0

        while i < len(headers):
            h = headers[i]
            heading = h["heading"]
            level = h["level"]
            section_path = f"{parent_path} > {heading}" if parent_path else heading

            # Collect children: all consecutive headers at deeper levels
            children_headers: list[dict[str, Any]] = []
            j = i + 1
            while j < len(headers) and headers[j]["level"] > level:
                children_headers.append(headers[j])
                j += 1

            # Determine this section's line_end:
            # - If there's a next sibling, it ends before the next sibling's line
            # - Otherwise, it ends at the parent's section_end
            if j < len(headers):
                h_line_end = headers[j]["line"] - 1
            else:
                h_line_end = section_end

            # Recursively build children with this section's boundary
            children = _build_tree(children_headers, section_path, h_line_end)

            # own_start: the header line itself (includes the header)
            own_start = h["line"]
            # own_end: line before first child header (or section end if leaf)
            if children_headers:
                own_end = children_headers[0]["line"] - 1
            else:
                own_end = h_line_end

            # Ensure own_end >= own_start
            if own_end < own_start:
                own_end = own_start

            node = {
                "id": _make_id(heading),
                "heading": heading,
                "path": section_path,
                "level": level,
                "line_start": h["line"],
                "line_end": h_line_end,
                "own_start": own_start,
                "own_end": own_end,
                "is_leaf": len(children) == 0,
                "children": children,
            }
            result.append(node)
            i = j  # Skip past all children

        return result

    return _build_tree(headers)


def build_sections(file_path: Path, is_markdown: bool) -> list[OutlineDict]:
    """Build the section tree for a file.

    Markdown files get a header-derived tree; every other file gets the
    zero-section fallback (one implicit section over the whole document).

    Args:
        file_path: Path to file
        is_markdown: Whether the file is Markdown

    Returns:
        Section tree (empty list for empty/unreadable non-markdown files).
    """
    if is_markdown:
        return extract_markdown_sections(file_path)

    total_lines = 0
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        pass

    if total_lines > 0:
        return [_implicit_document_section(total_lines)]
    return []


def count_tree(nodes: list[OutlineDict], depth: int = 1) -> tuple[int, int]:
    """Return (max_depth, total_sections) for a section tree."""
    max_d = depth if nodes else 0
    total = len(nodes)
    for node in nodes:
        child_max_d, child_total = count_tree(node.get("children", []), depth + 1)
        max_d = max(max_d, child_max_d)
        total += child_total
    return max_d, total
