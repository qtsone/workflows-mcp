"""Tests for ReadFiles structured sections and line-range features (TASK-275)."""

from pathlib import Path

import pytest

from workflows_mcp.engine.file_outline import (
    annotate_section_tokens,
    extract_markdown_code_blocks,
    extract_markdown_frontmatter,
    extract_markdown_references,
    extract_markdown_sections,
    generate_file_outline_with_sections,
)

# ── extract_markdown_sections ──────────────────────────────────────────


class TestExtractMarkdownSections:
    """Tests for extract_markdown_sections()."""

    def test_basic_flat_headers(self, tmp_path: Path) -> None:
        """Headers at the same level produce a flat list."""
        doc = tmp_path / "flat.md"
        doc.write_text("# Intro\nSome text\n# Methods\nMore text\n# Conclusion\nDone\n")

        sections = extract_markdown_sections(doc)

        assert len(sections) == 3
        assert sections[0]["heading"] == "Intro"
        assert sections[0]["is_leaf"] is True
        assert sections[0]["children"] == []
        assert sections[1]["heading"] == "Methods"
        assert sections[2]["heading"] == "Conclusion"

    def test_nested_headers(self, tmp_path: Path) -> None:
        """Nested headers produce children arrays."""
        doc = tmp_path / "nested.md"
        doc.write_text(
            "# Architecture\n"
            "Overview\n"
            "## Auth\n"
            "Auth details\n"
            "### JWT Config\n"
            "JWT stuff\n"
            "### OAuth Setup\n"
            "OAuth stuff\n"
            "## Database\n"
            "DB details\n"
        )

        sections = extract_markdown_sections(doc)

        assert len(sections) == 1  # Only top-level
        arch = sections[0]
        assert arch["heading"] == "Architecture"
        assert arch["is_leaf"] is False
        assert len(arch["children"]) == 2  # Auth, Database

        auth = arch["children"][0]
        assert auth["heading"] == "Auth"
        assert auth["path"] == "Architecture > Auth"
        assert auth["is_leaf"] is False
        assert len(auth["children"]) == 2  # JWT Config, OAuth Setup

        jwt = auth["children"][0]
        assert jwt["heading"] == "JWT Config"
        assert jwt["path"] == "Architecture > Auth > JWT Config"
        assert jwt["is_leaf"] is True

        db = arch["children"][1]
        assert db["heading"] == "Database"
        assert db["is_leaf"] is True

    def test_zero_section_fallback(self, tmp_path: Path) -> None:
        """Document without headers returns one implicit section."""
        doc = tmp_path / "plain.txt"
        doc.write_text("Just some plain text\nNo headers here\nThird line\n")

        # Treat as markdown (no headers found = fallback)
        sections = extract_markdown_sections(doc)

        assert len(sections) == 1
        implicit = sections[0]
        assert implicit["id"] == "document"
        assert implicit["heading"] == ""
        assert implicit["path"] == "(full document)"
        assert implicit["level"] == 0
        assert implicit["line_start"] == 1
        assert implicit["is_leaf"] is True
        assert implicit["children"] == []

    def test_line_ranges_correct(self, tmp_path: Path) -> None:
        """Verify line_start/line_end and own_start/own_end are correct."""
        doc = tmp_path / "ranges.md"
        doc.write_text(
            "# Section A\n"  # line 1
            "Line 2\n"  # line 2
            "Line 3\n"  # line 3
            "## Sub A1\n"  # line 4
            "Line 5\n"  # line 5
            "Line 6\n"  # line 6
            "# Section B\n"  # line 7
            "Line 8\n"  # line 8
        )

        sections = extract_markdown_sections(doc)

        assert len(sections) == 2
        sec_a = sections[0]
        assert sec_a["line_start"] == 1
        assert sec_a["line_end"] == 6  # Up to line before Section B
        assert sec_a["own_start"] == 1
        assert sec_a["own_end"] == 3  # Between header and first child

        sub_a1 = sec_a["children"][0]
        assert sub_a1["line_start"] == 4
        assert sub_a1["line_end"] == 6

        sec_b = sections[1]
        assert sec_b["line_start"] == 7
        # Trailing \n makes total_lines=9, so last section extends to line 9
        assert sec_b["line_end"] == 9

    def test_section_ids_are_slugified(self, tmp_path: Path) -> None:
        """Section IDs should be URL-safe slugs."""
        doc = tmp_path / "ids.md"
        doc.write_text("# Hello World!\n## JWT Config (v2)\n")

        sections = extract_markdown_sections(doc)

        assert sections[0]["id"] == "hello-world"
        assert sections[0]["children"][0]["id"] == "jwt-config-v2"

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file gets implicit section (split on newline gives [''] = 1 line)."""
        doc = tmp_path / "empty.md"
        doc.write_text("")

        sections = extract_markdown_sections(doc)
        # Empty string split produces [""] = 1 line, so fallback fires
        assert len(sections) == 1
        assert sections[0]["id"] == "document"

    def test_code_blocks_ignored(self, tmp_path: Path) -> None:
        """Headings inside fenced code blocks are not treated as sections."""
        doc = tmp_path / "codeblock.md"
        doc.write_text(
            "# Real Heading\n"
            "Some text\n"
            "```python\n"
            "# This is a Python comment, not a heading\n"
            "x = 1\n"
            "```\n"
            "## Another Real Heading\n"
            "More text\n"
        )

        sections = extract_markdown_sections(doc)

        assert len(sections) == 1
        assert sections[0]["heading"] == "Real Heading"
        assert len(sections[0]["children"]) == 1
        child = sections[0]["children"][0]
        assert child["heading"] == "Another Real Heading"
        assert child["is_leaf"] is True

    def test_tilde_code_blocks_ignored(self, tmp_path: Path) -> None:
        """Headings inside ~~~ fenced code blocks are not treated as sections."""
        doc = tmp_path / "tilde.md"
        doc.write_text("# Title\n~~~\n# Not a heading\n~~~\n## Subtitle\n")

        sections = extract_markdown_sections(doc)

        assert len(sections) == 1
        assert sections[0]["heading"] == "Title"
        assert len(sections[0]["children"]) == 1
        assert sections[0]["children"][0]["heading"] == "Subtitle"


# ── generate_file_outline_with_sections ────────────────────────────────


class TestGenerateFileOutlineWithSections:
    """Tests for generate_file_outline_with_sections()."""

    def test_markdown_returns_sections(self, tmp_path: Path) -> None:
        """Markdown file returns both outline string and sections tree."""
        doc = tmp_path / "test.md"
        doc.write_text("# Title\nContent\n## Subtitle\nMore\n")

        outline, sections, max_depth, total = generate_file_outline_with_sections(doc, "outline")

        assert "Title" in outline
        assert len(sections) == 1
        assert sections[0]["heading"] == "Title"
        assert len(sections[0]["children"]) == 1
        assert max_depth == 2
        assert total == 2

    def test_non_markdown_fallback(self, tmp_path: Path) -> None:
        """Non-markdown file returns implicit section."""
        doc = tmp_path / "test.py"
        doc.write_text("def foo():\n    pass\n")

        outline, sections, max_depth, total = generate_file_outline_with_sections(doc, "outline")

        assert len(sections) == 1
        assert sections[0]["id"] == "document"
        assert sections[0]["is_leaf"] is True
        assert max_depth == 1
        assert total == 1


# ── ReadFiles executor line-range ──────────────────────────────────────


class TestReadFilesLineRange:
    """Tests for ReadFiles line_start/line_end feature."""

    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> Path:
        """Create a sample file with numbered lines."""
        doc = tmp_path / "sample.txt"
        doc.write_text("\n".join(f"Line {i}" for i in range(1, 11)))
        return doc

    @pytest.mark.asyncio
    async def test_line_range_basic(self, sample_file: Path) -> None:
        """Read lines 3-5 from a file."""
        from workflows_mcp.engine.execution import Execution
        from workflows_mcp.engine.executors_file import ReadFilesExecutor, ReadFilesInput

        executor = ReadFilesExecutor()
        inputs = ReadFilesInput(path=str(sample_file), line_start=3, line_end=5)
        context = Execution()

        result = await executor.execute(inputs, context)

        assert result.total_files == 1
        content = result.content
        assert "Line 3" in content
        assert "Line 4" in content
        assert "Line 5" in content
        assert "Line 2" not in content
        assert "Line 6" not in content

    @pytest.mark.asyncio
    async def test_line_range_from_start(self, sample_file: Path) -> None:
        """Read from start to line 3."""
        from workflows_mcp.engine.execution import Execution
        from workflows_mcp.engine.executors_file import ReadFilesExecutor, ReadFilesInput

        executor = ReadFilesExecutor()
        inputs = ReadFilesInput(path=str(sample_file), line_end=3)
        context = Execution()

        result = await executor.execute(inputs, context)

        content = result.content
        assert "Line 1" in content
        assert "Line 3" in content
        assert "Line 4" not in content

    @pytest.mark.asyncio
    async def test_no_line_range_reads_full(self, sample_file: Path) -> None:
        """Without line_start/line_end, reads the full file."""
        from workflows_mcp.engine.execution import Execution
        from workflows_mcp.engine.executors_file import ReadFilesExecutor, ReadFilesInput

        executor = ReadFilesExecutor()
        inputs = ReadFilesInput(path=str(sample_file))
        context = Execution()

        result = await executor.execute(inputs, context)

        content = result.content
        assert "Line 1" in content
        assert "Line 10" in content

    @pytest.mark.asyncio
    async def test_outline_mode_returns_sections(self, tmp_path: Path) -> None:
        """Outline mode populates the sections field."""
        from workflows_mcp.engine.execution import Execution
        from workflows_mcp.engine.executors_file import ReadFilesExecutor, ReadFilesInput

        doc = tmp_path / "structured.md"
        doc.write_text("# Main\nIntro\n## Sub\nDetails\n")

        executor = ReadFilesExecutor()
        inputs = ReadFilesInput(path=str(doc), mode="outline")
        context = Execution()

        result = await executor.execute(inputs, context)

        assert result.total_sections == 2
        assert result.max_depth == 2
        assert len(result.sections) == 1
        assert result.sections[0]["heading"] == "Main"


# ── extract_markdown_frontmatter ───────────────────────────────────────


class TestExtractMarkdownFrontmatter:
    """Tests for extract_markdown_frontmatter()."""

    def test_basic_frontmatter(self) -> None:
        """Parse standard YAML frontmatter."""
        content = "---\nid: TASK-276\nstatus: backlog\ntags:\n  - document\n  - knowledge\n---\n# Title\nContent\n"
        result = extract_markdown_frontmatter(content)

        assert result is not None
        assert result["id"] == "TASK-276"
        assert result["status"] == "backlog"
        assert result["tags"] == ["document", "knowledge"]

    def test_no_frontmatter(self) -> None:
        """No frontmatter returns None."""
        assert extract_markdown_frontmatter("# Title\nContent\n") is None

    def test_empty_frontmatter(self) -> None:
        """Empty frontmatter block returns None."""
        assert extract_markdown_frontmatter("---\n---\n# Title\n") is None

    def test_invalid_yaml(self) -> None:
        """Invalid YAML returns None (no crash)."""
        assert extract_markdown_frontmatter("---\n[invalid: yaml:\n---\nContent\n") is None


# ── extract_markdown_references ────────────────────────────────────────


class TestExtractMarkdownReferences:
    """Tests for extract_markdown_references()."""

    def test_wikilinks_extracted(self) -> None:
        """Wikilinks are found."""
        content = "See [[document/CognitiveKnowledge]] and [[document/KnowledgeGraph]].\n"
        refs = extract_markdown_references(content)

        wikilinks = [r for r in refs if r["type"] == "wikilink"]
        assert len(wikilinks) == 2
        assert wikilinks[0]["target"] == "document/CognitiveKnowledge"

    def test_wikilinks_with_label(self) -> None:
        """Wikilinks with display labels extract the target."""
        content = "[[target|Display Label]]\n"
        refs = extract_markdown_references(content)
        assert len(refs) == 1
        assert refs[0]["target"] == "target"

    def test_file_paths_in_backticks(self) -> None:
        """File paths in backtick code spans are found."""
        content = "Edit `app/models/knowledge.py` and `app/executors/admin.py`\n"
        refs = extract_markdown_references(content)

        file_paths = [r for r in refs if r["type"] == "file_path"]
        assert len(file_paths) == 2
        assert file_paths[0]["target"] == "app/models/knowledge.py"

    def test_no_duplicates(self) -> None:
        """Same reference appearing twice is deduplicated."""
        content = "[[target]] and again [[target]]\n"
        refs = extract_markdown_references(content)
        assert len(refs) == 1


# ── extract_markdown_code_blocks ───────────────────────────────────────


class TestExtractMarkdownCodeBlocks:
    """Tests for extract_markdown_code_blocks()."""

    def test_basic_code_block(self) -> None:
        """Detect fenced code block with language."""
        content = "Text\n```python\nx = 1\n```\nMore text\n"
        blocks = extract_markdown_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0]["lang"] == "python"
        assert blocks[0]["start_line"] == 2
        assert blocks[0]["end_line"] == 4

    def test_tilde_code_block(self) -> None:
        """Detect ~~~ fenced code blocks."""
        content = "~~~yaml\nkey: value\n~~~\n"
        blocks = extract_markdown_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0]["lang"] == "yaml"

    def test_no_language_tag(self) -> None:
        """Code block without language tag has empty lang."""
        content = "```\nplain code\n```\n"
        blocks = extract_markdown_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0]["lang"] == ""


# ── annotate_section_tokens ────────────────────────────────────────────


class TestAnnotateSectionTokens:
    """Tests for annotate_section_tokens()."""

    def test_leaf_section(self, tmp_path: Path) -> None:
        """Leaf sections get own_tokens and subtree_tokens."""
        doc = tmp_path / "simple.md"
        doc.write_text("# Title\nLine 1\nLine 2\nLine 3\n")

        sections = extract_markdown_sections(doc)
        annotate_section_tokens(sections, total_lines=5)

        assert "own_tokens" in sections[0]
        assert "subtree_tokens" in sections[0]
        assert sections[0]["own_tokens"] == sections[0]["subtree_tokens"]

    def test_parent_child_tokens(self, tmp_path: Path) -> None:
        """Parent subtree_tokens includes children."""
        doc = tmp_path / "nested.md"
        doc.write_text("# Parent\nIntro\n## Child\nChild content\n")

        sections = extract_markdown_sections(doc)
        annotate_section_tokens(sections, total_lines=5)

        parent = sections[0]
        child = parent["children"][0]
        assert parent["subtree_tokens"] == parent["own_tokens"] + child["subtree_tokens"]


# ── ReadFiles enhanced outputs (integration) ──────────────────────────


class TestReadFilesEnhancedOutputs:
    """Integration tests for ReadFiles enhanced outputs."""

    @pytest.mark.asyncio
    async def test_outline_mode_returns_enhanced_fields(self, tmp_path: Path) -> None:
        """ReadFiles in outline mode returns frontmatter, references, code_blocks, tokens."""
        from workflows_mcp.engine.execution import Execution
        from workflows_mcp.engine.executors_file import ReadFilesExecutor, ReadFilesInput

        doc = tmp_path / "task.md"
        doc.write_text(
            "---\nid: TASK-001\nstatus: done\n---\n"
            "# Context\n"
            "See [[document/CognitiveKnowledge]] for details.\n"
            "Edit `app/models/knowledge.py` for the implementation.\n"
            "```python\n# comment\nx = 1\n```\n"
            "## Problem\n"
            "Details here.\n"
        )

        executor = ReadFilesExecutor()
        inputs = ReadFilesInput(path=str(doc), mode="outline")
        context = Execution()

        result = await executor.execute(inputs, context)

        # Frontmatter
        assert result.frontmatter is not None
        assert result.frontmatter["id"] == "TASK-001"

        # References
        assert len(result.references) >= 1
        wikilinks = [r for r in result.references if r["type"] == "wikilink"]
        assert any(r["target"] == "document/CognitiveKnowledge" for r in wikilinks)

        # Code blocks
        assert len(result.code_blocks) >= 1
        assert result.code_blocks[0]["lang"] == "python"

        # Token annotations on sections
        assert "own_tokens" in result.sections[0]
        assert "subtree_tokens" in result.sections[0]
