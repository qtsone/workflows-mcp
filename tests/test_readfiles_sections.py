"""Tests for ReadFiles structured sections and line-range features (TASK-275)."""

from pathlib import Path

import pytest

from workflows_mcp.engine.file_outline import (
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
