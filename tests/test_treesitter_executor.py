"""Tests for TreeSitter block executor - TDD (tests written before production code)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_PYTHON_SRC = """\
import os

class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}"

def main():
    g = Greeter()
    print(g.greet("world"))
"""


@pytest.fixture()
def python_file(tmp_path: Path) -> Path:
    f = tmp_path / "greeter.py"
    f.write_text(SIMPLE_PYTHON_SRC, encoding="utf-8")
    return f


@pytest.fixture()
def mock_execution() -> MagicMock:
    ctx = MagicMock()
    ctx.workflow_name = "test"
    return ctx


# ---------------------------------------------------------------------------
# detect_language tests
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_python_extensions(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.py") == "python"
        assert detect_language("foo.pyi") == "python"

    def test_typescript_extension(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.ts") == "typescript"

    def test_tsx_extension(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.tsx") == "tsx"

    def test_javascript_extensions(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.js") == "javascript"
        assert detect_language("foo.jsx") == "javascript"
        assert detect_language("foo.mjs") == "javascript"
        assert detect_language("foo.cjs") == "javascript"

    def test_go_extension(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.go") == "go"

    def test_rust_extension(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.rs") == "rust"

    def test_markdown_extensions(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.md") == "markdown"
        assert detect_language("foo.markdown") == "markdown"

    def test_yaml_extensions(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.yaml") == "yaml"
        assert detect_language("foo.yml") == "yaml"

    def test_json_extension(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.json") == "json"

    def test_unsupported_returns_unsupported(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.rb") == "unsupported"
        assert detect_language("Makefile") == "unsupported"
        assert detect_language("foo") == "unsupported"

    def test_case_insensitive_extension(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("foo.PY") == "python"
        assert detect_language("foo.JS") == "javascript"

    def test_path_with_directories(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import detect_language

        assert detect_language("/some/deep/path/module.py") == "python"
        assert detect_language("relative/path/index.ts") == "typescript"


# ---------------------------------------------------------------------------
# Parser loading tests
# ---------------------------------------------------------------------------


class TestParserLoading:
    def test_get_parser_returns_parser_for_python(self) -> None:
        from tree_sitter import Parser

        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        parser = get_parser("python")
        assert isinstance(parser, Parser)

    def test_get_parser_returns_parser_for_typescript(self) -> None:
        from tree_sitter import Parser

        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        parser = get_parser("typescript")
        assert isinstance(parser, Parser)

    def test_get_parser_returns_parser_for_tsx(self) -> None:
        from tree_sitter import Parser

        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        parser = get_parser("tsx")
        assert isinstance(parser, Parser)

    def test_get_parser_returns_parser_for_javascript(self) -> None:
        from tree_sitter import Parser

        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        parser = get_parser("javascript")
        assert isinstance(parser, Parser)

    def test_get_parser_returns_parser_for_go(self) -> None:
        from tree_sitter import Parser

        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        parser = get_parser("go")
        assert isinstance(parser, Parser)

    def test_get_parser_returns_parser_for_rust(self) -> None:
        from tree_sitter import Parser

        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        parser = get_parser("rust")
        assert isinstance(parser, Parser)

    def test_get_parser_returns_parser_for_markdown(self) -> None:
        from tree_sitter import Parser

        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        parser = get_parser("markdown")
        assert isinstance(parser, Parser)

    def test_get_parser_returns_parser_for_yaml(self) -> None:
        from tree_sitter import Parser

        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        parser = get_parser("yaml")
        assert isinstance(parser, Parser)

    def test_get_parser_returns_parser_for_json(self) -> None:
        from tree_sitter import Parser

        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        parser = get_parser("json")
        assert isinstance(parser, Parser)

    def test_get_parser_is_cached(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        p1 = get_parser("python")
        p2 = get_parser("python")
        assert p1 is p2

    def test_get_parser_raises_for_unsupported(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import get_parser

        with pytest.raises((ValueError, KeyError)):
            get_parser("unsupported")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# content_hash tests
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_content_hash_matches_tools_memory_implementation(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import content_hash

        text = "hello world\n"
        expected = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        assert content_hash(text) == expected

    def test_content_hash_empty_string(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import content_hash

        expected = hashlib.sha256(b"").hexdigest()
        assert content_hash("") == expected

    def test_content_hash_unicode(self) -> None:
        from workflows_mcp.code_intelligence.treesitter_languages import content_hash

        text = "caf\u00e9 \u4e2d\u6587"
        expected = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        assert content_hash(text) == expected


# ---------------------------------------------------------------------------
# TreeSitterInput model tests
# ---------------------------------------------------------------------------


class TestTreeSitterInput:
    def test_requires_path(self) -> None:
        from pydantic import ValidationError

        from workflows_mcp.code_intelligence.executors_treesitter import TreeSitterInput

        with pytest.raises(ValidationError):
            TreeSitterInput()  # type: ignore[call-arg]

    def test_minimal_valid_input(self) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import TreeSitterInput

        inp = TreeSitterInput(path="/some/file.py")
        assert inp.path == "/some/file.py"
        assert inp.language is None
        assert inp.repo_relative_path is None
        assert inp.palace is None
        assert inp.item_id is None

    def test_full_input(self) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import TreeSitterInput

        inp = TreeSitterInput(
            path="/repo/src/main.py",
            language="python",
            repo_relative_path="src/main.py",
            palace="mypalace",
            item_id="abc123",
        )
        assert inp.palace == "mypalace"
        assert inp.item_id == "abc123"

    def test_rejects_extra_fields(self) -> None:
        from pydantic import ValidationError

        from workflows_mcp.code_intelligence.executors_treesitter import TreeSitterInput

        with pytest.raises(ValidationError):
            TreeSitterInput(path="/file.py", unknown_field="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TreeSitterOutput model tests
# ---------------------------------------------------------------------------


class TestTreeSitterOutput:
    def test_output_has_required_fields(self) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import TreeSitterOutput

        out = TreeSitterOutput(
            language="python",
            content_hash="abc123",
            size_bytes=42,
            mtime_ns=1234567890,
            module_qualified_name="greeter",
            entities=[],
            relations=[],
            unresolved_imports=[],
        )
        assert out.language == "python"
        assert out.content_hash == "abc123"
        assert out.size_bytes == 42
        assert out.mtime_ns == 1234567890
        assert out.module_qualified_name == "greeter"
        assert out.entities == []
        assert out.relations == []
        assert out.unresolved_imports == []


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_treesitter_absent_from_dag_core_default_registry(self) -> None:
        """The DAG core no longer hard-wires the code-intelligence executor."""
        from workflows_mcp.engine.executor_base import create_default_registry

        registry = create_default_registry()
        assert not registry.has("TreeSitter")

    def test_treesitter_executor_registered_via_code_intelligence_seam(self) -> None:
        from workflows_mcp.code_intelligence import register_code_intelligence_executors
        from workflows_mcp.engine.executor_base import create_default_registry

        registry = create_default_registry()
        register_code_intelligence_executors(registry)
        assert registry.has("TreeSitter")

    def test_treesitter_executor_type_name(self) -> None:
        from workflows_mcp.code_intelligence import register_code_intelligence_executors
        from workflows_mcp.engine.executor_base import create_default_registry

        registry = create_default_registry()
        register_code_intelligence_executors(registry)
        executor = registry.get("TreeSitter")
        assert executor.type_name == "TreeSitter"


# ---------------------------------------------------------------------------
# Executor execution tests - Python file
# ---------------------------------------------------------------------------


class TestTreeSitterExecutorPythonFile:
    @pytest.mark.asyncio
    async def test_execute_python_file_returns_valid_output(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
            TreeSitterOutput,
        )

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(python_file))
        result = await executor.execute(inputs, mock_execution)

        assert isinstance(result, TreeSitterOutput)
        assert result.language == "python"

    @pytest.mark.asyncio
    async def test_execute_python_file_content_hash_matches(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )
        from workflows_mcp.code_intelligence.treesitter_languages import content_hash

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(python_file))
        result = await executor.execute(inputs, mock_execution)

        expected_hash = content_hash(python_file.read_text(encoding="utf-8"))
        assert result.content_hash == expected_hash

    @pytest.mark.asyncio
    async def test_execute_python_file_has_file_entity(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(python_file))
        result = await executor.execute(inputs, mock_execution)

        file_entities = [e for e in result.entities if e["entity_type"] == "File"]
        assert len(file_entities) == 1
        fe = file_entities[0]
        assert "qualified_name" in fe
        assert "stable_id" in fe
        assert "entity_type" in fe
        assert "metadata" in fe
        assert "confidence" in fe
        assert fe["entity_type"] == "File"

    @pytest.mark.asyncio
    async def test_execute_python_file_has_module_entity(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(python_file))
        result = await executor.execute(inputs, mock_execution)

        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert len(module_entities) == 1
        me = module_entities[0]
        assert "qualified_name" in me
        assert "stable_id" in me
        assert "entity_type" in me
        assert "metadata" in me
        assert "confidence" in me
        assert me["entity_type"] == "Module"

    @pytest.mark.asyncio
    async def test_execute_python_file_module_has_qualified_name(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(python_file))
        result = await executor.execute(inputs, mock_execution)

        assert result.module_qualified_name != ""
        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert module_entities[0]["qualified_name"] == result.module_qualified_name

    @pytest.mark.asyncio
    async def test_execute_python_file_has_contains_relation(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(python_file))
        result = await executor.execute(inputs, mock_execution)

        contains_rels = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        assert len(contains_rels) >= 1
        rel = contains_rels[0]
        assert "source_qname" in rel
        assert "target_qname" in rel
        assert "relation_type" in rel

        file_entities = [e for e in result.entities if e["entity_type"] == "File"]
        assert rel["source_qname"] == file_entities[0]["qualified_name"]

    @pytest.mark.asyncio
    async def test_execute_stable_id_is_deterministic(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(python_file), palace="palace1", item_id="item1")
        result1 = await executor.execute(inputs, mock_execution)
        result2 = await executor.execute(inputs, mock_execution)

        stable_ids_1 = {e["stable_id"] for e in result1.entities}
        stable_ids_2 = {e["stable_id"] for e in result2.entities}
        assert stable_ids_1 == stable_ids_2

    @pytest.mark.asyncio
    async def test_execute_stable_id_length_32(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(python_file))
        result = await executor.execute(inputs, mock_execution)

        for entity in result.entities:
            assert len(entity["stable_id"]) == 32


# ---------------------------------------------------------------------------
# Unsupported extension tests
# ---------------------------------------------------------------------------


class TestUnsupportedExtension:
    @pytest.mark.asyncio
    async def test_unsupported_extension_returns_unsupported_language(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "script.rb"
        f.write_text("puts 'hello'", encoding="utf-8")

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(f))
        result = await executor.execute(inputs, mock_execution)

        assert result.language == "unsupported"
        assert result.entities == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_unsupported_does_not_crash(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "Makefile"
        f.write_text("all:\n\techo hello", encoding="utf-8")

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(f))
        # Must not raise
        result = await executor.execute(inputs, mock_execution)
        assert result.language == "unsupported"

    @pytest.mark.asyncio
    async def test_language_override_in_input(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """When language is explicitly provided in input, use it instead of extension."""
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
            TreeSitterOutput,
        )

        # A file with .txt extension but explicitly declared as python
        f = tmp_path / "script.txt"
        f.write_text("x = 1\n", encoding="utf-8")

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(f), language="python")
        result = await executor.execute(inputs, mock_execution)

        assert isinstance(result, TreeSitterOutput)
        assert result.language == "python"


# ---------------------------------------------------------------------------
# ADR-006 exception propagation and stable_id sensitivity tests
# ---------------------------------------------------------------------------


class TestADR006ExceptionPropagation:
    @pytest.mark.asyncio
    async def test_missing_file_propagates_file_not_found_error(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """ADR-006: executor raises exceptions for failures; missing file must propagate."""
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(tmp_path / "does_not_exist.py"))

        with pytest.raises(FileNotFoundError):
            await executor.execute(inputs, mock_execution)


class TestStableIdSensitivity:
    @pytest.mark.asyncio
    async def test_stable_id_changes_when_palace_changes(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        """stable_id must differ when palace differs (same file, same item_id)."""
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result_a = await executor.execute(
            TreeSitterInput(path=str(python_file), palace="palace_a", item_id="item1"),
            mock_execution,
        )
        result_b = await executor.execute(
            TreeSitterInput(path=str(python_file), palace="palace_b", item_id="item1"),
            mock_execution,
        )

        ids_a = {e["stable_id"] for e in result_a.entities}
        ids_b = {e["stable_id"] for e in result_b.entities}
        assert ids_a != ids_b

    @pytest.mark.asyncio
    async def test_stable_id_changes_when_item_id_changes(
        self, python_file: Path, mock_execution: MagicMock
    ) -> None:
        """stable_id must differ when item_id differs (same file, same palace)."""
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result_a = await executor.execute(
            TreeSitterInput(path=str(python_file), palace="palace1", item_id="item_x"),
            mock_execution,
        )
        result_b = await executor.execute(
            TreeSitterInput(path=str(python_file), palace="palace1", item_id="item_y"),
            mock_execution,
        )

        ids_a = {e["stable_id"] for e in result_a.entities}
        ids_b = {e["stable_id"] for e in result_b.entities}
        assert ids_a != ids_b
