"""Tests for document-language extractor (Markdown, YAML, JSON) - TDD.

Document languages are file-only: they emit exactly one File entity and no
Module/Class/Function/Method entities. Relations and unresolved_imports are
also empty. `module_qualified_name` is non-empty (repo-relative path) for
output model compatibility but no Module entity is emitted.

Tests are written BEFORE the production code (strict TDD).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_MARKDOWN = """\
# Hello World

This is a simple markdown document.

## Section

Some content here.
"""

SIMPLE_YAML = """\
name: my-service
version: "1.0.0"
config:
  debug: true
  port: 8080
"""

SIMPLE_JSON = """\
{
  "name": "my-package",
  "version": "1.0.0",
  "dependencies": {
    "requests": "^2.28"
  }
}
"""


@pytest.fixture()
def md_file(tmp_path: Path) -> Path:
    f = tmp_path / "readme.md"
    f.write_text(SIMPLE_MARKDOWN, encoding="utf-8")
    return f


@pytest.fixture()
def markdown_ext_file(tmp_path: Path) -> Path:
    f = tmp_path / "notes.markdown"
    f.write_text(SIMPLE_MARKDOWN, encoding="utf-8")
    return f


@pytest.fixture()
def yaml_file(tmp_path: Path) -> Path:
    f = tmp_path / "config.yaml"
    f.write_text(SIMPLE_YAML, encoding="utf-8")
    return f


@pytest.fixture()
def yml_file(tmp_path: Path) -> Path:
    f = tmp_path / "values.yml"
    f.write_text(SIMPLE_YAML, encoding="utf-8")
    return f


@pytest.fixture()
def json_file(tmp_path: Path) -> Path:
    f = tmp_path / "package.json"
    f.write_text(SIMPLE_JSON, encoding="utf-8")
    return f


@pytest.fixture()
def mock_execution() -> MagicMock:
    ctx = MagicMock()
    ctx.workflow_name = "test"
    return ctx


# ---------------------------------------------------------------------------
# Unit tests: extract_document function
# ---------------------------------------------------------------------------


class TestExtractDocumentFunction:
    """Tests for the extract_document() function in the document extractor module."""

    def test_extract_document_is_importable(self) -> None:
        """extract_document must be importable from the extractors package."""
        from workflows_mcp.engine.treesitter_extractors import extract_document  # noqa: F401

    def test_extract_document_returns_tuple_of_entities_and_relations(
        self, tmp_path: Path
    ) -> None:
        from workflows_mcp.engine.treesitter_extractors import extract_document

        f = tmp_path / "readme.md"
        f.write_text(SIMPLE_MARKDOWN, encoding="utf-8")

        file_entity: dict = {
            "qualified_name": str(f),
            "stable_id": "abc123",
            "entity_type": "File",
            "name": "readme.md",
            "metadata": {"path": str(f), "language": "markdown"},
            "confidence": 1.0,
        }

        entities, relations = extract_document(
            file_entity=file_entity,
        )
        assert isinstance(entities, list)
        assert isinstance(relations, list)

    def test_extract_document_returns_exactly_one_file_entity(
        self, tmp_path: Path
    ) -> None:
        from workflows_mcp.engine.treesitter_extractors import extract_document

        f = tmp_path / "doc.md"
        f.write_text("# title", encoding="utf-8")

        file_entity: dict = {
            "qualified_name": str(f),
            "stable_id": "sid1",
            "entity_type": "File",
            "name": "doc.md",
            "metadata": {"path": str(f), "language": "markdown"},
            "confidence": 1.0,
        }

        entities, relations = extract_document(file_entity=file_entity)

        assert len(entities) == 1
        assert entities[0]["entity_type"] == "File"

    def test_extract_document_no_module_entity(self, tmp_path: Path) -> None:
        from workflows_mcp.engine.treesitter_extractors import extract_document

        f = tmp_path / "config.yaml"
        f.write_text(SIMPLE_YAML, encoding="utf-8")

        file_entity: dict = {
            "qualified_name": str(f),
            "stable_id": "sid2",
            "entity_type": "File",
            "name": "config.yaml",
            "metadata": {"path": str(f), "language": "yaml"},
            "confidence": 1.0,
        }

        entities, _ = extract_document(file_entity=file_entity)

        module_entities = [e for e in entities if e["entity_type"] == "Module"]
        assert module_entities == [], "Document languages must not emit Module entities"

    def test_extract_document_no_class_function_method_entities(
        self, tmp_path: Path
    ) -> None:
        from workflows_mcp.engine.treesitter_extractors import extract_document

        f = tmp_path / "data.json"
        f.write_text(SIMPLE_JSON, encoding="utf-8")

        file_entity: dict = {
            "qualified_name": str(f),
            "stable_id": "sid3",
            "entity_type": "File",
            "name": "data.json",
            "metadata": {"path": str(f), "language": "json"},
            "confidence": 1.0,
        }

        entities, _ = extract_document(file_entity=file_entity)

        symbol_types = {"Class", "Function", "Method"}
        symbol_entities = [e for e in entities if e["entity_type"] in symbol_types]
        assert symbol_entities == []

    def test_extract_document_no_relations(self, tmp_path: Path) -> None:
        from workflows_mcp.engine.treesitter_extractors import extract_document

        f = tmp_path / "readme.md"
        f.write_text(SIMPLE_MARKDOWN, encoding="utf-8")

        file_entity: dict = {
            "qualified_name": str(f),
            "stable_id": "sid4",
            "entity_type": "File",
            "name": "readme.md",
            "metadata": {"path": str(f), "language": "markdown"},
            "confidence": 1.0,
        }

        _, relations = extract_document(file_entity=file_entity)
        assert relations == [], "Document languages must not emit any relations"

    def test_extract_document_file_entity_has_required_fields(
        self, tmp_path: Path
    ) -> None:
        from workflows_mcp.engine.treesitter_extractors import extract_document

        f = tmp_path / "readme.md"
        f.write_text(SIMPLE_MARKDOWN, encoding="utf-8")

        file_entity: dict = {
            "qualified_name": str(f),
            "stable_id": "sid5",
            "entity_type": "File",
            "name": "readme.md",
            "metadata": {"path": str(f), "language": "markdown"},
            "confidence": 1.0,
        }

        entities, _ = extract_document(file_entity=file_entity)
        fe = entities[0]

        assert "entity_type" in fe
        assert "name" in fe
        assert "qualified_name" in fe
        assert "stable_id" in fe
        assert "metadata" in fe
        assert "confidence" in fe
        assert fe["entity_type"] == "File"


# ---------------------------------------------------------------------------
# Integration tests: full executor dispatch for document languages
# ---------------------------------------------------------------------------


class TestMarkdownExtraction:
    """Markdown (.md, .markdown) via TreeSitter executor must produce file-only output."""

    @pytest.mark.asyncio
    async def test_md_executor_returns_markdown_language(
        self, md_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result = await executor.execute(
            TreeSitterInput(path=str(md_file)), mock_execution
        )
        assert result.language == "markdown"

    @pytest.mark.asyncio
    async def test_md_executor_exactly_one_file_entity(
        self, md_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result = await executor.execute(
            TreeSitterInput(path=str(md_file)), mock_execution
        )

        file_entities = [e for e in result.entities if e["entity_type"] == "File"]
        assert len(file_entities) == 1

    @pytest.mark.asyncio
    async def test_md_executor_no_module_entity(
        self, md_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result = await executor.execute(
            TreeSitterInput(path=str(md_file)), mock_execution
        )

        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert module_entities == [], "Markdown must not emit Module entity"

    @pytest.mark.asyncio
    async def test_md_executor_no_relations(
        self, md_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result = await executor.execute(
            TreeSitterInput(path=str(md_file)), mock_execution
        )

        assert result.relations == [], "Markdown must not emit relations"

    @pytest.mark.asyncio
    async def test_md_executor_no_unresolved_imports(
        self, md_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result = await executor.execute(
            TreeSitterInput(path=str(md_file)), mock_execution
        )

        assert result.unresolved_imports == []

    @pytest.mark.asyncio
    async def test_md_executor_module_qualified_name_is_nonempty(
        self, md_file: Path, mock_execution: MagicMock
    ) -> None:
        """module_qualified_name must be non-empty for output model compatibility.

        No Module entity is emitted, but the field stores the repo-relative
        qualified name for traceability.
        """
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result = await executor.execute(
            TreeSitterInput(path=str(md_file)), mock_execution
        )

        assert result.module_qualified_name != ""

    @pytest.mark.asyncio
    async def test_md_executor_file_entity_has_required_fields(
        self, md_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result = await executor.execute(
            TreeSitterInput(path=str(md_file)), mock_execution
        )

        fe = next(e for e in result.entities if e["entity_type"] == "File")
        assert "entity_type" in fe
        assert "name" in fe
        assert "qualified_name" in fe
        assert "stable_id" in fe
        assert "metadata" in fe
        assert "confidence" in fe

    @pytest.mark.asyncio
    async def test_markdown_ext_is_supported(
        self, markdown_ext_file: Path, mock_execution: MagicMock
    ) -> None:
        """.markdown extension must also work."""
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        result = await executor.execute(
            TreeSitterInput(path=str(markdown_ext_file)), mock_execution
        )
        assert result.language == "markdown"
        assert len([e for e in result.entities if e["entity_type"] == "File"]) == 1

    @pytest.mark.asyncio
    async def test_md_stable_id_is_deterministic(
        self, md_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(md_file), palace="p1", item_id="i1")
        result1 = await executor.execute(inputs, mock_execution)
        result2 = await executor.execute(inputs, mock_execution)

        ids1 = {e["stable_id"] for e in result1.entities}
        ids2 = {e["stable_id"] for e in result2.entities}
        assert ids1 == ids2


class TestYamlExtraction:
    """YAML (.yaml, .yml) via TreeSitter executor must produce file-only output."""

    @pytest.mark.asyncio
    async def test_yaml_executor_returns_yaml_language(
        self, yaml_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(yaml_file)), mock_execution
        )
        assert result.language == "yaml"

    @pytest.mark.asyncio
    async def test_yaml_executor_exactly_one_file_entity(
        self, yaml_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(yaml_file)), mock_execution
        )
        file_entities = [e for e in result.entities if e["entity_type"] == "File"]
        assert len(file_entities) == 1

    @pytest.mark.asyncio
    async def test_yaml_executor_no_module_entity(
        self, yaml_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(yaml_file)), mock_execution
        )
        assert [e for e in result.entities if e["entity_type"] == "Module"] == []

    @pytest.mark.asyncio
    async def test_yaml_executor_no_relations(
        self, yaml_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(yaml_file)), mock_execution
        )
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_yml_extension_is_supported(
        self, yml_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(yml_file)), mock_execution
        )
        assert result.language == "yaml"
        assert len([e for e in result.entities if e["entity_type"] == "File"]) == 1

    @pytest.mark.asyncio
    async def test_yaml_executor_module_qualified_name_nonempty(
        self, yaml_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(yaml_file)), mock_execution
        )
        assert result.module_qualified_name != ""


class TestJsonExtraction:
    """JSON (.json) via TreeSitter executor must produce file-only output."""

    @pytest.mark.asyncio
    async def test_json_executor_returns_json_language(
        self, json_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(json_file)), mock_execution
        )
        assert result.language == "json"

    @pytest.mark.asyncio
    async def test_json_executor_exactly_one_file_entity(
        self, json_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(json_file)), mock_execution
        )
        file_entities = [e for e in result.entities if e["entity_type"] == "File"]
        assert len(file_entities) == 1

    @pytest.mark.asyncio
    async def test_json_executor_no_module_entity(
        self, json_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(json_file)), mock_execution
        )
        assert [e for e in result.entities if e["entity_type"] == "Module"] == []

    @pytest.mark.asyncio
    async def test_json_executor_no_relations(
        self, json_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(json_file)), mock_execution
        )
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_json_executor_no_unresolved_imports(
        self, json_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(json_file)), mock_execution
        )
        assert result.unresolved_imports == []

    @pytest.mark.asyncio
    async def test_json_executor_module_qualified_name_nonempty(
        self, json_file: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(json_file)), mock_execution
        )
        assert result.module_qualified_name != ""


# ---------------------------------------------------------------------------
# Cross-language: code language extractors remain unchanged
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Malformed document files: must not raise, syntax_errors reflected in metadata
# ---------------------------------------------------------------------------


class TestMalformedDocumentFiles:
    """Malformed YAML/JSON must not raise; File entity must carry syntax_errors=True."""

    @pytest.mark.asyncio
    async def test_malformed_yaml_does_not_raise(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "broken.yaml"
        f.write_text("key: [\n  - bad indent\n  missing: close\n{{{{", encoding="utf-8")

        # Must not raise
        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        assert result.language == "yaml"

    @pytest.mark.asyncio
    async def test_malformed_yaml_returns_exactly_one_file_entity(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "broken.yaml"
        f.write_text("key: [\n  - bad indent\n  missing: close\n{{{{", encoding="utf-8")

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        file_entities = [e for e in result.entities if e["entity_type"] == "File"]
        assert len(file_entities) == 1

    @pytest.mark.asyncio
    async def test_malformed_yaml_no_relations(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "broken.yaml"
        f.write_text("key: [\n  - bad indent\n  missing: close\n{{{{", encoding="utf-8")

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_malformed_yaml_syntax_errors_true_in_metadata(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """File entity metadata.syntax_errors must be True for malformed YAML."""
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "broken.yaml"
        f.write_text("key: [\n  - bad indent\n  missing: close\n{{{{", encoding="utf-8")

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        fe = next(e for e in result.entities if e["entity_type"] == "File")
        assert fe["metadata"]["syntax_errors"] is True

    @pytest.mark.asyncio
    async def test_malformed_json_does_not_raise(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "broken.json"
        f.write_text('{"key": "value", bad syntax here}', encoding="utf-8")

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        assert result.language == "json"

    @pytest.mark.asyncio
    async def test_malformed_json_returns_exactly_one_file_entity(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "broken.json"
        f.write_text('{"key": "value", bad syntax here}', encoding="utf-8")

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        file_entities = [e for e in result.entities if e["entity_type"] == "File"]
        assert len(file_entities) == 1

    @pytest.mark.asyncio
    async def test_malformed_json_no_relations(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "broken.json"
        f.write_text('{"key": "value", bad syntax here}', encoding="utf-8")

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_malformed_json_syntax_errors_true_in_metadata(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """File entity metadata.syntax_errors must be True for malformed JSON."""
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "broken.json"
        f.write_text('{"key": "value", bad syntax here}', encoding="utf-8")

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        fe = next(e for e in result.entities if e["entity_type"] == "File")
        assert fe["metadata"]["syntax_errors"] is True


# ---------------------------------------------------------------------------
# Cross-language: code language extractors remain unchanged
# ---------------------------------------------------------------------------


class TestCodeLanguageExtractorsUnchanged:
    """Verify Python extraction still emits Module and CONTAINS relation (not file-only)."""

    @pytest.mark.asyncio
    async def test_python_still_emits_module_entity(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "app.py"
        f.write_text("x = 1\n", encoding="utf-8")

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        assert result.language == "python"
        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert len(module_entities) == 1, "Python must still emit Module entity"

    @pytest.mark.asyncio
    async def test_python_still_emits_contains_relation(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Python must still emit a File->Module CONTAINS relation (not file-only)."""
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "app.py"
        f.write_text("x = 1\n", encoding="utf-8")

        result = await TreeSitterExecutor().execute(
            TreeSitterInput(path=str(f)), mock_execution
        )
        contains_rels = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        assert len(contains_rels) >= 1, "Python must emit at least one CONTAINS relation"
        file_qname = next(
            e["qualified_name"] for e in result.entities if e["entity_type"] == "File"
        )
        assert contains_rels[0]["source_qname"] == file_qname
