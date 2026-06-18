"""TDD tests for Go structural extraction via TreeSitterExecutor.

Tests written BEFORE production implementation (Phase 4).
Covers entities: File, Module, Class (struct/interface), Method, Function.
Covers relations: CONTAINS, IMPORTS, CALLS.
No INHERITS_FROM in v1 (Go has no class inheritance).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

FULL_GO_SRC = """\
package greeter

import (
\t"fmt"
\t"strings"
)

type Base struct {
\tID int
}

type Greeter struct {
\tname string
}

type Namer interface {
\tName() string
}

func (g *Greeter) Greet() string {
\treturn fmt.Sprintf("Hello, %s", g.name)
}

func (g *Greeter) helper() {
\tfmt.Println("helper")
}

func NewGreeter(name string) *Greeter {
\treturn &Greeter{name: name}
}

func main() {
\tg := NewGreeter("world")
\tg.Greet()
}
"""

SIMPLE_GO_SRC = """\
package main

func Add(a, b int) int {
\treturn a + b
}
"""

IMPORT_GO_SRC = """\
package mymod

import "fmt"
import (
\t"os"
\t"strings"
)
"""

CALLS_GO_SRC = """\
package mymod

func helper() {}

func foo() {
\thelper()
\tfmt.Println("hi")
}
"""


@pytest.fixture()
def full_go_file(tmp_path: Path) -> Path:
    f = tmp_path / "pkg" / "greeter.go"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(FULL_GO_SRC, encoding="utf-8")
    return f


@pytest.fixture()
def simple_go_file(tmp_path: Path) -> Path:
    f = tmp_path / "main.go"
    f.write_text(SIMPLE_GO_SRC, encoding="utf-8")
    return f


@pytest.fixture()
def mock_execution() -> MagicMock:
    ctx = MagicMock()
    ctx.workflow_name = "test"
    return ctx


def _run(path: Path, repo_relative: str, mock_execution: MagicMock) -> Any:
    """Execute TreeSitterExecutor synchronously via asyncio."""
    from workflows_mcp.code_intelligence.executors_treesitter import (
        TreeSitterExecutor,
        TreeSitterInput,
    )

    executor = TreeSitterExecutor()
    inputs = TreeSitterInput(
        path=str(path),
        repo_relative_path=repo_relative,
        palace="test_palace",
        item_id="test_item",
    )
    return asyncio.run(executor.execute(inputs, mock_execution))


# ---------------------------------------------------------------------------
# Module qname tests
# ---------------------------------------------------------------------------


class TestModuleQname:
    def test_go_module_qname_uses_package_name(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        """pkg/greeter.go with 'package greeter' -> module qname is 'greeter'."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert len(module_entities) == 1
        assert module_entities[0]["qualified_name"] == "greeter"

    def test_go_module_name_field(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        """Module entity name field equals the package name."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        module_entity = next(e for e in result.entities if e["entity_type"] == "Module")
        assert module_entity["name"] == "greeter"

    def test_go_language_detected(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        assert result.language == "go"

    def test_go_module_fallback_to_repo_relative_when_no_package(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Empty Go file (no package clause) -> falls back to repo-relative qname."""
        f = tmp_path / "pkg" / "empty.go"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("", encoding="utf-8")
        result = _run(f, "pkg/empty.go", mock_execution)
        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert len(module_entities) == 1
        # Fallback: repo-relative with dots (pkg.empty)
        assert module_entities[0]["qualified_name"] == "pkg.empty"


# ---------------------------------------------------------------------------
# File entity tests
# ---------------------------------------------------------------------------


class TestFileEntity:
    def test_file_entity_exists(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        file_entities = [e for e in result.entities if e["entity_type"] == "File"]
        assert len(file_entities) == 1

    def test_file_entity_name_is_basename(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        file_entity = next(e for e in result.entities if e["entity_type"] == "File")
        assert file_entity["name"] == "greeter.go"

    def test_file_entity_has_required_fields(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        file_entity = next(e for e in result.entities if e["entity_type"] == "File")
        assert "entity_type" in file_entity
        assert "name" in file_entity
        assert "qualified_name" in file_entity
        assert "stable_id" in file_entity
        assert "metadata" in file_entity
        assert "confidence" in file_entity
        assert len(file_entity["stable_id"]) == 32


# ---------------------------------------------------------------------------
# Class (struct) entity tests
# ---------------------------------------------------------------------------


class TestClassEntities:
    def test_struct_extracted_as_class(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        """type Greeter struct{} -> Class entity 'greeter.Greeter'."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        qnames = {e["qualified_name"] for e in class_entities}
        assert "greeter.Greeter" in qnames
        assert "greeter.Base" in qnames

    def test_interface_extracted_as_class_with_metadata(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        """type Namer interface{} -> Class entity with metadata.is_interface=True."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        namer = next(
            (e for e in class_entities if e["qualified_name"] == "greeter.Namer"),
            None,
        )
        assert namer is not None
        assert namer["metadata"].get("is_interface") is True

    def test_struct_class_has_is_interface_false(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        """Struct entities must not have is_interface=True."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        greeter = next(e for e in class_entities if e["qualified_name"] == "greeter.Greeter")
        assert greeter["metadata"].get("is_interface", False) is False

    def test_class_entity_has_required_fields(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        for entity in class_entities:
            assert "entity_type" in entity
            assert "name" in entity
            assert "qualified_name" in entity
            assert "stable_id" in entity
            assert "metadata" in entity
            assert "confidence" in entity
            assert len(entity["stable_id"]) == 32
            assert entity["confidence"] == 1.0

    def test_class_name_is_simple(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        greeter = next(e for e in result.entities if e.get("qualified_name") == "greeter.Greeter")
        assert greeter["name"] == "Greeter"

    def test_class_has_source_span(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        for entity in class_entities:
            assert "start_line" in entity["metadata"]
            assert "start_column" in entity["metadata"]


# ---------------------------------------------------------------------------
# Method entity tests
# ---------------------------------------------------------------------------


class TestMethodEntities:
    def test_method_declaration_extracted(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        """func (g *Greeter) Greet() -> Method 'greeter.Greeter.Greet'."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        qnames = {e["qualified_name"] for e in method_entities}
        assert "greeter.Greeter.Greet" in qnames
        assert "greeter.Greeter.helper" in qnames

    def test_method_entity_has_required_fields(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        for entity in method_entities:
            assert "entity_type" in entity
            assert "name" in entity
            assert "qualified_name" in entity
            assert "stable_id" in entity
            assert "metadata" in entity
            assert "confidence" in entity
            assert len(entity["stable_id"]) == 32

    def test_method_entity_has_parent_class_hint(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        """Method metadata must include parent_class_id hint."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        for entity in method_entities:
            meta = entity["metadata"]
            assert "parent_class_id" in meta
            assert meta["parent_class_id"].startswith("__class_qname__:")

    def test_greet_method_parent_class_hint(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        greet = next(
            e for e in result.entities if e.get("qualified_name") == "greeter.Greeter.Greet"
        )
        assert greet["metadata"]["parent_class_id"] == "__class_qname__:greeter.Greeter"

    def test_method_name_simple(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        greet = next(
            e for e in result.entities if e.get("qualified_name") == "greeter.Greeter.Greet"
        )
        assert greet["name"] == "Greet"


# ---------------------------------------------------------------------------
# Function entity tests
# ---------------------------------------------------------------------------


class TestFunctionEntities:
    def test_top_level_functions_extracted(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        """func NewGreeter() and func main() -> Function entities."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        qnames = {e["qualified_name"] for e in func_entities}
        assert "greeter.NewGreeter" in qnames
        assert "greeter.main" in qnames

    def test_methods_not_in_functions(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        """Methods must be Method type, not Function."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        qnames = {e["qualified_name"] for e in func_entities}
        assert "greeter.Greeter.Greet" not in qnames
        assert "greeter.Greeter.helper" not in qnames

    def test_function_entity_has_required_fields(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        for entity in func_entities:
            assert "entity_type" in entity
            assert "name" in entity
            assert "qualified_name" in entity
            assert "stable_id" in entity
            assert "metadata" in entity
            assert "confidence" in entity


# ---------------------------------------------------------------------------
# Entity ordering tests
# ---------------------------------------------------------------------------


class TestEntityOrdering:
    def test_entity_order_file_module_class_method_function(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        """Entities must appear in deterministic order: File, Module, Class, Method, Function."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        types = [e["entity_type"] for e in result.entities]
        assert types[0] == "File"
        assert types[1] == "Module"
        class_idx = [i for i, t in enumerate(types) if t == "Class"]
        method_idx = [i for i, t in enumerate(types) if t == "Method"]
        func_idx = [i for i, t in enumerate(types) if t == "Function"]
        if class_idx and method_idx:
            assert max(class_idx) < min(method_idx)
        if method_idx and func_idx:
            assert max(method_idx) < min(func_idx)
        if class_idx and func_idx:
            assert max(class_idx) < min(func_idx)


# ---------------------------------------------------------------------------
# Relation: CONTAINS
# ---------------------------------------------------------------------------


class TestContainsRelations:
    def test_contains_file_to_module(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        file_entity = next(e for e in result.entities if e["entity_type"] == "File")
        file_to_module = [
            r
            for r in contains
            if r["source_qname"] == file_entity["qualified_name"]
            and r["target_entity_type"] == "Module"
        ]
        assert len(file_to_module) == 1

    def test_contains_module_to_class(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        module_to_class = [
            r
            for r in contains
            if r["source_qname"] == "greeter" and r["target_entity_type"] == "Class"
        ]
        qnames = {r["target_qname"] for r in module_to_class}
        assert "greeter.Greeter" in qnames
        assert "greeter.Base" in qnames
        assert "greeter.Namer" in qnames

    def test_contains_module_to_function(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        module_to_func = [
            r
            for r in contains
            if r["source_qname"] == "greeter" and r["target_entity_type"] == "Function"
        ]
        qnames = {r["target_qname"] for r in module_to_func}
        assert "greeter.NewGreeter" in qnames
        assert "greeter.main" in qnames

    def test_contains_class_to_method(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        class_to_method = [
            r
            for r in contains
            if r["source_qname"] == "greeter.Greeter" and r["target_entity_type"] == "Method"
        ]
        qnames = {r["target_qname"] for r in class_to_method}
        assert "greeter.Greeter.Greet" in qnames
        assert "greeter.Greeter.helper" in qnames

    def test_contains_relation_has_required_fields(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        for rel in contains:
            assert "source_qname" in rel
            assert "target_qname" in rel
            assert "source_entity_type" in rel
            assert "target_entity_type" in rel
            assert "relation_type" in rel


# ---------------------------------------------------------------------------
# Relation: IMPORTS
# ---------------------------------------------------------------------------


class TestImports:
    def test_single_import_emitted(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """import 'fmt' -> IMPORTS relation to 'fmt'."""
        f = tmp_path / "main.go"
        f.write_text('package main\nimport "fmt"\n', encoding="utf-8")
        result = _run(f, "main.go", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        assert any(r["target_qname"] == "fmt" for r in imports)

    def test_grouped_imports_emitted(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """Grouped import block -> each package emitted as separate IMPORTS."""
        f = tmp_path / "main.go"
        f.write_text('package main\nimport (\n\t"os"\n\t"strings"\n)\n', encoding="utf-8")
        result = _run(f, "main.go", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        target_qnames = {r["target_qname"] for r in imports}
        assert "os" in target_qnames
        assert "strings" in target_qnames

    def test_import_confidence_is_unresolved(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """All Go imports are unresolved (confidence=0.5)."""
        f = tmp_path / "main.go"
        f.write_text('package main\nimport "fmt"\n', encoding="utf-8")
        result = _run(f, "main.go", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        for rel in imports:
            assert rel["confidence"] == 0.5
            assert rel["metadata"].get("resolution") == "unresolved"

    def test_import_source_is_module(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        f = tmp_path / "main.go"
        f.write_text('package main\nimport "fmt"\n', encoding="utf-8")
        result = _run(f, "main.go", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        for rel in imports:
            assert rel["source_entity_type"] == "Module"
            assert rel["target_entity_type"] == "Module"

    def test_import_deduplication(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """Duplicate import paths emitted only once."""
        f = tmp_path / "main.go"
        f.write_text('package main\nimport "fmt"\nimport "fmt"\n', encoding="utf-8")
        result = _run(f, "main.go", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        targets = [r["target_qname"] for r in imports]
        assert targets.count("fmt") == 1

    def test_unresolved_imports_populated(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        f = tmp_path / "main.go"
        f.write_text('package main\nimport (\n\t"os"\n\t"fmt"\n)\n', encoding="utf-8")
        result = _run(f, "main.go", mock_execution)
        assert "os" in result.unresolved_imports
        assert "fmt" in result.unresolved_imports

    def test_unresolved_imports_sorted(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        f = tmp_path / "main.go"
        f.write_text(
            'package main\nimport (\n\t"strings"\n\t"os"\n\t"fmt"\n)\n',
            encoding="utf-8",
        )
        result = _run(f, "main.go", mock_execution)
        assert result.unresolved_imports == sorted(result.unresolved_imports)

    def test_no_inherits_from_relations(
        self, full_go_file: Path, mock_execution: MagicMock
    ) -> None:
        """Go v1 extractor emits no INHERITS_FROM relations."""
        result = _run(full_go_file, "pkg/greeter.go", mock_execution)
        inherits = [r for r in result.relations if r["relation_type"] == "INHERITS_FROM"]
        assert len(inherits) == 0


# ---------------------------------------------------------------------------
# Relation: CALLS
# ---------------------------------------------------------------------------


class TestCalls:
    def test_same_package_call_resolved(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """foo() calls helper() -> resolved to mymod.helper, confidence=1.0."""
        f = tmp_path / "mymod.go"
        f.write_text(CALLS_GO_SRC, encoding="utf-8")
        result = _run(f, "mymod.go", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        foo_calls = [r for r in calls if r["source_qname"] == "mymod.foo"]
        helper_call = next((r for r in foo_calls if r["target_qname"] == "mymod.helper"), None)
        assert helper_call is not None
        assert helper_call["confidence"] == 1.0

    def test_selector_call_unresolved(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """fmt.Println call -> unresolved, confidence=0.5."""
        f = tmp_path / "mymod.go"
        f.write_text(CALLS_GO_SRC, encoding="utf-8")
        result = _run(f, "mymod.go", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        foo_calls = [r for r in calls if r["source_qname"] == "mymod.foo"]
        selector_calls = [r for r in foo_calls if r["metadata"].get("resolution") == "unresolved"]
        assert len(selector_calls) >= 1
        for c in selector_calls:
            assert c["confidence"] == 0.5

    def test_call_has_site_metadata(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """CALLS relations include call_line and call_column."""
        f = tmp_path / "mymod.go"
        f.write_text(CALLS_GO_SRC, encoding="utf-8")
        result = _run(f, "mymod.go", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        for rel in calls:
            assert "call_line" in rel["metadata"]
            assert "call_column" in rel["metadata"]

    def test_call_has_entity_type_fields(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        f = tmp_path / "mymod.go"
        f.write_text(CALLS_GO_SRC, encoding="utf-8")
        result = _run(f, "mymod.go", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        for rel in calls:
            assert "source_entity_type" in rel
            assert "target_entity_type" in rel


# ---------------------------------------------------------------------------
# Non-Go behavior unchanged
# ---------------------------------------------------------------------------


class TestNonGoUnchanged:
    @pytest.mark.asyncio
    async def test_python_file_unaffected(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """Python extractor continues to work after Go extractor is added."""
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "mymod.py"
        f.write_text("def foo(): pass\n", encoding="utf-8")
        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(f), repo_relative_path="mymod.py")
        result = await executor.execute(inputs, mock_execution)
        assert result.language == "python"
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        assert len(func_entities) == 1

    @pytest.mark.asyncio
    async def test_unsupported_extension_unchanged(
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


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_output_is_deterministic(self, full_go_file: Path, mock_execution: MagicMock) -> None:
        result1 = _run(full_go_file, "pkg/greeter.go", mock_execution)
        result2 = _run(full_go_file, "pkg/greeter.go", mock_execution)

        qnames1 = [e["qualified_name"] for e in result1.entities]
        qnames2 = [e["qualified_name"] for e in result2.entities]
        assert qnames1 == qnames2

        rel_keys1 = sorted(
            (r["relation_type"], r["source_qname"], r["target_qname"]) for r in result1.relations
        )
        rel_keys2 = sorted(
            (r["relation_type"], r["source_qname"], r["target_qname"]) for r in result2.relations
        )
        assert rel_keys1 == rel_keys2
