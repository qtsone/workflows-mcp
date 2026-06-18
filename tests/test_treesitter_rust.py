"""TDD tests for Rust structural extraction via TreeSitterExecutor.

Tests written BEFORE production implementation (Phase 4).
Covers entities: File, Module, Class (struct/trait), Method, Function.
Covers relations: CONTAINS, IMPORTS, CALLS, INHERITS_FROM.
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

FULL_RUST_SRC = """\
use std::fmt;
use std::io::Write;

pub struct Base {
    pub id: i32,
}

pub struct Greeter {
    name: String,
}

pub trait Namer {
    fn name(&self) -> &str;
}

impl Namer for Greeter {
    fn name(&self) -> &str {
        &self.name
    }
}

impl Greeter {
    pub fn new(name: String) -> Self {
        Greeter { name }
    }

    pub fn greet(&self) -> String {
        format!("Hello, {}", self.name)
    }
}

pub fn top_level(x: i32) -> i32 {
    x + 1
}

fn helper() {
    top_level(0);
}
"""

SIMPLE_RUST_SRC = """\
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""

CALLS_RUST_SRC = """\
fn helper() {}

fn foo() {
    helper();
    bar.baz();
}
"""

TRAIT_IMPL_RUST_SRC = """\
pub trait Display {
    fn fmt(&self);
}

pub struct Widget {
    value: i32,
}

impl Display for Widget {
    fn fmt(&self) {
    }
}
"""

MULTI_USE_RUST_SRC = """\
use std::fmt;
use std::collections::HashMap;
use std::fmt;
"""

PLAIN_IMPL_RUST_SRC = """\
pub struct Foo {
    x: i32,
}

impl Foo {
    pub fn bar(&self) -> i32 {
        self.x
    }
}
"""


@pytest.fixture()
def full_rust_file(tmp_path: Path) -> Path:
    f = tmp_path / "src" / "greeter.rs"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(FULL_RUST_SRC, encoding="utf-8")
    return f


@pytest.fixture()
def simple_rust_file(tmp_path: Path) -> Path:
    f = tmp_path / "main.rs"
    f.write_text(SIMPLE_RUST_SRC, encoding="utf-8")
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
    def test_rust_module_qname_uses_repo_relative_dotted(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """src/greeter.rs -> module qname is 'greeter' (src. prefix stripped)."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert len(module_entities) == 1
        assert module_entities[0]["qualified_name"] == "greeter"

    def test_rust_module_qname_nested_path(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """src/utils/parser.rs -> module qname 'utils.parser'."""
        f = tmp_path / "src" / "utils" / "parser.rs"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(SIMPLE_RUST_SRC, encoding="utf-8")
        result = _run(f, "src/utils/parser.rs", mock_execution)
        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert len(module_entities) == 1
        assert module_entities[0]["qualified_name"] == "utils.parser"

    def test_rust_language_detected(self, full_rust_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        assert result.language == "rust"

    def test_rust_module_qname_non_empty(
        self, simple_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(simple_rust_file, "main.rs", mock_execution)
        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert len(module_entities) == 1
        assert module_entities[0]["qualified_name"] != ""


# ---------------------------------------------------------------------------
# File entity tests
# ---------------------------------------------------------------------------


class TestFileEntity:
    def test_file_entity_exists(self, full_rust_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        file_entities = [e for e in result.entities if e["entity_type"] == "File"]
        assert len(file_entities) == 1

    def test_file_entity_name_is_basename(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        file_entity = next(e for e in result.entities if e["entity_type"] == "File")
        assert file_entity["name"] == "greeter.rs"

    def test_file_entity_has_required_fields(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
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


class TestStructEntities:
    def test_struct_extracted_as_class(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """pub struct Greeter -> Class entity 'greeter.Greeter'."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        qnames = {e["qualified_name"] for e in class_entities}
        assert "greeter.Greeter" in qnames
        assert "greeter.Base" in qnames

    def test_struct_class_entity_has_required_fields(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        structs = [e for e in class_entities if not e["metadata"].get("is_trait")]
        for entity in structs:
            assert "entity_type" in entity
            assert "name" in entity
            assert "qualified_name" in entity
            assert "stable_id" in entity
            assert "metadata" in entity
            assert "confidence" in entity
            assert len(entity["stable_id"]) == 32
            assert entity["confidence"] == 1.0

    def test_struct_class_has_source_span(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        for entity in class_entities:
            assert "start_line" in entity["metadata"]
            assert "start_column" in entity["metadata"]

    def test_struct_name_simple(self, full_rust_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        greeter = next(e for e in result.entities if e.get("qualified_name") == "greeter.Greeter")
        assert greeter["name"] == "Greeter"


# ---------------------------------------------------------------------------
# Class (trait) entity tests
# ---------------------------------------------------------------------------


class TestTraitEntities:
    def test_trait_extracted_as_class(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """pub trait Namer -> Class entity 'greeter.Namer' with is_trait=True."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        qnames = {e["qualified_name"] for e in class_entities}
        assert "greeter.Namer" in qnames

    def test_trait_has_is_trait_metadata(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """Trait Class entity must have metadata.is_trait=True."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        namer = next(
            (e for e in class_entities if e["qualified_name"] == "greeter.Namer"),
            None,
        )
        assert namer is not None
        assert namer["metadata"].get("is_trait") is True

    def test_struct_does_not_have_is_trait(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        greeter = next(e for e in class_entities if e["qualified_name"] == "greeter.Greeter")
        assert greeter["metadata"].get("is_trait", False) is False


# ---------------------------------------------------------------------------
# Method entity tests (functions inside impl blocks)
# ---------------------------------------------------------------------------


class TestMethodEntities:
    def test_impl_functions_extracted_as_methods(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """Functions inside impl Greeter -> Method entities."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        qnames = {e["qualified_name"] for e in method_entities}
        assert "greeter.Greeter.new" in qnames
        assert "greeter.Greeter.greet" in qnames

    def test_trait_impl_methods_extracted(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """Functions inside impl Namer for Greeter -> Method entities under Greeter."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        qnames = {e["qualified_name"] for e in method_entities}
        # impl Namer for Greeter -> methods under Greeter class
        assert "greeter.Greeter.name" in qnames

    def test_method_entity_has_required_fields(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        for entity in method_entities:
            assert "entity_type" in entity
            assert "name" in entity
            assert "qualified_name" in entity
            assert "stable_id" in entity
            assert "metadata" in entity
            assert "confidence" in entity
            assert len(entity["stable_id"]) == 32

    def test_method_has_parent_class_hint(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """Method metadata must include parent_class_id hint."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        for entity in method_entities:
            meta = entity["metadata"]
            assert "parent_class_id" in meta
            assert meta["parent_class_id"].startswith("__class_qname__:")

    def test_greeter_new_parent_class_hint(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        new_method = next(
            e for e in result.entities if e.get("qualified_name") == "greeter.Greeter.new"
        )
        assert new_method["metadata"]["parent_class_id"] == "__class_qname__:greeter.Greeter"

    def test_method_name_simple(self, full_rust_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        greet = next(
            e for e in result.entities if e.get("qualified_name") == "greeter.Greeter.greet"
        )
        assert greet["name"] == "greet"


# ---------------------------------------------------------------------------
# Function entity tests (top-level function_item)
# ---------------------------------------------------------------------------


class TestFunctionEntities:
    def test_top_level_functions_extracted(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """fn top_level() and fn helper() -> Function entities."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        qnames = {e["qualified_name"] for e in func_entities}
        assert "greeter.top_level" in qnames
        assert "greeter.helper" in qnames

    def test_impl_methods_not_in_functions(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """Methods inside impl blocks must be Method type, not Function."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        qnames = {e["qualified_name"] for e in func_entities}
        assert "greeter.Greeter.new" not in qnames
        assert "greeter.Greeter.greet" not in qnames
        assert "greeter.Greeter.name" not in qnames

    def test_function_entity_has_required_fields(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        for entity in func_entities:
            assert "entity_type" in entity
            assert "name" in entity
            assert "qualified_name" in entity
            assert "stable_id" in entity
            assert "metadata" in entity
            assert "confidence" in entity

    def test_simple_function_extracted(
        self, simple_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(simple_rust_file, "main.rs", mock_execution)
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        assert any(e["qualified_name"] == "main.add" for e in func_entities)


# ---------------------------------------------------------------------------
# Entity ordering tests
# ---------------------------------------------------------------------------


class TestEntityOrdering:
    def test_entity_order_file_module_class_method_function(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """Entities in deterministic order: File, Module, Class, Method, Function."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
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
    def test_contains_file_to_module(self, full_rust_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        file_entity = next(e for e in result.entities if e["entity_type"] == "File")
        file_to_module = [
            r
            for r in contains
            if r["source_qname"] == file_entity["qualified_name"]
            and r["target_entity_type"] == "Module"
        ]
        assert len(file_to_module) == 1

    def test_contains_module_to_struct(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
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
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        module_to_func = [
            r
            for r in contains
            if r["source_qname"] == "greeter" and r["target_entity_type"] == "Function"
        ]
        qnames = {r["target_qname"] for r in module_to_func}
        assert "greeter.top_level" in qnames
        assert "greeter.helper" in qnames

    def test_contains_class_to_method(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        class_to_method = [
            r
            for r in contains
            if r["source_qname"] == "greeter.Greeter" and r["target_entity_type"] == "Method"
        ]
        qnames = {r["target_qname"] for r in class_to_method}
        assert "greeter.Greeter.new" in qnames
        assert "greeter.Greeter.greet" in qnames
        assert "greeter.Greeter.name" in qnames

    def test_contains_relation_has_required_fields(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
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
    def test_use_declaration_emits_imports(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """use std::fmt; -> IMPORTS relation target 'std.fmt'."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        target_qnames = {r["target_qname"] for r in imports}
        assert "std.fmt" in target_qnames

    def test_scoped_use_emits_imports(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """use std::io::Write; -> IMPORTS target 'std.io.Write'."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        target_qnames = {r["target_qname"] for r in imports}
        assert "std.io.Write" in target_qnames

    def test_import_confidence_is_unresolved(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        """All Rust imports are unresolved (confidence=0.5)."""
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        for rel in imports:
            assert rel["confidence"] == 0.5
            assert rel["metadata"].get("resolution") == "unresolved"

    def test_import_source_is_module(self, full_rust_file: Path, mock_execution: MagicMock) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        for rel in imports:
            assert rel["source_entity_type"] == "Module"
            assert rel["target_entity_type"] == "Module"

    def test_import_deduplication(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """Duplicate use paths emitted only once."""
        f = tmp_path / "dup.rs"
        f.write_text(MULTI_USE_RUST_SRC, encoding="utf-8")
        result = _run(f, "dup.rs", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        targets = [r["target_qname"] for r in imports]
        assert targets.count("std.fmt") == 1

    def test_unresolved_imports_populated(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        assert "std.fmt" in result.unresolved_imports

    def test_unresolved_imports_sorted(
        self, full_rust_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_rust_file, "src/greeter.rs", mock_execution)
        assert result.unresolved_imports == sorted(result.unresolved_imports)


# ---------------------------------------------------------------------------
# Relation: INHERITS_FROM
# ---------------------------------------------------------------------------


class TestInheritsFrom:
    def test_trait_impl_emits_inherits_from(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """impl Display for Widget -> INHERITS_FROM Widget->Display."""
        f = tmp_path / "src" / "widget.rs"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(TRAIT_IMPL_RUST_SRC, encoding="utf-8")
        result = _run(f, "src/widget.rs", mock_execution)
        inherits = [r for r in result.relations if r["relation_type"] == "INHERITS_FROM"]
        assert len(inherits) >= 1
        assert any(
            r["source_qname"] == "widget.Widget" and "Display" in r["target_qname"]
            for r in inherits
        )

    def test_plain_impl_no_inherits_from(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """impl Foo { ... } (no trait) -> no INHERITS_FROM."""
        f = tmp_path / "foo.rs"
        f.write_text(PLAIN_IMPL_RUST_SRC, encoding="utf-8")
        result = _run(f, "foo.rs", mock_execution)
        inherits = [r for r in result.relations if r["relation_type"] == "INHERITS_FROM"]
        assert len(inherits) == 0

    def test_trait_impl_confidence(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """INHERITS_FROM from trait impl has confidence=0.8 and rust_trait=True."""
        f = tmp_path / "src" / "widget.rs"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(TRAIT_IMPL_RUST_SRC, encoding="utf-8")
        result = _run(f, "src/widget.rs", mock_execution)
        inherits = [r for r in result.relations if r["relation_type"] == "INHERITS_FROM"]
        for rel in inherits:
            assert rel["confidence"] == 0.8
            assert rel["metadata"].get("rust_trait") is True

    def test_trait_impl_same_module_resolved(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """When trait is defined in same module, resolution='resolved'."""
        f = tmp_path / "src" / "widget.rs"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(TRAIT_IMPL_RUST_SRC, encoding="utf-8")
        result = _run(f, "src/widget.rs", mock_execution)
        inherits = [r for r in result.relations if r["relation_type"] == "INHERITS_FROM"]
        # Display is defined in same module
        widget_inherits = [r for r in inherits if r["source_qname"] == "widget.Widget"]
        assert len(widget_inherits) >= 1
        assert widget_inherits[0]["metadata"].get("resolution") == "resolved"

    def test_trait_impl_external_trait_unresolved(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """impl std::fmt::Display for Foo must emit INHERITS_FROM unconditionally.

        Scoped trait path (scoped_type_identifier) must be handled: target qname
        uses dotted notation ('std.fmt.Display'), source is 'foo.Foo',
        confidence=0.5, resolution='unresolved', rust_trait=True.
        """
        src = """\
pub struct Foo {
    x: i32,
}

impl std::fmt::Display for Foo {
    fn fmt(&self, f: &mut std::fmt::Formatter) {}
}
"""
        f = tmp_path / "foo.rs"
        f.write_text(src, encoding="utf-8")
        result = _run(f, "foo.rs", mock_execution)
        inherits = [r for r in result.relations if r["relation_type"] == "INHERITS_FROM"]
        # Must emit exactly one INHERITS_FROM (scoped trait path must be handled)
        assert len(inherits) == 1, (
            f"Expected 1 INHERITS_FROM for scoped trait impl, got {len(inherits)}: {inherits}"
        )
        rel = inherits[0]
        assert rel["source_qname"] == "foo.Foo"
        assert rel["target_qname"] == "std.fmt.Display"
        assert rel["confidence"] == 0.5
        assert rel["metadata"].get("resolution") == "unresolved"
        assert rel["metadata"].get("rust_trait") is True

    def test_scoped_trait_impl_methods_extracted_under_struct(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Methods inside impl std::fmt::Display for Foo must be Method entities under Foo.

        The scoped trait path must not prevent method extraction.
        """
        src = """\
pub struct Foo {
    x: i32,
}

impl std::fmt::Display for Foo {
    fn fmt(&self, f: &mut std::fmt::Formatter) {}
}
"""
        f = tmp_path / "foo.rs"
        f.write_text(src, encoding="utf-8")
        result = _run(f, "foo.rs", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        qnames = {e["qualified_name"] for e in method_entities}
        assert "foo.Foo.fmt" in qnames, f"Expected 'foo.Foo.fmt' in method qnames, got: {qnames}"
        fmt_method = next(e for e in method_entities if e["qualified_name"] == "foo.Foo.fmt")
        assert fmt_method["metadata"]["parent_class_id"] == "__class_qname__:foo.Foo"


# ---------------------------------------------------------------------------
# Relation: CALLS
# ---------------------------------------------------------------------------


class TestCalls:
    def test_same_module_call_resolved(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """foo() calls helper() -> resolved to module.helper, confidence=1.0."""
        f = tmp_path / "mymod.rs"
        f.write_text(CALLS_RUST_SRC, encoding="utf-8")
        result = _run(f, "mymod.rs", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        foo_calls = [r for r in calls if r["source_qname"] == "mymod.foo"]
        helper_call = next((r for r in foo_calls if r["target_qname"] == "mymod.helper"), None)
        assert helper_call is not None
        assert helper_call["confidence"] == 1.0

    def test_field_expression_call_unresolved(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """bar.baz() -> unresolved CALLS, confidence=0.5."""
        f = tmp_path / "mymod.rs"
        f.write_text(CALLS_RUST_SRC, encoding="utf-8")
        result = _run(f, "mymod.rs", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        foo_calls = [r for r in calls if r["source_qname"] == "mymod.foo"]
        unresolved = [r for r in foo_calls if r["metadata"].get("resolution") == "unresolved"]
        assert len(unresolved) >= 1
        for c in unresolved:
            assert c["confidence"] == 0.5

    def test_call_has_site_metadata(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """CALLS relations include call_line and call_column."""
        f = tmp_path / "mymod.rs"
        f.write_text(CALLS_RUST_SRC, encoding="utf-8")
        result = _run(f, "mymod.rs", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        for rel in calls:
            assert "call_line" in rel["metadata"]
            assert "call_column" in rel["metadata"]

    def test_call_has_entity_type_fields(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        f = tmp_path / "mymod.rs"
        f.write_text(CALLS_RUST_SRC, encoding="utf-8")
        result = _run(f, "mymod.rs", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        for rel in calls:
            assert "source_entity_type" in rel
            assert "target_entity_type" in rel

    def test_calls_inside_scoped_trait_impl_method_are_emitted(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """CALLS from a method inside a scoped trait impl must not be dropped.

        impl std::fmt::Display for Foo { fn fmt(...) { helper(); } }
        must emit a CALLS relation: source=<module>.Foo.fmt, target=<module>.helper,
        confidence=1.0 (resolved same-module call).

        This guards against the _extract_calls_from_impl bug where
        scoped_type_identifier for the trait leaves only one type_identifier
        (the struct), causing the old len(type_ids) >= 2 check to fall through
        to 'return', silently dropping all CALLS inside the impl block.
        """
        src = """\
pub struct Foo {
    x: i32,
}

fn helper() {}

impl std::fmt::Display for Foo {
    fn fmt(&self, f: &mut std::fmt::Formatter) {
        helper();
    }
}
"""
        file = tmp_path / "foo.rs"
        file.write_text(src, encoding="utf-8")
        result = _run(file, "foo.rs", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        fmt_calls = [r for r in calls if r["source_qname"] == "foo.Foo.fmt"]
        helper_call = next((r for r in fmt_calls if r["target_qname"] == "foo.helper"), None)
        assert helper_call is not None, (
            f"Expected CALLS from foo.Foo.fmt to foo.helper, got fmt_calls={fmt_calls}"
        )
        assert helper_call["confidence"] == 1.0
        # Resolved calls omit 'resolution' from metadata (only unresolved carry it)
        assert "resolution" not in helper_call["metadata"]


# ---------------------------------------------------------------------------
# Non-Rust behavior unchanged
# ---------------------------------------------------------------------------


class TestNonRustUnchanged:
    @pytest.mark.asyncio
    async def test_python_file_unaffected(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """Python extractor continues to work after Rust extractor added."""
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
    async def test_go_file_unaffected(self, tmp_path: Path, mock_execution: MagicMock) -> None:
        """Go extractor continues to work after Rust extractor added."""
        from workflows_mcp.code_intelligence.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "main.go"
        f.write_text("package main\nfunc Add(a, b int) int { return a + b }\n", encoding="utf-8")
        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(f), repo_relative_path="main.go")
        result = await executor.execute(inputs, mock_execution)
        assert result.language == "go"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_output_is_deterministic(self, full_rust_file: Path, mock_execution: MagicMock) -> None:
        result1 = _run(full_rust_file, "src/greeter.rs", mock_execution)
        result2 = _run(full_rust_file, "src/greeter.rs", mock_execution)

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
