"""TDD tests for Python structural extraction via TreeSitterExecutor.

Tests written BEFORE production implementation (Phase 3).
Covers entities: File, Module, Class, Method, Function.
Covers relations: CONTAINS, INHERITS_FROM, IMPORTS, CALLS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

FULL_PYTHON_SRC = """\
import os
import sys as system
from pathlib import Path
from os.path import join, exists

class Base:
    pass

class Greeter(Base):
    def greet(self, name: str) -> str:
        return f"Hello, {name}"

    def _helper(self):
        print("helper")

def main():
    g = Greeter()
    g.greet("world")
    helper()

def helper():
    pass
"""

INIT_PYTHON_SRC = """\
"""  # empty __init__.py

EXTERNAL_INHERIT_SRC = """\
from external.lib import BaseClass

class MyClass(BaseClass):
    def method(self):
        pass
"""

SIMPLE_IMPORT_SRC = """\
import os
import sys as system
from pathlib import Path
from os.path import join, exists
"""


@pytest.fixture()
def full_python_file(tmp_path: Path) -> Path:
    f = tmp_path / "pkg" / "greeter.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(FULL_PYTHON_SRC, encoding="utf-8")
    return f


@pytest.fixture()
def init_py_file(tmp_path: Path) -> Path:
    f = tmp_path / "pkg" / "__init__.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(INIT_PYTHON_SRC, encoding="utf-8")
    return f


@pytest.fixture()
def mock_execution() -> MagicMock:
    ctx = MagicMock()
    ctx.workflow_name = "test"
    return ctx


def _run(path: Path, repo_relative: str, mock_execution: MagicMock) -> Any:
    """Helper: execute TreeSitterExecutor synchronously via asyncio."""
    import asyncio

    from workflows_mcp.engine.executors_treesitter import (
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
    def test_pkg_mod_py_becomes_pkg_mod(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """pkg/greeter.py -> pkg.greeter (not pkg/greeter)."""
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert len(module_entities) == 1
        assert module_entities[0]["qualified_name"] == "pkg.greeter"

    def test_pkg_init_py_becomes_pkg(
        self, init_py_file: Path, mock_execution: MagicMock
    ) -> None:
        """pkg/__init__.py -> pkg (strip __init__)."""
        result = _run(init_py_file, "pkg/__init__.py", mock_execution)
        module_entities = [e for e in result.entities if e["entity_type"] == "Module"]
        assert len(module_entities) == 1
        assert module_entities[0]["qualified_name"] == "pkg"


# ---------------------------------------------------------------------------
# Entity extraction tests
# ---------------------------------------------------------------------------


class TestClassEntities:
    def test_class_entities_extracted(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        qnames = {e["qualified_name"] for e in class_entities}
        assert "pkg.greeter.Base" in qnames
        assert "pkg.greeter.Greeter" in qnames

    def test_class_entity_has_required_fields(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
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

    def test_class_entity_name_is_simple_name(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        base = next(
            e for e in result.entities if e.get("qualified_name") == "pkg.greeter.Base"
        )
        greeter = next(
            e for e in result.entities if e.get("qualified_name") == "pkg.greeter.Greeter"
        )
        assert base["name"] == "Base"
        assert greeter["name"] == "Greeter"

    def test_class_entity_has_source_span(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        class_entities = [e for e in result.entities if e["entity_type"] == "Class"]
        for entity in class_entities:
            meta = entity["metadata"]
            assert "start_line" in meta
            assert "start_column" in meta


class TestMethodEntities:
    def test_method_entities_extracted(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        qnames = {e["qualified_name"] for e in method_entities}
        assert "pkg.greeter.Greeter.greet" in qnames
        assert "pkg.greeter.Greeter._helper" in qnames

    def test_method_entity_has_required_fields(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
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
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """Method metadata must include parent_class_id hint."""
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        method_entities = [e for e in result.entities if e["entity_type"] == "Method"]
        for entity in method_entities:
            meta = entity["metadata"]
            assert "parent_class_id" in meta
            assert meta["parent_class_id"].startswith("__class_qname__:")

    def test_greet_method_parent_class_hint(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        greet = next(
            e for e in result.entities
            if e.get("qualified_name") == "pkg.greeter.Greeter.greet"
        )
        assert greet["metadata"]["parent_class_id"] == "__class_qname__:pkg.greeter.Greeter"


class TestFunctionEntities:
    def test_top_level_functions_extracted(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        qnames = {e["qualified_name"] for e in func_entities}
        assert "pkg.greeter.main" in qnames
        assert "pkg.greeter.helper" in qnames

    def test_functions_not_methods(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """greet/_helper must be Method, not Function."""
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        func_entities = [e for e in result.entities if e["entity_type"] == "Function"]
        qnames = {e["qualified_name"] for e in func_entities}
        assert "pkg.greeter.Greeter.greet" not in qnames
        assert "pkg.greeter.Greeter._helper" not in qnames

    def test_function_entity_has_required_fields(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
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
    def test_entity_order_is_file_module_class_method_function(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """Entities must appear in deterministic type order: File, Module, Class, Method, Function."""  # noqa: E501
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        types = [e["entity_type"] for e in result.entities]
        # File first, Module second
        assert types[0] == "File"
        assert types[1] == "Module"
        # All Classes before Methods before Functions
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
# Relation tests: CONTAINS
# ---------------------------------------------------------------------------


class TestContainsRelations:
    def test_contains_module_to_class(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        # Module -> Class
        module_to_class = [
            r for r in contains
            if r["source_qname"] == "pkg.greeter"
            and r.get("target_entity_type") == "Class"
        ]
        target_qnames = {r["target_qname"] for r in module_to_class}
        assert "pkg.greeter.Base" in target_qnames
        assert "pkg.greeter.Greeter" in target_qnames

    def test_contains_module_to_function(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        module_to_func = [
            r for r in contains
            if r["source_qname"] == "pkg.greeter"
            and r.get("target_entity_type") == "Function"
        ]
        target_qnames = {r["target_qname"] for r in module_to_func}
        assert "pkg.greeter.main" in target_qnames
        assert "pkg.greeter.helper" in target_qnames

    def test_contains_class_to_method(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        class_to_method = [
            r for r in contains
            if r["source_qname"] == "pkg.greeter.Greeter"
            and r.get("target_entity_type") == "Method"
        ]
        target_qnames = {r["target_qname"] for r in class_to_method}
        assert "pkg.greeter.Greeter.greet" in target_qnames
        assert "pkg.greeter.Greeter._helper" in target_qnames

    def test_contains_relation_has_entity_type_fields(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        contains = [r for r in result.relations if r["relation_type"] == "CONTAINS"]
        for rel in contains:
            assert "source_qname" in rel
            assert "target_qname" in rel
            assert "source_entity_type" in rel
            assert "target_entity_type" in rel
            assert "relation_type" in rel


# ---------------------------------------------------------------------------
# Relation tests: INHERITS_FROM
# ---------------------------------------------------------------------------


class TestInheritsFrom:
    def test_same_module_inheritance(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """Greeter inherits from Base (same module)."""
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        inherits = [r for r in result.relations if r["relation_type"] == "INHERITS_FROM"]
        greeter_inherits = [r for r in inherits if r["source_qname"] == "pkg.greeter.Greeter"]
        assert len(greeter_inherits) == 1
        assert greeter_inherits[0]["target_qname"] == "pkg.greeter.Base"
        assert greeter_inherits[0]["confidence"] == 1.0

    def test_external_inheritance_is_unresolved(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Greeter inherits from external BaseClass -> unresolved, confidence=0.5."""
        f = tmp_path / "mymod.py"
        f.write_text(EXTERNAL_INHERIT_SRC, encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        inherits = [r for r in result.relations if r["relation_type"] == "INHERITS_FROM"]
        assert len(inherits) == 1
        rel = inherits[0]
        assert rel["source_qname"] == "mymod.MyClass"
        assert rel["confidence"] == 0.5
        assert rel["metadata"].get("resolution") == "unresolved"

    def test_inherits_from_has_entity_type_fields(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        inherits = [r for r in result.relations if r["relation_type"] == "INHERITS_FROM"]
        for rel in inherits:
            assert rel.get("source_entity_type") == "Class"
            assert rel.get("target_entity_type") == "Class"


# ---------------------------------------------------------------------------
# Relation tests: IMPORTS
# ---------------------------------------------------------------------------


class TestImports:
    def test_plain_import_emitted(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """import os -> IMPORTS relation from module to os."""
        f = tmp_path / "mymod.py"
        f.write_text("import os\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        assert any(r["target_qname"] == "os" for r in imports)

    def test_aliased_import_emitted(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """import sys as system -> target qname is 'sys' (original module)."""
        f = tmp_path / "mymod.py"
        f.write_text("import sys as system\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        assert any(r["target_qname"] == "sys" for r in imports)

    def test_from_import_emitted(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """from pathlib import Path -> target qname is 'pathlib'."""
        f = tmp_path / "mymod.py"
        f.write_text("from pathlib import Path\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        assert any(r["target_qname"] == "pathlib" for r in imports)

    def test_import_has_unresolved_confidence(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """External module imports get confidence=0.5 and resolution=unresolved."""
        f = tmp_path / "mymod.py"
        f.write_text("import os\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        for rel in imports:
            assert rel["confidence"] == 0.5
            assert rel["metadata"].get("resolution") == "unresolved"

    def test_import_source_is_module(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        f = tmp_path / "mymod.py"
        f.write_text("import os\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        for rel in imports:
            assert rel["source_qname"] == "mymod"
            assert rel.get("source_entity_type") == "Module"
            assert rel.get("target_entity_type") == "Module"

    def test_from_dotted_import(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """from os.path import join -> target qname 'os.path'."""
        f = tmp_path / "mymod.py"
        f.write_text("from os.path import join\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        assert any(r["target_qname"] == "os.path" for r in imports)


# ---------------------------------------------------------------------------
# Relation tests: CALLS
# ---------------------------------------------------------------------------


class TestCalls:
    def test_simple_call_resolved_same_module(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """main() calls helper() -> resolved to pkg.greeter.helper, confidence=1.0."""
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        main_calls = [r for r in calls if r["source_qname"] == "pkg.greeter.main"]
        helper_call = next(
            (r for r in main_calls if r["target_qname"] == "pkg.greeter.helper"),
            None,
        )
        assert helper_call is not None
        assert helper_call["confidence"] == 1.0

    def test_constructor_call_resolved(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """main() calls Greeter() -> resolved to pkg.greeter.Greeter."""
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        main_calls = [r for r in calls if r["source_qname"] == "pkg.greeter.main"]
        greeter_call = next(
            (r for r in main_calls if r["target_qname"] == "pkg.greeter.Greeter"),
            None,
        )
        assert greeter_call is not None

    def test_unresolved_call_confidence_half(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Call to unknown identifier -> confidence=0.5 with resolution=unresolved."""
        f = tmp_path / "mymod.py"
        f.write_text("def foo():\n    unknown_func()\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        foo_calls = [r for r in calls if r["source_qname"] == "mymod.foo"]
        assert len(foo_calls) >= 1
        unresolved = [c for c in foo_calls if c["metadata"].get("resolution") == "unresolved"]
        assert len(unresolved) >= 1
        assert unresolved[0]["confidence"] == 0.5

    def test_call_has_site_metadata(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """CALLS relations include line/column metadata."""
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        for rel in calls:
            meta = rel["metadata"]
            assert "call_line" in meta
            assert "call_column" in meta

    def test_call_has_entity_type_fields(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        for rel in calls:
            assert "source_entity_type" in rel
            assert "target_entity_type" in rel


# ---------------------------------------------------------------------------
# Non-Python behavior unchanged
# ---------------------------------------------------------------------------


class TestNonPythonUnchanged:
    @pytest.mark.asyncio
    async def test_unsupported_extension_still_returns_empty(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        from workflows_mcp.engine.executors_treesitter import (
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
    async def test_typescript_file_still_returns_file_module_only(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Non-Python languages still only get File+Module entities (Phase 1-2)."""
        from workflows_mcp.engine.executors_treesitter import (
            TreeSitterExecutor,
            TreeSitterInput,
        )

        f = tmp_path / "index.ts"
        f.write_text("const x = 1;\n", encoding="utf-8")
        executor = TreeSitterExecutor()
        inputs = TreeSitterInput(path=str(f), repo_relative_path="index.ts")
        result = await executor.execute(inputs, mock_execution)
        assert result.language == "typescript"
        assert len(result.entities) == 2
        types = {e["entity_type"] for e in result.entities}
        assert types == {"File", "Module"}


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_output_is_deterministic(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """Running the extractor twice yields identical results."""
        result1 = _run(full_python_file, "pkg/greeter.py", mock_execution)
        result2 = _run(full_python_file, "pkg/greeter.py", mock_execution)

        qnames1 = [e["qualified_name"] for e in result1.entities]
        qnames2 = [e["qualified_name"] for e in result2.entities]
        assert qnames1 == qnames2

        rel_types1 = sorted(
            (r["relation_type"], r["source_qname"], r["target_qname"])
            for r in result1.relations
        )
        rel_types2 = sorted(
            (r["relation_type"], r["source_qname"], r["target_qname"])
            for r in result2.relations
        )
        assert rel_types1 == rel_types2


# ---------------------------------------------------------------------------
# Spec-review gap fixes (Phase 3.1)
# ---------------------------------------------------------------------------


class TestFileAndModuleNameField:
    """Items 1 & 4: `name` field on File/Module; non-empty top-level __init__ qname."""

    def test_file_entity_has_name_field(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """File entity must have a `name` key equal to the path basename."""
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        file_entity = next(e for e in result.entities if e["entity_type"] == "File")
        assert "name" in file_entity
        assert file_entity["name"] == "greeter.py"

    def test_module_entity_has_name_field(
        self, full_python_file: Path, mock_execution: MagicMock
    ) -> None:
        """Module entity must have a `name` key equal to the last dotted component."""
        result = _run(full_python_file, "pkg/greeter.py", mock_execution)
        module_entity = next(e for e in result.entities if e["entity_type"] == "Module")
        assert "name" in module_entity
        assert module_entity["name"] == "greeter"

    def test_module_name_for_init_py_is_package_name(
        self, init_py_file: Path, mock_execution: MagicMock
    ) -> None:
        """pkg/__init__.py -> module qname=pkg, module name=pkg (not empty)."""
        result = _run(init_py_file, "pkg/__init__.py", mock_execution)
        module_entity = next(e for e in result.entities if e["entity_type"] == "Module")
        assert module_entity["qualified_name"] == "pkg"
        assert module_entity["name"] == "pkg"

    def test_top_level_init_py_qname_is_not_empty(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Top-level __init__.py (repo root) must produce a non-empty module qname."""
        f = tmp_path / "__init__.py"
        f.write_text("", encoding="utf-8")
        result = _run(f, "__init__.py", mock_execution)
        module_entity = next(e for e in result.entities if e["entity_type"] == "Module")
        assert module_entity["qualified_name"] != ""
        assert module_entity["name"] != ""

    def test_file_name_for_init_py_is_literal_filename(
        self, init_py_file: Path, mock_execution: MagicMock
    ) -> None:
        """File entity name is always the literal basename of the file path."""
        result = _run(init_py_file, "pkg/__init__.py", mock_execution)
        file_entity = next(e for e in result.entities if e["entity_type"] == "File")
        assert file_entity["name"] == "__init__.py"


class TestUnresolvedImports:
    """Item 2: unresolved_imports populated from Python IMPORTS relations."""

    def test_unresolved_imports_populated_for_python(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """unresolved_imports must list target qnames of IMPORTS relations."""
        f = tmp_path / "mymod.py"
        f.write_text("import os\nimport sys\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        assert "os" in result.unresolved_imports
        assert "sys" in result.unresolved_imports

    def test_unresolved_imports_deduplicated(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Duplicate import targets must appear only once."""
        f = tmp_path / "mymod.py"
        f.write_text("import os\nimport os\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        assert result.unresolved_imports.count("os") == 1

    def test_unresolved_imports_sorted(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """unresolved_imports must be in sorted order for determinism."""
        f = tmp_path / "mymod.py"
        f.write_text("import sys\nimport os\nimport pathlib\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        assert result.unresolved_imports == sorted(result.unresolved_imports)

    def test_unresolved_imports_empty_for_no_imports(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        f = tmp_path / "mymod.py"
        f.write_text("x = 1\n", encoding="utf-8")
        result = _run(f, "mymod.py", mock_execution)
        assert result.unresolved_imports == []


class TestRelativeImportNormalization:
    """Item 5: relative import qnames resolved from module_qname context."""

    def test_single_dot_sibling_import(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """from .sibling import x in pkg.mod -> target qname pkg.sibling."""
        f = tmp_path / "mod.py"
        f.write_text("from .sibling import foo\n", encoding="utf-8")
        result = _run(f, "pkg/mod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        assert any(r["target_qname"] == "pkg.sibling" for r in imports)

    def test_single_dot_bare_import(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """from . import foo in pkg.mod -> target qname pkg."""
        f = tmp_path / "mod.py"
        f.write_text("from . import foo\n", encoding="utf-8")
        result = _run(f, "pkg/mod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        assert any(r["target_qname"] == "pkg" for r in imports)

    def test_double_dot_sibling_import(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """from ..parent import x in pkg.sub.mod -> target qname pkg.parent."""
        f = tmp_path / "mod.py"
        f.write_text("from ..parent import x\n", encoding="utf-8")
        result = _run(f, "pkg/sub/mod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        assert any(r["target_qname"] == "pkg.parent" for r in imports)

    def test_relative_import_resolved_has_lower_confidence(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Resolved relative imports still get confidence=0.5 (cross-file unverified)."""
        f = tmp_path / "mod.py"
        f.write_text("from .sibling import foo\n", encoding="utf-8")
        result = _run(f, "pkg/mod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        sibling_import = next(
            r for r in imports if r["target_qname"] == "pkg.sibling"
        )
        assert sibling_import["confidence"] == 0.5

    def test_unresolvable_relative_import_kept_as_best_effort(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """from . import foo in a top-level module (no package) -> kept, not dropped."""
        f = tmp_path / "mod.py"
        # repo_relative_path has no package prefix
        f.write_text("from . import foo\n", encoding="utf-8")
        result = _run(f, "mod.py", mock_execution)
        imports = [r for r in result.relations if r["relation_type"] == "IMPORTS"]
        # Must still emit at least one IMPORTS relation, even if qname is best-effort
        assert len(imports) >= 1


class TestNestedCallExtraction:
    """Item 6: nested calls in argument lists are emitted as separate CALLS relations."""

    def test_nested_call_emits_both_calls(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """bar(baz()) in foo() -> both bar and baz appear as CALLS from foo."""
        f = tmp_path / "mymod.py"
        f.write_text(
            "def foo():\n    bar(baz())\n\ndef bar(x): pass\ndef baz(): pass\n",
            encoding="utf-8",
        )
        result = _run(f, "mymod.py", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        foo_calls = [r for r in calls if r["source_qname"] == "mymod.foo"]
        targets = {r["target_qname"] for r in foo_calls}
        assert "mymod.bar" in targets
        assert "mymod.baz" in targets

    def test_nested_call_inner_resolved_when_known(
        self, tmp_path: Path, mock_execution: MagicMock
    ) -> None:
        """Inner call baz() that is a known same-module function resolves at confidence=1.0."""
        f = tmp_path / "mymod.py"
        f.write_text(
            "def foo():\n    bar(baz())\n\ndef bar(x): pass\ndef baz(): pass\n",
            encoding="utf-8",
        )
        result = _run(f, "mymod.py", mock_execution)
        calls = [r for r in result.relations if r["relation_type"] == "CALLS"]
        baz_call = next(
            r for r in calls
            if r["source_qname"] == "mymod.foo" and r["target_qname"] == "mymod.baz"
        )
        assert baz_call["confidence"] == 1.0
