"""TDD tests for TypeScript/TSX structural extraction via TreeSitterExecutor.

Tests written BEFORE production implementation (TDD RED phase).
Covers entities: File, Module, Class, Method, Function.
Covers relations: CONTAINS, INHERITS_FROM, IMPORTS, CALLS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

FULL_TS_SRC = """\
import React from 'react';
import { useState, useEffect } from 'react';
import type { FC } from 'react';
import * as utils from './utils';

class Animal {
  name: string;
  constructor(name: string) {
    this.name = name;
  }
  speak(): void {
    console.log('roar');
  }
}

class Dog extends Animal {
  bark(): void {
    this.speak();
    greet();
  }
}

function greet(): void {
  console.log('hello');
}

function helper(): string {
  return greet();
}
"""

TSX_SRC = """\
import React from 'react';

class MyComponent {
  render(): string {
    return 'hello';
  }
}

function App(): string {
  return 'world';
}
"""

INTERFACE_TS_SRC = """\
interface Greeter {
  greet(name: string): string;
}

class SimpleGreeter {
  greet(name: string): string {
    return name;
  }
}
"""

SIMPLE_IMPORT_TS_SRC = """\
import React from 'react';
import { useState } from 'react';
import * as lodash from 'lodash';
import './side-effect';
"""

EMPTY_TS_SRC = """\
// empty file
"""

EXTERNAL_INHERIT_TS_SRC = """\
import { BaseClass } from 'external-lib';

class MyClass extends BaseClass {
  method(): void {
    console.log('hi');
  }
}
"""

SAME_MODULE_INHERIT_TS_SRC = """\
class Base {
  baseMethod(): void {}
}

class Child extends Base {
  childMethod(): void {}
}
"""

ARROW_FN_TS_SRC = """\
const arrowFn = (x: number): number => x + 1;

function named(): void {
  console.log('named');
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_entities(entities: list[dict[str, Any]], entity_type: str) -> list[dict[str, Any]]:
    return [e for e in entities if e["entity_type"] == entity_type]


def _find_relations(relations: list[dict[str, Any]], relation_type: str) -> list[dict[str, Any]]:
    return [r for r in relations if r["relation_type"] == relation_type]


def _entity_names(entities: list[dict[str, Any]], entity_type: str) -> list[str]:
    return [e["name"] for e in _find_entities(entities, entity_type)]


def _run_executor(
    tmp_path: Path,
    src: str,
    filename: str,
    repo_rel: str | None = None,
) -> Any:
    """Write src to a temp file and execute TreeSitterExecutor synchronously."""
    import asyncio

    from workflows_mcp.code_intelligence.executors_treesitter import (
        TreeSitterExecutor,
        TreeSitterInput,
    )

    f = tmp_path / filename
    f.write_text(src, encoding="utf-8")

    inp = TreeSitterInput(
        path=str(f),
        repo_relative_path=repo_rel,
        palace="test-palace",
        item_id="test-item",
    )
    executor = TreeSitterExecutor()
    ctx = MagicMock()

    return asyncio.run(executor.execute(inp, ctx))


# ---------------------------------------------------------------------------
# Test: language detection
# ---------------------------------------------------------------------------


def test_ts_file_detected_as_typescript(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_TS_SRC, "empty.ts", "src/empty.ts")
    assert result.language == "typescript"


def test_tsx_file_detected_as_tsx(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, TSX_SRC, "comp.tsx", "src/comp.tsx")
    assert result.language == "tsx"


# ---------------------------------------------------------------------------
# Test: Module qualified name
# ---------------------------------------------------------------------------


def test_module_qname_from_repo_relative(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "greeter.ts", "src/services/greeter.ts")
    assert result.module_qualified_name == "services.greeter"


def test_module_qname_no_empty(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_TS_SRC, "mod.ts", "mod.ts")
    assert result.module_qualified_name != ""
    assert result.module_qualified_name == "mod"


def test_module_qname_strips_src_prefix(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_TS_SRC, "utils.ts", "src/utils.ts")
    assert result.module_qualified_name == "utils"


# ---------------------------------------------------------------------------
# Test: Entities — File and Module
# ---------------------------------------------------------------------------


def test_file_entity_present(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_TS_SRC, "mod.ts", "src/mod.ts")
    files = _find_entities(result.entities, "File")
    assert len(files) == 1
    file_e = files[0]
    assert file_e["entity_type"] == "File"
    assert "stable_id" in file_e
    assert "qualified_name" in file_e


def test_module_entity_present(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_TS_SRC, "mod.ts", "src/mod.ts")
    modules = _find_entities(result.entities, "Module")
    assert len(modules) == 1
    mod = modules[0]
    assert mod["qualified_name"] == "mod"
    assert mod["name"] == "mod"


# ---------------------------------------------------------------------------
# Test: Entities — Classes and Methods
# ---------------------------------------------------------------------------


def test_class_entities_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    class_names = _entity_names(result.entities, "Class")
    assert "Animal" in class_names
    assert "Dog" in class_names


def test_class_has_required_fields(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    classes = _find_entities(result.entities, "Class")
    animal = next(c for c in classes if c["name"] == "Animal")
    assert animal["entity_type"] == "Class"
    assert animal["qualified_name"] == "animals.Animal"
    assert "stable_id" in animal
    assert "metadata" in animal
    assert animal["confidence"] == 1.0


def test_method_entities_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    method_names = _entity_names(result.entities, "Method")
    assert "speak" in method_names
    assert "bark" in method_names


def test_method_constructor_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    method_names = _entity_names(result.entities, "Method")
    assert "constructor" in method_names


def test_method_qname_qualified(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    methods = _find_entities(result.entities, "Method")
    speak = next(m for m in methods if m["name"] == "speak")
    assert speak["qualified_name"] == "animals.Animal.speak"


def test_method_has_start_line(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    methods = _find_entities(result.entities, "Method")
    for m in methods:
        assert "start_line" in m["metadata"]
        assert m["metadata"]["start_line"] >= 1


# ---------------------------------------------------------------------------
# Test: Entities — Functions
# ---------------------------------------------------------------------------


def test_function_entities_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    func_names = _entity_names(result.entities, "Function")
    assert "greet" in func_names
    assert "helper" in func_names


def test_function_qname_qualified(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    functions = _find_entities(result.entities, "Function")
    greet = next(f for f in functions if f["name"] == "greet")
    assert greet["qualified_name"] == "animals.greet"
    assert greet["confidence"] == 1.0


def test_arrow_function_not_extracted_as_function(tmp_path: Path) -> None:
    """Arrow function assigned to const is not extracted as a Function entity."""
    result = _run_executor(tmp_path, ARROW_FN_TS_SRC, "arr.ts", "src/arr.ts")
    func_names = _entity_names(result.entities, "Function")
    # named function IS extracted; arrow functions are NOT
    assert "named" in func_names
    assert "arrowFn" not in func_names


# ---------------------------------------------------------------------------
# Test: Entity ordering
# ---------------------------------------------------------------------------


def test_entity_ordering(tmp_path: Path) -> None:
    """File, Module, then Classes, Methods, Functions in deterministic order."""
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    types = [e["entity_type"] for e in result.entities]
    assert types[0] == "File"
    assert types[1] == "Module"
    # All Class entities appear before Method and Function entities
    class_indices = [i for i, t in enumerate(types) if t == "Class"]
    method_indices = [i for i, t in enumerate(types) if t == "Method"]
    function_indices = [i for i, t in enumerate(types) if t == "Function"]
    if class_indices and method_indices:
        assert max(class_indices) < min(method_indices)
    if method_indices and function_indices:
        assert max(method_indices) < min(function_indices)


# ---------------------------------------------------------------------------
# Test: Relations — CONTAINS
# ---------------------------------------------------------------------------


def test_contains_file_module(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    contains = _find_relations(result.relations, "CONTAINS")
    file_module = [
        r
        for r in contains
        if r["source_entity_type"] == "File" and r["target_entity_type"] == "Module"
    ]
    assert len(file_module) == 1


def test_contains_module_class(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    contains = _find_relations(result.relations, "CONTAINS")
    module_class = [
        r
        for r in contains
        if r["source_entity_type"] == "Module" and r["target_entity_type"] == "Class"
    ]
    class_targets = {r["target_qname"] for r in module_class}
    assert "animals.Animal" in class_targets
    assert "animals.Dog" in class_targets


def test_contains_module_function(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    contains = _find_relations(result.relations, "CONTAINS")
    module_fn = [
        r
        for r in contains
        if r["source_entity_type"] == "Module" and r["target_entity_type"] == "Function"
    ]
    fn_targets = {r["target_qname"] for r in module_fn}
    assert "animals.greet" in fn_targets
    assert "animals.helper" in fn_targets


def test_contains_class_method(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    contains = _find_relations(result.relations, "CONTAINS")
    class_method = [
        r
        for r in contains
        if r["source_entity_type"] == "Class" and r["target_entity_type"] == "Method"
    ]
    targets = {r["target_qname"] for r in class_method}
    assert "animals.Animal.speak" in targets
    assert "animals.Dog.bark" in targets


# ---------------------------------------------------------------------------
# Test: Relations — INHERITS_FROM
# ---------------------------------------------------------------------------


def test_inherits_from_same_module(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SAME_MODULE_INHERIT_TS_SRC, "inherit.ts", "src/inherit.ts")
    inherits = _find_relations(result.relations, "INHERITS_FROM")
    assert len(inherits) == 1
    rel = inherits[0]
    assert rel["source_qname"] == "inherit.Child"
    assert rel["target_qname"] == "inherit.Base"
    assert rel["confidence"] == 1.0


def test_inherits_from_external_unresolved(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EXTERNAL_INHERIT_TS_SRC, "myclass.ts", "src/myclass.ts")
    inherits = _find_relations(result.relations, "INHERITS_FROM")
    assert len(inherits) == 1
    rel = inherits[0]
    assert rel["source_qname"] == "myclass.MyClass"
    assert rel["confidence"] == 0.5
    assert rel["metadata"].get("resolution") == "unresolved"


def test_inherits_from_full_ts(tmp_path: Path) -> None:
    """Dog extends Animal — both in same module."""
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    inherits = _find_relations(result.relations, "INHERITS_FROM")
    dog_inherits = [r for r in inherits if r["source_qname"] == "animals.Dog"]
    assert len(dog_inherits) == 1
    assert dog_inherits[0]["target_qname"] == "animals.Animal"
    assert dog_inherits[0]["confidence"] == 1.0


# ---------------------------------------------------------------------------
# Test: Relations — IMPORTS
# ---------------------------------------------------------------------------


def test_imports_from_default_import(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_TS_SRC, "importer.ts", "src/importer.ts")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "react" in targets


def test_imports_from_named_import(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_TS_SRC, "importer.ts", "src/importer.ts")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "react" in targets


def test_imports_star_import(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_TS_SRC, "importer.ts", "src/importer.ts")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "lodash" in targets


def test_imports_side_effect(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_TS_SRC, "importer.ts", "src/importer.ts")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "./side-effect" in targets


def test_imports_confidence_unresolved(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_TS_SRC, "importer.ts", "src/importer.ts")
    imports = _find_relations(result.relations, "IMPORTS")
    for imp in imports:
        assert imp["confidence"] == 0.5
        assert imp["metadata"].get("resolution") == "unresolved"


def test_imports_source_is_module(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_TS_SRC, "importer.ts", "src/importer.ts")
    imports = _find_relations(result.relations, "IMPORTS")
    for imp in imports:
        assert imp["source_entity_type"] == "Module"
        assert imp["target_entity_type"] == "Module"


def test_unresolved_imports_populated(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_TS_SRC, "importer.ts", "src/importer.ts")
    assert len(result.unresolved_imports) > 0
    assert "react" in result.unresolved_imports


# ---------------------------------------------------------------------------
# Test: Relations — CALLS
# ---------------------------------------------------------------------------


def test_calls_same_module_function(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    calls = _find_relations(result.relations, "CALLS")
    # helper() calls greet()
    helper_calls = [r for r in calls if r["source_qname"] == "animals.helper"]
    assert any(r["target_qname"] == "animals.greet" for r in helper_calls)


def test_calls_resolved_confidence_1(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    calls = _find_relations(result.relations, "CALLS")
    resolved_calls = [
        r for r in calls if r["target_qname"] == "animals.greet" and r["confidence"] == 1.0
    ]
    assert len(resolved_calls) >= 1


def test_calls_unresolved_external(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    calls = _find_relations(result.relations, "CALLS")
    # console.log is a member expression — unresolved
    unresolved = [r for r in calls if r["confidence"] == 0.5]
    assert len(unresolved) > 0


def test_calls_has_call_line_metadata(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_TS_SRC, "animals.ts", "src/animals.ts")
    calls = _find_relations(result.relations, "CALLS")
    for call in calls:
        assert "call_line" in call["metadata"]
        assert call["metadata"]["call_line"] >= 1


def test_calls_target_entity_type_class_for_constructor(tmp_path: Path) -> None:
    """Calling a same-module class (constructor call) sets target_entity_type='Class'."""
    src = """\
class Foo {
  doSomething(): void {}
}

function factory(): Foo {
  return new Foo();
}
"""
    result = _run_executor(tmp_path, src, "factory.ts", "src/factory.ts")
    calls = _find_relations(result.relations, "CALLS")
    foo_calls = [r for r in calls if r["target_qname"] == "factory.Foo"]
    assert any(r["target_entity_type"] == "Class" for r in foo_calls)


def test_new_expression_on_known_function_is_unresolved(tmp_path: Path) -> None:
    """new greet() where greet is a known Function must be confidence 0.5 + unresolved.

    The bug: _emit_new set confidence=1.0 for any known same-module symbol,
    even when that symbol is a Function rather than a Class. A `new` expression
    applied to a Function is semantically unresolved — we cannot assert it is
    a constructor call with confidence 1.0.
    """
    src = """\
function greet(): void {
  console.log('hello');
}

function factory(): void {
  new greet();
}
"""
    result = _run_executor(tmp_path, src, "mod.ts", "src/mod.ts")
    calls = _find_relations(result.relations, "CALLS")
    new_greet_calls = [
        r for r in calls if r["target_qname"] == "mod.greet" and r["metadata"].get("is_constructor")
    ]
    assert len(new_greet_calls) == 1
    rel = new_greet_calls[0]
    assert rel["confidence"] == 0.5, (
        f"Expected confidence 0.5 for new-on-Function, got {rel['confidence']}"
    )
    assert rel["metadata"].get("resolution") == "unresolved", (
        f"Expected resolution='unresolved', got {rel['metadata']}"
    )


# ---------------------------------------------------------------------------
# Test: TSX extraction
# ---------------------------------------------------------------------------


def test_tsx_class_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, TSX_SRC, "comp.tsx", "src/comp.tsx")
    class_names = _entity_names(result.entities, "Class")
    assert "MyComponent" in class_names


def test_tsx_function_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, TSX_SRC, "comp.tsx", "src/comp.tsx")
    func_names = _entity_names(result.entities, "Function")
    assert "App" in func_names


def test_tsx_method_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, TSX_SRC, "comp.tsx", "src/comp.tsx")
    method_names = _entity_names(result.entities, "Method")
    assert "render" in method_names


def test_tsx_imports_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, TSX_SRC, "comp.tsx", "src/comp.tsx")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "react" in targets


def test_tsx_with_jsx_markup_does_not_crash(tmp_path: Path) -> None:
    """Actual JSX markup in a .tsx file must not crash the extractor."""
    src = """\
import React from 'react';

class Widget {
  render(): string {
    return 'hello';
  }
}

function App(): string {
  return '<div>hello</div>';
}
"""
    result = _run_executor(tmp_path, src, "widget.tsx", "src/widget.tsx")
    assert result.language == "tsx"
    assert "Widget" in _entity_names(result.entities, "Class")
    assert "App" in _entity_names(result.entities, "Function")


def test_import_type_extracted_as_import(tmp_path: Path) -> None:
    """import type { T } from 'pkg' should produce an IMPORTS relation."""
    src = """\
import type { Foo } from 'foo-pkg';
import type Bar from 'bar-pkg';
"""
    result = _run_executor(tmp_path, src, "types.ts", "src/types.ts")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "foo-pkg" in targets
    assert "bar-pkg" in targets


# ---------------------------------------------------------------------------
# Test: Interface handling
# ---------------------------------------------------------------------------


def test_interface_not_invented_as_new_entity_type(tmp_path: Path) -> None:
    """Interfaces should not introduce entity types outside Class/Method/Function."""
    result = _run_executor(tmp_path, INTERFACE_TS_SRC, "iface.ts", "src/iface.ts")
    entity_types = {e["entity_type"] for e in result.entities}
    allowed_types = {"File", "Module", "Class", "Method", "Function"}
    assert entity_types.issubset(allowed_types)


# ---------------------------------------------------------------------------
# Test: Non-TS languages unchanged
# ---------------------------------------------------------------------------


def test_non_ts_language_unchanged(tmp_path: Path) -> None:
    """Unsupported extensions still return language=unsupported."""
    import asyncio
    from unittest.mock import MagicMock

    from workflows_mcp.code_intelligence.executors_treesitter import (
        TreeSitterExecutor,
        TreeSitterInput,
    )

    f = tmp_path / "file.rb"
    f.write_text("puts 'hello'", encoding="utf-8")

    inp = TreeSitterInput(path=str(f))
    executor = TreeSitterExecutor()
    ctx = MagicMock()
    result = asyncio.run(executor.execute(inp, ctx))
    assert result.language == "unsupported"
    assert result.entities == []


# ---------------------------------------------------------------------------
# Test: IMPORTS deduplication (same target from multiple import statements)
# ---------------------------------------------------------------------------


MULTI_IMPORT_SAME_SOURCE_TS_SRC = """\
import React from 'react';
import { useState, useEffect } from 'react';
import type { FC } from 'react';
import * as lodash from 'lodash';
"""


def test_imports_no_duplicate_targets_for_same_source(tmp_path: Path) -> None:
    """Multiple import statements from the same source must emit exactly one IMPORTS relation.

    e.g. `import React from 'react'` and `import { useState } from 'react'`
    are two separate AST nodes but must produce a single IMPORTS -> 'react' relation.
    """
    result = _run_executor(
        tmp_path,
        MULTI_IMPORT_SAME_SOURCE_TS_SRC,
        "dedup.ts",
        "src/dedup.ts",
    )
    imports = _find_relations(result.relations, "IMPORTS")
    targets = [r["target_qname"] for r in imports]
    react_count = targets.count("react")
    assert react_count == 1, (
        f"Expected exactly 1 IMPORTS relation to 'react', got {react_count}. "
        f"All import targets: {targets}"
    )
