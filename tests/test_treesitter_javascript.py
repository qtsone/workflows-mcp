"""TDD tests for JavaScript/JSX structural extraction via TreeSitterExecutor.

Tests written BEFORE production implementation (TDD RED phase).
Covers entities: File, Module, Class, Method, Function.
Covers relations: CONTAINS, INHERITS_FROM, IMPORTS, CALLS.

Grammar notes verified against tree-sitter-javascript 0.25:
- class_declaration: field name='name' is an identifier (not type_identifier like TS)
- class_heritage: direct identifier child (no extends_clause wrapper like TS)
- method_definition: field name='name' is property_identifier
- function_declaration: field name='name' is identifier
- import_statement: same shape as TypeScript grammar
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

FULL_JS_SRC = """\
import React from 'react';
import { useState, useEffect } from 'react';
import * as utils from './utils';

class Animal {
  constructor(name) {
    this.name = name;
  }
  speak() {
    console.log('roar');
  }
}

class Dog extends Animal {
  bark() {
    this.speak();
    greet();
  }
}

function greet() {
  console.log('hello');
}

function helper() {
  return greet();
}
"""

JSX_SRC = """\
import React from 'react';

class MyComponent {
  render() {
    return 'hello';
  }
}

function App() {
  return 'world';
}
"""

SIMPLE_IMPORT_JS_SRC = """\
import React from 'react';
import { useState } from 'react';
import * as lodash from 'lodash';
import './side-effect';
"""

EMPTY_JS_SRC = """\
// empty file
"""

EXTERNAL_INHERIT_JS_SRC = """\
import { BaseClass } from 'external-lib';

class MyClass extends BaseClass {
  method() {
    console.log('hi');
  }
}
"""

SAME_MODULE_INHERIT_JS_SRC = """\
class Base {
  baseMethod() {}
}

class Child extends Base {
  childMethod() {}
}
"""

ARROW_FN_JS_SRC = """\
const arrowFn = (x) => x + 1;

function named() {
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


def test_js_file_detected_as_javascript(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_JS_SRC, "empty.js", "src/empty.js")
    assert result.language == "javascript"


def test_jsx_file_detected_as_javascript(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, JSX_SRC, "comp.jsx", "src/comp.jsx")
    assert result.language == "javascript"


def test_mjs_file_detected_as_javascript(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_JS_SRC, "mod.mjs", "src/mod.mjs")
    assert result.language == "javascript"


def test_cjs_file_detected_as_javascript(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_JS_SRC, "mod.cjs", "src/mod.cjs")
    assert result.language == "javascript"


# ---------------------------------------------------------------------------
# Test: Module qualified name
# ---------------------------------------------------------------------------


def test_module_qname_from_repo_relative(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "greeter.js", "src/services/greeter.js")
    assert result.module_qualified_name == "services.greeter"


def test_module_qname_no_empty(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_JS_SRC, "mod.js", "mod.js")
    assert result.module_qualified_name != ""
    assert result.module_qualified_name == "mod"


def test_module_qname_strips_src_prefix(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_JS_SRC, "utils.js", "src/utils.js")
    assert result.module_qualified_name == "utils"


# ---------------------------------------------------------------------------
# Test: Entities — File and Module
# ---------------------------------------------------------------------------


def test_file_entity_present(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_JS_SRC, "mod.js", "src/mod.js")
    files = _find_entities(result.entities, "File")
    assert len(files) == 1
    file_e = files[0]
    assert file_e["entity_type"] == "File"
    assert "stable_id" in file_e
    assert "qualified_name" in file_e


def test_module_entity_present(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EMPTY_JS_SRC, "mod.js", "src/mod.js")
    modules = _find_entities(result.entities, "Module")
    assert len(modules) == 1
    mod = modules[0]
    assert mod["qualified_name"] == "mod"
    assert mod["name"] == "mod"


# ---------------------------------------------------------------------------
# Test: Entities — Classes and Methods
# ---------------------------------------------------------------------------


def test_class_entities_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    class_names = _entity_names(result.entities, "Class")
    assert "Animal" in class_names
    assert "Dog" in class_names


def test_class_has_required_fields(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    classes = _find_entities(result.entities, "Class")
    animal = next(c for c in classes if c["name"] == "Animal")
    assert animal["entity_type"] == "Class"
    assert animal["qualified_name"] == "animals.Animal"
    assert "stable_id" in animal
    assert "metadata" in animal
    assert animal["confidence"] == 1.0


def test_method_entities_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    method_names = _entity_names(result.entities, "Method")
    assert "speak" in method_names
    assert "bark" in method_names


def test_method_constructor_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    method_names = _entity_names(result.entities, "Method")
    assert "constructor" in method_names


def test_method_qname_qualified(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    methods = _find_entities(result.entities, "Method")
    speak = next(m for m in methods if m["name"] == "speak")
    assert speak["qualified_name"] == "animals.Animal.speak"


def test_method_has_start_line(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    methods = _find_entities(result.entities, "Method")
    for m in methods:
        assert "start_line" in m["metadata"]
        assert m["metadata"]["start_line"] >= 1


# ---------------------------------------------------------------------------
# Test: Entities — Functions
# ---------------------------------------------------------------------------


def test_function_entities_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    func_names = _entity_names(result.entities, "Function")
    assert "greet" in func_names
    assert "helper" in func_names


def test_function_qname_qualified(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    functions = _find_entities(result.entities, "Function")
    greet = next(f for f in functions if f["name"] == "greet")
    assert greet["qualified_name"] == "animals.greet"
    assert greet["confidence"] == 1.0


def test_arrow_function_not_extracted_as_function(tmp_path: Path) -> None:
    """Arrow function assigned to const is NOT extracted as a Function entity.

    Limitation documented: only function_declaration nodes are extracted.
    Arrow functions (lexical_declaration with arrow_function) are skipped.
    """
    result = _run_executor(tmp_path, ARROW_FN_JS_SRC, "arr.js", "src/arr.js")
    func_names = _entity_names(result.entities, "Function")
    assert "named" in func_names
    assert "arrowFn" not in func_names


# ---------------------------------------------------------------------------
# Test: Entity ordering
# ---------------------------------------------------------------------------


def test_entity_ordering(tmp_path: Path) -> None:
    """File, Module, then Classes, Methods, Functions in deterministic order."""
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    types = [e["entity_type"] for e in result.entities]
    assert types[0] == "File"
    assert types[1] == "Module"
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
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    contains = _find_relations(result.relations, "CONTAINS")
    file_module = [
        r
        for r in contains
        if r["source_entity_type"] == "File" and r["target_entity_type"] == "Module"
    ]
    assert len(file_module) == 1


def test_contains_module_class(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
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
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
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
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
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
    result = _run_executor(tmp_path, SAME_MODULE_INHERIT_JS_SRC, "inherit.js", "src/inherit.js")
    inherits = _find_relations(result.relations, "INHERITS_FROM")
    assert len(inherits) == 1
    rel = inherits[0]
    assert rel["source_qname"] == "inherit.Child"
    assert rel["target_qname"] == "inherit.Base"
    assert rel["confidence"] == 1.0


def test_inherits_from_external_unresolved(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, EXTERNAL_INHERIT_JS_SRC, "myclass.js", "src/myclass.js")
    inherits = _find_relations(result.relations, "INHERITS_FROM")
    assert len(inherits) == 1
    rel = inherits[0]
    assert rel["source_qname"] == "myclass.MyClass"
    assert rel["confidence"] == 0.5
    assert rel["metadata"].get("resolution") == "unresolved"


def test_inherits_from_full_js(tmp_path: Path) -> None:
    """Dog extends Animal — both in same module, confidence 1.0."""
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    inherits = _find_relations(result.relations, "INHERITS_FROM")
    dog_inherits = [r for r in inherits if r["source_qname"] == "animals.Dog"]
    assert len(dog_inherits) == 1
    assert dog_inherits[0]["target_qname"] == "animals.Animal"
    assert dog_inherits[0]["confidence"] == 1.0


# ---------------------------------------------------------------------------
# Test: Relations — IMPORTS
# ---------------------------------------------------------------------------


def test_imports_from_default_import(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_JS_SRC, "importer.js", "src/importer.js")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "react" in targets


def test_imports_from_named_import(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_JS_SRC, "importer.js", "src/importer.js")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "react" in targets


def test_imports_star_import(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_JS_SRC, "importer.js", "src/importer.js")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "lodash" in targets


def test_imports_side_effect(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_JS_SRC, "importer.js", "src/importer.js")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "./side-effect" in targets


def test_imports_confidence_unresolved(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_JS_SRC, "importer.js", "src/importer.js")
    imports = _find_relations(result.relations, "IMPORTS")
    for imp in imports:
        assert imp["confidence"] == 0.5
        assert imp["metadata"].get("resolution") == "unresolved"


def test_imports_source_is_module(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_JS_SRC, "importer.js", "src/importer.js")
    imports = _find_relations(result.relations, "IMPORTS")
    for imp in imports:
        assert imp["source_entity_type"] == "Module"
        assert imp["target_entity_type"] == "Module"


def test_unresolved_imports_populated(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, SIMPLE_IMPORT_JS_SRC, "importer.js", "src/importer.js")
    assert len(result.unresolved_imports) > 0
    assert "react" in result.unresolved_imports


# ---------------------------------------------------------------------------
# Test: Relations — CALLS
# ---------------------------------------------------------------------------


def test_calls_same_module_function(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    calls = _find_relations(result.relations, "CALLS")
    # helper() calls greet()
    helper_calls = [r for r in calls if r["source_qname"] == "animals.helper"]
    assert any(r["target_qname"] == "animals.greet" for r in helper_calls)


def test_calls_resolved_confidence_1(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    calls = _find_relations(result.relations, "CALLS")
    resolved_calls = [
        r for r in calls if r["target_qname"] == "animals.greet" and r["confidence"] == 1.0
    ]
    assert len(resolved_calls) >= 1


def test_calls_unresolved_external(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    calls = _find_relations(result.relations, "CALLS")
    # console.log is a member_expression — unresolved
    unresolved = [r for r in calls if r["confidence"] == 0.5]
    assert len(unresolved) > 0


def test_calls_has_call_line_metadata(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, FULL_JS_SRC, "animals.js", "src/animals.js")
    calls = _find_relations(result.relations, "CALLS")
    for call in calls:
        assert "call_line" in call["metadata"]
        assert call["metadata"]["call_line"] >= 1


def test_calls_target_entity_type_class_for_constructor(tmp_path: Path) -> None:
    """new KnownClass() sets target_entity_type='Class' with confidence 1.0."""
    src = """\
class Foo {
  doSomething() {}
}

function factory() {
  return new Foo();
}
"""
    result = _run_executor(tmp_path, src, "factory.js", "src/factory.js")
    calls = _find_relations(result.relations, "CALLS")
    foo_calls = [r for r in calls if r["target_qname"] == "factory.Foo"]
    assert any(r["target_entity_type"] == "Class" for r in foo_calls)


def test_new_expression_on_known_function_is_unresolved(tmp_path: Path) -> None:
    """new greet() where greet is a known Function must be confidence 0.5 + unresolved.

    Mirrors the TS test to ensure the same bug cannot appear in the JS extractor:
    a `new` applied to a Function entity must NOT get confidence=1.0.
    """
    src = """\
function greet() {
  console.log('hello');
}

function factory() {
  new greet();
}
"""
    result = _run_executor(tmp_path, src, "mod.js", "src/mod.js")
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
# Test: JSX extraction
# ---------------------------------------------------------------------------


def test_jsx_class_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, JSX_SRC, "comp.jsx", "src/comp.jsx")
    class_names = _entity_names(result.entities, "Class")
    assert "MyComponent" in class_names


def test_jsx_function_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, JSX_SRC, "comp.jsx", "src/comp.jsx")
    func_names = _entity_names(result.entities, "Function")
    assert "App" in func_names


def test_jsx_method_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, JSX_SRC, "comp.jsx", "src/comp.jsx")
    method_names = _entity_names(result.entities, "Method")
    assert "render" in method_names


def test_jsx_imports_extracted(tmp_path: Path) -> None:
    result = _run_executor(tmp_path, JSX_SRC, "comp.jsx", "src/comp.jsx")
    imports = _find_relations(result.relations, "IMPORTS")
    targets = {r["target_qname"] for r in imports}
    assert "react" in targets


def test_jsx_does_not_crash(tmp_path: Path) -> None:
    """Actual JSX markup in a .jsx file must not crash the extractor."""
    src = """\
import React from 'react';

class Widget {
  render() {
    return <div className="widget"><span>Hello</span></div>;
  }
}

function App() {
  return <Widget />;
}
"""
    result = _run_executor(tmp_path, src, "widget.jsx", "src/widget.jsx")
    assert result.language == "javascript"
    assert "Widget" in _entity_names(result.entities, "Class")
    assert "App" in _entity_names(result.entities, "Function")


# ---------------------------------------------------------------------------
# Test: Non-JS behavior unchanged
# ---------------------------------------------------------------------------


def test_non_js_languages_still_return_unsupported(tmp_path: Path) -> None:
    """Adding JavaScript support must not affect other unsupported extensions."""
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


MULTI_IMPORT_SAME_SOURCE_JS_SRC = """\
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
        MULTI_IMPORT_SAME_SOURCE_JS_SRC,
        "dedup.js",
        "src/dedup.js",
    )
    imports = _find_relations(result.relations, "IMPORTS")
    targets = [r["target_qname"] for r in imports]
    react_count = targets.count("react")
    assert react_count == 1, (
        f"Expected exactly 1 IMPORTS relation to 'react', got {react_count}. "
        f"All import targets: {targets}"
    )
