"""E2E integration test for the system1-scan built-in workflow.

ADR-013 TDD — tests verify the thin orchestration pipeline:
  parse -> one derive_system1_topology Memory operation -> outputs.

Contract tested (Task 6 reshape):
- system1-scan.yaml is a valid built-in workflow (loaded from
  src/workflows_mcp/templates/memory/).
- It calls TreeSitter parse then one Memory derive_system1_topology operation.
- Topology uses only Palace/Wing/Room/Compartment; corridor is never a topology level.
- The workflow exposes derived_wing, derived_room, derived_compartment,
  derivation_source, provenance_id, claim_id outputs.
- No static wing/room/compartment defaults appear in the workflow definition.
- topology_override (not wing_hint) is the override contract.

YAML structural regression tests (pure static, no DB):
- TreeSitter block must not declare topology fields.
- Only one Memory block exists and uses derive_system1_topology.
- No hardcoded topology defaults.
- Corridor is not a topology level in inputs, outputs, or scope.
- topology_override is present; wing_hint is absent.
- Required derived output fields are declared.

E2E integration tests (requires PostgreSQL + full Tasks 1-5 service impl):
- system1-scan workflow is registered.
- First run stores structural evidence and derives topology.
- Second run is idempotent (no new provenance rows on unchanged file).
- Corridor never appears as entity_type in evidence store.

Execution model: the workflow is executed via execute_workflow() MCP tool
with a mock AppContext pointing at the real PostgreSQL test database
(env: MEMORY_DB_HOST / MEMORY_DB_PORT / MEMORY_DB_NAME / MEMORY_DB_USER /
MEMORY_DB_PASSWORD). The same connection is used for DB assertions so
test setup/teardown are consistent.

Database isolation: all rows belonging to PALACE = "palace_system1_scan_test"
are wiped before and after each test function.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from workflows_mcp.context import AppContext
from workflows_mcp.engine.block import BlockInput, BlockOutput
from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.executor_base import BlockExecutor, create_default_registry
from workflows_mcp.engine.io_queue import IOQueue
from workflows_mcp.engine.job_queue import JobQueue
from workflows_mcp.engine.knowledge.schema import ensure_schema
from workflows_mcp.engine.llm_config import LLMConfigLoader
from workflows_mcp.engine.memory_service import MemoryService, QueryMemoryRequest
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.schema import WorkflowSchema
from workflows_mcp.engine.sql.backend import ConnectionConfig, DatabaseEngine
from workflows_mcp.engine.sql.postgres_backend import PostgresBackend
from workflows_mcp.memory_runtime import register_memory_executors
from workflows_mcp.tools import execute_workflow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PALACE = "palace_system1_scan_test"

# Simple Python fixture — produces File/Module/Class/Function structural evidence.
FIXTURE_PY_SRC = """\
import os


class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}"


def main() -> None:
    g = Greeter()
    g.greet("world")
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _make_config() -> ConnectionConfig:
    return ConnectionConfig(
        dialect=DatabaseEngine.POSTGRESQL,
        host=os.environ.get("MEMORY_DB_HOST", "localhost"),
        port=int(os.environ.get("MEMORY_DB_PORT", "5432")),
        database=os.environ.get("MEMORY_DB_NAME", "workflows"),
        username=os.environ.get("MEMORY_DB_USER", "workflows"),
        password=os.environ.get("MEMORY_DB_PASSWORD", "supersecret"),
    )


@pytest_asyncio.fixture
async def knowledge_backend() -> AsyncIterator[PostgresBackend]:
    backend = PostgresBackend()
    await backend.connect(_make_config())
    await ensure_schema(backend)
    try:
        yield backend
    finally:
        await backend.disconnect()


@pytest_asyncio.fixture
async def clean_palace(knowledge_backend: PostgresBackend) -> AsyncIterator[None]:
    """Wipe all rows for PALACE before and after each test."""

    async def _wipe() -> None:
        pattern = f"{PALACE}%"
        # ADR-013 provenance join table (references structural evidence — delete first)
        await knowledge_backend.execute(
            "DELETE FROM knowledge_topology_provenance_evidence "
            "WHERE evidence_id IN "
            "(SELECT id FROM knowledge_structural_evidence WHERE palace LIKE $1)",
            (pattern,),
        )
        # ADR-013 provenance rows
        await knowledge_backend.execute(
            "DELETE FROM knowledge_topology_provenance WHERE palace LIKE $1",
            (pattern,),
        )
        # ADR-013 structural evidence rows
        await knowledge_backend.execute(
            "DELETE FROM knowledge_structural_evidence WHERE palace LIKE $1",
            (pattern,),
        )
        # Legacy pre-ADR tables (kept for safety; harmless if empty)
        await knowledge_backend.execute(
            "DELETE FROM knowledge_relations "
            "WHERE source_entity_id IN "
            "(SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_memories "
            "WHERE memory_id IN "
            "(SELECT id FROM knowledge_memories WHERE palace LIKE $1)",
            (pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entity_embeddings "
            "WHERE entity_id IN "
            "(SELECT id FROM knowledge_entities WHERE palace LIKE $1)",
            (pattern,),
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_memories WHERE palace LIKE $1", (pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_entities WHERE palace LIKE $1", (pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_items WHERE palace LIKE $1", (pattern,)
        )
        await knowledge_backend.execute(
            "DELETE FROM knowledge_sources WHERE palace LIKE $1", (pattern,)
        )

    await _wipe()
    yield
    await _wipe()


# ---------------------------------------------------------------------------
# Workflow execution context
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def workflow_context() -> AsyncIterator[MagicMock]:
    """AppContext wired to the built-in workflow registry."""
    builtin_dir = Path(__file__).parent.parent / "src" / "workflows_mcp" / "templates" / "memory"

    # Ensure MemoryExecutor picks up the same DB credentials as the test fixture.
    db_env_defaults = {
        "MEMORY_DB_HOST": os.environ.get("MEMORY_DB_HOST", "localhost"),
        "MEMORY_DB_PORT": os.environ.get("MEMORY_DB_PORT", "5432"),
        "MEMORY_DB_NAME": os.environ.get("MEMORY_DB_NAME", "workflows"),
        "MEMORY_DB_USER": os.environ.get("MEMORY_DB_USER", "workflows"),
        "MEMORY_DB_PASSWORD": os.environ.get("MEMORY_DB_PASSWORD", "supersecret"),
    }
    prev_env = {k: os.environ.get(k) for k in db_env_defaults}
    os.environ.update(db_env_defaults)

    registry = WorkflowRegistry()
    registry.load_from_directory(builtin_dir)

    executor_registry = create_default_registry()
    register_memory_executors(executor_registry)
    llm_config_loader = LLMConfigLoader()
    io_queue = IOQueue()

    app_context = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=None,
    )

    job_queue = JobQueue(app_context, num_workers=2)
    await job_queue._store.init()

    app_context = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=job_queue,
    )

    await io_queue.start()
    try:
        mock_ctx = MagicMock()
        mock_ctx.request_context.lifespan_context = app_context
        yield mock_ctx
    finally:
        await io_queue.stop()
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Helper: count rows
# ---------------------------------------------------------------------------


async def _count_structural_evidence(backend: PostgresBackend) -> int:
    rows = await backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_structural_evidence WHERE palace = $1",
        (PALACE,),
    )
    return int(rows.rows[0]["n"])


async def _count_structural_entities(backend: PostgresBackend) -> int:
    rows = await backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities"
        " WHERE palace = $1 AND source = 'STRUCTURAL'",
        (PALACE,),
    )
    return int(rows.rows[0]["n"])


async def _count_structural_relations(backend: PostgresBackend) -> int:
    rows = await backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_relations kr"
        " INNER JOIN knowledge_entities ke ON ke.id = kr.source_entity_id"
        " WHERE ke.palace = $1 AND ke.source = 'STRUCTURAL'",
        (PALACE,),
    )
    return int(rows.rows[0]["n"])


async def _fetch_evidence_categories(backend: PostgresBackend) -> set[str]:
    rows = await backend.query(
        "SELECT DISTINCT evidence_category FROM knowledge_structural_evidence WHERE palace = $1",
        (PALACE,),
    )
    return {r["evidence_category"] for r in rows.rows}


async def _fetch_evidence_entity_types(backend: PostgresBackend) -> set[str]:
    rows = await backend.query(
        "SELECT DISTINCT entity_type FROM knowledge_structural_evidence WHERE palace = $1",
        (PALACE,),
    )
    return {r["entity_type"] for r in rows.rows}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_system1_scan_workflow_is_registered(
    workflow_context: MagicMock,
) -> None:
    """system1-scan must be discoverable in the built-in workflow registry."""
    app_ctx: AppContext = workflow_context.request_context.lifespan_context
    assert app_ctx.registry.get("system1-scan") is not None, (
        "system1-scan workflow not found in built-in registry"
    )


async def test_system1_scan_first_run_stores_structural_evidence(
    workflow_context: MagicMock,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
    tmp_path: Path,
) -> None:
    """First scan of a Python file runs parse -> derive_system1_topology successfully.

    ADR-013 Task 6: the success path supplies an explicit complete topology_override
    so topology can be determined without structural derivation heuristics.
    The derive_system1_topology operation stores structural evidence and derives
    topology in one transaction.
    """
    fixture_file = tmp_path / "greeter.py"
    fixture_file.write_text(FIXTURE_PY_SRC, encoding="utf-8")

    result = await execute_workflow(
        workflow="system1-scan",
        inputs={
            "file_path": str(fixture_file),
            "repo_relative_path": "greeter.py",
            "palace": PALACE,
            "source_name": "test-repo",
            "topology_override": {
                "wing": "application",
                "room": "greeting",
                "compartment": "core",
                "override_reason": "Integration test explicit placement",
                "applied_by": "test_system1_scan_workflow",
            },
        },
        debug=False,
        mode="sync",
        timeout=60,
        ctx=workflow_context,
    )

    response: dict[str, Any] = result.structuredContent
    assert response.get("status") == "success", (
        f"Workflow did not complete successfully: {response}"
    )

    # Workflow must expose derived topology output fields
    outputs = response.get("outputs", {})
    assert "derived_wing" in outputs, (
        f"Workflow outputs missing 'derived_wing'; got: {list(outputs.keys())}"
    )
    assert "derivation_source" in outputs, (
        f"Workflow outputs missing 'derivation_source'; got: {list(outputs.keys())}"
    )
    assert "evidence_stored" in outputs, (
        f"Workflow outputs missing 'evidence_stored'; got: {list(outputs.keys())}"
    )
    assert "graph_diagnostics" in outputs, (
        f"Workflow outputs missing 'graph_diagnostics'; got: {list(outputs.keys())}"
    )
    graph_diagnostics = outputs["graph_diagnostics"]
    assert isinstance(graph_diagnostics, dict)
    for key in (
        "entities_submitted",
        "relations_submitted",
        "relations_by_type",
        "unresolved_imports",
    ):
        assert key in graph_diagnostics, (
            f"Expected graph_diagnostics to include '{key}'; got: {graph_diagnostics}"
        )

    # Structural evidence rows must exist in DB after successful derivation
    evidence_count = await _count_structural_evidence(knowledge_backend)
    assert evidence_count >= 1, (
        "Expected at least one knowledge_structural_evidence row after derive_system1_topology"
    )
    topology_rows = await knowledge_backend.query(
        """
        SELECT DISTINCT wing, room, compartment
          FROM knowledge_structural_evidence
         WHERE palace = $1
        """,
        (PALACE,),
    )
    assert topology_rows.rows, "Expected structural evidence rows to carry topology"
    observed_topologies = {
        (str(row["wing"]), str(row["room"]), str(row["compartment"])) for row in topology_rows.rows
    }
    non_empty_topologies = {topology for topology in observed_topologies if any(topology)}
    assert non_empty_topologies == {("application", "greeting", "core")}, (
        "Expected only the explicit override topology among non-empty topology rows; "
        f"got observed={observed_topologies}, non_empty={non_empty_topologies}"
    )
    assert all(
        "default-wing" not in topology and "default-room" not in topology
        for topology in observed_topologies
    ), f"Placeholder topology leaked into structural evidence rows: {observed_topologies}"

    # Corridor must not appear as an entity type (topology rule)
    entity_types = await _fetch_evidence_entity_types(knowledge_backend)
    assert "corridor" not in entity_types, (
        "Corridor must not be encoded as a topology entity_type in structural evidence"
    )


async def test_system1_scan_markdown_file_stores_file_structural_evidence(
    workflow_context: MagicMock,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
    tmp_path: Path,
) -> None:
    """Markdown files must not fail project sync with empty inline candidates."""
    fixture_file = tmp_path / ".impeccable.md"
    fixture_file.write_text(
        "# Project notes\n\nOperational guidance for the repository.\n",
        encoding="utf-8",
    )

    result = await execute_workflow(
        workflow="system1-scan",
        inputs={
            "file_path": str(fixture_file),
            "repo_relative_path": ".impeccable.md",
            "palace": PALACE,
            "source_name": "forge",
            "topology_override": {
                "wing": "application",
                "room": "documentation",
                "compartment": "forge",
                "override_reason": "Project default topology for System 1 project sync",
                "applied_by": "admin_sync",
            },
        },
        debug=False,
        mode="sync",
        timeout=60,
        ctx=workflow_context,
    )

    response: dict[str, Any] = result.structuredContent
    assert response.get("status") == "success", f"Markdown System 1 scan failed: {response}"


async def test_system1_scan_without_override_persists_graph_without_placeholder_topology(
    workflow_context: MagicMock,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
    tmp_path: Path,
) -> None:
    """No explicit topology_override persists graph data without placeholder topology values."""
    fixture_file = tmp_path / "derived_topology.py"
    fixture_file.write_text(FIXTURE_PY_SRC, encoding="utf-8")

    result = await execute_workflow(
        workflow="system1-scan",
        inputs={
            "file_path": str(fixture_file),
            "repo_relative_path": "derived_topology.py",
            "palace": PALACE,
            "source_name": "test-repo",
        },
        debug=False,
        mode="sync",
        timeout=60,
        ctx=workflow_context,
    )

    response: dict[str, Any] = result.structuredContent
    assert response.get("status") == "success", response

    evidence_count = await _count_structural_evidence(knowledge_backend)
    assert evidence_count > 0, "Expected structural evidence rows for no-override scan"

    placeholder_entities = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities "
        "WHERE palace = $1 AND (namespace = 'default-wing' OR room = 'default-room')",
        (PALACE,),
    )
    assert int(placeholder_entities.rows[0]["n"]) == 0

    placeholder_evidence = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_structural_evidence "
        "WHERE palace = $1 AND (wing = 'default-wing' OR room = 'default-room')",
        (PALACE,),
    )
    assert int(placeholder_evidence.rows[0]["n"]) == 0


@pytest.mark.asyncio
async def test_system1_project_sync_preserves_null_topology_override_for_system1_scan(
    tmp_path: Path,
) -> None:
    """Runtime wiring keeps null topology_override as null through for_each inputs.

    This guards against template coercion where {{each.value.topology_override}} could
    become a non-null string/object and accidentally force explicit override.
    """

    class _CaptureInput(BlockInput):
        topology_override: dict[str, str] | None = None

    class _CaptureOutput(BlockOutput):
        seen_none: bool

    captured: list[dict[str, str] | None] = []

    class _CaptureTopologyExecutor(BlockExecutor):
        type_name = "CaptureTopology"
        input_type = _CaptureInput
        output_type = _CaptureOutput

        async def execute(self, inputs: BlockInput, context: Execution) -> _CaptureOutput:
            assert isinstance(inputs, _CaptureInput)
            captured.append(inputs.topology_override)
            return _CaptureOutput(seen_none=inputs.topology_override is None)

    class _NoOpMemoryInput(BlockInput):
        operation: str
        scope: dict[str, Any] | None = None
        record: dict[str, Any] | None = None

    class _NoOpMemoryOutput(BlockOutput):
        operation: str
        result: dict[str, Any]

    class _NoOpMemoryExecutor(BlockExecutor):
        type_name = "Memory"
        input_type = _NoOpMemoryInput
        output_type = _NoOpMemoryOutput

        async def execute(self, inputs: BlockInput, context: Execution) -> _NoOpMemoryOutput:
            assert isinstance(inputs, _NoOpMemoryInput)
            return _NoOpMemoryOutput(
                operation=inputs.operation,
                result={"manage": {"cycle_id": "cycle-test-noop"}},
            )

    project_root = tmp_path / "project-sync-null-override"
    fixture_file = project_root / "src" / "app.py"
    fixture_file.parent.mkdir(parents=True, exist_ok=True)
    fixture_file.write_text("def main() -> int:\n    return 1\n", encoding="utf-8")

    builtin_dir = Path(__file__).parent.parent / "src" / "workflows_mcp" / "templates" / "memory"
    registry = WorkflowRegistry()
    registry.load_from_directory(builtin_dir)

    registry.unregister("system1-scan")
    registry.register(
        WorkflowSchema(
            name="system1-scan",
            description="Test shim that captures topology_override from parent workflow",
            inputs={
                "topology_override": {
                    "type": "dict",
                    "required": False,
                    "description": "Optional explicit topology override for system1-scan",
                },
            },
            blocks=[
                {
                    "id": "capture",
                    "type": "CaptureTopology",
                    "inputs": {
                        "topology_override": "{{inputs.topology_override}}",
                    },
                }
            ],
            outputs={
                "seen_none": {"value": "{{blocks.capture.outputs.seen_none}}", "type": "bool"}
            },
        )
    )

    executor_registry = create_default_registry()
    executor_registry._executors.pop("Memory", None)
    executor_registry.register(_NoOpMemoryExecutor())
    executor_registry.register(_CaptureTopologyExecutor())

    app_context = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=LLMConfigLoader(),
        io_queue=IOQueue(),
        job_queue=None,
    )
    mock_ctx = MagicMock()
    mock_ctx.request_context.lifespan_context = app_context

    result = await execute_workflow(
        workflow="system1-project-sync",
        inputs={
            "project_root": str(project_root),
            "fs_allowlist": [str(project_root)],
            "candidate_paths": [],
            "palace": "palace_null_override_runtime",
            "source_name": "test-project-sync",
            "default_wing": None,
            "default_room": None,
            "default_compartment": "forge",
            "sync_scope": "rebuild",
        },
        debug=False,
        mode="sync",
        timeout=60,
        ctx=mock_ctx,
    )

    response: dict[str, Any] = result.structuredContent
    assert response.get("status") == "success", response
    assert captured, "Expected at least one discovered file to trigger system1-scan"
    assert captured[0] is None


async def test_system1_scan_second_run_is_idempotent(
    workflow_context: MagicMock,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
    tmp_path: Path,
) -> None:
    """Second run on an unchanged file must not grow structural evidence rows.

    ADR-013: derive_system1_topology upserts structural evidence rows.
    A second run on the same file content must not insert new evidence rows.
    Provenance rows are append-only (new row per derivation), but evidence
    rows must be idempotent.
    """
    fixture_file = tmp_path / "greeter.py"
    fixture_file.write_text(FIXTURE_PY_SRC, encoding="utf-8")

    common_inputs: dict[str, Any] = {
        "file_path": str(fixture_file),
        "repo_relative_path": "greeter.py",
        "palace": PALACE,
        "source_name": "test-repo",
        "topology_override": {
            "wing": "application",
            "room": "greeting",
            "compartment": "core",
            "override_reason": "Idempotency integration test explicit placement",
            "applied_by": "test_system1_scan_workflow",
        },
    }

    # First run
    r1 = await execute_workflow(
        workflow="system1-scan",
        inputs=common_inputs,
        debug=False,
        mode="sync",
        timeout=60,
        ctx=workflow_context,
    )
    assert r1.structuredContent.get("status") == "success", (
        f"First run failed: {r1.structuredContent}"
    )

    evidence_after_1 = await _count_structural_evidence(knowledge_backend)
    entities_after_1 = await _count_structural_entities(knowledge_backend)
    relations_after_1 = await _count_structural_relations(knowledge_backend)

    # Second run (same file, same content)
    r2 = await execute_workflow(
        workflow="system1-scan",
        inputs=common_inputs,
        debug=False,
        mode="sync",
        timeout=60,
        ctx=workflow_context,
    )
    assert r2.structuredContent.get("status") == "success", (
        f"Second run failed: {r2.structuredContent}"
    )

    evidence_after_2 = await _count_structural_evidence(knowledge_backend)
    entities_after_2 = await _count_structural_entities(knowledge_backend)
    relations_after_2 = await _count_structural_relations(knowledge_backend)

    assert evidence_after_2 == evidence_after_1, (
        "knowledge_structural_evidence grew on second run (not idempotent): "
        f"{evidence_after_1} -> {evidence_after_2}"
    )
    assert entities_after_2 == entities_after_1, (
        "knowledge_entities (source=STRUCTURAL) grew on second run (not idempotent): "
        f"{entities_after_1} -> {entities_after_2}"
    )
    assert relations_after_2 == relations_after_1, (
        "knowledge_relations (source=STRUCTURAL) grew on second run (not idempotent): "
        f"{relations_after_1} -> {relations_after_2}"
    )


async def test_system1_scan_persists_graph_and_neighbors_metadata_e2e(
    workflow_context: MagicMock,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
    tmp_path: Path,
) -> None:
    """E2E: scan persists structural graph rows and neighbors exposes provenance metadata."""
    callee = tmp_path / "callee.py"
    callee.write_text("def greet() -> str:\n    return 'hello'\n", encoding="utf-8")

    caller = tmp_path / "caller.py"
    caller.write_text(
        "from callee import greet\n\ndef run() -> str:\n    return greet()\n",
        encoding="utf-8",
    )

    explicit_topology = {
        "wing": "application",
        "room": "integration",
        "compartment": "core",
        "override_reason": "Task10 final integration verification",
        "applied_by": "test_system1_scan_workflow",
    }

    for file_name in ("callee.py", "caller.py"):
        result = await execute_workflow(
            workflow="system1-scan",
            inputs={
                "file_path": str(tmp_path / file_name),
                "repo_relative_path": file_name,
                "palace": PALACE,
                "source_name": "test-repo",
                "topology_override": explicit_topology,
            },
            debug=False,
            mode="sync",
            timeout=60,
            ctx=workflow_context,
        )

        response: dict[str, Any] = result.structuredContent
        assert response.get("status") == "success", response
        outputs = response.get("outputs", {})
        assert isinstance(outputs.get("graph_diagnostics"), dict), outputs
        assert outputs["graph_diagnostics"].get("entities_submitted", 0) > 0, outputs
        for field in ("derived_wing", "derived_room", "derived_compartment", "derivation_source"):
            assert field in outputs, (
                f"Expected topology derivation output '{field}' in scan outputs: {outputs}"
            )

    structural_entities = await _count_structural_entities(knowledge_backend)
    assert structural_entities >= 2, (
        f"Expected at least two STRUCTURAL entities after two-file scan, got {structural_entities}"
    )

    relation_rows = await knowledge_backend.query(
        "SELECT relation_type FROM knowledge_relations kr"
        " INNER JOIN knowledge_entities ke ON ke.id = kr.source_entity_id"
        " WHERE ke.palace = $1 AND ke.source = 'STRUCTURAL'",
        (PALACE,),
    )
    relation_types = {str(row["relation_type"]) for row in relation_rows.rows}
    allowed_relation_types = {"CONTAINS", "IMPORTS", "CALLS", "INHERITS_FROM"}
    assert relation_types & allowed_relation_types, (
        "Expected at least one structural relation type among "
        f"{sorted(allowed_relation_types)}; got {sorted(relation_types)}"
    )

    non_empty_topology_rows = await knowledge_backend.query(
        "SELECT wing, room FROM knowledge_structural_evidence"
        " WHERE palace = $1 AND (wing IS NOT NULL OR room IS NOT NULL)",
        (PALACE,),
    )
    assert all(
        row["wing"] != "default-wing" and row["room"] != "default-room"
        for row in non_empty_topology_rows.rows
    ), "default-wing/default-room must never be written as topology"

    start_entity_rows = await knowledge_backend.query(
        "SELECT id, name FROM knowledge_entities"
        " WHERE palace = $1 AND source = 'STRUCTURAL'"
        " ORDER BY id LIMIT 1",
        (PALACE,),
    )
    assert start_entity_rows.rows, "Expected at least one STRUCTURAL entity for neighbors query"
    start_name = str(start_entity_rows.rows[0]["name"])

    graph_result = await MemoryService(backend=knowledge_backend, context=Execution())._query_graph(
        QueryMemoryRequest(
            query="neighbors",
            strategy="graph",
            graph_op="neighbors",
            start_entity=start_name,
            palace=PALACE,
        )
    )

    edges = graph_result.evidence[0].get("edges", []) if graph_result.evidence else []
    matching_edges = [
        edge
        for edge in edges
        if edge.get("relation_type") in allowed_relation_types
        and isinstance(edge.get("metadata"), dict)
        and edge["metadata"].get("source_file")
        and edge["metadata"].get("content_hash")
    ]
    assert matching_edges, (
        "Expected neighbors graph query to return at least one System1 edge "
        "with provenance metadata source_file and content_hash"
    )


async def test_system1_scan_no_corridor_topology_output(
    workflow_context: MagicMock,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
    tmp_path: Path,
) -> None:
    """Corridor must never appear as a topology level in workflow scope or evidence.

    ADR-013 §4.1: corridors are directed typed relations only, never placement levels.
    After derive_system1_topology, no evidence row must have entity_type='corridor'.
    """
    fixture_file = tmp_path / "greeter.py"
    fixture_file.write_text(FIXTURE_PY_SRC, encoding="utf-8")

    result = await execute_workflow(
        workflow="system1-scan",
        inputs={
            "file_path": str(fixture_file),
            "repo_relative_path": "greeter.py",
            "palace": PALACE,
            "source_name": "test-repo",
            "topology_override": {
                "wing": "application",
                "room": "greeting",
                "compartment": "core",
                "override_reason": "Corridor topology integration test explicit placement",
                "applied_by": "test_system1_scan_workflow",
            },
        },
        debug=False,
        mode="sync",
        timeout=60,
        ctx=workflow_context,
    )

    assert result.structuredContent.get("status") == "success"

    # Verify no corridor stored as entity type in evidence table
    entity_types = await _fetch_evidence_entity_types(knowledge_backend)
    assert "corridor" not in entity_types, (
        f"Corridor encoded as entity_type in structural evidence: {entity_types}"
    )

    # Verify that knowledge_structural_evidence rows have the four topology
    # columns (palace/wing/room/compartment); no 'corridor' column should exist.
    rows = await knowledge_backend.query(
        "SELECT palace, wing, room, compartment FROM knowledge_structural_evidence"
        " WHERE palace = $1 LIMIT 1",
        (PALACE,),
    )
    assert rows.rows, "Expected at least one evidence row after derive_system1_topology"
    row = rows.rows[0]
    for col in ("palace", "wing", "room", "compartment"):
        assert col in row, f"Expected topology column '{col}' in evidence row"


# ---------------------------------------------------------------------------
# ADR-013 Task 6: YAML structural regression tests
# ---------------------------------------------------------------------------

_WORKFLOW_YAML_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "workflows_mcp"
    / "templates"
    / "memory"
    / "system1-scan.yaml"
)

_SYSTEM2_PROJECT_SYNC_YAML_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "workflows_mcp"
    / "templates"
    / "memory"
    / "system2-project-sync.yaml"
)


def _load_workflow_yaml() -> dict:
    import yaml  # noqa: PLC0415

    return yaml.safe_load(_WORKFLOW_YAML_PATH.read_text(encoding="utf-8"))


def _load_system2_project_sync_yaml() -> dict:
    import yaml  # noqa: PLC0415

    return yaml.safe_load(_SYSTEM2_PROJECT_SYNC_YAML_PATH.read_text(encoding="utf-8"))


def test_system2_project_sync_template_suppresses_placeholder_topology_defaults() -> None:
    """system2-project-sync.yaml must not ship default-wing/default-room placeholders.

    Final QA blocker guard: default_wing/default_room must use suppressed semantics
    (empty string) and never static placeholder literals.
    """

    wf = _load_system2_project_sync_yaml()
    inputs: dict[str, dict[str, Any]] = wf.get("inputs", {})

    assert inputs["default_wing"]["default"] == ""
    assert inputs["default_room"]["default"] == ""
    assert inputs["default_wing"]["default"] != "default-wing"
    assert inputs["default_room"]["default"] != "default-room"


# ---- 1. TreeSitter block must not include topology fields ------------------


def test_treesitter_block_output_schema_has_no_topology_fields() -> None:
    """TreeSitter parse block must not declare wing/room/compartment/placement outputs.

    ADR-013: structural parse produces entities and relations only; topology
    is derived later by the memory service, not the parser.
    """
    wf = _load_workflow_yaml()
    blocks: list[dict] = wf.get("blocks", [])
    ts_blocks = [b for b in blocks if b.get("type") == "TreeSitter"]
    assert ts_blocks, "Expected at least one TreeSitter block in system1-scan.yaml"

    forbidden_fields = {"wing", "room", "compartment", "placement"}
    for block in ts_blocks:
        # Check block-level outputs section (if any)
        block_outputs = block.get("outputs", {})
        if isinstance(block_outputs, dict):
            found = forbidden_fields & set(block_outputs.keys())
            assert not found, (
                f"TreeSitter block '{block.get('id')}' declares forbidden topology "
                f"output field(s): {found}"
            )
        # Check block inputs — TreeSitter must not receive topology fields
        block_inputs = block.get("inputs", {})
        if isinstance(block_inputs, dict):
            found = forbidden_fields & set(block_inputs.keys())
            assert not found, (
                f"TreeSitter block '{block.get('id')}' receives forbidden topology "
                f"input field(s): {found}"
            )


# ---- 2. Flow is parse -> persist graph -> derive topology -> outputs --------


def test_workflow_sequence_persists_graph_before_topology_derivation() -> None:
    """system1-scan.yaml must persist graph before topology derivation.

    ADR-013 Task 6 contract:
    - parse_file runs first,
    - first Memory op persists raw structural graph,
    - derive_system1_topology remains present as downstream consumer.
    """
    wf = _load_workflow_yaml()
    blocks: list[dict] = wf.get("blocks", [])
    assert [b.get("type") for b in blocks[:2]] == ["TreeSitter", "Memory"]

    memory_blocks = [b for b in blocks if b.get("type") == "Memory"]
    operations = [b.get("inputs", {}).get("operation") for b in memory_blocks]
    assert operations[0] == "store_system1_structural_graph"
    assert "derive_system1_topology" in operations

    persist = memory_blocks[0].get("inputs", {}).get("record", {}).get("system1_graph", {})
    assert persist["entities"] == "{{blocks.parse_file.outputs.entities}}"
    assert persist["relations"] == "{{blocks.parse_file.outputs.relations}}"
    assert persist["unresolved_imports"] == "{{blocks.parse_file.outputs.unresolved_imports}}"
    assert (
        persist["structural_evidence"] == "{{blocks.parse_file.outputs.structural_evidence_items}}"
    )


# ---- 3. No static topology defaults (wing/room/compartment) ----------------


def test_workflow_inputs_have_no_static_topology_defaults() -> None:
    """system1-scan.yaml must not define wing, room, or compartment with static defaults.

    ADR-013: static defaults like wing='code', room='default',
    compartment='system1scan' allow topology to leak from scope literals.
    These inputs must be absent or have no default.
    """
    wf = _load_workflow_yaml()
    inputs: dict = wf.get("inputs", {})

    forbidden_defaults = {
        "wing": "code",
        "room": "default",
        "compartment": "system1scan",
    }

    for field, forbidden_value in forbidden_defaults.items():
        if field in inputs:
            actual_default = inputs[field].get("default")
            assert actual_default != forbidden_value, (
                f"Input '{field}' still has hardcoded default '{forbidden_value}' — "
                "remove static topology defaults per ADR-013"
            )


def test_workflow_blocks_have_no_static_topology_defaults() -> None:
    """Memory blocks must not pass hardcoded wing/room/compartment scope literals.

    ADR-013: any scope passed to the Memory derive operation must come from
    inputs or derived values, never hardcoded strings. If a topology scope
    value is present and non-empty, it must be a template reference (starts
    with '{{'), otherwise the test fails regardless of the literal value.
    """
    wf = _load_workflow_yaml()
    blocks: list[dict] = wf.get("blocks", [])
    memory_blocks = [b for b in blocks if b.get("type") == "Memory"]

    for block in memory_blocks:
        scope = block.get("inputs", {}).get("scope", {})
        if not isinstance(scope, dict):
            continue
        for key in ("wing", "room", "compartment"):
            val = scope.get(key)
            if val is None or val == "":
                continue
            assert isinstance(val, str) and val.startswith("{{"), (
                f"Memory block '{block.get('id')}' scope.{key}='{val}' is a "
                "hardcoded topology value — must be a template reference (e.g. "
                "'{{inputs.topology_override.wing}}') per ADR-013"
            )


# ---- 4. Corridor fixture: corridor stays relation, not topology level -------


def test_corridor_is_not_a_topology_field_in_workflow_contract() -> None:
    """Corridor must not appear as a topology level field in workflow inputs or outputs.

    ADR-013 §4.1: corridors are directed typed relations only; never
    placement levels. The workflow YAML must not expose 'corridor' as an
    input, output, or scope field.
    """
    wf = _load_workflow_yaml()
    inputs: dict = wf.get("inputs", {})
    outputs: dict = wf.get("outputs", {})
    blocks: list[dict] = wf.get("blocks", [])

    assert "corridor" not in inputs, (
        "Workflow input 'corridor' found — corridor must not be a topology level"
    )
    assert "corridor" not in outputs, (
        "Workflow output 'corridor' found — corridor must not be a topology level"
    )

    for block in blocks:
        scope = block.get("inputs", {}).get("scope", {})
        if isinstance(scope, dict):
            assert "corridor" not in scope, (
                f"Block '{block.get('id')}' scope contains 'corridor' — "
                "corridor must not be a topology level in scope fields"
            )


def test_corridor_shaped_relation_in_treesitter_output_is_not_topology() -> None:
    """Corridor-shaped fixture: TreeSitter output reference must not alias to topology.

    The workflow YAML must not reference blocks.parse_file.outputs.corridor
    or any corridor-named field as a topology value passed to Memory scope.
    """
    wf = _load_workflow_yaml()
    blocks: list[dict] = wf.get("blocks", [])
    import json  # noqa: PLC0415

    memory_blocks = [b for b in blocks if b.get("type") == "Memory"]
    for block in memory_blocks:
        # Ensure no corridor reference in scope of any Memory block
        scope = block.get("inputs", {}).get("scope", {})
        scope_text = json.dumps(scope)
        assert "corridor" not in scope_text.lower(), (
            f"Memory block '{block.get('id')}' scope references 'corridor' — "
            "corridor is a relation, not a topology level"
        )


# ---- 5. topology_override contract: nested override, not wing_hint ----------


def test_workflow_memory_block_uses_topology_override_not_wing_hint() -> None:
    """Memory block must use nested derivation envelope with topology_override (not wing_hint).

    ADR-013 Task 6: topology_override must live inside the derivation: envelope, not at
    the top-level of Memory block inputs. wing_hint must be absent everywhere.
    """
    wf = _load_workflow_yaml()
    import json  # noqa: PLC0415

    wf_text = json.dumps(wf)
    assert "wing_hint" not in wf_text, (
        "workflow YAML references 'wing_hint' — this was removed in ADR-013 Task 2a; "
        "use nested derivation.topology_override instead"
    )

    blocks: list[dict] = wf.get("blocks", [])
    memory_blocks = [b for b in blocks if b.get("type") == "Memory"]
    assert memory_blocks, "Expected at least one Memory block"

    memory_block = next(
        b
        for b in memory_blocks
        if b.get("inputs", {}).get("operation") == "derive_system1_topology"
    )
    block_inputs = memory_block.get("inputs", {})

    # topology_override must NOT be at the top level of block inputs (extra=forbid in MemoryInput)
    assert "topology_override" not in block_inputs, (
        f"Memory block '{memory_block.get('id')}' has top-level 'topology_override' — "
        "ADR-013 Task 6 requires topology_override inside the derivation: envelope"
    )

    # derivation envelope must be present and contain topology_override
    derivation = block_inputs.get("derivation", {})
    assert derivation, (
        f"Memory block '{memory_block.get('id')}' has no 'derivation' envelope — "
        "ADR-013 Task 6 requires a nested derivation: payload"
    )
    assert "topology_override" in derivation, (
        f"Memory block '{memory_block.get('id')}' derivation envelope "
        "missing 'topology_override' — override must be declared inside derivation:"
    )
    assert "inline_candidates" in derivation, (
        f"Memory block '{memory_block.get('id')}' derivation envelope "
        "missing 'inline_candidates' — parsed structural evidence items must be "
        "passed as inline_candidates"
    )
    assert "palace" in derivation, (
        f"Memory block '{memory_block.get('id')}' derivation envelope missing 'palace'"
    )


# ---- 5b. Fail-closed: absent topology_override with insufficient evidence --


async def test_system1_scan_skips_per_file_derivation_without_topology_override(
    workflow_context: MagicMock,
    knowledge_backend: PostgresBackend,
    clean_palace: None,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Absent topology_override skips per-file topology derivation without failing workflow.

    No-override scans still persist System 1 structural graph evidence, but
    per-file derive_system1_topology is intentionally skipped. Project-level
    topology derivation remains authoritative in system1-project-sync.

    Observable behaviour:
    - Structural graph persistence succeeds and workflow status is success.
    - Per-file derive is skipped, so no MEM_INSUFFICIENT_EVIDENCE ERROR log is emitted.
    - Derivation outcome remains non-success (`evidence_stored` is false).

    Assertions:
    1. Primary: workflow remains successful because persist_structural_graph is
       the blocking step.
    2. Derivation result is non-success (`evidence_stored` is false).
    3. Secondary: MEM_INSUFFICIENT_EVIDENCE is absent from captured ERROR logs.
    """
    import logging  # noqa: PLC0415

    minimal_src = "x = 1\n"
    fixture_file = tmp_path / "minimal.py"
    fixture_file.write_text(minimal_src, encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        result = await execute_workflow(
            workflow="system1-scan",
            inputs={
                "file_path": str(fixture_file),
                "repo_relative_path": "minimal.py",
                "palace": PALACE,
                "source_name": "test-repo",
                # topology_override intentionally absent
            },
            debug=False,
            mode="sync",
            timeout=60,
            ctx=workflow_context,
        )

    response: dict[str, Any] = result.structuredContent

    # Primary: persist_structural_graph is the blocking unit. Derivation can
    # fail closed without failing the overall workflow.
    status = response.get("status")
    assert status == "success", (
        "Expected workflow success when topology_override is absent and evidence "
        "is insufficient for derivation because graph persistence is primary. "
        f"Got status={status!r}. Full response: {response}"
    )

    outputs = response.get("outputs", {})
    assert outputs.get("evidence_stored") is False, (
        "Expected derive_topology fail-closed outcome when insufficient evidence "
        f"is present; outputs={outputs}"
    )

    # Secondary: per-file derive is skipped, so no insufficient-evidence error log.
    assert "MEM_INSUFFICIENT_EVIDENCE" not in caplog.text, (
        "Did not expect MEM_INSUFFICIENT_EVIDENCE in executor logs when "
        "topology_override is absent and per-file derivation is skipped. "
        f"Captured log: {caplog.text!r}"
    )


# ---- 6. Outputs expose derived topology/provenance -------------------------


@pytest.mark.asyncio
async def test_treesitter_output_relations_are_schema_valid(tmp_path: Path) -> None:
    """TreeSitter extraction emits schema-valid System1 entities/relations."""
    from workflows_mcp.engine.executors_treesitter import (  # noqa: PLC0415
        TreeSitterExecutor,
        TreeSitterInput,
        validate_system1_extraction_payload,
    )

    source = tmp_path / "service.py"
    source.write_text(
        "import os\n\nclass Service:\n    def run(self):\n        return os.getcwd()\n",
        encoding="utf-8",
    )

    output = await TreeSitterExecutor().execute(
        TreeSitterInput(
            path=str(source),
            repo_relative_path="src/service.py",
            palace=PALACE,
            item_id="item-service",
        ),
        Execution(),
    )

    diagnostics = validate_system1_extraction_payload(output.entities, output.relations)
    assert diagnostics["valid"] is True
    assert diagnostics["entity_count"] >= 2
    assert diagnostics["relation_count"] >= 1
    assert set(diagnostics["relation_types"]).issubset(
        {"CONTAINS", "IMPORTS", "CALLS", "INHERITS_FROM"}
    )


@pytest.mark.asyncio
async def test_treesitter_entities_and_relations_include_source_provenance(
    tmp_path: Path,
) -> None:
    from workflows_mcp.engine.executors_treesitter import (  # noqa: PLC0415
        TreeSitterExecutor,
        TreeSitterInput,
    )

    source = tmp_path / "svc.py"
    source.write_text("import os\n\ndef run():\n    return os.getcwd()\n", encoding="utf-8")

    output = await TreeSitterExecutor().execute(
        TreeSitterInput(
            path=str(source),
            repo_relative_path="src/svc.py",
            palace=PALACE,
            item_id="item-svc",
        ),
        Execution(),
    )

    assert output.entities
    for entity in output.entities:
        metadata = entity["metadata"]
        assert metadata["source_file"] == str(source)
        assert metadata["repo_relative_path"] == "src/svc.py"
        assert metadata["content_hash"] == output.content_hash
        assert "source_range" in metadata
        assert metadata["language"] == "python"

    assert output.relations
    for relation in output.relations:
        metadata = relation["metadata"]
        assert metadata["source_file"] == str(source)
        assert metadata["repo_relative_path"] == "src/svc.py"
        assert metadata["content_hash"] == output.content_hash
        assert metadata["provenance"] == "treesitter"
        assert "source_range" in metadata


@pytest.mark.asyncio
async def test_structural_import_evidence_lists_targets_and_resolution(
    tmp_path: Path,
) -> None:
    from workflows_mcp.engine.executors_treesitter import (  # noqa: PLC0415
        TreeSitterExecutor,
        TreeSitterInput,
    )

    source = tmp_path / "svc.py"
    source.write_text("import os\nimport missing_pkg\n", encoding="utf-8")

    output = await TreeSitterExecutor().execute(
        TreeSitterInput(path=str(source), repo_relative_path="src/svc.py", palace=PALACE),
        Execution(),
    )

    import_items = [
        item
        for item in output.structural_evidence_items
        if item["evidence_category"] == "structural_import"
    ]
    assert import_items
    payload = import_items[0]["evidence_data"]
    assert payload["import_count"] >= 2
    assert {entry["target_qname"] for entry in payload["imports"]} >= {
        "os",
        "missing_pkg",
    }
    assert all("resolution" in entry for entry in payload["imports"])


@pytest.mark.asyncio
async def test_treesitter_source_range_defaults_are_not_shared(tmp_path: Path) -> None:
    from workflows_mcp.engine.executors_treesitter import (  # noqa: PLC0415
        TreeSitterExecutor,
        TreeSitterInput,
    )

    source = tmp_path / "svc.py"
    source.write_text("import os\n\ndef run():\n    return os.getcwd()\n", encoding="utf-8")

    output = await TreeSitterExecutor().execute(
        TreeSitterInput(
            path=str(source),
            repo_relative_path="src/svc.py",
            palace=PALACE,
            item_id="item-svc",
        ),
        Execution(),
    )

    entity_ranges = [entity["metadata"]["source_range"] for entity in output.entities]
    assert len(entity_ranges) >= 2
    assert len({id(source_range) for source_range in entity_ranges}) == len(entity_ranges)

    if output.relations:
        relation_ranges = [relation["metadata"]["source_range"] for relation in output.relations]
        assert len({id(source_range) for source_range in relation_ranges}) == len(relation_ranges)


@pytest.mark.asyncio
async def test_treesitter_execute_raises_for_invalid_schema_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid extraction payload fails at the extraction boundary."""
    from workflows_mcp.engine.executors_treesitter import (  # noqa: PLC0415
        TreeSitterExecutor,
        TreeSitterInput,
    )

    source = tmp_path / "broken.py"
    source.write_text("def run() -> None:\n    return None\n", encoding="utf-8")

    def _invalid_extract_code(
        *args: object, **kwargs: object
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        del args, kwargs
        return ([{"qualified_name": "mod.X"}], [])

    monkeypatch.setattr(
        "workflows_mcp.engine.executors_treesitter.extract_code",
        _invalid_extract_code,
    )

    with pytest.raises(ValueError, match="SYSTEM1_EXTRACTION_SCHEMA_INVALID"):
        await TreeSitterExecutor().execute(
            TreeSitterInput(
                path=str(source),
                repo_relative_path="src/broken.py",
                palace=PALACE,
                item_id="item-broken",
            ),
            Execution(),
        )


def test_validate_system1_extraction_payload_allows_empty_payload() -> None:
    """Unsupported/no-op extraction paths may produce empty payloads without error."""
    from workflows_mcp.engine.executors_treesitter import (  # noqa: PLC0415
        validate_system1_extraction_payload,
    )

    diagnostics = validate_system1_extraction_payload([], [])
    assert diagnostics["valid"] is True
    assert diagnostics["errors"] == []
    assert diagnostics["entity_count"] == 0
    assert diagnostics["relation_count"] == 0


def test_treesitter_schema_error_message_is_bounded() -> None:
    """Schema invalid exceptions are summarized and size-bounded."""
    from workflows_mcp.engine.executors_treesitter import (  # noqa: PLC0415
        _raise_if_invalid_system1_extraction_payload,
    )

    invalid_entities = [{"qualified_name": f"mod.Entity{i}"} for i in range(5000)]
    invalid_relations = [{"source_qname": "src.only"} for _ in range(200)]

    with pytest.raises(ValueError) as exc:
        _raise_if_invalid_system1_extraction_payload(invalid_entities, invalid_relations)

    message = str(exc.value)
    assert "SYSTEM1_EXTRACTION_SCHEMA_INVALID" in message
    assert "validation errors" in message
    assert "showing first" in message
    assert "and" in message and "more" in message
    assert "entity_count=" in message
    assert "relation_count=" in message
    assert len(message) < 5000


def test_workflow_outputs_expose_derived_topology_fields() -> None:
    """Workflow outputs must expose derived topology and provenance result fields.

    ADR-013 §8: ManageMemoryResult includes derived_wing, derived_room,
    derived_compartment, derivation_source, provenance_id, claim_id.
    """
    wf = _load_workflow_yaml()
    outputs: dict = wf.get("outputs", {})

    required_derived_outputs = {
        "derived_wing",
        "derived_room",
        "derived_compartment",
        "derivation_source",
        "graph_diagnostics",
    }

    missing = required_derived_outputs - set(outputs.keys())
    assert not missing, (
        f"Workflow outputs missing required derived topology fields: {missing}. "
        f"Current outputs: {list(outputs.keys())}"
    )
