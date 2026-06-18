from __future__ import annotations

import json
import os
import uuid
from typing import Any

import pytest
import pytest_asyncio

from workflows_mcp.engine.execution import Execution
from workflows_mcp.engine.knowledge.schema import ensure_schema
from workflows_mcp.engine.memory_errors import MemoryContractError
from workflows_mcp.engine.memory_service import (
    _PROJECT_DEFAULT_TOPOLOGY_APPLIED_BY,
    _PROJECT_DEFAULT_TOPOLOGY_OVERRIDE_REASON,
    MemoryRequest,
    MemoryService,
)
from workflows_mcp.engine.sql.backend import ConnectionConfig, DatabaseEngine
from workflows_mcp.engine.sql.postgres_backend import PostgresBackend


class _NoDbBackend:
    """Backend guard: this contract test must not touch DB."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected backend usage: {name}")


def _service() -> MemoryService:
    return MemoryService(backend=_NoDbBackend(), context=Execution())


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
async def knowledge_backend() -> PostgresBackend:
    backend = PostgresBackend()
    await backend.connect(_make_config())
    await ensure_schema(backend)
    try:
        yield backend
    finally:
        await backend.disconnect()


async def _wipe_palace(knowledge_backend: PostgresBackend, palace: str) -> None:
    await knowledge_backend.execute(
        "DELETE FROM knowledge_topology_provenance_evidence "
        "WHERE evidence_id IN "
        "(SELECT id FROM knowledge_structural_evidence WHERE palace = $1)",
        (palace,),
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_topology_provenance WHERE palace = $1", (palace,)
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_structural_evidence WHERE palace = $1", (palace,)
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_relations "
        "WHERE source_entity_id IN (SELECT id FROM knowledge_entities WHERE palace = $1)",
        (palace,),
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_entity_embeddings "
        "WHERE entity_id IN (SELECT id FROM knowledge_entities WHERE palace = $1)",
        (palace,),
    )
    await knowledge_backend.execute("DELETE FROM knowledge_entities WHERE palace = $1", (palace,))


def _valid_entity() -> dict[str, Any]:
    return {
        "qualified_name": "svc",
        "stable_id": "stable-svc",
        "entity_type": "Module",
        "name": "svc",
        "metadata": {
            "source_file": "src/svc.py",
            "source_range": {"start": {"line": 1, "column": 0}},
            "content_hash": "abc",
        },
        "confidence": 1.0,
    }


def _valid_relation() -> dict[str, Any]:
    return {
        "source_qname": "svc",
        "source_entity_type": "Module",
        "target_qname": "other",
        "target_entity_type": "Module",
        "relation_type": "IMPORTS",
        "confidence": 1.0,
        "metadata": {
            "source_file": "src/svc.py",
            "source_range": {"start": {"line": 1, "column": 0}},
            "content_hash": "abc",
            "provenance": {"line": 1, "column": 0},
        },
    }


@pytest.mark.asyncio
async def test_store_system1_structural_graph_requires_palace() -> None:
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "record": {
                "system1_graph": {
                    "parser_metadata": {"language": "python"},
                    "entities": [_valid_entity()],
                    "relations": [_valid_relation()],
                }
            },
        }
    )

    result = await _service().execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_PALACE_REQUIRED" in (result.manage.error or "")


@pytest.mark.asyncio
async def test_store_system1_structural_graph_rejects_invalid_relation_type() -> None:
    invalid_relation = _valid_relation()
    invalid_relation["relation_type"] = "DEPENDS_MAYBE"

    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": "palace_structural_graph_contract"},
            "record": {
                "system1_graph": {
                    "parser_metadata": {"language": "python", "content_hash": "abc"},
                    "entities": [_valid_entity()],
                    "relations": [invalid_relation],
                }
            },
        }
    )

    result = await _service().execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_INVALID_RELATION_TYPE" in (result.manage.error or "")


@pytest.mark.asyncio
async def test_store_system1_structural_graph_rejects_empty_entities_for_supported_code() -> None:
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": "palace_structural_graph_contract"},
            "record": {
                "system1_graph": {
                    "parser_metadata": {"language": "python", "source_file": "src/svc.py"},
                    "entities": [],
                    "relations": [],
                }
            },
        }
    )

    result = await _service().execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_EMPTY_SYSTEM1_ENTITIES" in (result.manage.error or "")


@pytest.mark.asyncio
async def test_store_system1_structural_graph_requires_relation_provenance() -> None:
    relation_missing_provenance = _valid_relation()
    relation_missing_provenance["metadata"] = {"source_file": "src/svc.py", "content_hash": "abc"}

    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": "palace_structural_graph_contract"},
            "record": {
                "system1_graph": {
                    "parser_metadata": {"language": "python", "content_hash": "abc"},
                    "entities": [_valid_entity()],
                    "relations": [relation_missing_provenance],
                }
            },
        }
    )

    result = await _service().execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_RELATION_PROVENANCE_REQUIRED" in (result.manage.error or "")


@pytest.mark.asyncio
async def test_store_system1_structural_graph_returns_diagnostics_counts(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_contract"
    await _wipe_palace(knowledge_backend, palace)

    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "content_hash": "abc",
                    },
                    "entities": [_valid_entity()],
                    "relations": [_valid_relation()],
                    "unresolved_imports": ["missing.module"],
                }
            },
        }
    )

    result = await MemoryService(backend=knowledge_backend, context=Execution()).execute(request)
    assert result.manage is not None
    assert result.manage.success is True
    assert result.manage.diagnostics["entities_submitted"] == 1
    assert result.manage.diagnostics["relations_submitted"] == 1
    assert set(result.manage.diagnostics["unresolved_imports"]) == {"missing.module", "other"}
    assert result.manage.diagnostics["relations_by_type"] == {"IMPORTS": 1}
    await _wipe_palace(knowledge_backend, palace)


def build_three_entity_graph_payload(palace: str) -> dict[str, Any]:
    return {
        "operation": "store_system1_structural_graph",
        "scope": {"palace": palace},
        "record": {
            "system1_graph": {
                "palace": palace,
                "source_name": "repo",
                "repo_relative_path": "src/app.py",
                "parser_metadata": {
                    "language": "python",
                    "source_file": "src/app.py",
                    "content_hash": "graph123",
                    "source_range": {"start": {"line": 1, "column": 0}},
                },
                "entities": [
                    {
                        "qualified_name": "pkg",
                        "stable_id": "stable-pkg",
                        "entity_type": "Module",
                        "name": "pkg",
                        "metadata": {
                            "source_file": "src/app.py",
                            "source_range": {"start": {"line": 1, "column": 0}},
                            "content_hash": "graph123",
                        },
                        "confidence": 1.0,
                    },
                    {
                        "qualified_name": "pkg.mod",
                        "stable_id": "stable-mod",
                        "entity_type": "Module",
                        "name": "mod",
                        "metadata": {
                            "source_file": "src/app.py",
                            "source_range": {"start": {"line": 2, "column": 0}},
                            "content_hash": "graph123",
                        },
                        "confidence": 1.0,
                    },
                    {
                        "qualified_name": "pkg.mod.fn",
                        "stable_id": "stable-fn",
                        "entity_type": "Function",
                        "name": "fn",
                        "metadata": {
                            "source_file": "src/app.py",
                            "source_range": {"start": {"line": 3, "column": 0}},
                            "content_hash": "graph123",
                        },
                        "confidence": 1.0,
                    },
                ],
                "relations": [
                    {
                        "source_qname": "pkg",
                        "source_entity_type": "Module",
                        "target_qname": "pkg.mod",
                        "target_entity_type": "Module",
                        "relation_type": "CONTAINS",
                        "confidence": 1.0,
                        "metadata": {
                            "source_file": "src/app.py",
                            "source_range": {"start": {"line": 10, "column": 0}},
                            "content_hash": "graph123",
                            "provenance": {"line": 10, "column": 0},
                        },
                    },
                    {
                        "source_qname": "pkg.mod.fn",
                        "source_entity_type": "Function",
                        "target_qname": "pkg.mod",
                        "target_entity_type": "Module",
                        "relation_type": "CALLS",
                        "confidence": 1.0,
                        "metadata": {
                            "source_file": "src/app.py",
                            "source_range": {"start": {"line": 11, "column": 0}},
                            "content_hash": "graph123",
                            "provenance": {"line": 11, "column": 0},
                        },
                    },
                ],
                "unresolved_imports": ["z.missing", "a.missing", "z.missing"],
            }
        },
    }


@pytest.mark.asyncio
async def test_store_system1_structural_graph_returns_deterministic_diagnostics(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_diagnostics"
    await _wipe_palace(knowledge_backend, palace)

    request = MemoryRequest.model_validate(build_three_entity_graph_payload(palace))
    result = await MemoryService(backend=knowledge_backend, context=Execution()).execute(request)

    assert result.manage is not None
    assert result.manage.success is True
    diagnostics = result.manage.diagnostics
    assert diagnostics["entities_submitted"] == 3
    assert diagnostics["relations_submitted"] == 2
    assert diagnostics["relations_by_type"] == {"CALLS": 1, "CONTAINS": 1}
    assert diagnostics["relations_unresolved"] == 0
    assert diagnostics["cross_scope_edges_rejected"] == 0
    assert diagnostics["unresolved_imports"] == ["a.missing", "z.missing"]
    assert diagnostics["isolates"] == sorted(diagnostics["isolates"])
    assert diagnostics["hubs"] == sorted(
        diagnostics["hubs"],
        key=lambda item: (-int(item["degree"]), str(item["qname"])),
    )

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_rejects_non_dict_relation_item_without_pydantic_crash() -> (  # noqa: E501
    None
):
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": "palace_structural_graph_contract"},
            "record": {
                "system1_graph": {
                    "parser_metadata": {"language": "python", "content_hash": "abc"},
                    "entities": [_valid_entity()],
                    "relations": ["bad"],
                }
            },
        }
    )

    result = await _service().execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_INVALID_SYSTEM1_GRAPH_PAYLOAD" in (result.manage.error or "")


@pytest.mark.asyncio
async def test_store_system1_structural_graph_rejects_non_dict_entity_item_without_pydantic_crash() -> (  # noqa: E501
    None
):
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": "palace_structural_graph_contract"},
            "record": {
                "system1_graph": {
                    "parser_metadata": {"language": "python", "content_hash": "abc"},
                    "entities": ["bad"],
                    "relations": [_valid_relation()],
                }
            },
        }
    )

    result = await _service().execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_INVALID_SYSTEM1_GRAPH_PAYLOAD" in (result.manage.error or "")


@pytest.mark.asyncio
async def test_store_system1_structural_graph_rejects_relation_dict_missing_required_keys() -> None:
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": "palace_structural_graph_contract"},
            "record": {
                "system1_graph": {
                    "parser_metadata": {"language": "python", "content_hash": "abc"},
                    "entities": [_valid_entity()],
                    "relations": [{"metadata": {"provenance": {"line": 1}}}],
                }
            },
        }
    )

    result = await _service().execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_INVALID_SYSTEM1_GRAPH_PAYLOAD" in (result.manage.error or "")


@pytest.mark.asyncio
async def test_store_system1_structural_graph_upserts_entities_with_provenance(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_entities"
    await _wipe_palace(knowledge_backend, palace)

    service = MemoryService(backend=knowledge_backend, context=Execution())
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "source_range": {"start": {"line": 1, "column": 0}},
                        "content_hash": "hash-1",
                    },
                    "entities": [
                        {
                            "qualified_name": "src.svc",
                            "stable_id": "stable-module-svc",
                            "entity_type": "Module",
                            "name": "svc",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 1, "column": 0}},
                                "content_hash": "hash-1",
                            },
                            "confidence": 0.99,
                        }
                    ],
                    "relations": [_valid_relation()],
                    "structural_evidence": [
                        {
                            "entity_stable_id": "stable-module-svc",
                            "entity_type": "module",
                            "evidence_category": "structural_module",
                            "evidence_data": {
                                "source_file": "src/svc.py",
                                "content_hash": "hash-1",
                            },
                        }
                    ],
                }
            },
        }
    )

    result = await service.execute(request)
    assert result.manage is not None
    assert result.manage.success is True
    assert result.manage.entities_stored_count == 1

    rows = await knowledge_backend.query(
        """
        SELECT source, stable_id, qualified_name, metadata, namespace, room, corridor
          FROM knowledge_entities
         WHERE palace = $1
        """,
        (palace,),
    )
    assert len(rows.rows) == 1
    row = rows.rows[0]
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert row["source"] == "STRUCTURAL"
    assert row["stable_id"] == "stable-module-svc"
    assert row["qualified_name"] == "src.svc"
    assert metadata["content_hash"] == "hash-1"
    assert "source_file" in metadata
    assert "source_range" in metadata
    assert row["namespace"] == "src"
    assert row["room"] == "svc.py"
    assert row["corridor"] == "src.svc"
    assert row["namespace"] != "default-wing"
    assert row["room"] != "default-room"

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_entity_rescan_is_idempotent(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_entities_idempotent"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "content_hash": "hash-1",
                    },
                    "entities": [
                        {
                            "qualified_name": "src.svc",
                            "stable_id": "stable-module-svc",
                            "entity_type": "Module",
                            "name": "svc",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": 1},
                                "content_hash": "hash-1",
                            },
                            "confidence": 0.99,
                        }
                    ],
                    "relations": [_valid_relation()],
                    "structural_evidence": [
                        {
                            "entity_stable_id": "stable-module-svc",
                            "entity_type": "module",
                            "evidence_category": "structural_module",
                            "evidence_data": {
                                "source_file": "src/svc.py",
                                "content_hash": "hash-1",
                            },
                        }
                    ],
                }
            },
        }
    )

    first = await service.execute(request)
    second = await service.execute(request)
    assert first.manage is not None and first.manage.success is True
    assert second.manage is not None and second.manage.success is True

    entity_count = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities WHERE palace = $1 AND stable_id = $2",
        (palace, "stable-module-svc"),
    )
    assert entity_count.rows[0]["n"] == 1

    evidence_count = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = $2
           AND evidence_category = $3
        """,
        (palace, "stable-module-svc", "structural_module"),
    )
    assert evidence_count.rows[0]["n"] == 1

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_rejects_entity_missing_stable_id(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_missing_stable_id"
    await _wipe_palace(knowledge_backend, palace)

    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "content_hash": "hash-1",
                    },
                    "entities": [
                        {
                            "qualified_name": "src.svc",
                            "stable_id": "",
                            "entity_type": "Module",
                            "name": "svc",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 1, "column": 0}},
                                "content_hash": "hash-1",
                            },
                            "confidence": 0.99,
                        }
                    ],
                    "relations": [_valid_relation()],
                }
            },
        }
    )

    service = MemoryService(backend=knowledge_backend, context=Execution())
    result = await service.execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_ENTITY_STABLE_ID_REQUIRED" in (result.manage.error or "")

    entity_count = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities WHERE palace = $1",
        (palace,),
    )
    assert entity_count.rows[0]["n"] == 0

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_rejects_entity_missing_required_metadata(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_missing_metadata"
    await _wipe_palace(knowledge_backend, palace)

    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "content_hash": "hash-1",
                    },
                    "entities": [
                        {
                            "qualified_name": "src.svc",
                            "stable_id": "stable-module-svc",
                            "entity_type": "Module",
                            "name": "svc",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "content_hash": "hash-1",
                            },
                            "confidence": 0.99,
                        }
                    ],
                    "relations": [_valid_relation()],
                }
            },
        }
    )

    service = MemoryService(backend=knowledge_backend, context=Execution())
    result = await service.execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_ENTITY_METADATA_REQUIRED" in (result.manage.error or "")
    assert "source_range" in (result.manage.error or "")

    entity_count = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_entities WHERE palace = $1",
        (palace,),
    )
    assert entity_count.rows[0]["n"] == 0

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_entity_upsert_handles_name_unique_conflict(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_name_unique_conflict"
    await _wipe_palace(knowledge_backend, palace)

    service = MemoryService(backend=knowledge_backend, context=Execution())
    base_graph = {
        "palace": palace,
        "parser_metadata": {
            "language": "python",
            "source_file": "src/greeter.py",
            "source_range": {"start": {"line": 1, "column": 0}},
            "content_hash": "hash-1",
        },
        "relations": [_valid_relation()],
    }

    first_request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    **base_graph,
                    "entities": [
                        {
                            "qualified_name": "src.greeter",
                            "stable_id": "stable-file-greeter-v1",
                            "entity_type": "File",
                            "name": "greeter.py",
                            "metadata": {
                                "source_file": "src/greeter.py",
                                "source_range": {"start": {"line": 1, "column": 0}},
                                "content_hash": "hash-1",
                            },
                            "confidence": 1.0,
                        }
                    ],
                }
            },
        }
    )
    first_result = await service.execute(first_request)
    assert first_result.manage is not None
    assert first_result.manage.success is True

    second_request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    **base_graph,
                    "entities": [
                        {
                            "qualified_name": "src.greeter",
                            "stable_id": "stable-file-greeter-v2",
                            "entity_type": "File",
                            "name": "greeter.py",
                            "metadata": {
                                "source_file": "src/greeter.py",
                                "source_range": {"start": {"line": 1, "column": 0}},
                                "content_hash": "hash-2",
                            },
                            "confidence": 1.0,
                        }
                    ],
                }
            },
        }
    )
    second_result = await service.execute(second_request)
    assert second_result.manage is not None
    assert second_result.manage.success is True

    rows = await knowledge_backend.query(
        """
        SELECT stable_id, metadata
          FROM knowledge_entities
         WHERE palace = $1
           AND entity_type = $2
           AND name = $3
        """,
        (palace, "File", "greeter.py"),
    )
    assert len(rows.rows) == 1
    row = rows.rows[0]
    assert row["stable_id"] == "stable-file-greeter-v2"
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert metadata["content_hash"] == "hash-2"

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_persists_same_name_distinct_structural_entities(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_same_name_distinct_entities"
    await _wipe_palace(knowledge_backend, palace)

    service = MemoryService(backend=knowledge_backend, context=Execution())
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/api/handlers.py",
                        "source_range": {"start": {"line": 1, "column": 0}},
                        "content_hash": "same-name-hash",
                    },
                    "entities": [
                        {
                            "qualified_name": "src.api.handlers.run",
                            "stable_id": "stable-func-run-api",
                            "entity_type": "Function",
                            "name": "run",
                            "metadata": {
                                "source_file": "src/api/handlers.py",
                                "source_range": {"start": {"line": 10, "column": 0}},
                                "content_hash": "same-name-hash",
                            },
                            "confidence": 1.0,
                        },
                        {
                            "qualified_name": "src.worker.jobs.run",
                            "stable_id": "stable-func-run-worker",
                            "entity_type": "Function",
                            "name": "run",
                            "metadata": {
                                "source_file": "src/worker/jobs.py",
                                "source_range": {"start": {"line": 22, "column": 0}},
                                "content_hash": "same-name-hash",
                            },
                            "confidence": 1.0,
                        },
                    ],
                    "relations": [],
                }
            },
        }
    )

    result = await service.execute(request)
    assert result.manage is not None
    assert result.manage.success is True

    rows = await knowledge_backend.query(
        """
        SELECT stable_id, qualified_name, name
          FROM knowledge_entities
         WHERE palace = $1
           AND source = 'STRUCTURAL'
           AND entity_type = 'Function'
           AND name = 'run'
         ORDER BY stable_id ASC
        """,
        (palace,),
    )
    assert [str(row["stable_id"]) for row in rows.rows] == [
        "stable-func-run-api",
        "stable-func-run-worker",
    ]
    assert [str(row["qualified_name"]) for row in rows.rows] == [
        "src.api.handlers.run",
        "src.worker.jobs.run",
    ]

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True

    evidence_rows = await knowledge_backend.query(
        """
        SELECT entity_stable_id
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id IN ('stable-func-run-api', 'stable-func-run-worker')
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    assert [str(row["entity_stable_id"]) for row in evidence_rows.rows] == [
        "stable-func-run-api",
        "stable-func-run-worker",
    ]

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_persists_relations_with_provenance(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_relations"
    await _wipe_palace(knowledge_backend, palace)

    service = MemoryService(backend=knowledge_backend, context=Execution())
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "source_range": {"start": {"line": 1, "column": 0}},
                        "content_hash": "hash-rel-1",
                    },
                    "entities": [
                        {
                            "qualified_name": "pkg.module",
                            "stable_id": "stable-module",
                            "entity_type": "Module",
                            "name": "module",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 1, "column": 0}},
                                "content_hash": "hash-rel-1",
                            },
                            "confidence": 0.98,
                        },
                        {
                            "qualified_name": "pkg.module.ClassA",
                            "stable_id": "stable-class-a",
                            "entity_type": "Class",
                            "name": "ClassA",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 3, "column": 0}},
                                "content_hash": "hash-rel-1",
                            },
                            "confidence": 0.97,
                        },
                    ],
                    "relations": [
                        {
                            "source_qname": "pkg.module",
                            "source_entity_type": "Module",
                            "target_qname": "pkg.module.ClassA",
                            "target_entity_type": "Class",
                            "relation_type": "CONTAINS",
                            "confidence": 0.91,
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 3, "column": 0}},
                                "content_hash": "hash-rel-1",
                                "provenance": {"kind": "treesitter", "line": 3, "column": 0},
                            },
                        }
                    ],
                }
            },
        }
    )

    result = await service.execute(request)
    assert result.manage is not None
    assert result.manage.success is True
    assert result.manage.relations_stored_count == 1

    rows = await knowledge_backend.query(
        """
        SELECT
            kr.relation_type,
            kr.confidence,
            kr.metadata,
            src.palace AS src_palace,
            tgt.palace AS tgt_palace
        FROM knowledge_relations kr
        JOIN knowledge_entities src ON src.id = kr.source_entity_id
        JOIN knowledge_entities tgt ON tgt.id = kr.target_entity_id
        WHERE src.palace = $1 AND tgt.palace = $1
        """,
        (palace,),
    )
    assert len(rows.rows) == 1
    relation = rows.rows[0]
    metadata = relation["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert relation["relation_type"] == "CONTAINS"
    assert relation["confidence"] == pytest.approx(0.91)
    assert metadata["source_qname"] == "pkg.module"
    assert metadata["target_qname"] == "pkg.module.ClassA"
    assert metadata["content_hash"] == "hash-rel-1"
    assert metadata["source_file"] == "src/svc.py"
    assert "source_range" in metadata
    assert "provenance" in metadata
    assert metadata["resolution"] in {"exact", "resolved"}

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_relation_rescan_is_idempotent(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_relation_idempotent"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "source_range": {"start": {"line": 1, "column": 0}},
                        "content_hash": "hash-rel-2",
                    },
                    "entities": [
                        {
                            "qualified_name": "pkg.m",
                            "stable_id": "stable-mod-m",
                            "entity_type": "Module",
                            "name": "m",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 1, "column": 0}},
                                "content_hash": "hash-rel-2",
                            },
                            "confidence": 0.95,
                        },
                        {
                            "qualified_name": "pkg.m.F",
                            "stable_id": "stable-f",
                            "entity_type": "Function",
                            "name": "F",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 2, "column": 0}},
                                "content_hash": "hash-rel-2",
                            },
                            "confidence": 0.95,
                        },
                    ],
                    "relations": [
                        {
                            "source_qname": "pkg.m",
                            "source_entity_type": "Module",
                            "target_qname": "pkg.m.F",
                            "target_entity_type": "Function",
                            "relation_type": "CONTAINS",
                            "confidence": 0.9,
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 2, "column": 0}},
                                "content_hash": "hash-rel-2",
                                "provenance": {"kind": "treesitter", "line": 2, "column": 0},
                            },
                        }
                    ],
                }
            },
        }
    )

    first = await service.execute(request)
    second = await service.execute(request)
    assert first.manage is not None and first.manage.success is True
    assert second.manage is not None and second.manage.success is True

    rel_count = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
        FROM knowledge_relations kr
        JOIN knowledge_entities src ON src.id = kr.source_entity_id
        JOIN knowledge_entities tgt ON tgt.id = kr.target_entity_id
        WHERE src.palace = $1
          AND tgt.palace = $1
          AND kr.relation_type IN ('CONTAINS', 'IMPORTS', 'CALLS', 'INHERITS_FROM')
        """,
        (palace,),
    )
    assert rel_count.rows[0]["n"] == 1

    rel_rows = await knowledge_backend.query(
        """
        SELECT kr.metadata
        FROM knowledge_relations kr
        JOIN knowledge_entities src ON src.id = kr.source_entity_id
        WHERE src.palace = $1 AND kr.relation_type = 'CONTAINS'
        """,
        (palace,),
    )
    assert len(rel_rows.rows) == 1
    metadata = rel_rows.rows[0]["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert metadata["source_qname"] == "pkg.m"
    assert metadata["target_qname"] == "pkg.m.F"
    assert metadata["provenance"]["kind"] == "treesitter"

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_reports_unresolved_import_relations(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_unresolved_import"
    await _wipe_palace(knowledge_backend, palace)

    service = MemoryService(backend=knowledge_backend, context=Execution())
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "source_range": {"start": {"line": 1, "column": 0}},
                        "content_hash": "hash-rel-3",
                    },
                    "entities": [
                        {
                            "qualified_name": "pkg.source",
                            "stable_id": "stable-source",
                            "entity_type": "Module",
                            "name": "source",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 1, "column": 0}},
                                "content_hash": "hash-rel-3",
                            },
                            "confidence": 0.95,
                        }
                    ],
                    "relations": [
                        {
                            "source_qname": "pkg.source",
                            "source_entity_type": "Module",
                            "target_qname": "missing.module",
                            "target_entity_type": "Module",
                            "relation_type": "IMPORTS",
                            "confidence": 0.84,
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 9, "column": 0}},
                                "content_hash": "hash-rel-3",
                                "provenance": {"kind": "treesitter", "line": 9, "column": 0},
                            },
                        }
                    ],
                    "unresolved_imports": ["missing.module"],
                }
            },
        }
    )

    result = await service.execute(request)
    assert result.manage is not None
    assert result.manage.success is True
    diagnostics = result.manage.diagnostics
    assert diagnostics["unresolved_import_count"] == 1
    assert diagnostics["unresolved_imports"] == ["missing.module"]

    false_target = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
        FROM knowledge_entities
        WHERE palace = $1
          AND source = 'STRUCTURAL'
          AND qualified_name = $2
        """,
        (palace, "missing.module"),
    )
    assert false_target.rows[0]["n"] == 0

    rel_count = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
        FROM knowledge_relations kr
        JOIN knowledge_entities src ON src.id = kr.source_entity_id
        WHERE src.palace = $1 AND kr.relation_type = 'IMPORTS'
        """,
        (palace,),
    )
    assert rel_count.rows[0]["n"] == 0

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_rejects_relation_missing_required_metadata(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_missing_relation_metadata"
    await _wipe_palace(knowledge_backend, palace)

    service = MemoryService(backend=knowledge_backend, context=Execution())
    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "content_hash": "hash-rel-4",
                    },
                    "entities": [
                        {
                            "qualified_name": "pkg.src",
                            "stable_id": "stable-src",
                            "entity_type": "Module",
                            "name": "src",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 1, "column": 0}},
                                "content_hash": "hash-rel-4",
                            },
                            "confidence": 0.95,
                        },
                        {
                            "qualified_name": "pkg.dst",
                            "stable_id": "stable-dst",
                            "entity_type": "Module",
                            "name": "dst",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 2, "column": 0}},
                                "content_hash": "hash-rel-4",
                            },
                            "confidence": 0.95,
                        },
                    ],
                    "relations": [
                        {
                            "source_qname": "pkg.src",
                            "source_entity_type": "Module",
                            "target_qname": "pkg.dst",
                            "target_entity_type": "Module",
                            "relation_type": "IMPORTS",
                            "confidence": 0.8,
                            "metadata": {
                                "source_file": "src/svc.py",
                                "content_hash": "hash-rel-4",
                                "provenance": {"kind": "treesitter", "line": 9, "column": 0},
                            },
                        }
                    ],
                }
            },
        }
    )

    result = await service.execute(request)
    assert result.manage is not None
    assert result.manage.success is False
    assert "MEM_RELATION_METADATA_REQUIRED" in (result.manage.error or "")

    rel_count = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
        FROM knowledge_relations kr
        JOIN knowledge_entities src ON src.id = kr.source_entity_id
        WHERE src.palace = $1
        """,
        (palace,),
    )
    assert rel_count.rows[0]["n"] == 0

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_relation_metadata_drift_updates_existing_edge(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_relation_metadata_drift"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    base_graph = {
        "palace": palace,
        "parser_metadata": {
            "language": "python",
            "source_file": "src/svc.py",
            "source_range": {"start": {"line": 1, "column": 0}},
            "content_hash": "hash-rel-a",
        },
        "entities": [
            {
                "qualified_name": "pkg.src",
                "stable_id": "stable-src-drift",
                "entity_type": "Module",
                "name": "src",
                "metadata": {
                    "source_file": "src/svc.py",
                    "source_range": {"start": {"line": 1, "column": 0}},
                    "content_hash": "hash-rel-a",
                },
                "confidence": 0.95,
            },
            {
                "qualified_name": "pkg.dst",
                "stable_id": "stable-dst-drift",
                "entity_type": "Module",
                "name": "dst",
                "metadata": {
                    "source_file": "src/svc.py",
                    "source_range": {"start": {"line": 2, "column": 0}},
                    "content_hash": "hash-rel-a",
                },
                "confidence": 0.95,
            },
        ],
        "relations": [
            {
                "source_qname": "pkg.src",
                "source_entity_type": "Module",
                "target_qname": "pkg.dst",
                "target_entity_type": "Module",
                "relation_type": "IMPORTS",
                "confidence": 0.81,
                "metadata": {
                    "source_file": "src/svc.py",
                    "source_range": {"start": {"line": 1, "column": 0}},
                    "content_hash": "hash-rel-a",
                    "provenance": {"kind": "treesitter", "line": 1, "column": 0},
                },
            }
        ],
    }

    first_request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {"system1_graph": base_graph},
        }
    )
    first_result = await service.execute(first_request)
    assert first_result.manage is not None
    assert first_result.manage.success is True

    updated_graph = dict(base_graph)
    updated_graph["relations"] = [dict(base_graph["relations"][0])]
    updated_graph["relations"][0]["metadata"] = {
        "source_file": "src/svc.py",
        "source_range": {"start": {"line": 99, "column": 0}},
        "content_hash": "hash-rel-b",
        "provenance": {"kind": "treesitter", "line": 99, "column": 0},
    }
    updated_graph["relations"][0]["confidence"] = 0.66

    second_request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {"system1_graph": updated_graph},
        }
    )
    second_result = await service.execute(second_request)
    assert second_result.manage is not None
    assert second_result.manage.success is True

    rel_rows = await knowledge_backend.query(
        """
        SELECT kr.id, kr.confidence, kr.metadata
        FROM knowledge_relations kr
        JOIN knowledge_entities src ON src.id = kr.source_entity_id
        JOIN knowledge_entities tgt ON tgt.id = kr.target_entity_id
        WHERE src.palace = $1
          AND tgt.palace = $1
          AND src.qualified_name = 'pkg.src'
          AND tgt.qualified_name = 'pkg.dst'
          AND kr.relation_type = 'IMPORTS'
        """,
        (palace,),
    )
    assert len(rel_rows.rows) == 1
    row = rel_rows.rows[0]
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert row["confidence"] == pytest.approx(0.66)
    assert metadata["content_hash"] == "hash-rel-b"
    assert metadata["source_range"]["start"]["line"] == 99

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_store_system1_structural_graph_rejects_ambiguous_relation_endpoint(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_structural_graph_ambiguous_endpoint"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    request = MemoryRequest.model_validate(
        {
            "operation": "store_system1_structural_graph",
            "scope": {"palace": palace},
            "record": {
                "system1_graph": {
                    "palace": palace,
                    "parser_metadata": {
                        "language": "python",
                        "source_file": "src/svc.py",
                        "source_range": {"start": {"line": 1, "column": 0}},
                        "content_hash": "hash-rel-amb",
                    },
                    "entities": [
                        {
                            "qualified_name": "pkg.source",
                            "stable_id": "stable-source-amb",
                            "entity_type": "Module",
                            "name": "source",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 1, "column": 0}},
                                "content_hash": "hash-rel-amb",
                            },
                            "confidence": 0.95,
                        },
                        {
                            "qualified_name": "pkg.target",
                            "stable_id": "stable-target-a",
                            "entity_type": "Module",
                            "name": "targetA",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 2, "column": 0}},
                                "content_hash": "hash-rel-amb",
                            },
                            "confidence": 0.95,
                        },
                        {
                            "qualified_name": "pkg.target",
                            "stable_id": "stable-target-b",
                            "entity_type": "Module",
                            "name": "targetB",
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 3, "column": 0}},
                                "content_hash": "hash-rel-amb",
                            },
                            "confidence": 0.95,
                        },
                    ],
                    "relations": [
                        {
                            "source_qname": "pkg.source",
                            "source_entity_type": "Module",
                            "target_qname": "pkg.target",
                            "target_entity_type": "Module",
                            "relation_type": "IMPORTS",
                            "confidence": 0.88,
                            "metadata": {
                                "source_file": "src/svc.py",
                                "source_range": {"start": {"line": 9, "column": 0}},
                                "content_hash": "hash-rel-amb",
                                "provenance": {"kind": "treesitter", "line": 9, "column": 0},
                            },
                        }
                    ],
                }
            },
        }
    )

    result = await service.execute(request)
    assert result.manage is not None
    assert result.manage.success is True
    diagnostics = result.manage.diagnostics
    unresolved_relations = diagnostics.get("unresolved_relations", [])
    assert any(item.get("reason") == "ambiguous_target" for item in unresolved_relations)
    ambiguous_entry = next(
        item for item in unresolved_relations if item.get("reason") == "ambiguous_target"
    )
    assert len(ambiguous_entry.get("candidates", [])) >= 2

    rel_count = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
        FROM knowledge_relations kr
        JOIN knowledge_entities src ON src.id = kr.source_entity_id
        WHERE src.palace = $1
        """,
        (palace,),
    )
    assert rel_count.rows[0]["n"] == 0

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_namespaces_forbidden_path_labels_and_reports_path_diagnostics(  # noqa: E501
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_forbidden_path_labels"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    payload = build_three_entity_graph_payload(palace)
    entities = payload["record"]["system1_graph"]["entities"]
    for entity in entities:
        stable_id = str(entity.get("stable_id"))
        metadata = entity.setdefault("metadata", {})
        if stable_id == "stable-fn":
            metadata["repo_relative_path"] = "default-wing/default-room/default.py"
            metadata["source_file"] = "default-wing/default-room/default.py"
        elif stable_id == "stable-mod":
            metadata["repo_relative_path"] = "graph-component-9/cluster-2/reasoning-unit-3.py"
            metadata["source_file"] = "graph-component-9/cluster-2/reasoning-unit-3.py"
        else:
            metadata["repo_relative_path"] = "code/code/code.py"
            metadata["source_file"] = "code/code/code.py"

    store_request = MemoryRequest.model_validate(payload)
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert diagnostics.get("assignment_strategy") == "path_metadata"
    assert diagnostics.get("path_deferred_entities") == []
    assert diagnostics.get("path_deferred_count") == 0

    rows = await knowledge_backend.query(
        """
        SELECT entity_stable_id, wing, room, compartment
          FROM knowledge_structural_evidence
         WHERE palace = $1
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    assert len(rows.rows) == 3
    forbidden_exact = {"default-wing", "default-room", "default", "code"}
    forbidden_prefixes = ("graph-component-", "cluster-", "reasoning-unit-")
    for row in rows.rows:
        for value in (str(row["wing"]), str(row["room"]), str(row["compartment"])):
            assert value not in forbidden_exact
            assert all(not value.startswith(prefix) for prefix in forbidden_prefixes)
            assert value.startswith("path-")

    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    payload = build_three_entity_graph_payload(palace)
    entities = payload["record"]["system1_graph"]["entities"]
    for entity in entities:
        stable_id = str(entity.get("stable_id"))
        metadata = entity.setdefault("metadata", {})
        if stable_id == "stable-fn":
            metadata["repo_relative_path"] = "src/api/handlers/user.py"
            metadata["source_file"] = "src/api/handlers/user.py"
        else:
            metadata["repo_relative_path"] = "src/api/models/profile.py"
            metadata["source_file"] = "src/api/models/profile.py"

    store_request = MemoryRequest.model_validate(payload)
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert diagnostics.get("selected_entities") == 3
    assert diagnostics.get("selected_relations") == 2
    assert diagnostics.get("guard_blocked_entities") == 0
    assert diagnostics.get("eligible_entities") == 3
    assert diagnostics.get("updated_rows") == 3

    rows = await knowledge_backend.query(
        """
        SELECT entity_stable_id, wing, room, compartment
          FROM knowledge_structural_evidence
         WHERE palace = $1
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    expected = {
        "stable-fn": ("src", "api", "user"),
        "stable-mod": ("src", "api", "profile"),
        "stable-pkg": ("src", "api", "profile"),
    }
    assert len(rows.rows) == 3
    for row in rows.rows:
        stable_id = str(row["entity_stable_id"])
        assert (str(row["wing"]), str(row["room"]), str(row["compartment"])) == expected[stable_id]
        assert str(row["wing"]) not in {
            "default-wing",
            "default",
            "code",
            "graph-component-1",
            "cluster-1",
            "reasoning-unit-1",
        }
        assert str(row["room"]) not in {
            "default-room",
            "default",
            "code",
            "graph-component-1",
            "cluster-1",
            "reasoning-unit-1",
        }
        assert str(row["compartment"]) not in {
            "default",
            "code",
            "graph-component-1",
            "cluster-1",
            "reasoning-unit-1",
        }

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_ignores_non_structural_relations(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_ignore_non_structural_relations"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    store_request = MemoryRequest.model_validate(build_three_entity_graph_payload(palace))
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    initial_derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    initial_derive_result = await service.execute(initial_derive_request)
    assert initial_derive_result.manage is not None
    assert initial_derive_result.manage.success is True

    initial_derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    initial_derive_result = await service.execute(initial_derive_request)
    assert initial_derive_result.manage is not None
    assert initial_derive_result.manage.success is True

    await knowledge_backend.execute(
        (
            "DELETE FROM knowledge_relations "
            "WHERE source_entity_id IN ("
            "SELECT id FROM knowledge_entities WHERE palace = $1)"
        ),
        (palace,),
    )

    relation_endpoints = await knowledge_backend.query(
        (
            "SELECT stable_id, id "
            "FROM knowledge_entities "
            "WHERE palace = $1 "
            "AND stable_id IN ('stable-pkg', 'stable-mod')"
        ),
        (palace,),
    )
    endpoint_by_stable_id = {
        str(row["stable_id"]): str(row["id"]) for row in relation_endpoints.rows
    }
    assert set(endpoint_by_stable_id) == {"stable-pkg", "stable-mod"}

    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_relations
            (id, source_entity_id, target_entity_id, relation_type,
             confidence, evidence_memory_ids, metadata)
        VALUES
            ($1::uuid, $2::uuid, $3::uuid, $4,
             $5, $6::uuid[], $7::jsonb)
        """,
        (
            str(uuid.uuid4()),
            endpoint_by_stable_id["stable-pkg"],
            endpoint_by_stable_id["stable-mod"],
            "MENTIONS",
            1.0,
            [],
            json.dumps({"reason": "non_structural_test_relation"}),
        ),
    )

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert diagnostics.get("selected_entities") == 3
    assert diagnostics.get("selected_relations") == 0
    assert diagnostics.get("guard_blocked_entities") == 0
    assert diagnostics.get("eligible_entities") == 3
    assert diagnostics.get("updated_rows") == 3
    deferred = diagnostics.get("deferred_isolates", [])
    assert deferred == []

    rows = await knowledge_backend.query(
        """
        SELECT entity_stable_id, wing, room, compartment
          FROM knowledge_structural_evidence
         WHERE palace = $1
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    assert len(rows.rows) == 3
    for row in rows.rows:
        assert row["wing"] == "src"
        assert row["room"] == "app.py"
        assert row["compartment"] == "app"

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_falls_back_to_source_file_when_repo_relative_path_missing(  # noqa: E501
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_source_file_fallback"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    payload = build_three_entity_graph_payload(palace)
    entities = payload["record"]["system1_graph"]["entities"]
    for entity in entities:
        metadata = entity.setdefault("metadata", {})
        metadata.pop("repo_relative_path", None)
        metadata["source_file"] = "src/domain/model/user.py"

    store_request = MemoryRequest.model_validate(payload)
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True

    rows = await knowledge_backend.query(
        """
        SELECT entity_stable_id, wing, room, compartment
          FROM knowledge_structural_evidence
         WHERE palace = $1
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    assert len(rows.rows) == 3
    for row in rows.rows:
        assert row["wing"] == "src"
        assert row["room"] == "domain"
        assert row["compartment"] == "user"

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_requires_palace() -> None:
    service = _service()
    request = MemoryRequest.model_validate({"operation": "derive_system1_project_topology"})

    with pytest.raises(MemoryContractError) as exc_info:
        await service.execute(request)
    assert "MEM_PALACE_REQUIRED" in str(exc_info.value)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_ignores_non_structural_entities(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_structural_only"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    store_request = MemoryRequest.model_validate(build_three_entity_graph_payload(palace))
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    # First derive creates structural evidence rows needed for explicit override evidence_ids.
    initial_derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    initial_derive_result = await service.execute(initial_derive_request)
    assert initial_derive_result.manage is not None
    assert initial_derive_result.manage.success is True

    await knowledge_backend.execute(
        "UPDATE knowledge_entities SET source = 'USER' WHERE palace = $1 AND stable_id = $2",
        (palace, "stable-fn"),
    )
    await knowledge_backend.execute(
        "DELETE FROM knowledge_structural_evidence WHERE palace = $1 AND entity_stable_id = $2",
        (palace, "stable-fn"),
    )

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert diagnostics.get("selected_entities") == 2
    assert diagnostics.get("selected_relations") == 1
    assert diagnostics.get("component_count") == 0
    assert diagnostics.get("isolate_count") == 0
    assert diagnostics.get("non_isolate_entities") == 2
    assert diagnostics.get("guard_blocked_entities") == 0
    assert diagnostics.get("eligible_entities") == 2
    assert diagnostics.get("updated_rows") == 2

    non_structural = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = $2
        """,
        (palace, "stable-fn"),
    )
    assert non_structural.rows[0]["n"] == 0

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_treats_admin_sync_default_as_non_blocking(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_admin_sync_default_non_blocking"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    store_request = MemoryRequest.model_validate(build_three_entity_graph_payload(palace))
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    initial_derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    initial_derive_result = await service.execute(initial_derive_request)
    assert initial_derive_result.manage is not None
    assert initial_derive_result.manage.success is True

    evidence_rows = await knowledge_backend.query(
        """
        SELECT id
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id IN ('stable-pkg', 'stable-mod')
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    assert len(evidence_rows.rows) == 2
    evidence_ids = [str(row["id"]) for row in evidence_rows.rows]

    project_default_override_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": palace,
                "evidence_ids": evidence_ids,
                "topology_override": {
                    "wing": "default-wing",
                    "room": "default-room",
                    "compartment": "default-compartment",
                    "override_reason": "Project default topology for System 1 project sync",
                    "applied_by": "admin_sync",
                },
            },
        }
    )
    project_default_override_result = await service.execute(project_default_override_request)
    assert project_default_override_result.manage is not None
    assert project_default_override_result.manage.success is True

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True

    diagnostics = derive_result.manage.diagnostics
    assert diagnostics.get("guard_blocked_entities") == 0
    assert diagnostics.get("eligible_entities") == 3
    assert diagnostics.get("updated_rows") == 3

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_coalesces_duplicate_evidence_rows_before_scope_update(  # noqa: E501
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_coalesces_duplicate_evidence"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    store_request = MemoryRequest.model_validate(build_three_entity_graph_payload(palace))
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    initial_derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    initial_derive_result = await service.execute(initial_derive_request)
    assert initial_derive_result.manage is not None
    assert initial_derive_result.manage.success is True

    category_row = await knowledge_backend.query(
        """
        SELECT evidence_category
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = 'stable-pkg'
         ORDER BY scanned_at DESC, id DESC
         LIMIT 1
        """,
        (palace,),
    )
    assert len(category_row.rows) == 1
    evidence_category = str(category_row.rows[0]["evidence_category"])

    duplicate_id = str(uuid.uuid4())
    await knowledge_backend.execute(
        """
        UPDATE knowledge_structural_evidence
           SET wing = '',
               room = '',
               compartment = '',
               scanned_at = NOW() - INTERVAL '2 days'
         WHERE palace = $1
            AND entity_stable_id = 'stable-pkg'
            AND evidence_category = $2
        """,
        (palace, evidence_category),
    )
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_structural_evidence
            (id, palace, wing, room, compartment,
             entity_stable_id, entity_type, evidence_category,
             evidence_payload, scanned_at)
        VALUES (
            $1::uuid, $2, 'default-wing', 'default-room', 'forge',
            'stable-pkg', 'Module', $3,
            '{"source_file": "src/app.py"}'::jsonb,
            NOW() - INTERVAL '1 day'
        )
        """,
        (duplicate_id, palace, evidence_category),
    )
    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert diagnostics.get("coalesced_evidence_rows") == 1

    rows = await knowledge_backend.query(
        """
        SELECT id::text AS id, wing, room, compartment, scanned_at
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = 'stable-pkg'
           AND evidence_category = $2
         ORDER BY scanned_at DESC, id DESC
        """,
        (palace, evidence_category),
    )
    assert len(rows.rows) == 1
    row = rows.rows[0]
    assert row["id"] == duplicate_id
    assert row["wing"] == "src"
    assert row["room"] == "app.py"
    assert row["compartment"] == "app"

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_preserves_referenced_duplicate_evidence_row(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_preserves_referenced_duplicate"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    store_request = MemoryRequest.model_validate(build_three_entity_graph_payload(palace))
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    initial_derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    initial_derive_result = await service.execute(initial_derive_request)
    assert initial_derive_result.manage is not None
    assert initial_derive_result.manage.success is True

    linked_row = await knowledge_backend.query(
        """
        SELECT id::text AS id, evidence_category
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = 'stable-pkg'
         ORDER BY scanned_at DESC, id DESC
         LIMIT 1
        """,
        (palace,),
    )
    assert len(linked_row.rows) == 1
    linked_evidence_id = str(linked_row.rows[0]["id"])
    evidence_category = str(linked_row.rows[0]["evidence_category"])

    explicit_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": palace,
                "evidence_ids": [linked_evidence_id],
                "topology_override": {
                    "wing": "explicit-wing",
                    "room": "explicit-room",
                    "compartment": "explicit-compartment",
                    "override_reason": _PROJECT_DEFAULT_TOPOLOGY_OVERRIDE_REASON,
                    "applied_by": _PROJECT_DEFAULT_TOPOLOGY_APPLIED_BY,
                },
            },
        }
    )
    explicit_result = await service.execute(explicit_request)
    assert explicit_result.manage is not None
    assert explicit_result.manage.success is True

    await knowledge_backend.execute(
        """
        UPDATE knowledge_structural_evidence
           SET wing = '',
               room = '',
               compartment = '',
               scanned_at = NOW() - INTERVAL '2 days'
         WHERE id = $1::uuid
        """,
        (linked_evidence_id,),
    )

    duplicate_id = str(uuid.uuid4())
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_structural_evidence
            (id, palace, wing, room, compartment,
             entity_stable_id, entity_type, evidence_category,
             evidence_payload, scanned_at)
        VALUES (
            $1::uuid, $2, 'default-wing', 'default-room', 'forge',
            'stable-pkg', 'Module', $3,
            '{"source_file": "src/app.py"}'::jsonb,
            NOW() - INTERVAL '1 day'
        )
        """,
        (duplicate_id, palace, evidence_category),
    )

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert diagnostics.get("coalesced_evidence_rows") == 1

    remaining_rows = await knowledge_backend.query(
        """
        SELECT id::text AS id, wing, room, compartment
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = 'stable-pkg'
           AND evidence_category = $2
         ORDER BY scanned_at DESC, id DESC
        """,
        (palace, evidence_category),
    )
    assert len(remaining_rows.rows) == 1
    remaining_row = remaining_rows.rows[0]
    assert remaining_row["id"] == linked_evidence_id
    assert remaining_row["wing"] == "src"
    assert remaining_row["room"] == "app.py"
    assert remaining_row["compartment"] == "app"

    provenance_link_rows = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS link_count
          FROM knowledge_topology_provenance_evidence
         WHERE evidence_id = $1::uuid
        """,
        (linked_evidence_id,),
    )
    assert int(provenance_link_rows.rows[0]["link_count"]) >= 1

    duplicate_lookup = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS duplicate_count
          FROM knowledge_structural_evidence
         WHERE id = $1::uuid
        """,
        (duplicate_id,),
    )
    assert int(duplicate_lookup.rows[0]["duplicate_count"]) == 0

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_preserves_explicit_topology_override(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_preserve_explicit"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    store_request = MemoryRequest.model_validate(build_three_entity_graph_payload(palace))
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    initial_derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    initial_derive_result = await service.execute(initial_derive_request)
    assert initial_derive_result.manage is not None
    assert initial_derive_result.manage.success is True

    evidence_rows = await knowledge_backend.query(
        """
        SELECT id
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id IN ('stable-pkg', 'stable-mod')
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    assert len(evidence_rows.rows) == 2
    evidence_ids = [str(row["id"]) for row in evidence_rows.rows]

    explicit_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_topology",
            "derivation": {
                "palace": palace,
                "evidence_ids": evidence_ids,
                "topology_override": {
                    "wing": "explicit-wing",
                    "room": "explicit-room",
                    "compartment": "explicit-compartment",
                    "override_reason": "manual override for test",
                    "applied_by": "test-suite",
                },
            },
        }
    )
    explicit_result = await service.execute(explicit_request)
    assert explicit_result.manage is not None
    assert explicit_result.manage.success is True

    await knowledge_backend.execute(
        """
        UPDATE knowledge_structural_evidence
           SET wing = 'explicit-wing',
               room = 'explicit-room',
               compartment = 'explicit-compartment'
         WHERE palace = $1
           AND entity_stable_id IN ('stable-pkg', 'stable-mod')
        """,
        (palace,),
    )

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert diagnostics.get("guard_blocked_entities") == 2
    assert diagnostics.get("eligible_entities") == 1
    assert diagnostics.get("updated_rows") == 1

    rows = await knowledge_backend.query(
        """
        SELECT entity_stable_id, wing, room, compartment
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id IN ('stable-pkg', 'stable-mod')
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    assert len(rows.rows) == 2
    for row in rows.rows:
        assert row["wing"] == "explicit-wing"
        assert row["room"] == "explicit-room"
        assert row["compartment"] == "explicit-compartment"

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_prunes_orphan_empty_scope_evidence_without_provenance(  # noqa: E501
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_prune_orphan_empty"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    store_request = MemoryRequest.model_validate(build_three_entity_graph_payload(palace))
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    await knowledge_backend.execute(
        "DELETE FROM knowledge_entities WHERE palace = $1 AND stable_id = 'stable-pkg'",
        (palace,),
    )
    await knowledge_backend.execute(
        """
        INSERT INTO knowledge_structural_evidence
            (id, palace, wing, room, compartment,
             entity_stable_id, entity_type, evidence_category,
             evidence_payload, scanned_at)
        VALUES (
            $1::uuid, $2, '', '', '',
            'stable-pkg', 'Module', 'structural_module',
            '{"source_file": "src/app.py"}'::jsonb,
            NOW()
        )
        """,
        (str(uuid.uuid4()), palace),
    )

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert int(diagnostics.get("orphan_empty_evidence_pruned", 0)) >= 1

    orphan_lookup = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = 'stable-pkg'
           AND wing = ''
           AND room = ''
           AND compartment = ''
        """,
        (palace,),
    )
    assert int(orphan_lookup.rows[0]["n"]) == 0

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_prunes_legacy_default_orphans_and_preserves_explicit_orphans(  # noqa: E501
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_prune_legacy_default_orphan"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    store_payload = build_three_entity_graph_payload(palace)
    store_payload["record"]["system1_graph"]["entities"].append(
        {
            "qualified_name": "pkg.explicit_override_target",
            "stable_id": "stable-explicit-target",
            "entity_type": "Module",
            "name": "explicit_override_target",
            "metadata": {
                "source_file": "src/explicit_target.py",
                "source_range": {"start": {"line": 1, "column": 0}},
                "content_hash": "graph123",
            },
            "confidence": 0.97,
        }
    )
    store_result = await service.execute(MemoryRequest.model_validate(store_payload))
    assert store_result.manage is not None
    assert store_result.manage.success is True

    initial_derive = await service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system1_project_topology",
                "scope": {"palace": palace},
            }
        )
    )
    assert initial_derive.manage is not None
    assert initial_derive.manage.success is True

    evidence_rows = await knowledge_backend.query(
        """
        SELECT id::text AS id, entity_stable_id
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id IN ('stable-pkg', 'stable-explicit-target')
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    assert len(evidence_rows.rows) == 2
    by_stable = {str(row["entity_stable_id"]): str(row["id"]) for row in evidence_rows.rows}
    legacy_evidence_id = by_stable["stable-pkg"]
    explicit_evidence_id = by_stable["stable-explicit-target"]

    legacy_override = await service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system1_topology",
                "derivation": {
                    "palace": palace,
                    "evidence_ids": [legacy_evidence_id],
                    "topology_override": {
                        "wing": "default-wing",
                        "room": "default-room",
                        "compartment": "forge",
                        "override_reason": _PROJECT_DEFAULT_TOPOLOGY_OVERRIDE_REASON,
                        "applied_by": _PROJECT_DEFAULT_TOPOLOGY_APPLIED_BY,
                    },
                },
            }
        )
    )
    assert legacy_override.manage is not None
    assert legacy_override.manage.success is True

    explicit_override = await service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system1_topology",
                "derivation": {
                    "palace": palace,
                    "evidence_ids": [explicit_evidence_id],
                    "topology_override": {
                        "wing": "custom-wing",
                        "room": "custom-room",
                        "compartment": "custom-compartment",
                        "override_reason": "manual override",
                        "applied_by": "test-suite",
                    },
                },
            }
        )
    )
    assert explicit_override.manage is not None
    assert explicit_override.manage.success is True

    await knowledge_backend.execute(
        """
        UPDATE knowledge_structural_evidence
           SET wing = 'default-wing', room = 'default-room', compartment = 'forge'
         WHERE id = $1::uuid
        """,
        (legacy_evidence_id,),
    )
    await knowledge_backend.execute(
        """
        UPDATE knowledge_structural_evidence
           SET wing = 'custom-wing', room = 'custom-room', compartment = 'custom-compartment'
         WHERE id = $1::uuid
        """,
        (explicit_evidence_id,),
    )

    await knowledge_backend.execute(
        "DELETE FROM knowledge_entities "
        "WHERE palace = $1 "
        "AND stable_id IN ('stable-pkg', 'stable-explicit-target')",
        (palace,),
    )

    derive_result = await service.execute(
        MemoryRequest.model_validate(
            {
                "operation": "derive_system1_project_topology",
                "scope": {"palace": palace},
            }
        )
    )
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert int(diagnostics.get("legacy_default_evidence_pruned", 0)) >= 1

    legacy_after = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_structural_evidence WHERE id = $1::uuid",
        (legacy_evidence_id,),
    )
    assert int(legacy_after.rows[0]["n"]) == 0

    explicit_after = await knowledge_backend.query(
        "SELECT COUNT(*)::int AS n FROM knowledge_structural_evidence WHERE id = $1::uuid",
        (explicit_evidence_id,),
    )
    assert int(explicit_after.rows[0]["n"]) == 1

    explicit_links = await knowledge_backend.query(
        """
        SELECT COUNT(*)::int AS n
          FROM knowledge_topology_provenance_evidence
         WHERE evidence_id = $1::uuid
        """,
        (explicit_evidence_id,),
    )
    assert int(explicit_links.rows[0]["n"]) >= 1

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_is_idempotent(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_idempotent"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    store_request = MemoryRequest.model_validate(build_three_entity_graph_payload(palace))
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    first = await service.execute(derive_request)
    second = await service.execute(derive_request)
    assert first.manage is not None and first.manage.success is True
    assert second.manage is not None and second.manage.success is True

    rows = await knowledge_backend.query(
        """
        SELECT entity_stable_id, wing, room, compartment
          FROM knowledge_structural_evidence
         WHERE palace = $1
         ORDER BY entity_stable_id ASC
        """,
        (palace,),
    )
    assert len(rows.rows) == 3
    expected_by_entity = {
        str(row["entity_stable_id"]): (str(row["wing"]), str(row["room"]), str(row["compartment"]))
        for row in rows.rows
    }
    assert expected_by_entity == first.manage.diagnostics.get("applied_topology_by_entity")
    assert expected_by_entity == second.manage.diagnostics.get("applied_topology_by_entity")

    await _wipe_palace(knowledge_backend, palace)


@pytest.mark.asyncio
async def test_derive_system1_project_topology_defers_isolates_without_path_only_topology(
    knowledge_backend: PostgresBackend,
) -> None:
    palace = "palace_system1_project_topology_isolates"
    await _wipe_palace(knowledge_backend, palace)
    service = MemoryService(backend=knowledge_backend, context=Execution())

    payload = build_three_entity_graph_payload(palace)
    payload["record"]["system1_graph"]["entities"].append(
        {
            "qualified_name": "pkg.isolated",
            "stable_id": "stable-isolated",
            "entity_type": "Module",
            "name": "isolated",
            "metadata": {
                "source_file": "",
                "source_range": {"start": {"line": 1, "column": 0}},
                "content_hash": "graph123",
            },
            "confidence": 1.0,
        }
    )
    payload["record"]["system1_graph"].setdefault("structural_evidence", [])
    payload["record"]["system1_graph"]["structural_evidence"].append(
        {
            "entity_stable_id": "stable-isolated",
            "entity_type": "module",
            "evidence_category": "structural_module",
            "evidence_data": {
                "content_hash": "graph123",
            },
        }
    )

    store_request = MemoryRequest.model_validate(payload)
    store_result = await service.execute(store_request)
    assert store_result.manage is not None
    assert store_result.manage.success is True

    derive_request = MemoryRequest.model_validate(
        {
            "operation": "derive_system1_project_topology",
            "scope": {"palace": palace},
        }
    )
    derive_result = await service.execute(derive_request)
    assert derive_result.manage is not None
    assert derive_result.manage.success is True
    diagnostics = derive_result.manage.diagnostics
    assert diagnostics.get("assignment_strategy") == "path_metadata"
    assert diagnostics.get("selected_entities") == 4
    assert diagnostics.get("selected_relations") == 2
    assert diagnostics.get("guard_blocked_entities") == 0
    assert diagnostics.get("eligible_entities") == 3
    assert diagnostics.get("updated_rows") == 3
    deferred = diagnostics.get("deferred_isolates", [])
    assert "stable-isolated" in deferred
    assert diagnostics.get("path_deferred_entities") == deferred
    assert diagnostics.get("path_deferred_count") == len(deferred)

    isolated_row = await knowledge_backend.query(
        """
        SELECT wing, room, compartment
          FROM knowledge_structural_evidence
         WHERE palace = $1
           AND entity_stable_id = 'stable-isolated'
         LIMIT 1
        """,
        (palace,),
    )
    assert len(isolated_row.rows) == 1
    assert isolated_row.rows[0]["wing"] == ""
    assert isolated_row.rows[0]["room"] == ""
    assert isolated_row.rows[0]["compartment"] == ""

    await _wipe_palace(knowledge_backend, palace)
