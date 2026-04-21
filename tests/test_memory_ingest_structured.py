"""Focused tests for structured memory ingestion.

These tests cover the new shared ingest_structured manage operation at the
service layer. Database interactions are mocked; no real PostgreSQL required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from workflows_mcp.engine.memory_service import (
    ManageMemoryRequest,
    ManageMemoryResult,
    MemoryRequest,
    MemoryService,
)


class TestStructuredIngestService:
    """Tests for MemoryService ingest_structured behavior."""

    def _embedding_patch(self) -> Any:
        """Stub embedding generation for structured ingest service tests."""
        return patch(
            "workflows_mcp.engine.memory_service.compute_embedding",
            new=AsyncMock(return_value=([0.1, 0.2, 0.3], "embed-model", 3, None)),
        )

    @pytest.mark.asyncio
    async def test_ingest_structured_writes_memories_entities_relations_and_links_atomically(
        self,
    ) -> None:
        """Structured ingest writes all graph tables in one transaction and reports affected upsert IDs/counts."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[{"id": "source-id"}]),
                MagicMock(rows=[{"id": "item-id"}]),
                MagicMock(rows=[{"id": "entity-alice-id"}]),
                MagicMock(rows=[{"id": "entity-acme-id"}]),
                MagicMock(rows=[{"id": "relation-id"}]),
            ]
        )

        context = MagicMock()
        context.execution_context = None

        generated_ids = [
            "source-upsert-id",
            "item-upsert-id",
            "memory-0-id",
            "entity-alice-upsert-id",
            "entity-acme-upsert-id",
            "relation-upsert-id",
        ]

        service = MemoryService(backend=backend, context=context)

        with (
            patch(
                "workflows_mcp.engine.memory_service.uuid.uuid4",
                side_effect=generated_ids,
            ),
            self._embedding_patch(),
        ):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="ingest_structured",
                    source="spec-tests",
                    path="docs/spec.md",
                    namespace="engineering",
                    room="memory",
                    memories=[
                        {
                            "content": "Alice works at Acme.",
                            "confidence": 0.91,
                            "authority": "AGENT",
                            "lifecycle_state": "ACTIVE",
                            "metadata": {"origin": "unit-test"},
                        }
                    ],
                    entities=[
                        {
                            "name": "Alice",
                            "entity_type": "PERSON",
                            "confidence": 0.95,
                            "memory_indices": [0],
                        },
                        {
                            "name": "Acme",
                            "entity_type": "ORG",
                            "confidence": 0.89,
                            "memory_indices": [0],
                        },
                    ],
                    relations=[
                        {
                            "source_name": "Alice",
                            "source_type": "PERSON",
                            "target_name": "Acme",
                            "target_type": "ORG",
                            "relation_type": "works_at",
                            "confidence": 0.88,
                            "evidence_memory_index": 0,
                        }
                    ],
                )
            )

        assert result.success is True
        assert result.operation == "ingest_structured"
        assert result.stored_count == 1
        assert result.memory_ids == ["memory-0-id"]
        assert result.entity_ids == ["entity-alice-id", "entity-acme-id"]
        assert result.relation_ids == ["relation-id"]
        assert result.entities_stored_count == 2
        assert result.relations_stored_count == 1
        assert result.entities_stored_count == len(result.entity_ids)
        assert result.relations_stored_count == len(result.relation_ids)
        backend.begin_transaction.assert_awaited_once()
        backend.commit.assert_awaited_once()
        backend.rollback.assert_not_awaited()

        executed_sql = [call.args[0] for call in backend.execute.await_args_list]
        assert any("INSERT INTO knowledge_memories" in sql for sql in executed_sql)
        assert any("INSERT INTO knowledge_entity_memories" in sql for sql in executed_sql)
        assert any(
            "INSERT INTO knowledge_entity_memories" in sql and call.args[1] == ("memory-0-id", "entity-alice-id", "mentioned", 0.95)
            for call, sql in zip(backend.execute.await_args_list, executed_sql, strict=False)
        )
        assert any(
            "INSERT INTO knowledge_entity_memories" in sql and call.args[1] == ("memory-0-id", "entity-acme-id", "mentioned", 0.89)
            for call, sql in zip(backend.execute.await_args_list, executed_sql, strict=False)
        )

        queried_sql = [call.args[0] for call in backend.query.await_args_list]
        assert any("INSERT INTO knowledge_sources" in sql for sql in queried_sql)
        assert any("INSERT INTO knowledge_items" in sql for sql in queried_sql)
        assert any("INSERT INTO knowledge_entities" in sql for sql in queried_sql)
        assert any("INSERT INTO knowledge_relations" in sql for sql in queried_sql)

    @pytest.mark.asyncio
    async def test_ingest_structured_computes_embeddings_and_logs_creation_audits(self) -> None:
        """Structured ingest must store searchable memory fields and emit creation audits."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock()

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with (
            patch(
                "workflows_mcp.engine.memory_service.uuid.uuid4",
                side_effect=["memory-0-id"],
            ),
            patch(
                "workflows_mcp.engine.memory_service.compute_embedding",
                new=AsyncMock(return_value=([0.1, 0.2, 0.3], "embed-model", 3, None)),
            ),
        ):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="ingest_structured",
                    memories=[{"content": "Searchable memory."}],
                    entities=[],
                    relations=[],
                )
            )

        assert result.success is True
        memory_insert_calls = [
            call for call in backend.execute.await_args_list if "INSERT INTO knowledge_memories" in call.args[0]
        ]
        assert len(memory_insert_calls) == 1
        memory_insert_sql = memory_insert_calls[0].args[0]
        memory_insert_params = memory_insert_calls[0].args[1]
        assert "embedding" in memory_insert_sql
        assert "search_vector" in memory_insert_sql
        assert "embedding_model" in memory_insert_sql
        assert "to_tsvector('english', $3)" in memory_insert_sql
        assert memory_insert_params[3] == "[0.1, 0.2, 0.3]"
        assert memory_insert_params[7] == "embed-model"

        audit_calls = [
            call for call in backend.execute.await_args_list if "INSERT INTO knowledge_memory_audits" in call.args[0]
        ]
        assert len(audit_calls) == 1

    @pytest.mark.asyncio
    async def test_ingest_structured_defaults_entities_and_relations_when_omitted(self) -> None:
        """Structured ingest should accept payloads with only memories and default graph inputs to empty."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock()

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with (
            patch(
                "workflows_mcp.engine.memory_service.uuid.uuid4",
                side_effect=["memory-0-id"],
            ),
            self._embedding_patch(),
        ):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="ingest_structured",
                    memories=[{"content": "Memory-only ingest."}],
                )
            )

        assert result.success is True
        assert result.stored_count == 1
        assert result.memory_ids == ["memory-0-id"]
        assert result.entity_ids == []
        assert result.relation_ids == []
        assert result.entities_stored_count == 0
        assert result.relations_stored_count == 0
        backend.begin_transaction.assert_awaited_once()
        backend.commit.assert_awaited_once()
        backend.rollback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ingest_structured_without_memories_returns_explicit_error(self) -> None:
        """Structured ingest should fail with an explicit contract error when memories is omitted."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        result = await service.manage(
            ManageMemoryRequest(
                operation="ingest_structured",
                entities=[],
                relations=[],
            )
        )

        assert result.success is False
        assert result.error is not None
        assert "'memories' is required" in result.error
        backend.begin_transaction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ingest_structured_rolls_back_on_write_failure(self) -> None:
        """Structured ingest rolls back the transaction when any write fails."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[{"id": "entity-id"}]))
        backend.execute = AsyncMock(side_effect=RuntimeError("link insert failed"))

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with (
            patch(
                "workflows_mcp.engine.memory_service.uuid.uuid4",
                side_effect=["memory-0-id", "entity-upsert-id"],
            ),
            self._embedding_patch(),
        ):
            with pytest.raises(RuntimeError, match="link insert failed"):
                await service.manage(
                    ManageMemoryRequest(
                        operation="ingest_structured",
                        memories=[{"content": "Alice works at Acme."}],
                        entities=[
                            {
                                "name": "Alice",
                                "entity_type": "PERSON",
                                "memory_indices": [0],
                            }
                        ],
                        relations=[],
                    )
                )

        backend.begin_transaction.assert_awaited_once()
        backend.commit.assert_not_awaited()
        backend.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ingest_structured_links_relation_only_entities_to_evidence_memory(self) -> None:
        """Relation-only endpoint auto-upserts still create direct memory-entity links."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[{"id": "entity-alice-id"}]),
                MagicMock(rows=[{"id": "entity-acme-id"}]),
                MagicMock(rows=[{"id": "relation-id"}]),
            ]
        )

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with (
            patch(
                "workflows_mcp.engine.memory_service.uuid.uuid4",
                side_effect=["memory-0-id", "entity-alice-upsert-id", "entity-acme-upsert-id", "relation-upsert-id"],
            ),
            self._embedding_patch(),
        ):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="ingest_structured",
                    memories=[{"content": "Alice works at Acme."}],
                    entities=[],
                    relations=[
                        {
                            "source_name": "Alice",
                            "source_type": "PERSON",
                            "target_name": "Acme",
                            "target_type": "ORG",
                            "relation_type": "works_at",
                            "confidence": 0.88,
                            "evidence_memory_index": 0,
                        }
                    ],
                )
            )

        assert result.success is True
        link_params = [
            call.args[1]
            for call in backend.execute.await_args_list
            if "INSERT INTO knowledge_entity_memories" in call.args[0]
        ]
        assert ("memory-0-id", "entity-alice-id", "mentioned", 0.88) in link_params
        assert ("memory-0-id", "entity-acme-id", "mentioned", 0.88) in link_params

    @pytest.mark.asyncio
    async def test_ingest_structured_reports_actual_memory_count(self) -> None:
        """stored_count reflects the number of inserted memories, not a fixed bundle count."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock()

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with (
            patch(
                "workflows_mcp.engine.memory_service.uuid.uuid4",
                side_effect=["memory-0-id", "memory-1-id"],
            ),
            self._embedding_patch(),
        ):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="ingest_structured",
                    memories=[
                        {"content": "First memory."},
                        {"content": "Second memory."},
                    ],
                    entities=[],
                    relations=[],
                )
            )

        assert result.success is True
        assert result.memory_ids == ["memory-0-id", "memory-1-id"]
        assert result.stored_count == 2

    @pytest.mark.asyncio
    async def test_ingest_structured_uses_request_confidence_when_nested_confidence_missing(self) -> None:
        """Omitted entity and relation confidence should fall back to request.confidence, not 1.0."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[{"id": "entity-alice-id"}]),
                MagicMock(rows=[{"id": "entity-acme-id"}]),
                MagicMock(rows=[{"id": "relation-id"}]),
            ]
        )

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with (
            patch(
                "workflows_mcp.engine.memory_service.uuid.uuid4",
                side_effect=["memory-0-id", "entity-alice-upsert-id", "entity-acme-upsert-id", "relation-upsert-id"],
            ),
            self._embedding_patch(),
        ):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="ingest_structured",
                    confidence=0.8,
                    memories=[{"content": "Alice works at Acme."}],
                    entities=[],
                    relations=[
                        {
                            "source_name": "Alice",
                            "source_type": "PERSON",
                            "target_name": "Acme",
                            "target_type": "ORG",
                            "relation_type": "works_at",
                            "evidence_memory_index": 0,
                        }
                    ],
                )
            )

        assert result.success is True
        relation_insert_call = next(
            call for call in backend.query.await_args_list if "INSERT INTO knowledge_relations" in call.args[0]
        )
        assert relation_insert_call.args[1][4] == 0.8
        link_params = [
            call.args[1]
            for call in backend.execute.await_args_list
            if "INSERT INTO knowledge_entity_memories" in call.args[0]
        ]
        assert ("memory-0-id", "entity-alice-id", "mentioned", 0.8) in link_params
        assert ("memory-0-id", "entity-acme-id", "mentioned", 0.8) in link_params

    @pytest.mark.asyncio
    async def test_ingest_structured_propagates_source_type_to_knowledge_sources(self) -> None:
        """Structured ingest should classify source rows with the same source_type as memory rows."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[{"id": "source-id"}]),
                MagicMock(rows=[{"id": "item-id"}]),
            ]
        )

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with (
            patch(
                "workflows_mcp.engine.memory_service.uuid.uuid4",
                side_effect=["source-upsert-id", "item-upsert-id", "memory-0-id"],
            ),
            self._embedding_patch(),
        ):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="ingest_structured",
                    source="workflow-source",
                    path="docs/spec.md",
                    source_type="WORKFLOW",
                    memories=[{"content": "Workflow memory."}],
                    entities=[],
                    relations=[],
                )
            )

        assert result.success is True
        source_insert_call = next(
            call for call in backend.query.await_args_list if "INSERT INTO knowledge_sources" in call.args[0]
        )
        assert source_insert_call.args[1][2] == "WORKFLOW"

    def test_ingest_structured_rejects_unknown_nested_memory_fields(self) -> None:
        """Nested structured memory records must reject extra legacy or unsupported keys."""
        with pytest.raises(ValidationError):
            ManageMemoryRequest(
                operation="ingest_structured",
                memories=[
                    {
                        "content": "Alice works at Acme.",
                        "proposition_id": "legacy-alias",
                    }
                ],
                entities=[],
                relations=[],
            )

    def test_ingest_structured_rejects_unsupported_entity_properties(self) -> None:
        """Entity properties are not part of the Task 3 persisted contract and must be rejected."""
        with pytest.raises(ValidationError):
            ManageMemoryRequest(
                operation="ingest_structured",
                memories=[{"content": "Alice works at Acme."}],
                entities=[
                    {
                        "name": "Alice",
                        "entity_type": "PERSON",
                        "properties": {"role": "engineer"},
                    }
                ],
                relations=[],
            )

    @pytest.mark.asyncio
    async def test_ingest_structured_scopes_entity_upsert_by_topology(self) -> None:
        """Entity upserts must include namespace/room/corridor scope to avoid cross-service collisions."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[{"id": "entity-alice-id"}]))

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with (
            patch(
                "workflows_mcp.engine.memory_service.uuid.uuid4",
                side_effect=["memory-0-id", "entity-alice-upsert-id"],
            ),
            self._embedding_patch(),
        ):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="ingest_structured",
                    namespace="service-a",
                    room="auth",
                    corridor="incident",
                    memories=[{"content": "Alice triaged auth incident."}],
                    entities=[
                        {
                            "name": "Alice",
                            "entity_type": "PERSON",
                            "memory_indices": [0],
                        }
                    ],
                    relations=[],
                )
            )

        assert result.success is True
        entity_insert_call = next(
            call for call in backend.query.await_args_list if "INSERT INTO knowledge_entities" in call.args[0]
        )
        entity_insert_sql = entity_insert_call.args[0]
        entity_insert_params = entity_insert_call.args[1]
        assert "namespace, room, corridor" in entity_insert_sql
        assert "ON CONFLICT (namespace, room, corridor, entity_type, name)" in entity_insert_sql
        assert entity_insert_params[3] == "service-a"
        assert entity_insert_params[4] == "auth"
        assert entity_insert_params[5] == "incident"

    @pytest.mark.asyncio
    async def test_store_rejects_unknown_categories_without_explicit_opt_in(self) -> None:
        """Unknown category names should fail by default (no silent auto-create)."""
        backend = MagicMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with self._embedding_patch():
            with pytest.raises(ValueError, match="Unknown categories"):
                await service.manage(
                    ManageMemoryRequest(
                        operation="store",
                        content="Runbook update",
                        categories=["  Incident   Response  "],
                    )
                )

    @pytest.mark.asyncio
    async def test_store_can_create_normalized_categories_when_explicitly_enabled(self) -> None:
        """Category creation should be opt-in and persist normalized category names."""
        backend = MagicMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[]),
                MagicMock(rows=[{"id": "cat-id", "was_inserted": True}]),
            ]
        )

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with (
            patch("workflows_mcp.engine.memory_service.uuid.uuid4", side_effect=["cat-upsert-id", "memory-0-id"]),
            self._embedding_patch(),
        ):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="store",
                    content="Runbook update",
                    categories=["  Incident   Response  "],
                    allow_create_categories=True,
                )
            )

        assert result.success is True
        category_upsert_call = next(
            call for call in backend.query.await_args_list if "INSERT INTO knowledge_categories" in call.args[0]
        )
        assert category_upsert_call.args[1][1] == "incident response"

    @pytest.mark.asyncio
    async def test_graph_store_relation_resolves_named_entities_with_scope_filters(self) -> None:
        """Graph relation resolution by entity name must be scoped by topology keys."""
        backend = MagicMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[{"id": "source-id"}]),
                MagicMock(rows=[{"id": "target-id"}]),
                MagicMock(rows=[]),
                MagicMock(rows=[{"id": "relation-id"}]),
            ]
        )

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with patch("workflows_mcp.engine.memory_service.uuid.uuid4", return_value="relation-upsert-id"):
            result = await service.manage(
                ManageMemoryRequest(
                    operation="graph_store_relation",
                    source_entity="Alice",
                    target_entity="Acme",
                    relation_type="works_at",
                    namespace="service-a",
                    room="auth",
                    corridor="incident",
                )
            )

        assert result.success is True
        source_lookup_sql = backend.query.await_args_list[0].args[0]
        target_lookup_sql = backend.query.await_args_list[1].args[0]
        assert "name = $1" in source_lookup_sql
        assert "namespace = $2" in source_lookup_sql
        assert "room = $3" in source_lookup_sql
        assert "corridor = $4" in source_lookup_sql
        assert "namespace = $2" in target_lookup_sql
        assert backend.query.await_args_list[0].args[1] == ("Alice", "service-a", "auth", "incident")
        assert backend.query.await_args_list[1].args[1] == ("Acme", "service-a", "auth", "incident")

    @pytest.mark.asyncio
    async def test_supersede_requires_superseded_by_reference(self) -> None:
        """Supersede must fail fast when replacement memory id is missing."""
        backend = MagicMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))
        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        result = await service.manage(
            ManageMemoryRequest(
                operation="supersede",
                memory_ids=["11111111-1111-1111-1111-111111111111"],
            )
        )

        assert result.success is False
        assert result.error == "'superseded_by' is required for supersede operation"

    @pytest.mark.asyncio
    async def test_supersede_auto_closes_validity_to_now_when_not_provided(self) -> None:
        """Supersede should close validity window immediately when valid_to is omitted."""
        backend = MagicMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[{"id": "m-1"}]))
        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        result = await service.manage(
            ManageMemoryRequest(
                operation="supersede",
                memory_ids=["11111111-1111-1111-1111-111111111111"],
                superseded_by="22222222-2222-2222-2222-222222222222",
            )
        )

        assert result.success is True
        update_call = backend.query.await_args_list[0]
        assert "valid_to = COALESCE" in update_call.args[0]
        assert update_call.args[1][-1] is None

    @pytest.mark.asyncio
    async def test_supersede_honors_explicit_valid_to_override(self) -> None:
        """Supersede should use caller-provided valid_to when provided."""
        backend = MagicMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[{"id": "m-1"}]))
        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        result = await service.manage(
            ManageMemoryRequest(
                operation="supersede",
                memory_ids=["11111111-1111-1111-1111-111111111111"],
                superseded_by="22222222-2222-2222-2222-222222222222",
                valid_to="2026-04-30T23:59:59Z",
            )
        )

        assert result.success is True
        update_call = backend.query.await_args_list[0]
        assert "valid_to = COALESCE" in update_call.args[0]
        assert str(update_call.args[1][-1]) == "2026-04-30 23:59:59+00:00"

    @pytest.mark.asyncio
    async def test_archive_auto_closes_validity_to_now_when_not_provided(self) -> None:
        """Archive/forget should close validity window immediately when valid_to is omitted."""
        backend = MagicMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[{"cnt": 0}]),
                MagicMock(rows=[{"id": "m-1"}]),
            ]
        )
        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        result = await service.manage(
            ManageMemoryRequest(
                operation="forget",
                memory_ids=["11111111-1111-1111-1111-111111111111"],
            )
        )

        assert result.success is True
        update_call = backend.query.await_args_list[1]
        assert "valid_to = COALESCE" in update_call.args[0]
        assert update_call.args[1][-1] is None

    @pytest.mark.asyncio
    async def test_execute_archive_forwards_record_valid_to_override(self) -> None:
        """Unified archive operation should pass record.valid_to to forget handler."""
        backend = MagicMock()
        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)
        service.manage = AsyncMock(  # type: ignore[method-assign]
            return_value=ManageMemoryResult(operation="forget")
        )

        await service.execute(
            MemoryRequest(
                operation="archive",
                record={
                    "ids": ["11111111-1111-1111-1111-111111111111"],
                    "valid_to": "2026-04-30T23:59:59Z",
                },
            )
        )

        called_request = service.manage.await_args.args[0]  # type: ignore[attr-defined]
        assert called_request.operation == "forget"
        assert called_request.valid_to == "2026-04-30T23:59:59Z"

    @pytest.mark.asyncio
    async def test_graph_upsert_link_is_idempotent_for_same_semantic_edge(self) -> None:
        """Repeated link upserts for the same edge should update existing relation instead of inserting duplicates."""
        backend = MagicMock()
        relation_id_by_edge: dict[tuple[str, str, str], str] = {}

        async def query_side_effect(sql: str, params: tuple[object, ...]) -> MagicMock:
            if "FROM knowledge_entities" in sql:
                if params[0] == "Alice":
                    return MagicMock(rows=[{"id": "source-id"}])
                if params[0] == "Acme":
                    return MagicMock(rows=[{"id": "target-id"}])
                return MagicMock(rows=[])

            if "SELECT id" in sql and "FROM knowledge_relations" in sql:
                key = ("source-id", "target-id", "depends_on")
                relation_id = relation_id_by_edge.get(key)
                return MagicMock(rows=[] if relation_id is None else [{"id": relation_id}])

            if "UPDATE knowledge_relations" in sql:
                key = ("source-id", "target-id", "depends_on")
                relation_id = relation_id_by_edge[key]
                return MagicMock(rows=[{"id": relation_id}])

            if "INSERT INTO knowledge_relations" in sql:
                relation_id = str(params[0])
                key = ("source-id", "target-id", "depends_on")
                relation_id_by_edge[key] = relation_id
                return MagicMock(rows=[{"id": relation_id}])

            raise AssertionError(f"Unexpected SQL: {sql}")

        backend.query = AsyncMock(side_effect=query_side_effect)
        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with patch("workflows_mcp.engine.memory_service.uuid.uuid4", return_value="relation-created-id"):
            first = await service.execute(
                MemoryRequest(
                    operation="graph_upsert",
                    graph={
                        "kind": "link",
                        "from": "Alice",
                        "to": "Acme",
                        "link_type": "depends_on",
                    },
                )
            )
            second = await service.execute(
                MemoryRequest(
                    operation="graph_upsert",
                    graph={
                        "kind": "link",
                        "from": "Alice",
                        "to": "Acme",
                        "link_type": "depends_on",
                    },
                )
            )

        assert first.manage is not None
        assert second.manage is not None
        assert first.manage.relation_id == second.manage.relation_id
        relation_insert_calls = [
            call for call in backend.query.await_args_list if "INSERT INTO knowledge_relations" in call.args[0]
        ]
        assert len(relation_insert_calls) == 1

    @pytest.mark.asyncio
    async def test_graph_delete_place_reports_cascaded_relation_deletes(self) -> None:
        """Entity deletes should include cascaded relation rows in deleted_relation_count."""
        backend = MagicMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[{"cnt": 3}]),
                MagicMock(rows=[{"id": "entity-1"}]),
            ]
        )
        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        result = await service.execute(
            MemoryRequest(
                operation="graph_delete",
                graph={"kind": "place", "ids": ["11111111-1111-1111-1111-111111111111"]},
            )
        )

        assert result.manage is not None
        assert result.manage.deleted_entity_count == 1
        assert result.manage.deleted_relation_count == 3
