"""Focused tests for community refresh and community retrieval in MemoryService."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workflows_mcp.engine.knowledge.constants import Authority
from workflows_mcp.engine.memory_service import (
    ManageMemoryRequest,
    MemoryRequest,
    MemoryService,
    QueryMemoryRequest,
)


def _embedding_patch() -> Any:
    """Stub embedding generation for community refresh and retrieval tests."""
    return patch(
        "workflows_mcp.engine.memory_service.compute_embedding",
        new=AsyncMock(return_value=([0.2, 0.6], "embed-model", 2, None)),
    )


class TestCommunityRefresh:
    """Tests for consolidate/community_refresh behavior in MemoryService."""

    @pytest.mark.asyncio
    async def test_community_refresh_writes_communities_and_assignments(self) -> None:
        """Refreshing communities should persist community rows and assign entities and memories."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(
                    rows=[
                        {"id": "entity-alice", "entity_type": "PERSON", "name": "Alice"},
                        {"id": "entity-acme", "entity_type": "ORG", "name": "Acme"},
                    ]
                ),
                MagicMock(
                    rows=[{"source_entity_id": "entity-alice", "target_entity_id": "entity-acme"}]
                ),
                MagicMock(
                    rows=[
                        {
                            "id": "memory-1",
                            "content": "Alice works at Acme.",
                            "embedding": "[0.1,0.3]",
                        }
                    ]
                ),
                MagicMock(rows=[{"id": "community-1"}]),
                MagicMock(rows=[{"id": "memory-1", "memory_tier": "direct"}]),
                MagicMock(
                    rows=[
                        {
                            "memory_id": "memory-1",
                            "community_id": "community-1",
                            "link_confidence": 0.95,
                        }
                    ]
                ),
            ]
        )

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with _embedding_patch():
            result = await service.manage(
                ManageMemoryRequest(
                    operation="consolidate",
                    mode="community_refresh",
                )
            )

        assert result.success is True
        assert result.communities_updated == 1
        assert result.diagnostics["community_count"] == result.communities_updated
        backend.begin_transaction.assert_awaited_once()
        backend.commit.assert_awaited_once()
        backend.rollback.assert_not_awaited()

        query_sql = [call.args[0] for call in backend.query.await_args_list]
        execute_calls = backend.execute.await_args_list
        assert any("INSERT INTO knowledge_communities" in sql for sql in query_sql)
        assert any("UPDATE knowledge_entities" in call.args[0] for call in execute_calls)
        assert any("UPDATE knowledge_memories" in call.args[0] for call in execute_calls)
        derived_insert_calls = [
            call for call in execute_calls if "INSERT INTO knowledge_memories" in call.args[0]
        ]
        assert len(derived_insert_calls) == 1
        assert derived_insert_calls[0].args[1][4] == Authority.COMMUNITY_SUMMARY
        assert derived_insert_calls[0].args[1][14] == "derived"
        assert derived_insert_calls[0].args[1][15] == "community"

    @pytest.mark.asyncio
    async def test_community_refresh_honors_namespace_and_room_scope(self) -> None:
        """Scoped refresh should restrict source graph discovery to the requested memory scope."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(return_value=MagicMock(rows=[]))

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        result = await service.manage(
            ManageMemoryRequest(
                operation="consolidate",
                mode="community_refresh",
                namespace="engineering",
                room="memory",
            )
        )

        assert result.success is True
        assert result.communities_updated == 0
        assert result.diagnostics["community_count"] == result.communities_updated
        first_query = backend.query.await_args_list[0]
        assert "km.namespace = $1" in first_query.args[0]
        assert "km.room = $2" in first_query.args[0]
        assert first_query.args[1] == ("engineering", "memory")

    @pytest.mark.asyncio
    async def test_community_refresh_propagates_memories_by_strongest_linked_community(
        self,
    ) -> None:
        """Memories linked to multiple communities should keep the strongest linked community."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(
                    rows=[
                        {"id": "entity-alice", "entity_type": "PERSON", "name": "Alice"},
                        {"id": "entity-acme", "entity_type": "ORG", "name": "Acme"},
                        {"id": "entity-ops", "entity_type": "TEAM", "name": "Ops"},
                    ]
                ),
                MagicMock(
                    rows=[{"source_entity_id": "entity-alice", "target_entity_id": "entity-acme"}]
                ),
                MagicMock(
                    rows=[
                        {
                            "id": "memory-1",
                            "content": "Alice works at Acme.",
                            "embedding": "[0.1,0.3]",
                        },
                        {"id": "memory-2", "content": "Acme roadmap.", "embedding": "[0.1,0.4]"},
                    ]
                ),
                MagicMock(rows=[{"id": "community-1"}]),
                MagicMock(
                    rows=[
                        {"id": "memory-1", "memory_tier": "direct"},
                        {"id": "memory-2", "memory_tier": "direct"},
                    ]
                ),
                MagicMock(
                    rows=[
                        {"id": "memory-1", "content": "Ops update.", "embedding": "[0.9,0.9]"},
                        {"id": "memory-3", "content": "Ops status.", "embedding": "[0.8,0.8]"},
                    ]
                ),
                MagicMock(rows=[{"id": "community-2"}]),
                MagicMock(
                    rows=[
                        {"id": "memory-1", "memory_tier": "direct"},
                        {"id": "memory-3", "memory_tier": "direct"},
                    ]
                ),
                MagicMock(
                    rows=[
                        {
                            "memory_id": "memory-1",
                            "community_id": "community-1",
                            "link_confidence": 0.95,
                        },
                        {
                            "memory_id": "memory-1",
                            "community_id": "community-2",
                            "link_confidence": 0.35,
                        },
                        {
                            "memory_id": "memory-2",
                            "community_id": "community-1",
                            "link_confidence": 0.7,
                        },
                        {
                            "memory_id": "memory-3",
                            "community_id": "community-2",
                            "link_confidence": 0.6,
                        },
                    ]
                ),
            ]
        )

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with _embedding_patch():
            result = await service.manage(
                ManageMemoryRequest(
                    operation="consolidate",
                    mode="community_refresh",
                )
            )

        assert result.success is True
        assert result.communities_updated == 2
        assert result.diagnostics["community_count"] == result.communities_updated

        memory_updates = [
            call.args[1]
            for call in backend.execute.await_args_list
            if "UPDATE knowledge_memories" in call.args[0] and "SET community_id" in call.args[0]
        ]
        assert ("community-1", "memory-1") in memory_updates
        assert ("community-1", "memory-2") in memory_updates
        assert ("community-2", "memory-3") in memory_updates

    @pytest.mark.asyncio
    async def test_execute_maintain_community_refresh_routes_to_refresh_semantics(self) -> None:
        """Maintain accepts community_refresh and returns refresh-shaped data."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(
                    rows=[
                        {"id": "entity-alice", "entity_type": "PERSON", "name": "Alice"},
                        {"id": "entity-acme", "entity_type": "ORG", "name": "Acme"},
                    ]
                ),
                MagicMock(
                    rows=[{"source_entity_id": "entity-alice", "target_entity_id": "entity-acme"}]
                ),
                MagicMock(
                    rows=[
                        {
                            "id": "memory-1",
                            "content": "Alice works at Acme.",
                            "embedding": "[0.1,0.3]",
                        }
                    ]
                ),
                MagicMock(rows=[{"id": "community-1"}]),
                MagicMock(rows=[{"id": "memory-1", "memory_tier": "direct"}]),
                MagicMock(
                    rows=[
                        {
                            "memory_id": "memory-1",
                            "community_id": "community-1",
                            "link_confidence": 0.95,
                        }
                    ]
                ),
            ]
        )

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        with _embedding_patch():
            result = await service.execute(
                MemoryRequest(operation="maintain", maintenance={"mode": "community_refresh"})
            )

        assert result.operation == "maintain"
        assert result.manage is not None
        assert result.manage.success is True
        assert result.manage.operation == "maintain"
        assert result.manage.communities_updated == 1
        assert result.manage.diagnostics["mode"] == "community_refresh"
        assert result.manage.diagnostics["community_count"] == result.manage.communities_updated

    @pytest.mark.asyncio
    async def test_community_refresh_fails_deterministically_without_parent_memories(self) -> None:
        """Derived community summaries fail when no parent memories exist."""
        backend = MagicMock()
        backend.begin_transaction = AsyncMock()
        backend.commit = AsyncMock()
        backend.rollback = AsyncMock()
        backend.execute = AsyncMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(rows=[{"id": "entity-alice", "entity_type": "PERSON", "name": "Alice"}]),
                MagicMock(rows=[]),
                MagicMock(rows=[]),
                MagicMock(rows=[{"id": "community-1"}]),
            ]
        )

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        result = await service.manage(
            ManageMemoryRequest(
                operation="consolidate",
                mode="community_refresh",
            )
        )

        assert result.success is False
        assert result.error is not None
        assert result.error.startswith("MEM_LINEAGE_PARENT_REQUIRED:")
        backend.commit.assert_not_awaited()
        backend.rollback.assert_awaited_once()


class TestCommunityRetrieval:
    """Tests for dedicated communities retrieval strategy."""

    @pytest.mark.asyncio
    async def test_strategy_communities_returns_distinct_community_results(self) -> None:
        """Communities strategy should return selected communities and their member memories."""
        backend = MagicMock()
        backend.query = AsyncMock(
            side_effect=[
                MagicMock(
                    rows=[
                        {
                            "id": "community-1",
                            "content": "Community: Alice, Acme",
                            "member_count": 2,
                            "memory_count": 1,
                            "namespace": "engineering",
                            "room": "memory",
                            "corridor": "cluster-a",
                            "similarity": 0.98,
                        }
                    ]
                ),
            ]
        )
        backend.execute = AsyncMock()

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)
        room_search = AsyncMock(
            return_value=[
                {
                    "id": "memory-1",
                    "content": "Alice works at Acme",
                    "confidence": 0.93,
                    "authority": None,
                    "source_name": "repo/docs/a.md",
                    "item_path": "repo/docs/a.md",
                    "namespace": "engineering",
                    "rrf_score": 0.98,
                }
            ]
        )

        with (
            _embedding_patch(),
            patch("workflows_mcp.engine.memory_service.room_scoped_search", new=room_search),
        ):
            result = await service.query(
                QueryMemoryRequest(
                    query="Alice organization",
                    strategy="communities",
                    namespace="engineering",
                    room="memory",
                    scope={"corridor": "cluster-a"},
                )
            )

        assert result.facts == []
        assert result.memories == [
            {
                "id": "memory-1",
                "content": "Alice works at Acme",
                "confidence": 0.93,
                "authority": None,
                "rrf_score": 0.98,
                "path": "repo/docs/a.md",
                "source": "repo/docs/a.md",
                "namespace": "engineering",
            }
        ]
        assert result.communities == [
            {
                "id": "community-1",
                "content": "Community: Alice, Acme",
                "member_count": 2,
                "memory_count": 1,
                "namespace": "engineering",
                "room": "memory",
                "corridor": "cluster-a",
                "similarity": 0.98,
            }
        ]
        assert result.diagnostics["strategy"] == "communities"
        assert result.diagnostics["scope_mode"] == "community_exists_filter"
        assert result.diagnostics["scope_applied"] is True
        assert result.diagnostics["scope_status"] == "applied_with_results"
        assert result.diagnostics["authority_routing"] == "facts_user_validated"
        assert "knowledge_communities" in backend.query.await_args_list[0].args[0]
        assert len(backend.query.await_args_list) == 1
        all_sql = "\n".join(call.args[0] for call in backend.query.await_args_list)
        assert "km.item_path" not in all_sql
        assert "COALESCE(km.confidence, 1.0)" not in all_sql
        assert "ORDER BY score DESC" not in all_sql
        room_search.assert_awaited_once()
        room_kwargs = room_search.await_args.kwargs
        assert room_kwargs["community_ids"] == ["community-1"]
        assert room_kwargs["namespace"] == "engineering"
        assert room_kwargs["room"] == "memory"
        assert room_kwargs["corridor"] == "cluster-a"
        assert room_kwargs["include_global_companion"] is False
        backend.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_strategy_palace_requires_explicit_scope(self) -> None:
        """Palace strategy should reject missing namespace/room instead of falling back."""
        backend = MagicMock()
        backend.query = AsyncMock()
        backend.execute = AsyncMock()

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)

        result = await service.query(
            QueryMemoryRequest(
                query="unscoped palace query",
                strategy="palace",
            )
        )

        assert result.facts == []
        assert result.memories == []
        assert result.communities == []
        assert (
            result.diagnostics["error"]
            == "palace strategy requires namespace or room for scoped retrieval"
        )
        assert result.diagnostics["scope_mode"] == "strict_scoped"
        assert result.diagnostics["scope_applied"] is False
        assert result.diagnostics["scope_status"] == "missing_scope"
        backend.query.assert_not_awaited()
        backend.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_strategy_palace_scoped_no_data_reports_scope_status(self) -> None:
        """Palace scoped queries should report explicit no_data_in_scope diagnostics."""
        backend = MagicMock()
        backend.query = AsyncMock()
        backend.execute = AsyncMock()

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)
        room_search = AsyncMock(return_value=[])

        with (
            _embedding_patch(),
            patch("workflows_mcp.engine.memory_service.room_scoped_search", new=room_search),
        ):
            result = await service.query(
                QueryMemoryRequest(
                    query="scoped palace no data",
                    strategy="palace",
                    namespace="engineering",
                    room="memory",
                    scope={"corridor": "cluster-a"},
                )
            )

        assert result.facts == []
        assert result.memories == []
        assert result.diagnostics["scope_mode"] == "strict_scoped"
        assert result.diagnostics["scope_applied"] is True
        assert result.diagnostics["scope_status"] == "no_data_in_scope"
        assert result.diagnostics["authority_routing"] == "facts_user_validated"
        room_search.assert_awaited_once()
        room_kwargs = room_search.await_args.kwargs
        assert room_kwargs["include_global_companion"] is False
        assert room_kwargs["corridor"] == "cluster-a"

    @pytest.mark.asyncio
    async def test_strategy_palace_forwards_corridor_from_request_field(self) -> None:
        """Palace strategy should forward corridor when explicitly present on request."""
        backend = MagicMock()
        backend.query = AsyncMock()
        backend.execute = AsyncMock()

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)
        room_search = AsyncMock(return_value=[])

        request = QueryMemoryRequest(
            query="scoped palace no data",
            strategy="palace",
            namespace="engineering",
            room="memory",
        )
        object.__setattr__(request, "corridor", "cluster-b")

        with (
            _embedding_patch(),
            patch("workflows_mcp.engine.memory_service.room_scoped_search", new=room_search),
        ):
            result = await service.query(request)

        assert result.facts == []
        assert result.memories == []
        assert result.diagnostics["scope_mode"] == "strict_scoped"
        assert result.diagnostics["scope_applied"] is True
        assert result.diagnostics["scope_status"] == "no_data_in_scope"
        assert result.diagnostics["authority_routing"] == "facts_user_validated"
        room_search.assert_awaited_once()
        room_kwargs = room_search.await_args.kwargs
        assert room_kwargs["include_global_companion"] is False
        assert room_kwargs["corridor"] == "cluster-b"

    @pytest.mark.asyncio
    async def test_strategy_auto_sets_authority_routing_diagnostic(self) -> None:
        """Auto strategy should publish authority lane routing diagnostics."""
        backend = MagicMock()
        backend.query = AsyncMock()
        backend.execute = AsyncMock()

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)
        room_search = AsyncMock(return_value=[])

        with (
            _embedding_patch(),
            patch("workflows_mcp.engine.memory_service.room_scoped_search", new=room_search),
        ):
            result = await service.query(
                QueryMemoryRequest(
                    query="auto strategy no data",
                    strategy="auto",
                )
            )

        assert result.facts == []
        assert result.memories == []
        assert result.diagnostics["scope_mode"] == "dual_lane_with_companion"
        assert result.diagnostics["scope_applied"] is False
        assert result.diagnostics["scope_status"] == "unscoped"
        assert result.diagnostics["authority_routing"] == "facts_user_validated"
        room_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_strategy_auto_forwards_corridor_scope_to_room_search(self) -> None:
        """Auto strategy forwards corridor scope to room search."""
        backend = MagicMock()
        backend.query = AsyncMock()
        backend.execute = AsyncMock()

        context = MagicMock()
        context.execution_context = None
        service = MemoryService(backend=backend, context=context)
        room_search = AsyncMock(return_value=[])

        with (
            _embedding_patch(),
            patch("workflows_mcp.engine.memory_service.room_scoped_search", new=room_search),
        ):
            result = await service.query(
                QueryMemoryRequest(
                    query="auto strategy corridor scoped no data",
                    strategy="auto",
                    scope={"corridor": "cluster-auto"},
                )
            )

        assert result.facts == []
        assert result.memories == []
        assert result.diagnostics["scope_mode"] == "dual_lane_with_companion"
        assert result.diagnostics["scope_applied"] is True
        assert result.diagnostics["scope_status"] == "no_data_in_scope"
        assert result.diagnostics["authority_routing"] == "facts_user_validated"
        room_search.assert_awaited_once()
        room_kwargs = room_search.await_args.kwargs
        assert room_kwargs["corridor"] == "cluster-auto"
