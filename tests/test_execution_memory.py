"""Tests for ExecutionMemory — ephemeral per-execution SQLite memory."""

from pathlib import Path

import pytest

from workflows_mcp.engine.execution_memory import ExecutionMemory, Turn


@pytest.fixture
async def memory(tmp_path: Path) -> ExecutionMemory:
    """Create and initialize a fresh ExecutionMemory instance."""
    mem = ExecutionMemory(tmp_path / "test_memory.db")
    await mem.initialize()
    yield mem  # type: ignore[misc]
    await mem.close()


# -----------------------------------------------------------------------
# Key-Value Context
# -----------------------------------------------------------------------


class TestKeyValueContext:
    """Tests for get/set key-value API."""

    @pytest.mark.asyncio
    async def test_get_set_global(self, memory: ExecutionMemory) -> None:
        await memory.set("greeting", "hello")
        assert await memory.get("greeting") == "hello"

    @pytest.mark.asyncio
    async def test_get_missing_key_returns_none(self, memory: ExecutionMemory) -> None:
        assert await memory.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_overwrite(self, memory: ExecutionMemory) -> None:
        await memory.set("key", "v1")
        await memory.set("key", "v2")
        assert await memory.get("key") == "v2"

    @pytest.mark.asyncio
    async def test_scoped_keys_isolated(self, memory: ExecutionMemory) -> None:
        """Global and block-scoped keys with the same name don't interfere."""
        await memory.set("data", "global_value", scope="global")
        await memory.set("data", "block_value", scope="block:step_1")

        assert await memory.get("data", scope="global") == "global_value"
        assert await memory.get("data", scope="block:step_1") == "block_value"

    @pytest.mark.asyncio
    async def test_scoped_key_missing(self, memory: ExecutionMemory) -> None:
        await memory.set("key", "val", scope="block:a")
        assert await memory.get("key", scope="block:b") is None


# -----------------------------------------------------------------------
# Conversation Turns
# -----------------------------------------------------------------------


class TestConversationTurns:
    """Tests for add_turn/get_turns API."""

    @pytest.mark.asyncio
    async def test_add_and_get_turns(self, memory: ExecutionMemory) -> None:
        await memory.add_turn("llm_1", "user", "What is 2+2?")
        await memory.add_turn("llm_1", "assistant", "4")

        turns = await memory.get_turns(block_id="llm_1")
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[0].content == "What is 2+2?"
        assert turns[1].role == "assistant"
        assert turns[1].content == "4"

    @pytest.mark.asyncio
    async def test_get_turns_filters_by_block(self, memory: ExecutionMemory) -> None:
        await memory.add_turn("block_a", "user", "A prompt")
        await memory.add_turn("block_b", "user", "B prompt")

        turns_a = await memory.get_turns(block_id="block_a")
        turns_b = await memory.get_turns(block_id="block_b")

        assert len(turns_a) == 1
        assert turns_a[0].content == "A prompt"
        assert len(turns_b) == 1
        assert turns_b[0].content == "B prompt"

    @pytest.mark.asyncio
    async def test_get_all_turns(self, memory: ExecutionMemory) -> None:
        await memory.add_turn("block_a", "user", "A")
        await memory.add_turn("block_b", "user", "B")

        all_turns = await memory.get_turns()
        assert len(all_turns) == 2

    @pytest.mark.asyncio
    async def test_turn_ordering(self, memory: ExecutionMemory) -> None:
        for i in range(5):
            await memory.add_turn("block", "user", f"msg_{i}")

        turns = await memory.get_turns(block_id="block")
        contents = [t.content for t in turns]
        assert contents == [f"msg_{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_turn_dataclass(self, memory: ExecutionMemory) -> None:
        await memory.add_turn("b", "system", "You are helpful.")

        turns = await memory.get_turns(block_id="b")
        t = turns[0]
        assert isinstance(t, Turn)
        assert t.block_id == "b"
        assert t.role == "system"
        assert t.created_at  # non-empty timestamp


# -----------------------------------------------------------------------
# Context Propagation
# -----------------------------------------------------------------------


class TestContextPropagation:
    """Verify that child execution contexts share the same memory instance."""

    @pytest.mark.asyncio
    async def test_child_context_shares_memory(self, memory: ExecutionMemory) -> None:
        from workflows_mcp.engine.execution_context import ExecutionContext
        from workflows_mcp.engine.executor_base import create_default_registry
        from workflows_mcp.engine.llm_config import LLMConfigLoader
        from workflows_mcp.engine.registry import WorkflowRegistry

        parent_ctx = ExecutionContext(
            workflow_registry=WorkflowRegistry(),
            executor_registry=create_default_registry(),
            llm_config_loader=LLMConfigLoader(),
            io_queue=None,
            execution_memory=memory,
        )

        # Simulate fractal composition
        from workflows_mcp.engine.execution import Execution

        parent_exec = Execution()
        child_ctx = parent_ctx.create_child_context(
            parent_execution=parent_exec,
            workflow_name="child-wf",
        )

        assert child_ctx.execution_memory is parent_ctx.execution_memory
        assert child_ctx.execution_memory is memory

    @pytest.mark.asyncio
    async def test_none_memory_by_default(self) -> None:
        from workflows_mcp.engine.execution_context import ExecutionContext
        from workflows_mcp.engine.executor_base import create_default_registry
        from workflows_mcp.engine.llm_config import LLMConfigLoader
        from workflows_mcp.engine.registry import WorkflowRegistry

        ctx = ExecutionContext(
            workflow_registry=WorkflowRegistry(),
            executor_registry=create_default_registry(),
            llm_config_loader=LLMConfigLoader(),
            io_queue=None,
        )
        assert ctx.execution_memory is None


# -----------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------


class TestLifecycle:
    """Tests for initialize/close lifecycle."""

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, memory: ExecutionMemory) -> None:
        await memory.close()
        await memory.close()  # should not raise

    @pytest.mark.asyncio
    async def test_reopen_preserves_data(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen.db"

        mem1 = ExecutionMemory(db_path)
        await mem1.initialize()
        await mem1.set("key", "persisted")
        await mem1.close()

        mem2 = ExecutionMemory(db_path)
        await mem2.initialize()
        assert await mem2.get("key") == "persisted"
        await mem2.close()
