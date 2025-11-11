"""Tests for IO Queue."""

import asyncio
import json

import pytest

from workflows_mcp.engine.io_queue import IOQueue


@pytest.fixture
async def io_queue():
    """Create and start IO queue."""
    queue = IOQueue()
    await queue.start()
    yield queue
    await queue.stop()


@pytest.mark.asyncio
async def test_io_queue_sequential_execution(io_queue, tmp_path):
    """Test that operations execute sequentially (no races)."""
    counter_file = tmp_path / "counter.json"
    counter_file.write_text('{"count": 0}')

    async def increment():
        # Read-modify-write (race-prone operation)
        data = json.loads(counter_file.read_text())
        await asyncio.sleep(0.01)  # Simulate slow I/O
        data["count"] += 1
        counter_file.write_text(json.dumps(data))
        return data["count"]

    # Launch 10 parallel increments
    tasks = [io_queue.submit(increment) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # Verify: all increments applied (no lost updates)
    final_data = json.loads(counter_file.read_text())
    assert final_data["count"] == 10
    assert results == list(range(1, 11))  # Sequential: 1, 2, 3, ..., 10


@pytest.mark.asyncio
async def test_io_queue_exception_handling(io_queue):
    """Test that exceptions propagate to caller."""

    def failing_operation():
        raise ValueError("Operation failed")

    with pytest.raises(ValueError, match="Operation failed"):
        await io_queue.submit(failing_operation)


@pytest.mark.asyncio
async def test_io_queue_stats(io_queue):
    """Test queue statistics tracking."""

    def successful_op():
        return 42

    def failing_op():
        raise RuntimeError("Test error")

    # Execute operations
    await io_queue.submit(successful_op)
    await io_queue.submit(successful_op)

    try:
        await io_queue.submit(failing_op)
    except RuntimeError:
        pass

    stats = io_queue.get_stats()
    assert stats["total_operations"] == 3
    assert stats["successful_operations"] == 2
    assert stats["failed_operations"] == 1


@pytest.mark.asyncio
async def test_io_queue_sync_function():
    """Test that sync functions are executed correctly."""
    queue = IOQueue()
    await queue.start()

    def sync_operation():
        return "sync result"

    result = await queue.submit(sync_operation)
    assert result == "sync result"

    await queue.stop()


@pytest.mark.asyncio
async def test_io_queue_async_function():
    """Test that async functions are executed correctly."""
    queue = IOQueue()
    await queue.start()

    async def async_operation():
        await asyncio.sleep(0.01)
        return "async result"

    result = await queue.submit(async_operation)
    assert result == "async result"

    await queue.stop()


@pytest.mark.asyncio
async def test_io_queue_not_started():
    """Test that submitting to non-started queue raises error."""
    queue = IOQueue()

    def operation():
        return "result"

    with pytest.raises(RuntimeError, match="IOQueue not started"):
        await queue.submit(operation)


@pytest.mark.asyncio
async def test_io_queue_stop_waits_for_completion():
    """Test that stop waits for in-progress operations."""
    queue = IOQueue()
    await queue.start()

    completed = []

    async def slow_operation_0():
        await asyncio.sleep(0.05)
        completed.append(0)
        return 0

    async def slow_operation_1():
        await asyncio.sleep(0.05)
        completed.append(1)
        return 1

    async def slow_operation_2():
        await asyncio.sleep(0.05)
        completed.append(2)
        return 2

    # Submit multiple operations BEFORE stopping
    task0 = asyncio.create_task(queue.submit(slow_operation_0))
    task1 = asyncio.create_task(queue.submit(slow_operation_1))
    task2 = asyncio.create_task(queue.submit(slow_operation_2))

    # Give queue time to accept all operations
    await asyncio.sleep(0.01)

    # Stop queue (should wait for completion)
    await queue.stop()

    # Wait for all operations
    results = await asyncio.gather(task0, task1, task2)

    assert results == [0, 1, 2]
    assert len(completed) == 3


@pytest.mark.asyncio
async def test_io_queue_parallel_file_writes(tmp_path):
    """Test parallel writes to different files are serialized."""
    queue = IOQueue()
    await queue.start()

    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    execution_order = []

    async def write_file1():
        execution_order.append("start-content1")
        await asyncio.sleep(0.01)
        file1.write_text("content1")
        execution_order.append("end-content1")
        return "content1"

    async def write_file2():
        execution_order.append("start-content2")
        await asyncio.sleep(0.01)
        file2.write_text("content2")
        execution_order.append("end-content2")
        return "content2"

    # Launch parallel writes
    task1 = queue.submit(write_file1)
    task2 = queue.submit(write_file2)

    await asyncio.gather(task1, task2)

    # Verify execution was sequential (one completes before next starts)
    assert len(execution_order) == 4
    # Either file1 completes first, or file2 completes first
    assert execution_order == [
        "start-content1",
        "end-content1",
        "start-content2",
        "end-content2",
    ] or execution_order == ["start-content2", "end-content2", "start-content1", "end-content1"]

    await queue.stop()


@pytest.mark.asyncio
async def test_io_queue_double_start():
    """Test that starting an already-running queue is safe."""
    queue = IOQueue()
    await queue.start()
    await queue.start()  # Should log warning but not fail

    def operation():
        return "result"

    result = await queue.submit(operation)
    assert result == "result"

    await queue.stop()


@pytest.mark.asyncio
async def test_io_queue_double_stop():
    """Test that stopping a non-running queue is safe."""
    queue = IOQueue()
    await queue.start()
    await queue.stop()
    await queue.stop()  # Should be safe (no-op)
