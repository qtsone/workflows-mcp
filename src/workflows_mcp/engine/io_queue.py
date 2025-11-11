"""IO Queue for serialized file operations.

Prevents race conditions when multiple parallel workflows access shared state files.
Uses single-worker pattern to guarantee sequential execution.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class IOQueue:
    """Sequential execution queue for file I/O operations.

    Architecture:
    - Single worker coroutine processes operations sequentially
    - Callers await Future for synchronous response
    - Guarantees no concurrent file access

    Usage:
        queue = IOQueue()
        await queue.start()

        # Submit operation, wait for result
        result = await queue.submit(lambda: json_operation())
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[Callable[[], Any], asyncio.Future[Any]]] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._running = False
        self._stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
        }

    async def start(self) -> None:
        """Start IO worker coroutine."""
        if self._running:
            logger.warning("IOQueue already running")
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("IOQueue started")

    async def stop(self) -> None:
        """Stop IO worker and wait for queue to drain."""
        if not self._running:
            return

        self._running = False

        # Wait for queue to empty (with timeout to prevent hanging)
        try:
            await asyncio.wait_for(self._queue.join(), timeout=30.0)
        except TimeoutError:
            logger.warning("IO queue drain timeout (30s) - forcing shutdown")

        # Cancel worker task
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"IOQueue stopped. Stats: {self._stats['total_operations']} total, "
            f"{self._stats['successful_operations']} successful, "
            f"{self._stats['failed_operations']} failed"
        )

    async def submit(self, operation: Callable[[], Any]) -> Any:
        """Submit operation to queue and wait for result.

        Args:
            operation: Callable that performs I/O operation

        Returns:
            Result from operation

        Raises:
            Exception: If operation fails
        """
        if not self._running:
            raise RuntimeError("IOQueue not started. Call start() first.")

        future: asyncio.Future[Any] = asyncio.Future()
        await self._queue.put((operation, future))

        # Wait for operation to complete (blocks caller)
        return await future

    async def _worker(self) -> None:
        """Worker coroutine - processes operations sequentially."""
        logger.info("IOQueue worker started")

        while True:
            try:
                # Get next operation (with timeout to periodically check _running flag)
                timeout = 1.0 if self._running else 0.1
                try:
                    operation, future = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except TimeoutError:
                    if not self._running:
                        break  # Exit when stopping and no more items
                    continue  # Keep waiting when running

                self._stats["total_operations"] += 1

                try:
                    # Execute operation (synchronous, not async)
                    if asyncio.iscoroutinefunction(operation):
                        result = await operation()
                    else:
                        result = operation()

                    future.set_result(result)
                    self._stats["successful_operations"] += 1

                except Exception as e:
                    logger.error(f"IOQueue operation failed: {e}", exc_info=True)
                    future.set_exception(e)
                    self._stats["failed_operations"] += 1

                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                logger.info("IOQueue worker cancelled")
                break
            except Exception as e:
                logger.error(f"IOQueue worker error: {e}", exc_info=True)

        logger.info("IOQueue worker stopped")

    def get_stats(self) -> dict[str, int]:
        """Get queue statistics."""
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
        }
