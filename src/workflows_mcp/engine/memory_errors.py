"""Shared memory contract errors.

This module exists so memory contract helpers can be reused by focused engine
modules without importing the large memory_service module and creating cycles.
"""

from __future__ import annotations

from typing import NoReturn


class MemoryContractError(ValueError):
    """Deterministic contract error with machine-readable code."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        actionable_fix: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.actionable_fix = actionable_fix


def _raise_contract_error(
    *,
    code: str,
    message: str,
    retryable: bool = False,
    actionable_fix: str | None = None,
) -> NoReturn:
    raise MemoryContractError(
        code=code,
        message=f"{code}: {message}",
        retryable=retryable,
        actionable_fix=actionable_fix,
    )
