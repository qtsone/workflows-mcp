"""Readiness evaluation service.

Derives the service readiness state from:
1. Presence of the ``~/.workflows/`` directory.
2. Presence of ``~/.workflows/llm-config.yml``.
3. Result of the PostgreSQL connectivity probe.

States (spec section 9.1):
- ``UNCONFIGURED``: directory or config file absent.
- ``PARTIALLY_CONFIGURED``: config artifacts present but DB probe fails.
- ``READY``: all prerequisites satisfied.

The ``ReadinessService`` accepts any object that satisfies the ``Probe``
protocol, making it straightforward to inject test doubles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .http_models import ReadinessState


def _canonicalize_blocker(blocker: str) -> str:
    """Return canonical blocker identifiers for readiness responses.

    Keeps existing blocker strings stable except for legacy/variant schema
    incompatibility names, which are normalized to the canonical
    ``knowledge_schema_incompatible`` code used by control-plane readiness.
    """

    schema_incompatible_aliases = {
        "knowledge_schema_incompatible",
        "postgresql_schema_incompatible",
        "knowledge_schema_invalid",
    }
    if blocker in schema_incompatible_aliases:
        return "knowledge_schema_incompatible"
    return blocker


def _canonicalize_blockers(blockers: list[str]) -> list[str]:
    """Normalize blocker identifiers while preserving order and uniqueness."""

    canonical: list[str] = []
    seen: set[str] = set()
    for blocker in blockers:
        mapped = _canonicalize_blocker(blocker)
        if mapped not in seen:
            seen.add(mapped)
            canonical.append(mapped)
    return canonical


@runtime_checkable
class Probe(Protocol):
    """Protocol for dependency readiness probes.

    Implementations return a ``(ok, blockers)`` tuple where ``blockers``
    is a list of string identifiers describing what failed. The list must
    be empty when ``ok`` is ``True``.
    """

    async def check(self) -> tuple[bool, list[str]]: ...


@dataclass(slots=True)
class ReadinessReport:
    """Snapshot of the current readiness evaluation.

    Attributes
    ----------
    state:
        The computed readiness state.
    blockers:
        Human-opaque string identifiers describing unmet prerequisites.
        Always empty when ``state`` is ``READY``.
    """

    state: ReadinessState
    blockers: list[str]


@dataclass(slots=True)
class ReadinessService:
    """Evaluates service readiness on demand.

    Parameters
    ----------
    base_dir:
        The ``~/.workflows/`` directory (or equivalent for tests).
    probe:
        A ``Probe`` implementation for the external dependency check
        (PostgreSQL in production; a test double in unit tests).
    """

    base_dir: Path
    probe: Probe

    async def evaluate(self) -> ReadinessReport:
        """Compute and return a ``ReadinessReport`` reflecting current state.

        The evaluation is stateless: each call re-reads the filesystem and
        re-executes the probe. Background polling (spec section 12.2) is
        handled by callers, not here.
        """
        if not self.base_dir.exists():
            return ReadinessReport(
                state=ReadinessState.UNCONFIGURED,
                blockers=["workflows_dir"],
            )

        llm_config = self.base_dir / "llm-config.yml"
        if not llm_config.exists():
            return ReadinessReport(
                state=ReadinessState.UNCONFIGURED,
                blockers=["llm_config"],
            )

        ok, db_blockers = await self.probe.check()
        if not ok:
            return ReadinessReport(
                state=ReadinessState.PARTIALLY_CONFIGURED,
                blockers=_canonicalize_blockers(db_blockers),
            )

        return ReadinessReport(state=ReadinessState.READY, blockers=[])
