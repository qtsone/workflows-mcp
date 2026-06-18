"""Shared constants and scope helpers for the memory executor-op tests.

Imported by the per-concern ``test_*`` modules in this package and by the
``conftest`` fixtures, so the test scope and identifiers live in one place.
"""

from __future__ import annotations

import os

from workflows_mcp.engine.sql.backend import ConnectionConfig, DatabaseEngine

PALACE = "palace_ops_test"
WING = "default"
CODE_WING = "code"
ROOM = "default"
COMPARTMENT = "ops"


def _scope() -> dict[str, str]:
    return {"palace": PALACE, "wing": WING, "room": ROOM, "compartment": COMPARTMENT}


def _scope_key(
    palace: str = PALACE,
    wing: str = WING,
    room: str = ROOM,
    compartment: str = COMPARTMENT,
) -> str:
    from workflows_mcp.engine.memory_scope_resolver import scope_key as _sk

    return _sk({"palace": palace, "wing": wing, "room": room, "compartment": compartment})


def _make_config() -> ConnectionConfig:
    return ConnectionConfig(
        dialect=DatabaseEngine.POSTGRESQL,
        host=os.environ.get("MEMORY_DB_HOST", "localhost"),
        port=int(os.environ.get("MEMORY_DB_PORT", "5432")),
        database=os.environ.get("MEMORY_DB_NAME", "workflows"),
        username=os.environ.get("MEMORY_DB_USER", "workflows"),
        password=os.environ.get("MEMORY_DB_PASSWORD", "supersecret"),
    )
