from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ConfigService:
    """Validates and atomically writes ``~/.workflows/llm-config.yml``.

    ``apply_payload`` uses a temp-file + fsync + rename sequence to ensure
    the config artifact is never left in a partially-written state.  A
    failed validation never touches the config file.
    """

    base_dir: Path
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def validate_payload(self, payload: dict[str, Any]) -> dict[str, str]:
        """Return a mapping of field names to error messages.

        An empty dict means the payload is valid.
        """
        errors: dict[str, str] = {}
        if not isinstance(payload.get("profiles"), list):
            errors["profiles"] = "profiles must be a list"
        return errors

    async def apply_payload(self, payload: dict[str, Any]) -> None:
        """Atomically write *payload* to ``<base_dir>/llm-config.yml``.

        Uses an asyncio lock to prevent concurrent writes.  Raises
        ``RuntimeError("CONFIG_WRITE_IN_PROGRESS")`` if a write is already in
        flight.  Uses ``NamedTemporaryFile -> fsync -> rename`` so the target
        file is never observed in a partially-written state by concurrent readers.
        """
        if self._lock.locked():
            raise RuntimeError("CONFIG_WRITE_IN_PROGRESS")
        async with self._lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            target = self.base_dir / "llm-config.yml"
            with tempfile.NamedTemporaryFile(
                "w",
                delete=False,
                dir=self.base_dir,
                suffix=".tmp",
            ) as handle:
                yaml.safe_dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_name = handle.name
            Path(tmp_name).replace(target)
