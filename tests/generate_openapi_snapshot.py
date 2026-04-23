#!/usr/bin/env python3
"""Regenerate and overwrite the committed OpenAPI snapshot.

Usage:
    uv run python tests/generate_openapi_snapshot.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure the src package is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "openapi.json"


def main() -> None:
    os.environ.setdefault("WORKFLOWS_BOOTSTRAP_TOKEN", "0123456789abcdef0123456789abcdef")

    from workflows_mcp.auth import TokenStore
    from workflows_mcp.server import build_app

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store = TokenStore(tmp_path / "auth.json")
        store.write_token("0123456789abcdef0123456789abcdef")
        app = build_app(base_dir=tmp_path)

    schema = app.openapi()
    _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_PATH.write_text(json.dumps(schema, sort_keys=True, indent=2) + "\n")
    print(f"OpenAPI snapshot written to {_SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
