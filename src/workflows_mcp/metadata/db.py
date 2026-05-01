from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def connect_metadata_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    os.chmod(path, 0o600)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    return conn
