"""Smoke validation for multi-service memory benchmark script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_multiservice_benchmark_dry_run_succeeds() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "workflows_memory_multiservice_bench.py"

    result = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--operations", "5"],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["operations"] == 5
    assert payload["scopes"] > 0
    assert payload["initial_records"] > 0
