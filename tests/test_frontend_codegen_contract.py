"""Contract test: frontend generated client must be fresh.

This test executes the repository-supported freshness check in ``web``:

    npm run check:client
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"


def test_frontend_generated_client_is_fresh() -> None:
    """Generated frontend client must match committed OpenAPI snapshot + schema types."""
    npm = shutil.which("npm")
    assert npm is not None, "npm is required to run frontend codegen contract checks"

    result = subprocess.run(
        [npm, "run", "check:client"],
        cwd=WEB_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Frontend generated client freshness check failed. "
        "From repo root run: cd web && npm run generate:client && npm run check:client\n\n"
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
