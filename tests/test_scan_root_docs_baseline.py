"""Documentation baseline checks for WORKFLOWS_SCAN_ROOT hardening guidance."""

from __future__ import annotations

from pathlib import Path


def test_readme_documents_scan_root_hardening_baseline() -> None:
    readme = Path("README.md").read_text(encoding="utf-8").lower()

    assert "default allowed root is `/`." in readme
    assert "hardened deployments should not leave this at `/`" in readme
    assert (
        "local development: `workflows_scan_root=/absolute/path/to/your/workflows-mcp-repo`"
        in readme
    )
    assert (
        "ci/release runners: `workflows_scan_root=$ci_project_dir` "
        "(or runner workspace root for this repository only)"
        in readme
    )
    assert (
        "effective knowledge file access is an intersection: "
        "`workflows_scan_root ∩ fs_root`, plus explicitly approved extra allowlist roots."
        in readme
    )
