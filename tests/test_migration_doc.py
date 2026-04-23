"""Release-blocking tests for docs/MIGRATION.md content.

These tests ensure the migration documentation covers all required topics for
the stdio→HTTP cutover before any release is made.
"""

from pathlib import Path

_MIGRATION_DOC = Path(__file__).parent.parent / "docs" / "MIGRATION.md"


def test_migration_doc_exists():
    """docs/MIGRATION.md must exist."""
    assert _MIGRATION_DOC.exists(), f"docs/MIGRATION.md not found at {_MIGRATION_DOC}"


def test_migration_doc_covers_http_cutover():
    """Migration doc must cover the HTTP transport, /mcp endpoint, bootstrap token, and stdio."""
    text = _MIGRATION_DOC.read_text()

    assert "HTTP" in text or "http" in text, (
        "docs/MIGRATION.md must mention HTTP transport"
    )
    assert "/mcp" in text or "MCP endpoint" in text, (
        "docs/MIGRATION.md must mention the /mcp endpoint"
    )
    assert "WORKFLOWS_BOOTSTRAP_TOKEN" in text or "bootstrap token" in text, (
        "docs/MIGRATION.md must document the WORKFLOWS_BOOTSTRAP_TOKEN or bootstrap token"
    )
    assert "stdio" in text, (
        "docs/MIGRATION.md must mention stdio to confirm the stdio→HTTP migration is documented"
    )


def test_migration_doc_covers_breaking_changes():
    """Migration doc must flag a breaking change."""
    text = _MIGRATION_DOC.read_text()

    has_breaking_keyword = "breaking" in text.lower()
    # Alternatively a version bump marker such as "10.0.0" satisfies the gate.
    has_version_bump = "10.0.0" in text

    assert has_breaking_keyword or has_version_bump, (
        "docs/MIGRATION.md must mention a breaking change "
        "(look for 'breaking' case-insensitive or a version bump marker like '10.0.0')"
    )


def test_migration_doc_not_placeholder():
    """Migration doc must not be a stub — must have substantial content."""
    text = _MIGRATION_DOC.read_text()

    assert len(text) > 500, (
        f"docs/MIGRATION.md is too short ({len(text)} chars); expected > 500 chars"
    )

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    stub_lines = [
        line for line in non_empty_lines if line.upper() in {"TODO", "TBD"}
    ]
    assert len(stub_lines) < len(non_empty_lines), (
        "docs/MIGRATION.md appears to consist only of TODO/TBD placeholder lines"
    )
