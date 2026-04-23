"""Documentation verification tests for HTTP-only transport cutover.

These tests verify that user-facing documentation accurately reflects the
HTTP-only service, provides bearer-token guidance, and contains no stale
stdio startup instructions.
"""

from pathlib import Path


def _readme() -> str:
    return (Path(__file__).parent.parent / "README.md").read_text()


def _migration_doc() -> str:
    return (Path(__file__).parent.parent / "docs" / "MIGRATION.md").read_text()


def _adr_013() -> str:
    return (
        Path(__file__).parent.parent
        / "docs"
        / "adr"
        / "ADR-013-http-transport-cutover.md"
    ).read_text()


def test_readme_documents_http_only_startup() -> None:
    """README must reference the HTTP bind address and bearer token usage."""
    text = _readme()
    assert "http://127.0.0.1" in text, "README must show the HTTP endpoint"
    assert "Bearer" in text, "README must document bearer token header"


def test_readme_contains_no_stdio_references() -> None:
    """README must not contain any stdio startup instructions."""
    text = _readme()
    # Check for stdio transport references (case-insensitive)
    # Allow 'stdio' in comments or code that is explicitly about removal/legacy
    assert "stdio" not in text.lower(), (
        "README must not contain stdio references — this is an HTTP-only service"
    )


def test_readme_documents_bootstrap_token() -> None:
    """README must mention WORKFLOWS_BOOTSTRAP_TOKEN for first-start configuration."""
    text = _readme()
    assert "WORKFLOWS_BOOTSTRAP_TOKEN" in text


def test_readme_documents_public_endpoints() -> None:
    """README must document /docs, /health, and /ready endpoints."""
    text = _readme()
    assert "/docs" in text
    assert "/health" in text
    assert "/ready" in text


def test_migration_doc_exists() -> None:
    """docs/MIGRATION.md must exist and document the stdio-to-HTTP cutover."""
    doc = _migration_doc()
    assert len(doc) > 100, "MIGRATION.md must be substantive"


def test_migration_doc_covers_client_reconfiguration() -> None:
    """Migration guide must cover bearer token and endpoint reconfiguration."""
    doc = _migration_doc()
    assert "Bearer" in doc or "bearer" in doc.lower()
    assert "127.0.0.1" in doc or "endpoint" in doc.lower()


def test_migration_doc_references_stdio_removal() -> None:
    """Migration guide must explicitly mention the stdio transport removal."""
    doc = _migration_doc()
    assert "stdio" in doc.lower(), (
        "MIGRATION.md must mention the stdio transport that was removed"
    )


def test_adr_013_exists() -> None:
    """ADR-013 must exist and document the HTTP transport decision."""
    doc = _adr_013()
    assert len(doc) > 100, "ADR-013 must be substantive"


def test_adr_013_documents_rationale() -> None:
    """ADR-013 must contain a rationale or consequences section."""
    doc = _adr_013()
    lower = doc.lower()
    assert "rationale" in lower or "decision" in lower or "consequence" in lower
