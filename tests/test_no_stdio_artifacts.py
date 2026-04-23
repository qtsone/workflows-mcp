"""Sentinel: verify no stdio transport artifacts remain in the codebase.

These tests act as regression guards. If any of them fail, a stdio reference has
been re-introduced and must be removed before merging.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"

# Patterns that must not appear in source or test files
BANNED_PATTERNS: list[tuple[str, str]] = [
    ("stdio_server", "mcp stdio_server import or call"),
    ("run_stdio_server", "run_stdio_server entry point"),
    ("StdioServerParameters", "StdioServerParameters usage"),
    ("stdio_client", "stdio_client usage"),
]

# Skips whose reason mentions stdio transport retirement are banned; all
# legitimate skips must use a different reason string.
BANNED_SKIP_REASON_SUBSTRINGS = [
    "stdio transport retired",
    "pending HTTP TestClient conversion",
]


def _git_grep(pattern: str, *paths: Path) -> list[str]:
    """Return lines matching pattern across the given paths using git grep."""
    result = subprocess.run(
        ["git", "grep", "-rn", "--", pattern, *(str(p) for p in paths)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_no_stdio_server_import() -> None:
    """stdio_server must not be imported or called anywhere."""
    matches = _git_grep("stdio_server", SRC_DIR, TESTS_DIR)
    assert not matches, (
        "stdio_server reference(s) found — remove before merging:\n"
        + "\n".join(matches)
    )


def test_no_run_stdio_server() -> None:
    """run_stdio_server entry point must not exist."""
    matches = _git_grep("run_stdio_server", SRC_DIR, TESTS_DIR)
    assert not matches, (
        "run_stdio_server reference(s) found — remove before merging:\n"
        + "\n".join(matches)
    )


def test_no_stdio_server_parameters() -> None:
    """StdioServerParameters must not be used anywhere."""
    matches = _git_grep("StdioServerParameters", SRC_DIR, TESTS_DIR)
    assert not matches, (
        "StdioServerParameters reference(s) found — remove before merging:\n"
        + "\n".join(matches)
    )


def test_no_stdio_client() -> None:
    """stdio_client must not be used anywhere."""
    matches = _git_grep("stdio_client", SRC_DIR, TESTS_DIR)
    assert not matches, (
        "stdio_client reference(s) found — remove before merging:\n"
        + "\n".join(matches)
    )


def test_no_stdio_skip_reasons() -> None:
    """Skips citing stdio transport retirement must all have been converted."""
    for substring in BANNED_SKIP_REASON_SUBSTRINGS:
        matches = _git_grep(substring, TESTS_DIR)
        assert not matches, (
            f"Skipped test(s) still citing '{substring}' — convert or remove:\n"
            + "\n".join(matches)
        )
