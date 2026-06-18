"""File-filtering seam for ReadFiles: gitignore, glob exclusion, binary detection.

Decides which files the executor should read and how to encode them — entirely
independent of outline extraction. Pure path/byte predicates, unit-testable
without touching the format or section seams.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pathspec

# Default exclusion patterns (version control, dependencies, build artifacts)
DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    # Version control
    "**/.git/*",
    "**/.git/**",
    "**/.svn/*",
    "**/.hg/*",
    "**/.CVS/*",
    "**/.DS_Store",
    # Dependencies
    "**/node_modules/**",
    "**/bower_components/**",
    "**/vendor/**",
    "**/venv/**",
    "**/env/**",
    "**/.venv/**",
    # Python specific
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.egg-info/**",
    # Build artifacts and logs
    "**/build/**",
    "**/dist/**",
    "**/*.log",
    "**/*.tmp",
    "**/*.swp",
    # Compiled files and archives
    "**/*.so",
    "**/*.dll",
    "**/*.exe",
    "**/*.jar",
    "**/*.class",
    "**/*.zip",
    "**/*.tar.gz",
    "**/*.rar",
]

# Binary file extensions to base64 encode
BASE64_ENCODE_EXTENSIONS: set[str] = {
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    # Documents
    ".pdf",
    # Audio
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    # Video
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


def is_binary(file_path: Path) -> bool:
    """Check if file is binary by reading first 4KB.

    Args:
        file_path: Path to file to check

    Returns:
        True if file contains null bytes (binary), False otherwise
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(4096)
            return b"\x00" in chunk
    except OSError:
        return True


def load_gitignore_patterns(base_path: Path) -> list[str]:
    """Load .gitignore patterns from base directory.

    Args:
        base_path: Base directory containing .gitignore file

    Returns:
        List of gitignore patterns (empty if no .gitignore found)
    """
    gitignore_path = base_path / ".gitignore"
    patterns: list[str] = []

    if not gitignore_path.exists():
        return patterns

    try:
        with open(gitignore_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    except OSError:
        pass

    return patterns


def matches_pattern(file_path: Path, base_path: Path, pattern: str) -> bool:
    """Check if file matches a glob pattern (handles ** recursive patterns).

    Args:
        file_path: Absolute file path to check
        base_path: Base directory for relative path calculation
        pattern: Glob pattern (may contain ** for recursive matching)

    Returns:
        True if file matches pattern, False otherwise
    """
    try:
        relative_path = file_path.relative_to(base_path)
        relative_str = str(relative_path)

        # Handle different pattern types
        if pattern.startswith("**/"):
            # Pattern like **/foo matches foo at any depth
            sub_pattern = pattern[3:]
            if fnmatch.fnmatch(relative_str, f"*/{sub_pattern}") or fnmatch.fnmatch(
                relative_str, sub_pattern
            ):
                return True
        elif pattern.endswith("/**"):
            # Pattern like foo/** matches everything under foo/
            dir_pattern = pattern[:-3]
            if relative_str.startswith(dir_pattern + "/") or relative_str == dir_pattern:
                return True
        elif "**" in pattern:
            # General ** pattern - convert to simpler form
            converted = pattern.replace("**/", "*/").replace("/**", "/*")
            if fnmatch.fnmatch(relative_str, converted):
                return True

        # Direct pattern match
        if fnmatch.fnmatch(relative_str, pattern):
            return True

        # Check path components
        parts = relative_path.parts
        for i in range(len(parts)):
            partial = "/".join(parts[: i + 1])
            if fnmatch.fnmatch(partial, pattern.rstrip("/**")):
                return True

    except ValueError:
        pass

    return False


def create_gitignore_spec(patterns: list[str]) -> pathspec.PathSpec | None:
    """Create PathSpec for gitignore pattern matching.

    Args:
        patterns: List of gitignore patterns

    Returns:
        PathSpec instance, or None if no patterns or creation fails
    """
    if not patterns:
        return None

    try:
        return pathspec.GitIgnoreSpec.from_lines(patterns)
    except Exception:
        return None


def matches_gitignore(file_path: Path, base_path: Path, spec: pathspec.PathSpec) -> bool:
    """Check if file matches gitignore patterns using pathspec.

    Args:
        file_path: Absolute file path to check
        base_path: Base directory for relative path calculation
        spec: PathSpec instance from create_gitignore_spec

    Returns:
        True if file should be ignored, False otherwise
    """
    try:
        relative_path = str(file_path.relative_to(base_path))
        return spec.match_file(relative_path)
    except (ValueError, AttributeError):
        return False
