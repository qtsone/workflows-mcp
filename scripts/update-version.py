#!/usr/bin/env python3
"""Update version in src/workflows_mcp/__init__.py

Usage: python scripts/update-version.py <version>

This script uses the official Python packaging library for robust version validation,
supporting all PEP 440 version formats including:
- Standard releases: 1.2.3
- Pre-releases: 1.2.3a1, 1.2.3b2, 1.2.3rc1
- Post-releases: 1.2.3.post1
- Dev releases: 1.2.3.dev1
- Epochs: 1!1.0
- Local versions: 1.2.3+local.version
"""

import sys
from pathlib import Path

try:
    from packaging.version import InvalidVersion, Version
except ImportError:
    print("Error: 'packaging' library not found", file=sys.stderr)
    print("Install it with: pip install packaging", file=sys.stderr)
    sys.exit(1)


def validate_version(version_string: str) -> Version:
    """Validate version string using PEP 440 specification.

    Args:
        version_string: Version string to validate

    Returns:
        Validated Version object

    Raises:
        InvalidVersion: If version string is invalid
    """
    try:
        return Version(version_string)
    except InvalidVersion as e:
        print(f"Error: Invalid version format: {version_string}", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        print(
            "\nValid formats include:",
            "  - Standard: 1.2.3",
            "  - Pre-release: 1.2.3a1, 1.2.3b2, 1.2.3rc1",
            "  - Post-release: 1.2.3.post1",
            "  - Dev release: 1.2.3.dev1",
            "  - With epoch: 1!1.0",
            "  - With local: 1.2.3+local.version",
            sep="\n",
            file=sys.stderr,
        )
        sys.exit(1)


def update_version_in_file(file_path: Path, new_version: str) -> tuple[str, bool]:
    """Update __version__ variable in a Python file.

    Args:
        file_path: Path to the Python file
        new_version: New version string

    Returns:
        Tuple of (old_version, was_updated)
    """
    if not file_path.exists():
        print(f"Error: {file_path} not found", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text(encoding="utf-8")

    old_version = None
    updated = False

    # Find and update __version__ line while preserving all formatting
    for line in content.splitlines():
        # Match __version__ = "..." with any amount of whitespace
        if line.strip().startswith("__version__") and "=" in line:
            # Extract old version (everything between quotes)
            parts = line.split("=", 1)
            if len(parts) == 2:
                value_part = parts[1].strip()
                if value_part.startswith('"') or value_part.startswith("'"):
                    quote_char = value_part[0]
                    # Find the version string between quotes
                    start = value_part.index(quote_char)
                    end = value_part.index(quote_char, start + 1)
                    old_version = value_part[start + 1 : end]

                    if old_version != new_version:
                        # Replace only the version string in the entire content
                        # This preserves all whitespace and line endings
                        old_full_line = line
                        new_value_part = (
                            value_part[: start + 1] + new_version + value_part[end:]
                        )
                        new_full_line = parts[0] + "= " + new_value_part
                        content = content.replace(old_full_line, new_full_line, 1)
                        updated = True
                    break

    if old_version is None:
        print(f"Error: Could not find __version__ in {file_path}", file=sys.stderr)
        sys.exit(1)

    if updated:
        file_path.write_text(content, encoding="utf-8")

    return old_version, updated


def main() -> None:
    """Update version in __init__.py file."""
    if len(sys.argv) != 2:
        print("Error: Version argument required", file=sys.stderr)
        print("Usage: python scripts/update-version.py <version>", file=sys.stderr)
        sys.exit(1)

    new_version_str = sys.argv[1]

    # Validate using official Python packaging library (PEP 440)
    new_version = validate_version(new_version_str)

    print(f"\n📦 Updating version to {new_version}\n")

    # Update __init__.py
    init_file = Path("src/workflows_mcp/__init__.py")
    old_version, was_updated = update_version_in_file(init_file, str(new_version))

    if was_updated:
        print(f"✓ {init_file}: {old_version} → {new_version}")
    else:
        print(f"  {init_file}: already {new_version}")

    print("\n✅ Version update complete\n")


if __name__ == "__main__":
    main()
