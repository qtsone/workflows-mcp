#!/usr/bin/env python3
"""
Analyze git diff and file status to generate semantic commit message.

This script replaces the 216-line bash script with clean, testable Python code.
"""

import re
import sys
from pathlib import Path

# Semantic commit type patterns
COMMIT_PATTERNS = {
    "feat": {
        "files": [r".*"],
        "patterns": [
            r"add|new|create|implement",
            r"introduce",
            r"\+.*class\s+\w+",
            r"\+.*def\s+\w+",
            r"\+.*function\s+\w+",
        ],
        "priority": 5,
    },
    "fix": {
        "files": [r".*"],
        "patterns": [
            r"fix|repair|resolve|correct",
            r"bug|issue|error",
            r"patch",
            r"-.*assert|raise|throw",
        ],
        "priority": 8,
    },
    "refactor": {
        "files": [r".*\.py$", r".*\.js$", r".*\.ts$"],
        "patterns": [
            r"refactor|restructure|reorganize",
            r"rename|move",
            r"extract|split",
            r"-.*def\s+\w+.*\+.*def\s+\w+",
        ],
        "priority": 4,
    },
    "docs": {
        "files": [r".*\.md$", r".*\.rst$", r".*\.txt$", r"README", r"CHANGELOG"],
        "patterns": [
            r"document|doc",
            r"\+.*#.*",
            r"\+.*docstring",
            r"\+.*\/\/\/",
            r"\+.*\/\*\*",
        ],
        "priority": 6,
    },
    "test": {
        "files": [r"test_.*\.py$", r".*_test\.py$", r".*\.test\.js$", r".*\.spec\.ts$"],
        "patterns": [
            r"test|spec",
            r"\+.*def\s+test_",
            r"\+.*it\(|describe\(",
            r"assert|expect",
        ],
        "priority": 7,
    },
    "style": {
        "files": [r".*\.css$", r".*\.scss$", r".*\.less$"],
        "patterns": [
            r"format|formatting",
            r"whitespace|indent",
            r"style|styling",
            r"lint|linting",
        ],
        "priority": 3,
    },
    "chore": {
        "files": [
            r"package\.json$",
            r"pyproject\.toml$",
            r"Cargo\.toml$",
            r"go\.mod$",
            r"\.gitignore$",
        ],
        "patterns": [
            r"chore|maintenance",
            r"update.*dependency|dependencies",
            r"bump|upgrade",
            r"config|configuration",
        ],
        "priority": 2,
    },
    "ci": {
        "files": [
            r"\.github/workflows/.*",
            r"\.gitlab-ci\.yml$",
            r"Jenkinsfile",
            r"\.circleci/.*",
        ],
        "patterns": [
            r"ci|cd|pipeline",
            r"build|deploy",
            r"continuous",
        ],
        "priority": 6,
    },
    "perf": {
        "files": [r".*"],
        "patterns": [
            r"performance|perf|optimize|optimization",
            r"speed|faster|slow",
            r"cache|caching",
            r"efficient|efficiency",
        ],
        "priority": 7,
    },
}


def analyze_files(file_status: str) -> list[str]:
    """Extract list of changed files from git status output."""
    files = []
    for line in file_status.strip().split("\n"):
        if line:
            # Parse git status --short format
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                files.append(parts[1])
    return files


def detect_commit_type(diff_content: str, file_status: str) -> tuple[str, int]:
    """
    Detect semantic commit type based on diff content and files.

    Returns:
        Tuple of (commit_type, confidence_score)
    """
    files = analyze_files(file_status)
    scores: dict[str, int] = {ctype: 0 for ctype in COMMIT_PATTERNS}

    # Analyze files
    for file in files:
        for commit_type, config in COMMIT_PATTERNS.items():
            for file_pattern in config["files"]:
                if re.search(file_pattern, file):
                    scores[commit_type] += config["priority"]

    # Analyze diff content
    diff_lower = diff_content.lower()
    for commit_type, config in COMMIT_PATTERNS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, diff_lower, re.IGNORECASE):
                scores[commit_type] += config["priority"] * 2

    # Find best match
    if not any(scores.values()):
        return "chore", 0

    best_type = max(scores.items(), key=lambda x: x[1])
    return best_type[0], best_type[1]


def extract_scope(files: list[str]) -> str:
    """Extract scope from changed files."""
    if not files:
        return ""

    # Try to find common directory
    common_parts = []
    if len(files) == 1:
        path_parts = Path(files[0]).parts
        if len(path_parts) > 1:
            return path_parts[0]
        return ""

    # Find common prefix for multiple files
    path_parts_list = [Path(f).parts for f in files]
    for parts in zip(*path_parts_list):
        if len(set(parts)) == 1:
            common_parts.append(parts[0])
        else:
            break

    if common_parts:
        return common_parts[0]
    return ""


def generate_summary(diff_content: str, commit_type: str, files: list[str]) -> str:
    """Generate commit message summary based on diff analysis."""
    # Extract key changes
    added_lines = len([line for line in diff_content.split("\n") if line.startswith("+")])
    removed_lines = len([line for line in diff_content.split("\n") if line.startswith("-")])

    if commit_type == "feat":
        if len(files) == 1:
            return f"add {Path(files[0]).stem} implementation"
        return f"add new functionality ({len(files)} files changed)"

    elif commit_type == "fix":
        return "correct implementation issues"

    elif commit_type == "refactor":
        if removed_lines > added_lines:
            return "simplify implementation"
        return "restructure code organization"

    elif commit_type == "docs":
        return "update documentation"

    elif commit_type == "test":
        return f"add test coverage ({len(files)} test files)"

    elif commit_type == "style":
        return "apply code formatting"

    elif commit_type == "chore":
        if any("package" in f or "toml" in f for f in files):
            return "update dependencies"
        return "update project configuration"

    elif commit_type == "ci":
        return "update CI/CD configuration"

    elif commit_type == "perf":
        return "improve performance"

    return "update implementation"


def main():
    """Main entry point for semantic commit analysis."""
    if len(sys.argv) < 3:
        print("Usage: analyze_semantic_commit.py <diff_content> <file_status>", file=sys.stderr)
        sys.exit(1)

    diff_content = sys.argv[1]
    file_status = sys.argv[2]

    # Detect commit type
    commit_type, confidence = detect_commit_type(diff_content, file_status)

    # Extract scope
    files = analyze_files(file_status)
    scope = extract_scope(files)

    # Generate summary
    summary = generate_summary(diff_content, commit_type, files)

    # Format commit message
    if scope:
        message = f"{commit_type}({scope}): {summary}"
    else:
        message = f"{commit_type}: {summary}"

    print(message)


if __name__ == "__main__":
    main()
