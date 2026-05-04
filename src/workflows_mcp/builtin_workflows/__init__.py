"""Packaged built-in workflows.

Workflows in this directory are loaded by the engine before any user-managed
source path. Names defined here cannot be shadowed by user workflows; the
loader rejects user workflows whose names collide with a built-in name.

Track 4 (TreeSitter + system1-scan) and later tracks populate this directory.
This package itself contains no workflows in this plan.
"""
