#!/usr/bin/env python3
"""Snapshot generation script — retired as part of HTTP transport cutover (ADR-013).

The stdio-based snapshot generation workflow (TestWorkflowSnapshots + this script)
has been removed. Snapshot-based regression tests will be re-introduced using the
HTTP TestClient in a follow-up task.

Do not run this script. It will exit with a non-zero status code.
"""

import sys


def main() -> None:
    print(
        "generate_snapshots.py: retired — stdio transport removed (ADR-013).\n"
        "Snapshot-based regression tests will be re-introduced via HTTP TestClient."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
