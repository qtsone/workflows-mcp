from __future__ import annotations

import os
from pathlib import Path

from .ignore import WatcherIgnorePolicy


def scan_project_files(root: Path, policy: WatcherIgnorePolicy | None = None) -> list[Path]:
    project_root = root.resolve(strict=True)
    active_policy = policy or WatcherIgnorePolicy.from_project_root(project_root)

    scanned: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(project_root, topdown=True, followlinks=False):
        current_dir = Path(dirpath)

        dirnames.sort()
        filenames.sort()

        retained_dirs: list[str] = []
        for dirname in dirnames:
            candidate_dir = current_dir / dirname
            if active_policy.is_ignored(candidate_dir):
                continue
            retained_dirs.append(dirname)
        dirnames[:] = retained_dirs

        for filename in filenames:
            candidate_file = current_dir / filename
            if active_policy.is_ignored(candidate_file):
                continue
            scanned.append(candidate_file.resolve().relative_to(project_root))

    return sorted(scanned, key=lambda path: path.as_posix())
