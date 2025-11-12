"""State directory configuration and management.

Provides centralized state directory management based on current working directory.
Uses SHA256 hash of CWD for path-based isolation across different projects.

Architecture:
    ~/.workflows/
      states/
        <hash-of-cwd>/
          state.db          # SQLite database for job metadata
          jobs/             # Individual job JSON files
            job_abc123.json
            job_def456.json
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class StateConfig:
    """State directory configuration for workflows MCP server.

    Provides path-based isolation where each project (working directory)
    has its own state directory. Multiple MCP server instances started from
    the same directory share the same state.

    Example:
        # Project A: /home/user/project-a
        state_dir = StateConfig.get_state_dir()
        # Returns: ~/.workflows/states/a1b2c3d4e5f6/

        # Project B: /home/user/project-b
        state_dir = StateConfig.get_state_dir()
        # Returns: ~/.workflows/states/f6e5d4c3b2a1/
    """

    @staticmethod
    def get_state_dir() -> Path:
        """Get state directory for current working directory.

        Returns path-based isolated state directory using hash of CWD.
        Creates directory structure if it doesn't exist.

        Returns:
            Path to state directory: ~/.workflows/states/<hash-of-cwd>/

        Example:
            >>> StateConfig.get_state_dir()
            Path('/home/user/.workflows/states/a1b2c3d4e5f6')
        """
        # Get current working directory
        cwd = Path.cwd()

        # Create deterministic hash of CWD (first 16 chars of SHA256)
        cwd_hash = hashlib.sha256(str(cwd).encode()).hexdigest()[:16]

        # Build state directory path
        state_dir = Path.home() / ".workflows" / "states" / cwd_hash

        # Create directory structure if it doesn't exist
        state_dir.mkdir(parents=True, exist_ok=True)

        return state_dir

    @staticmethod
    def get_jobs_dir() -> Path:
        """Get jobs directory for current working directory.

        Returns directory where individual job JSON files are stored.
        Creates directory if it doesn't exist.

        Returns:
            Path to jobs directory: ~/.workflows/states/<hash-of-cwd>/jobs/

        Example:
            >>> StateConfig.get_jobs_dir()
            Path('/home/user/.workflows/states/a1b2c3d4e5f6/jobs')
        """
        jobs_dir = StateConfig.get_state_dir() / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        return jobs_dir

    @staticmethod
    def get_db_path() -> Path:
        """Get SQLite database path for current working directory.

        Returns path to SQLite database file containing job metadata.

        Returns:
            Path to database: ~/.workflows/states/<hash-of-cwd>/state.db

        Example:
            >>> StateConfig.get_db_path()
            Path('/home/user/.workflows/states/a1b2c3d4e5f6/state.db')
        """
        return StateConfig.get_state_dir() / "state.db"


__all__ = ["StateConfig"]
