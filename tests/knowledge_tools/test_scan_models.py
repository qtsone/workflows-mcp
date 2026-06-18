"""Scan model unit tests and scan path-safety / sensitive-default guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from _helpers import onboard

from workflows_mcp.engine.executors_file import _SENSITIVE_EXCLUDE_PATTERNS
from workflows_mcp.memory.memory_service import MemoryContractError
from workflows_mcp.tools_memory import (
    ScanConfig,
    ScanSnapshot,
    _compute_scan_delta,
    _validate_scan_path_within_workspace,
)

pytestmark = pytest.mark.asyncio


# ============================================================================
# Scan models unit tests
# ============================================================================


class TestScanConfigModel:
    def test_valid_minimal(self) -> None:
        cfg = ScanConfig(patterns=["**/*.py"])
        assert cfg.patterns == ["**/*.py"]
        assert cfg.root == "."
        assert cfg.mode == "full"
        assert cfg.deletion_policy == "archive"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            ScanConfig.model_validate({"patterns": ["*.py"], "unknown_field": True})

    def test_deletion_policy_values(self) -> None:
        for policy in ("archive", "supersede", "ignore"):
            cfg = ScanConfig(patterns=["*.py"], deletion_policy=policy)  # type: ignore[arg-type]
            assert cfg.deletion_policy == policy


class TestScanSnapshot:
    def test_extra_fields_forbidden(self) -> None:
        scan_cfg = ScanConfig(patterns=["*.py"])
        with pytest.raises(Exception):
            ScanSnapshot.model_validate(
                {
                    "scan_config": scan_cfg.model_dump(),
                    "entries": [],
                    "unexpected": True,
                }
            )

    def test_round_trip_serialization(self) -> None:
        scan_cfg = ScanConfig(patterns=["src/**/*.py"], root="/tmp")
        snap = ScanSnapshot(
            scan_config=scan_cfg,
            entries=[],
        )
        dumped = snap.model_dump()
        restored = ScanSnapshot.model_validate(dumped)
        assert restored.scan_config.patterns == ["src/**/*.py"]


class TestComputeScanDelta:
    def _make_snap(
        self,
        entries: list[dict[str, Any]],
        patterns: list[str] | None = None,
    ) -> ScanSnapshot:
        from workflows_mcp.tools_memory import FileSnapshotEntry

        return ScanSnapshot(
            scan_config=ScanConfig(patterns=patterns or ["*.py"]),
            entries=[FileSnapshotEntry(**e) for e in entries],
        )

    def test_added_files(self) -> None:
        old = self._make_snap([])
        new = self._make_snap(
            [{"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "abc"}]
        )
        added, modified, deleted = _compute_scan_delta(old, new)
        assert added == ["a.py"]
        assert modified == []
        assert deleted == []

    def test_deleted_files(self) -> None:
        old = self._make_snap(
            [{"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "abc"}]
        )
        new = self._make_snap([])
        added, modified, deleted = _compute_scan_delta(old, new)
        assert added == []
        assert modified == []
        assert deleted == ["a.py"]

    def test_modified_files_hash_differs(self) -> None:
        old = self._make_snap(
            [{"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "old"}]
        )
        new = self._make_snap(
            [{"path": "a.py", "size_bytes": 12, "mtime_ns": 2, "content_hash": "new"}]
        )
        added, modified, deleted = _compute_scan_delta(old, new)
        assert modified == ["a.py"]

    def test_no_change_same_hash(self) -> None:
        entry = {"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "same"}
        old = self._make_snap([entry])
        new = self._make_snap([entry])
        added, modified, deleted = _compute_scan_delta(old, new)
        assert added == modified == deleted == []

    def test_mtime_size_same_but_hash_differs_counts_as_modified(self) -> None:
        # Fix 3: same mtime AND size but different hash MUST be detected as modified.
        # Previously a prefilter skipped hash comparison when mtime+size matched; removed.
        entry_old = {"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "old"}
        entry_new = {"path": "a.py", "size_bytes": 10, "mtime_ns": 1, "content_hash": "new"}
        old = self._make_snap([entry_old])
        new = self._make_snap([entry_new])
        added, modified, deleted = _compute_scan_delta(old, new)
        assert "a.py" in modified  # same mtime+size, different hash → must be modified


# ============================================================================
# Security: path containment, sensitive defaults
# ============================================================================


class TestScanPathSafety:
    """Verify scan path containment enforcement with MEM_SCAN_PATH_OUT_OF_ROOT."""

    def test_out_of_override_root_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A scan.root outside WORKFLOWS_SCAN_ROOT override must raise MemoryContractError."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(allowed))

        with pytest.raises(MemoryContractError) as exc_info:
            _validate_scan_path_within_workspace(outside.resolve(), "scan.root")
        assert exc_info.value.code == "MEM_SCAN_PATH_OUT_OF_ROOT"
        assert "scan.root" in exc_info.value.message

    def test_in_root_accepted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A scan.root inside WORKFLOWS_SCAN_ROOT must not raise."""
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(tmp_path))
        _validate_scan_path_within_workspace(tmp_path / "src", "scan.root")

    def test_workspace_root_itself_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The configured root itself must be accepted."""
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(tmp_path))
        _validate_scan_path_within_workspace(tmp_path.resolve(), "scan.root")

    @pytest.mark.asyncio
    async def test_project_onboard_rejects_out_of_override_root_scan_root(
        self, monkeypatch: pytest.MonkeyPatch, mock_ctx: MagicMock, tmp_path: Path
    ) -> None:
        """project_onboard must return MEM_SCAN_PATH_OUT_OF_ROOT when scan.root is
        outside the WORKFLOWS_SCAN_ROOT override."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setenv("WORKFLOWS_SCAN_ROOT", str(allowed))

        result = await onboard(
            scan={"patterns": ["*.py"], "root": str(outside)},
            ctx=mock_ctx,
        )
        payload = json.loads(result.content[0].text)
        assert payload["error"]["code"] == "MEM_SCAN_PATH_OUT_OF_ROOT"


class TestSensitiveExcludeDefaults:
    """Verify _SENSITIVE_EXCLUDE_PATTERNS contains the required secret-file patterns."""

    def test_env_files_excluded(self) -> None:
        assert "**/.env" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/.env.*" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_pem_key_excluded(self) -> None:
        assert "**/*.pem" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/*.key" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_rsa_key_excluded(self) -> None:
        assert "**/id_rsa" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/id_rsa.*" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_credentials_secrets_excluded(self) -> None:
        assert "**/credentials" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/secrets" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_p12_pfx_excluded(self) -> None:
        assert "**/*.p12" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/*.pfx" in _SENSITIVE_EXCLUDE_PATTERNS

    def test_sqlite_db_excluded(self) -> None:
        assert "**/*.sqlite" in _SENSITIVE_EXCLUDE_PATTERNS
        assert "**/*.db" in _SENSITIVE_EXCLUDE_PATTERNS

    @pytest.mark.asyncio
    async def test_run_readfiles_scan_excludes_env_file(self, workspace_tmp: Path) -> None:
        """run_readfiles_scan must never return .env files even when they match the glob."""
        (workspace_tmp / ".env").write_text("SECRET=hunter2\n")
        (workspace_tmp / "app.py").write_text("print('hello')\n")

        from workflows_mcp.engine.executors_file import run_readfiles_scan

        files = await run_readfiles_scan(
            patterns=["**/*", "**/.env"],
            base_path=str(workspace_tmp),
            max_files=50,
        )
        paths = {f["path"] for f in files}
        assert ".env" not in paths
        assert "app.py" in paths
