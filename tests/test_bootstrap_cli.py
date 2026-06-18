from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows_mcp.bootstrap import bootstrap_if_needed
from workflows_mcp.metadata.migrations import CURRENT_SCHEMA_VERSION
from workflows_mcp.security.passwords import (
    ALGORITHM,
    DEFAULT_ITERATIONS,
    MAX_ITERATIONS,
    MIN_ITERATIONS,
    hash_password,
    verify_password,
)


def _fetch_one(db_path: Path, query: str, params: tuple[object, ...] = ()) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(query, params).fetchone()
        assert row is not None
        return row
    finally:
        conn.close()


def test_bootstrap_creates_db_key_schema_admin_and_settings(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"

    result = bootstrap_if_needed(
        config_dir=config_dir,
        host="127.0.0.1",
        port=8000,
        admin_password="s3cure-passphrase",
    )

    assert result.created is True

    db_path = config_dir / "server.db"
    key_path = config_dir / "secrets.key"
    assert db_path.exists()
    assert key_path.exists()
    assert (db_path.stat().st_mode & 0o077) == 0
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    assert (key_path.stat().st_mode & 0o077) == 0

    schema_row = _fetch_one(db_path, "SELECT MAX(version) AS v FROM schema_migrations")
    assert int(schema_row["v"]) == CURRENT_SCHEMA_VERSION

    credentials = _fetch_one(
        db_path,
        "SELECT id, password_hash FROM admin_credentials WHERE id = 1",
    )
    assert int(credentials["id"]) == 1
    password_hash = str(credentials["password_hash"])
    assert "s3cure-passphrase" not in password_hash
    assert verify_password("s3cure-passphrase", password_hash)

    host_row = _fetch_one(
        db_path,
        "SELECT value FROM server_settings WHERE key = ?",
        ("host",),
    )
    port_row = _fetch_one(
        db_path,
        "SELECT value FROM server_settings WHERE key = ?",
        ("port",),
    )
    assert json.loads(str(host_row["value"])) == "127.0.0.1"
    assert json.loads(str(port_row["value"])) == 8000


def test_bootstrap_is_idempotent_and_preserves_password_hash_and_key(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"

    first = bootstrap_if_needed(
        config_dir=config_dir,
        host="0.0.0.0",
        port=8080,
        admin_password="first-password",
    )
    assert first.created is True

    db_path = config_dir / "server.db"
    key_path = config_dir / "secrets.key"

    original_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    original_key = key_path.read_bytes()

    second = bootstrap_if_needed(
        config_dir=config_dir,
        host="localhost",
        port=9999,
        admin_password="second-password",
    )
    assert second.created is False

    after_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    after_key = key_path.read_bytes()

    assert after_hash == original_hash
    assert verify_password("first-password", after_hash)
    assert not verify_password("second-password", after_hash)
    assert after_key == original_key


def test_bootstrap_direct_call_rejects_empty_admin_password(tmp_path: Path) -> None:
    try:
        bootstrap_if_needed(
            config_dir=tmp_path / "cfg",
            host="127.0.0.1",
            port=8000,
            admin_password="",
        )
    except ValueError as exc:
        assert "admin_password" in str(exc)
        assert "must not be empty" in str(exc)
    else:
        msg = "expected ValueError when admin_password is empty"
        raise AssertionError(msg)


def test_bootstrap_direct_call_rejects_whitespace_only_admin_password(
    tmp_path: Path,
) -> None:
    try:
        bootstrap_if_needed(
            config_dir=tmp_path / "cfg",
            host="127.0.0.1",
            port=8000,
            admin_password="   ",
        )
    except ValueError as exc:
        assert "admin_password" in str(exc)
        assert "must not be empty" in str(exc)
    else:
        msg = "expected ValueError when admin_password is whitespace only"
        raise AssertionError(msg)


def test_bootstrap_direct_call_rejects_out_of_range_ports_without_persisting(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "cfg"

    for invalid_port in (0, -1, 65536):
        try:
            bootstrap_if_needed(
                config_dir=config_dir,
                host="127.0.0.1",
                port=invalid_port,
                admin_password="valid-password",
            )
        except ValueError as exc:
            assert "port" in str(exc).lower()
            assert "1" in str(exc)
            assert "65535" in str(exc)
        else:
            msg = f"expected ValueError for invalid port {invalid_port}"
            raise AssertionError(msg)

    assert not (config_dir / "server.db").exists()
    assert not (config_dir / "secrets.key").exists()


def test_password_hash_and_verify_roundtrip() -> None:
    password = "CorrectHorseBatteryStaple"
    password_hash = hash_password(password)

    assert password not in password_hash
    assert verify_password(password, password_hash)
    assert not verify_password("wrong-password", password_hash)


def test_password_defaults_meet_iteration_policy_and_hash_encodes_it() -> None:
    assert DEFAULT_ITERATIONS >= 600_000

    password_hash = hash_password("policy-check")
    parts = password_hash.split("$")
    assert len(parts) == 4
    encoded_iterations = int(parts[1])
    assert encoded_iterations >= 600_000


def test_hash_password_rejects_iterations_outside_policy_bounds() -> None:
    assert MIN_ITERATIONS >= 600_000
    assert MAX_ITERATIONS >= MIN_ITERATIONS

    too_low = MIN_ITERATIONS - 1
    too_high = MAX_ITERATIONS + 1

    try:
        hash_password("pw", iterations=too_low)
    except ValueError:
        pass
    else:
        msg = "expected ValueError for too-low iteration count"
        raise AssertionError(msg)

    try:
        hash_password("pw", iterations=too_high)
    except ValueError:
        pass
    else:
        msg = "expected ValueError for too-high iteration count"
        raise AssertionError(msg)


def test_verify_password_rejects_out_of_policy_iterations_without_pbkdf2(
    monkeypatch,
) -> None:
    called = False

    def _pbkdf2_fail_if_called(*args: object, **kwargs: object) -> bytes:
        nonlocal called
        called = True
        msg = "pbkdf2_hmac should not be called for out-of-policy iteration count"
        raise AssertionError(msg)

    monkeypatch.setattr(hashlib, "pbkdf2_hmac", _pbkdf2_fail_if_called)

    too_high_hash = f"{ALGORITHM}${MAX_ITERATIONS + 1}$c2FsdA$ZXhwZWN0ZWQ"
    too_low_hash = f"{ALGORITHM}${MIN_ITERATIONS - 1}$c2FsdA$ZXhwZWN0ZWQ"

    assert verify_password("pw", too_high_hash) is False
    assert verify_password("pw", too_low_hash) is False
    assert called is False


def test_verify_password_rejects_malformed_hashes() -> None:
    malformed = [
        "",
        "not-even-close",
        f"{ALGORITHM}$abc$salt$expected",
        f"{ALGORITHM}$600000$***$expected",
    ]

    for value in malformed:
        assert verify_password("pw", value) is False


def test_cli_bootstrap_parses_args_and_calls_bootstrap(monkeypatch, tmp_path: Path) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    called: dict[str, object] = {}

    def _fake_bootstrap_if_needed(
        *,
        config_dir: Path,
        host: str | None,
        port: int | None,
        admin_password: str | None,
        reconfigure: bool = False,
    ) -> object:
        called["config_dir"] = config_dir
        called["host"] = host
        called["port"] = port
        called["admin_password"] = admin_password
        called["reconfigure"] = reconfigure
        return SimpleNamespace(created=True, reconfigured=False)

    monkeypatch.setattr(cli, "bootstrap_if_needed", _fake_bootstrap_if_needed)

    cli.main(
        [
            "bootstrap",
            "--config-dir",
            str(tmp_path / "cfg"),
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--admin-password",
            "top-secret",
        ]
    )

    assert called == {
        "config_dir": tmp_path / "cfg",
        "host": "0.0.0.0",
        "port": 9000,
        "admin_password": "top-secret",
        "reconfigure": False,
    }


def test_cli_bootstrap_prompts_host_port_and_password_when_flags_omitted(
    monkeypatch, tmp_path: Path
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    called: dict[str, object] = {}

    answers = iter(["0.0.0.0", "9001"])

    def _fake_input(prompt: str) -> str:
        called.setdefault("prompts", []).append(prompt)
        return next(answers)

    def _fake_getpass(prompt: str) -> str:
        called["password_prompt"] = prompt
        return "prompted-secret"

    def _fake_bootstrap_if_needed(
        *,
        config_dir: Path,
        host: str | None,
        port: int | None,
        admin_password: str | None,
        reconfigure: bool = False,
    ) -> object:
        called["config_dir"] = config_dir
        called["host"] = host
        called["port"] = port
        called["admin_password"] = admin_password
        called["reconfigure"] = reconfigure
        return SimpleNamespace(created=True, reconfigured=False)

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)
    monkeypatch.setattr(cli, "bootstrap_if_needed", _fake_bootstrap_if_needed)

    cli.main(
        [
            "bootstrap",
            "--config-dir",
            str(tmp_path / "cfg"),
        ]
    )

    assert called["prompts"] == [
        "Host [127.0.0.1]: ",
        "Port [8000]: ",
    ]
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 9001
    assert called["password_prompt"] == "Admin password: "
    assert called["admin_password"] == "prompted-secret"
    assert called["reconfigure"] is False


def test_cli_bootstrap_first_time_enter_uses_factory_host_port_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    called: dict[str, object] = {}
    answers = iter(["", ""])

    def _fake_input(prompt: str) -> str:
        called.setdefault("prompts", []).append(prompt)
        return next(answers)

    def _fake_getpass(prompt: str) -> str:
        called["password_prompt"] = prompt
        return "prompted-secret"

    def _fake_bootstrap_if_needed(
        *,
        config_dir: Path,
        host: str | None,
        port: int | None,
        admin_password: str | None,
        reconfigure: bool = False,
    ) -> object:
        called["config_dir"] = config_dir
        called["host"] = host
        called["port"] = port
        called["admin_password"] = admin_password
        called["reconfigure"] = reconfigure
        return SimpleNamespace(created=True, reconfigured=False)

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)
    monkeypatch.setattr(cli, "bootstrap_if_needed", _fake_bootstrap_if_needed)

    cli.main(["bootstrap", "--config-dir", str(tmp_path / "cfg")])

    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8000
    assert called["admin_password"] == "prompted-secret"


def test_cli_bootstrap_first_time_eof_on_host_port_prompts_uses_factory_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    called: dict[str, object] = {}
    prompts: list[str] = []

    def _fake_input(prompt: str) -> str:
        prompts.append(prompt)
        raise EOFError("stdin closed")

    def _fake_getpass(_prompt: str) -> str:
        return "prompted-secret"

    def _fake_bootstrap_if_needed(
        *,
        config_dir: Path,
        host: str | None,
        port: int | None,
        admin_password: str | None,
        reconfigure: bool = False,
    ) -> object:
        called["config_dir"] = config_dir
        called["host"] = host
        called["port"] = port
        called["admin_password"] = admin_password
        called["reconfigure"] = reconfigure
        return SimpleNamespace(created=True, reconfigured=False)

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)
    monkeypatch.setattr(cli, "bootstrap_if_needed", _fake_bootstrap_if_needed)

    cli.main(["bootstrap", "--config-dir", str(tmp_path / "cfg")])

    assert prompts == ["Host [127.0.0.1]: ", "Port [8000]: "]
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8000
    assert called["admin_password"] == "prompted-secret"
    assert called["reconfigure"] is False


def test_cli_bootstrap_skips_prompt_when_metadata_db_exists(monkeypatch, tmp_path: Path) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    config_dir = tmp_path / "cfg"
    bootstrap_if_needed(
        config_dir=config_dir,
        host="127.0.0.1",
        port=8080,
        admin_password="first-password",
    )

    db_path = config_dir / "server.db"
    key_path = config_dir / "secrets.key"
    original_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    original_key = key_path.read_bytes()

    called_getpass = False

    def _fake_getpass(prompt: str) -> str:
        nonlocal called_getpass
        called_getpass = True
        msg = f"getpass should not be called when metadata DB exists: {prompt}"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)

    cli.main(["bootstrap", "--config-dir", str(config_dir)])

    after_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    after_key = key_path.read_bytes()

    assert called_getpass is False
    assert after_hash == original_hash
    assert after_key == original_key


def test_cli_bootstrap_existing_state_prints_message_and_exits_without_reconfigure(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    config_dir = tmp_path / "cfg"
    bootstrap_if_needed(
        config_dir=config_dir,
        host="127.0.0.1",
        port=8080,
        admin_password="first-password",
    )

    db_path = config_dir / "server.db"
    original_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )

    called_getpass = False

    def _fake_getpass(prompt: str) -> str:
        nonlocal called_getpass
        called_getpass = True
        msg = f"getpass should not be called when already bootstrapped: {prompt}"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)

    cli.main(["bootstrap", "--config-dir", str(config_dir)])

    out = capsys.readouterr().err
    assert "already initialized" in out.lower()
    assert "--reconfigure" in out
    assert called_getpass is False

    after_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    assert after_hash == original_hash


def test_cli_bootstrap_reconfigure_existing_state_updates_password_with_flag(
    tmp_path: Path, capsys
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    config_dir = tmp_path / "cfg"
    bootstrap_if_needed(
        config_dir=config_dir,
        host="127.0.0.1",
        port=8080,
        admin_password="first-password",
    )

    db_path = config_dir / "server.db"

    cli.main(
        [
            "bootstrap",
            "--config-dir",
            str(config_dir),
            "--reconfigure",
            "--admin-password",
            "second-password",
        ]
    )

    out = capsys.readouterr().err
    assert "reconfigure" in out.lower()
    assert "admin password" in out.lower()
    assert "updated" in out.lower()

    updated_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    assert verify_password("second-password", updated_hash)
    assert not verify_password("first-password", updated_hash)


def test_cli_bootstrap_reconfigure_prompts_with_existing_defaults_and_enter_keeps_values(
    monkeypatch, tmp_path: Path
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    config_dir = tmp_path / "cfg"
    bootstrap_if_needed(
        config_dir=config_dir,
        host="10.0.0.5",
        port=8123,
        admin_password="first-password",
    )

    db_path = config_dir / "server.db"
    original_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )

    answers = iter(["", ""])
    prompts: list[str] = []

    def _fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    def _fake_getpass(_prompt: str) -> str:
        return ""

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)

    cli.main(["bootstrap", "--config-dir", str(config_dir), "--reconfigure"])

    assert prompts == ["Host [10.0.0.5]: ", "Port [8123]: "]
    host_row = _fetch_one(
        db_path,
        "SELECT value FROM server_settings WHERE key = ?",
        ("host",),
    )
    port_row = _fetch_one(
        db_path,
        "SELECT value FROM server_settings WHERE key = ?",
        ("port",),
    )
    assert json.loads(str(host_row["value"])) == "10.0.0.5"
    assert json.loads(str(port_row["value"])) == 8123

    after_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    assert after_hash == original_hash


def test_cli_bootstrap_reconfigure_prompt_password_updates_when_entered(
    monkeypatch, tmp_path: Path
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    config_dir = tmp_path / "cfg"
    bootstrap_if_needed(
        config_dir=config_dir,
        host="127.0.0.1",
        port=8080,
        admin_password="first-password",
    )

    db_path = config_dir / "server.db"
    answers = iter(["", ""])

    def _fake_input(_prompt: str) -> str:
        return next(answers)

    def _fake_getpass(_prompt: str) -> str:
        return "second-password"

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)

    cli.main(["bootstrap", "--config-dir", str(config_dir), "--reconfigure"])

    updated_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    assert verify_password("second-password", updated_hash)
    assert not verify_password("first-password", updated_hash)


def test_cli_bootstrap_reconfigure_password_prompt_eof_keeps_existing_password(
    monkeypatch, tmp_path: Path
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    config_dir = tmp_path / "cfg"
    bootstrap_if_needed(
        config_dir=config_dir,
        host="127.0.0.1",
        port=8080,
        admin_password="first-password",
    )

    db_path = config_dir / "server.db"
    original_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    answers = iter(["", ""])

    def _fake_input(_prompt: str) -> str:
        return next(answers)

    def _fake_getpass(_prompt: str) -> str:
        raise EOFError("stdin closed")

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)

    cli.main(["bootstrap", "--config-dir", str(config_dir), "--reconfigure"])

    after_hash = str(
        _fetch_one(
            db_path,
            "SELECT password_hash FROM admin_credentials WHERE id = 1",
        )["password_hash"]
    )
    assert after_hash == original_hash
    assert verify_password("first-password", after_hash)


def test_cli_bootstrap_rejects_empty_admin_password_flag(tmp_path: Path, capsys) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    try:
        cli.main(
            [
                "bootstrap",
                "--config-dir",
                str(tmp_path / "cfg"),
                "--admin-password",
                "",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        msg = "expected SystemExit(2) when --admin-password is empty"
        raise AssertionError(msg)

    captured = capsys.readouterr()
    assert "--admin-password" in captured.err
    assert "must not be empty" in captured.err.lower()


def test_cli_bootstrap_rejects_whitespace_only_admin_password_flag(tmp_path: Path, capsys) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    try:
        cli.main(
            [
                "bootstrap",
                "--config-dir",
                str(tmp_path / "cfg"),
                "--admin-password",
                "   ",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        msg = "expected SystemExit(2) when --admin-password is whitespace only"
        raise AssertionError(msg)

    captured = capsys.readouterr()
    assert "--admin-password" in captured.err
    assert "must not be empty" in captured.err.lower()


def test_cli_bootstrap_rejects_out_of_range_port_flag_with_friendly_error(
    tmp_path: Path, capsys
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    try:
        cli.main(
            [
                "bootstrap",
                "--config-dir",
                str(tmp_path / "cfg"),
                "--port",
                "0",
                "--admin-password",
                "s3cure-passphrase",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        msg = "expected SystemExit(2) when --port is out of range"
        raise AssertionError(msg)

    captured = capsys.readouterr()
    assert "port" in captured.err.lower()
    assert "1" in captured.err
    assert "65535" in captured.err
    assert "traceback" not in captured.err.lower()


def test_cli_bootstrap_prompt_rejects_out_of_range_port_then_accepts_default_on_enter(
    monkeypatch, tmp_path: Path
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    called: dict[str, object] = {}
    answers = iter(["0.0.0.0", "65536", ""])
    prompts: list[str] = []

    def _fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    def _fake_getpass(_prompt: str) -> str:
        return "prompted-secret"

    def _fake_bootstrap_if_needed(
        *,
        config_dir: Path,
        host: str | None,
        port: int | None,
        admin_password: str | None,
        reconfigure: bool = False,
    ) -> object:
        called["config_dir"] = config_dir
        called["host"] = host
        called["port"] = port
        called["admin_password"] = admin_password
        called["reconfigure"] = reconfigure
        return SimpleNamespace(created=True, reconfigured=False)

    monkeypatch.setattr("builtins.input", _fake_input)
    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)
    monkeypatch.setattr(cli, "bootstrap_if_needed", _fake_bootstrap_if_needed)

    cli.main(["bootstrap", "--config-dir", str(tmp_path / "cfg")])

    assert prompts == ["Host [127.0.0.1]: ", "Port [8000]: ", "Port [8000]: "]
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 8000
    assert called["admin_password"] == "prompted-secret"


def test_cli_bootstrap_friendly_message_when_password_prompt_eof_non_interactive(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    def _fake_getpass(prompt: str) -> str:
        msg = f"non-interactive prompt failed: {prompt}"
        raise EOFError(msg)

    monkeypatch.setattr(cli.getpass, "getpass", _fake_getpass)

    try:
        cli.main(["bootstrap", "--config-dir", str(tmp_path / "cfg")])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        msg = "expected SystemExit(2) when password prompt receives EOF"
        raise AssertionError(msg)

    captured = capsys.readouterr()
    assert "admin password" in captured.err.lower()
    assert "--admin-password" in captured.err


def test_cli_bootstrap_uses_env_config_dir_when_flag_omitted(monkeypatch, tmp_path: Path) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    called: dict[str, object] = {}
    env_config_dir = tmp_path / "env-cfg"

    def _fake_bootstrap_if_needed(
        *,
        config_dir: Path,
        host: str | None,
        port: int | None,
        admin_password: str | None,
        reconfigure: bool = False,
    ) -> object:
        called["config_dir"] = config_dir
        called["host"] = host
        called["port"] = port
        called["admin_password"] = admin_password
        called["reconfigure"] = reconfigure
        return SimpleNamespace(created=True, reconfigured=False)

    monkeypatch.setenv("WORKFLOWS_CONFIG_DIR", str(env_config_dir))
    monkeypatch.setattr(cli, "bootstrap_if_needed", _fake_bootstrap_if_needed)

    cli.main(
        [
            "bootstrap",
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
            "--admin-password",
            "env-secret",
        ]
    )

    assert called == {
        "config_dir": env_config_dir.expanduser(),
        "host": "127.0.0.1",
        "port": 8123,
        "admin_password": "env-secret",
        "reconfigure": False,
    }


def test_cli_bootstrap_explicit_config_dir_wins_over_env(monkeypatch, tmp_path: Path) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    called: dict[str, object] = {}
    env_config_dir = tmp_path / "env-cfg"
    explicit_config_dir = tmp_path / "explicit-cfg"

    def _fake_bootstrap_if_needed(
        *,
        config_dir: Path,
        host: str | None,
        port: int | None,
        admin_password: str | None,
        reconfigure: bool = False,
    ) -> object:
        called["config_dir"] = config_dir
        called["host"] = host
        called["port"] = port
        called["admin_password"] = admin_password
        called["reconfigure"] = reconfigure
        return SimpleNamespace(created=True, reconfigured=False)

    monkeypatch.setenv("WORKFLOWS_CONFIG_DIR", str(env_config_dir))
    monkeypatch.setattr(cli, "bootstrap_if_needed", _fake_bootstrap_if_needed)

    cli.main(
        [
            "bootstrap",
            "--config-dir",
            str(explicit_config_dir),
            "--host",
            "127.0.0.1",
            "--port",
            "8124",
            "--admin-password",
            "explicit-secret",
        ]
    )

    assert called == {
        "config_dir": explicit_config_dir,
        "host": "127.0.0.1",
        "port": 8124,
        "admin_password": "explicit-secret",
        "reconfigure": False,
    }


def test_cli_bootstrap_rejects_postgres_flags_and_does_not_bootstrap(
    monkeypatch, tmp_path: Path
) -> None:
    """Public CLI behavior: postgres args are not accepted/collected by bootstrap."""
    cli = importlib.import_module("workflows_mcp.cli")

    called = False

    def _fake_bootstrap_if_needed(**kwargs: object) -> object:
        nonlocal called
        called = True
        msg = f"bootstrap_if_needed must not be called for invalid args: {kwargs}"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "bootstrap_if_needed", _fake_bootstrap_if_needed)

    try:
        cli.main(
            [
                "bootstrap",
                "--config-dir",
                str(tmp_path / "cfg"),
                "--postgres-dsn",
                "postgresql://example.invalid/db",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        msg = "expected SystemExit(2) because --postgres-dsn is unsupported"
        raise AssertionError(msg)

    assert called is False


def test_cli_no_args_delegates_to_server_main(monkeypatch) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    called = False

    def _fake_server_main() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli.server, "main", _fake_server_main)

    cli.main([])

    assert called is True


def test_cli_no_args_does_not_invoke_ui_build(monkeypatch) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    called_server = False
    called_build = False

    def _fake_server_main() -> None:
        nonlocal called_server
        called_server = True

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal called_build
        called_build = True
        msg = f"build command should not run without --build-ui: args={args} kwargs={kwargs}"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.server, "main", _fake_server_main)
    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    cli.main([])

    assert called_server is True
    assert called_build is False


def test_cli_build_ui_invokes_npm_build_in_web_dir(monkeypatch) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    seen: dict[str, object] = {}

    def _fake_server_main() -> None:
        seen["server_called"] = True

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(["npm", "run", "build"], returncode=0)

    monkeypatch.setattr(cli.server, "main", _fake_server_main)
    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    cli.main(["--build-ui"])

    assert seen["server_called"] is True
    assert seen["args"] == (["npm", "run", "build"],)
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["cwd"] == cli._repo_root_dir() / "web"
    assert kwargs["check"] is False


def test_cli_build_ui_fails_when_web_package_json_missing(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    monkeypatch.setattr(cli, "_repo_root_dir", lambda: tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--build-ui"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "web/package.json was not found" in captured.err


def test_cli_build_ui_fails_with_actionable_message_when_npm_missing(monkeypatch, capsys) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    web_dir = cli._repo_root_dir() / "web"
    package_json = web_dir / "package.json"
    if not package_json.exists():
        pytest.skip("web/package.json missing in checkout")

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise FileNotFoundError("npm not found")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--build-ui"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "cd web && npm install" in captured.err
    assert "uv run workflows-mcp --build-ui" in captured.err
    assert "npm run build" in captured.err


def test_cli_build_ui_fails_with_actionable_message_when_build_fails(monkeypatch, capsys) -> None:
    cli = importlib.import_module("workflows_mcp.cli")

    web_dir = cli._repo_root_dir() / "web"
    package_json = web_dir / "package.json"
    if not package_json.exists():
        pytest.skip("web/package.json missing in checkout")

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args
        del kwargs
        return subprocess.CompletedProcess(["npm", "run", "build"], returncode=7, stderr="boom")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--build-ui"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "cd web && npm install" in captured.err
    assert "uv run workflows-mcp --build-ui" in captured.err
    assert "npm run build" in captured.err
    assert "stderr=boom" not in captured.err


def test_build_app_uses_workflows_config_dir_when_base_dir_not_explicit(
    monkeypatch, tmp_path: Path
) -> None:
    from workflows_mcp import server

    expected = tmp_path / "wf-config"
    monkeypatch.setenv("WORKFLOWS_CONFIG_DIR", str(expected))
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "x" * 32)

    app = server.build_app()

    assert app.state.resources.metadata.base_dir == expected.expanduser()
