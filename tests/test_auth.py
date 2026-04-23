from pathlib import Path

import pytest

from workflows_mcp.auth import BootstrapTokenError, TokenStore, ensure_bootstrap_token


def test_bootstrap_token_requires_minimum_entropy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", "short-token")
    with pytest.raises(BootstrapTokenError):
        ensure_bootstrap_token(Path("/tmp/workflows-test"))


def test_bootstrap_token_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKFLOWS_BOOTSTRAP_TOKEN", raising=False)
    with pytest.raises(BootstrapTokenError):
        ensure_bootstrap_token(Path("/tmp/workflows-test"))


def test_token_store_never_persists_plaintext(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("a" * 40)
    contents = (tmp_path / "auth.json").read_text()
    assert "a" * 40 not in contents


def test_valid_token_round_trip(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("b" * 40)
    assert store.validate("b" * 40) is True
    assert store.validate("c" * 40) is False


def test_token_store_file_has_restricted_permissions(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("d" * 40)
    mode = (tmp_path / "auth.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_token_store_validate_missing_file_returns_false(tmp_path: Path) -> None:
    store = TokenStore(tmp_path / "nonexistent.json")
    assert store.validate("any-token") is False


def test_ensure_bootstrap_token_idempotent_when_store_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second call with store already present should not overwrite token."""
    token = "e" * 40
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", token)
    store1 = ensure_bootstrap_token(tmp_path)
    hash_before = (tmp_path / "auth.json").read_text()
    store2 = ensure_bootstrap_token(tmp_path)
    hash_after = (tmp_path / "auth.json").read_text()
    assert hash_before == hash_after
    assert store1.validate(token) is True
    assert store2.validate(token) is True


def test_ensure_bootstrap_token_succeeds_without_env_when_store_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After auth state exists, startup must not fail if env var is absent.

    Spec §7.4: bootstrap token is only required on first start when no
    persisted auth state exists.
    """
    token = "f" * 40
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", token)
    ensure_bootstrap_token(tmp_path)  # first start — writes store

    monkeypatch.delenv("WORKFLOWS_BOOTSTRAP_TOKEN")
    # Must not raise even though env var is now absent.
    store = ensure_bootstrap_token(tmp_path)
    assert store.validate(token) is True
