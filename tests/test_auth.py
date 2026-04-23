from pathlib import Path

import pytest

from workflows_mcp.auth import BootstrapTokenError, TokenStore, ensure_bootstrap_token

# ---------------------------------------------------------------------------
# Token lifecycle: rotate
# ---------------------------------------------------------------------------


def test_rotate_token_invalidates_old_token(tmp_path: Path) -> None:
    """After rotation the previous token must not authenticate."""
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("a" * 40)
    store.rotate_token("b" * 40)
    assert store.validate("a" * 40) is False
    assert store.validate("b" * 40) is True


def test_rotate_token_file_mode_is_0600(tmp_path: Path) -> None:
    """rotate_token must preserve 0600 permissions on the store file."""
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("a" * 40)
    store.rotate_token("b" * 40)
    mode = (tmp_path / "auth.json").stat().st_mode & 0o777
    assert mode == 0o600


# ---------------------------------------------------------------------------
# Token lifecycle: revoke
# ---------------------------------------------------------------------------


def test_revoke_token_invalidates_current_token(tmp_path: Path) -> None:
    """After revocation the current token must not authenticate."""
    store = TokenStore(tmp_path / "auth.json")
    store.write_token("a" * 40)
    store.revoke()
    assert store.validate("a" * 40) is False


def test_revoke_when_store_absent_does_not_raise(tmp_path: Path) -> None:
    """Revoking when no store file exists must succeed silently."""
    store = TokenStore(tmp_path / "nonexistent.json")
    store.revoke()  # must not raise


# ---------------------------------------------------------------------------
# Recovery / bootstrap path
# ---------------------------------------------------------------------------


def test_recovery_path_bootstrap_token_when_store_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the store is absent and WORKFLOWS_BOOTSTRAP_TOKEN is set,
    ensure_bootstrap_token must write a new token and allow validation.

    Addendum §8.2: recovery path must be testable, not just implied.
    """
    token = "g" * 40
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", token)
    store = ensure_bootstrap_token(tmp_path)
    assert store.validate(token) is True
    assert (tmp_path / "auth.json").exists()


def test_recovery_path_after_explicit_revoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After revocation (store absent), supplying a bootstrap token must
    issue a new admin token without requiring a reinstall.

    Addendum §8.2: recovery path covers explicitly-reset auth state.
    """
    token_initial = "h" * 40
    token_recovery = "i" * 40

    # First start: write token
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", token_initial)
    store = ensure_bootstrap_token(tmp_path)
    assert store.validate(token_initial) is True

    # Explicit revocation
    store.revoke()
    assert not (tmp_path / "auth.json").exists()

    # Recovery: new bootstrap token while store is absent
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", token_recovery)
    recovered_store = ensure_bootstrap_token(tmp_path)
    assert recovered_store.validate(token_recovery) is True
    assert recovered_store.validate(token_initial) is False


def test_recovery_file_mode_is_0600_after_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auth store created during recovery must have mode 0600."""
    token = "j" * 40
    monkeypatch.setenv("WORKFLOWS_BOOTSTRAP_TOKEN", token)
    ensure_bootstrap_token(tmp_path)
    mode = (tmp_path / "auth.json").stat().st_mode & 0o777
    assert mode == 0o600


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
