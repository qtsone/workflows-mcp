"""Packaging regression tests.

Verifies that both wheel and sdist artifacts contain required files.
Run with: uv run pytest tests/test_package_artifacts.py -v -m slow

These tests build real artifacts and are therefore slow.
Artifacts are built once per test session via a module-scoped fixture.
"""

import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build wheel and sdist once per module into a shared temp directory."""
    out_dir = tmp_path_factory.mktemp("artifacts", numbered=False)
    result = subprocess.run(
        ["uv", "build", "--wheel", "--sdist", "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"uv build failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return out_dir


@pytest.mark.slow
def test_wheel_contains_template_memory_yaml(built_artifacts: Path) -> None:
    """Wheel must include workflows_mcp/templates/memory/system1-scan.yaml."""
    wheel = _find_artifact(built_artifacts, "*.whl")
    members = _wheel_members(wheel)
    target = "workflows_mcp/templates/memory/system1-scan.yaml"
    assert any(m == target or m.endswith("/" + target) for m in members), (
        f"Wheel missing {target!r}. Members with 'templates':\n"
        + "\n".join(m for m in members if "templates" in m or "memory" in m)
    )


@pytest.mark.slow
def test_wheel_contains_python_source(built_artifacts: Path) -> None:
    """Wheel must include workflows_mcp/server.py."""
    wheel = _find_artifact(built_artifacts, "*.whl")
    members = _wheel_members(wheel)
    target = "workflows_mcp/server.py"
    assert any(m == target or m.endswith("/" + target) for m in members), (
        f"Wheel missing {target!r}."
    )


@pytest.mark.slow
def test_sdist_contains_template_memory_yaml(built_artifacts: Path) -> None:
    """Sdist must include src/workflows_mcp/templates/memory/system1-scan.yaml."""
    sdist = _find_artifact(built_artifacts, "*.tar.gz")
    members = _sdist_members(sdist)
    target = "src/workflows_mcp/templates/memory/system1-scan.yaml"
    assert any(_member_endswith(m, target) for m in members), (
        f"Sdist missing {target!r}. Members with 'templates':\n"
        + "\n".join(m for m in members if "templates" in m or "memory" in m)
    )


@pytest.mark.slow
def test_sdist_contains_python_source(built_artifacts: Path) -> None:
    """Sdist must include src/workflows_mcp/server.py."""
    sdist = _find_artifact(built_artifacts, "*.tar.gz")
    members = _sdist_members(sdist)
    target = "src/workflows_mcp/server.py"
    assert any(_member_endswith(m, target) for m in members), (
        f"Sdist missing {target!r}."
    )


@pytest.mark.slow
def test_wheel_contains_memory_system2_derive_yaml(built_artifacts: Path) -> None:
    """Wheel must include workflows_mcp/templates/memory/system2-derive.yaml."""
    wheel = _find_artifact(built_artifacts, "*.whl")
    members = _wheel_members(wheel)
    target = "workflows_mcp/templates/memory/system2-derive.yaml"
    assert any(m == target or m.endswith("/" + target) for m in members), (
        f"Wheel missing {target!r}. Members with 'templates':\n"
        + "\n".join(m for m in members if "templates" in m or "memory" in m)
    )


@pytest.mark.slow
def test_sdist_contains_memory_system2_derive_yaml(built_artifacts: Path) -> None:
    """Sdist must include src/workflows_mcp/templates/memory/system2-derive.yaml."""
    sdist = _find_artifact(built_artifacts, "*.tar.gz")
    members = _sdist_members(sdist)
    target = "src/workflows_mcp/templates/memory/system2-derive.yaml"
    assert any(_member_endswith(m, target) for m in members), (
        f"Sdist missing {target!r}. Members with 'templates':\n"
        + "\n".join(m for m in members if "templates" in m or "memory" in m)
    )


@pytest.mark.slow
def test_wheel_contains_memory_system2_verify_lifecycle_yaml(built_artifacts: Path) -> None:
    """Wheel must include workflows_mcp/templates/memory/system2-verify-lifecycle.yaml."""
    wheel = _find_artifact(built_artifacts, "*.whl")
    members = _wheel_members(wheel)
    target = "workflows_mcp/templates/memory/system2-verify-lifecycle.yaml"
    assert any(m == target or m.endswith("/" + target) for m in members), (
        f"Wheel missing {target!r}. Members with 'templates':\n"
        + "\n".join(m for m in members if "templates" in m or "memory" in m)
    )


@pytest.mark.slow
def test_sdist_contains_memory_system2_verify_lifecycle_yaml(built_artifacts: Path) -> None:
    """Sdist must include src/workflows_mcp/templates/memory/system2-verify-lifecycle.yaml."""
    sdist = _find_artifact(built_artifacts, "*.tar.gz")
    members = _sdist_members(sdist)
    target = "src/workflows_mcp/templates/memory/system2-verify-lifecycle.yaml"
    assert any(_member_endswith(m, target) for m in members), (
        f"Sdist missing {target!r}. Members with 'templates':\n"
        + "\n".join(m for m in members if "templates" in m or "memory" in m)
    )


@pytest.mark.slow
def test_wheel_does_not_contain_builtin_workflows(built_artifacts: Path) -> None:
    """Wheel must not contain any builtin_workflows/system1-scan.yaml path."""
    wheel = _find_artifact(built_artifacts, "*.whl")
    members = _wheel_members(wheel)
    bad = [m for m in members if "builtin_workflows" in m and "system1-scan" in m]
    assert not bad, f"Wheel contains deleted builtin_workflows paths: {bad}"


@pytest.mark.slow
def test_sdist_does_not_contain_builtin_workflows(built_artifacts: Path) -> None:
    """Sdist must not contain any builtin_workflows/system1-scan.yaml path."""
    sdist = _find_artifact(built_artifacts, "*.tar.gz")
    members = _sdist_members(sdist)
    bad = [m for m in members if "builtin_workflows" in m and "system1-scan" in m]
    assert not bad, f"Sdist contains deleted builtin_workflows paths: {bad}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_artifact(out_dir: Path, pattern: str) -> Path:
    matches = list(out_dir.glob(pattern))
    assert matches, f"No artifact matching {pattern!r} in {out_dir}"
    return matches[0]


def _wheel_members(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def _sdist_members(sdist: Path) -> list[str]:
    with tarfile.open(sdist, "r:gz") as tf:
        return tf.getnames()


def _member_endswith(member: str, suffix: str) -> bool:
    """Return True if member equals suffix or ends with /suffix."""
    return member == suffix or member.endswith("/" + suffix)
