"""Deterministic scope/path/scan foundation for onboard and sync tools.

Phase 2 — scope normalization, scope_key stability, sorted scan manifest,
and sync({}) zero-argument context resolution with structured error envelopes.

Design invariants:
- scope_key is a stable, deterministic string derived from normalized scope fields.
  It does NOT depend on insertion order, caller formatting, or whitespace.
- Scan manifests are always returned in sorted (lexicographic by path) order so
  that delta computation between two snapshots is deterministic.
- sync({}) resolution: looks up available contexts (onboard checkpoints) for the
  current execution principal. Returns exactly one on success, NO_CONTEXT when
  none match, AMBIGUOUS_CONTEXT when multiple match.
- Boundary marker discovery: searches the filesystem upward from base_path for
  the nearest marker file (default strategy) or any file in a custom list.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Scope normalisation
# ---------------------------------------------------------------------------

#: Containment hierarchy for the MemPalace scope, top to bottom.
_SCOPE_FIELDS: tuple[str, str, str, str] = ("palace", "wing", "room", "compartment")

#: Default-palace label used when a scope omits the top-level palace.
DEFAULT_PALACE_LABEL: str = "default-palace"

#: Placeholder labels that signal "no real lower-topology value was supplied".
#: Project onboarding seeds these defaults; the graph builder and sync paths
#: treat them as absent so they are never persisted as literal node labels.
TOPOLOGY_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {"default-wing", "default-room", "default", "code"}
)


def normalize_topology_label(value: str | None) -> str:
    """Collapse blank or placeholder topology labels to the empty string.

    Returns the stripped label, or ``""`` when *value* is absent, blank, or a
    known placeholder (case-insensitive). Used wherever a project's seeded
    ``default-*`` topology values must not surface as real labels.
    """
    if value is None:
        return ""
    stripped = value.strip()
    if not stripped or stripped.casefold() in TOPOLOGY_PLACEHOLDER_VALUES:
        return ""
    return stripped


def _normalize_field(value: str | None) -> str | None:
    """Strip surrounding whitespace; return None for blank or absent values."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def normalize_scope(scope: dict[str, Any]) -> dict[str, str | None]:
    """Return a canonical scope dict with all four fields present and whitespace stripped.

    Keys not in _SCOPE_FIELDS are silently ignored.
    Ordering is fixed (palace, wing, room, compartment) regardless of input ordering.
    """
    return {f: _normalize_field(scope.get(f)) for f in _SCOPE_FIELDS}


def scope_key(scope: dict[str, Any]) -> str:
    """Derive a stable, deterministic string key from a scope dict.

    The key is a SHA-256 hex digest of the canonical serialisation
    ``palace=<v>|wing=<v>|room=<v>|compartment=<v>`` where each <v> is the
    normalised field value or the empty string for absent/blank fields.

    This guarantees:
    - Input-order independence (always iterates _SCOPE_FIELDS in fixed order).
    - Whitespace independence (values are stripped before hashing).
    - Collision resistance (SHA-256 of the full canonical string).
    """
    normalized = normalize_scope(scope)
    canonical = "|".join(f"{f}={normalized[f] or ''}" for f in _SCOPE_FIELDS)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Scan manifest ordering
# ---------------------------------------------------------------------------


def sorted_scan_manifest(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return entries sorted lexicographically by their 'path' field.

    Entries missing a 'path' key sort before those with one (path treated as
    empty string).  Sorting is stable for equal paths.
    """
    return sorted(entries, key=lambda e: (e.get("path") or ""))


# ---------------------------------------------------------------------------
# Boundary marker discovery
# ---------------------------------------------------------------------------

#: Default boundary marker filenames searched (in order) when ``marker_files``
#: is not customised. The first file found walking upward wins.
DEFAULT_BOUNDARY_MARKERS: tuple[str, ...] = (
    ".workflows-root",
    ".memory-root",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    ".git",
)


@dataclass(frozen=True)
class BoundaryResolution:
    """Result of a boundary marker search."""

    path: Path | None
    """Absolute path to the directory containing the marker, or None if not found."""

    marker: str | None
    """Name of the marker file/directory that was found, or None."""

    strategy: Literal["nearest", "custom"]
    """Strategy used to find the boundary."""

    searched_from: Path
    """Directory from which the upward search started."""


def find_boundary_marker(
    start: Path,
    *,
    strategy: Literal["nearest", "custom"] = "nearest",
    marker_files: list[str] | None = None,
) -> BoundaryResolution:
    """Walk upward from *start* to find a project boundary marker.

    Args:
        start: Directory to begin upward search from (resolved to absolute).
        strategy: ``"nearest"`` uses DEFAULT_BOUNDARY_MARKERS (first match wins);
                  ``"custom"`` requires *marker_files* to be a non-empty list and
                  uses those names instead.
        marker_files: Custom list of marker filenames (only used when
                      ``strategy="custom"``).

    Returns:
        BoundaryResolution describing the outcome.

    Raises:
        ValueError: When ``strategy="custom"`` but *marker_files* is empty/None.
    """
    start = start.expanduser().resolve()
    searched_from = start

    if strategy == "custom":
        if not marker_files:
            raise ValueError("marker_files must be a non-empty list when strategy='custom'")
        candidates: tuple[str, ...] = tuple(marker_files)
    else:
        candidates = DEFAULT_BOUNDARY_MARKERS

    current = start if start.is_dir() else start.parent

    while True:
        for marker in candidates:
            if (current / marker).exists():
                return BoundaryResolution(
                    path=current,
                    marker=marker,
                    strategy=strategy,
                    searched_from=searched_from,
                )
        parent = current.parent
        if parent == current:
            # Reached filesystem root without finding a marker.
            return BoundaryResolution(
                path=None,
                marker=None,
                strategy=strategy,
                searched_from=searched_from,
            )
        current = parent


# ---------------------------------------------------------------------------
# sync({}) context resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncContextCandidate:
    """Minimal information about a stored onboard checkpoint context."""

    scope: dict[str, str | None]
    """Normalised scope dict."""

    scope_key_value: str
    """Precomputed scope_key for this candidate."""

    checkpoint_data: dict[str, Any]
    """Full checkpoint payload (passed through, not inspected here)."""

    source: str = "stored_checkpoint"
    """Human-readable source label for diagnostics."""


@dataclass
class SyncContextResolution:
    """Outcome of sync({}) context resolution."""

    status: Literal["success", "NO_CONTEXT", "AMBIGUOUS_CONTEXT"]

    # Populated only on status == "success"
    context: SyncContextCandidate | None = None

    # Populated only on status == "AMBIGUOUS_CONTEXT"
    candidates: list[SyncContextCandidate] = field(default_factory=list)

    # Human-readable description for error envelopes
    message: str = ""


def resolve_sync_context(
    candidates: list[SyncContextCandidate],
    *,
    requested_scope: dict[str, Any] | None = None,
) -> SyncContextResolution:
    """Resolve a sync({}) call to exactly one context or an error state.

    When *requested_scope* is provided (non-empty after normalisation), the
    candidates are filtered to those whose scope_key matches the requested
    scope_key.  Otherwise all candidates are considered.

    Resolution rules:
    - 0 matching candidates → NO_CONTEXT
    - 1 matching candidate  → success
    - 2+ matching candidates → AMBIGUOUS_CONTEXT (candidates list included)

    Args:
        candidates: All available stored contexts for the current principal.
        requested_scope: Optional scope filter from the caller.

    Returns:
        SyncContextResolution with status and populated fields.
    """
    if requested_scope:
        requested_key = scope_key(requested_scope)
        filtered = [c for c in candidates if c.scope_key_value == requested_key]
    else:
        filtered = list(candidates)

    if len(filtered) == 0:
        return SyncContextResolution(
            status="NO_CONTEXT",
            message=(
                "No stored onboard context found. "
                "Run onboard() first to create a context before calling sync({})."
            ),
        )

    if len(filtered) == 1:
        return SyncContextResolution(
            status="success",
            context=filtered[0],
            message="",
        )

    # Multiple candidates — surface their scopes so the caller can disambiguate.
    return SyncContextResolution(
        status="AMBIGUOUS_CONTEXT",
        candidates=filtered,
        message=(
            f"sync({{}}) matched {len(filtered)} contexts; "
            "provide a scope to disambiguate. "
            f"Candidates: {[c.scope for c in filtered]}"
        ),
    )


def build_no_context_envelope() -> dict[str, Any]:
    """Return a deterministic NO_CONTEXT error envelope for sync({})."""
    return {
        "error": {
            "code": "NO_CONTEXT",
            "message": (
                "sync({}) found no stored onboard context. "
                "Run onboard() first to initialise a project context."
            ),
            "retryable": False,
        }
    }


def build_ambiguous_context_envelope(candidates: list[SyncContextCandidate]) -> dict[str, Any]:
    """Return a deterministic AMBIGUOUS_CONTEXT error envelope for sync({})."""
    candidate_info = [
        {"scope": c.scope, "scope_key": c.scope_key_value, "source": c.source} for c in candidates
    ]
    return {
        "error": {
            "code": "AMBIGUOUS_CONTEXT",
            "message": (
                f"sync({{}}) matched {len(candidates)} stored contexts; "
                "supply a scope to select one."
            ),
            "retryable": False,
            "candidates": candidate_info,
        }
    }
