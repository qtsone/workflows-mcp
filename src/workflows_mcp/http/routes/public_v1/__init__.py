from __future__ import annotations

import ipaddress
import os

from fastapi import APIRouter

router = APIRouter(prefix="/api/public/v1", tags=["public-v1"])


def _is_loopback_bind_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        # Unknown hostnames are treated as non-loopback for safer warnings.
        return False


@router.get("/system/status")
async def system_status() -> dict[str, str]:
    """Return minimal public service status payload for v1 namespace."""
    bind_host = os.getenv("WORKFLOWS_BIND_HOST", "127.0.0.1")
    payload: dict[str, str] = {"status": "ok"}
    if not _is_loopback_bind_host(bind_host):
        payload["warning"] = (
            f"Service bind host is set to '{bind_host}', which is not loopback. "
            "The service may be reachable from the local network."
        )
    return payload
