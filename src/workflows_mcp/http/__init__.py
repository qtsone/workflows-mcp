"""HTTP shared resource helpers for FastAPI and FastMCP entrypoints."""

from .dependencies import get_resources
from .lifespan import AppResources, MetadataResources, build_resources

__all__ = ["AppResources", "MetadataResources", "build_resources", "get_resources"]
