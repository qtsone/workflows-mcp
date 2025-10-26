"""Entry point for workflows-mcp MCP server.

This module provides the main entry point for running the MCP server
via `uv run` or direct execution. Follows the official Anthropic MCP
Python SDK patterns.

Orchestrates the import sequence to ensure tool decorators are registered
before the server starts:
1. Import tools module (triggers @mcp.tool() decorator registration)
2. Import server module (provides FastMCP instance and entry point)
3. Call server.main() to start the MCP server
"""


def main() -> None:
    """Entry point for direct execution.

    This function orchestrates the startup sequence:
    - Imports tools module for decorator registration
    - Imports server main function
    - Starts the MCP server
    """
    # Import tools first to register @mcp.tool() decorators
    from . import tools  # noqa: F401 - imported for side effects (decorator registration)

    # Import and call server main
    from .server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
