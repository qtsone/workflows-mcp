"""Entry point for workflows-mcp HTTP server.

Delegates to ``server.main()`` which starts the Uvicorn/FastAPI HTTP service.
"""


def main() -> None:
    """Entry point for direct execution."""
    from .server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
