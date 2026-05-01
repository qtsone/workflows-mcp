"""Entry point for workflows-mcp CLI."""


def main() -> None:
    """Entry point for direct execution through CLI router."""
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
