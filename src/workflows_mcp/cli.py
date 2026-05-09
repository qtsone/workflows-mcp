"""CLI entrypoint routing for workflows-mcp."""

from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from . import server
from .bootstrap import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    bootstrap_if_needed,
    read_bootstrap_defaults,
    validate_port,
)


def _prompt_with_default(prompt: str, default: str) -> str:
    try:
        raw = input(f"{prompt} [{default}]: ")
    except (EOFError, OSError):
        return default

    value = raw.strip()
    return value if value else default


def _prompt_port_with_default(default: int, parser: argparse.ArgumentParser) -> int:
    while True:
        value = _prompt_with_default("Port", str(default))
        try:
            return validate_port(int(value))
        except ValueError as exc:
            parser._print_message(f"Invalid port value: {value!r}. {exc}\n")


def _resolve_password(
    *,
    parser: argparse.ArgumentParser,
    has_existing_state: bool,
    explicit_password: str | None,
) -> str | None:
    if explicit_password is not None:
        return explicit_password

    try:
        prompted = getpass.getpass("Admin password: ")
    except EOFError:
        if has_existing_state:
            return None
        parser.exit(
            2,
            "Unable to read admin password from prompt. "
            "Provide --admin-password in non-interactive environments.\n",
        )

    if has_existing_state and prompted == "":
        return None

    if not prompted.strip():
        parser.exit(
            2,
            "Invalid value for admin password: must not be empty or whitespace-only.\n",
        )

    return prompted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workflows-mcp")
    parser.add_argument(
        "--build-ui",
        action="store_true",
        help="Build frontend assets from web/ before starting the server.",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--config-dir", type=Path)
    bootstrap_parser.add_argument("--host")
    bootstrap_parser.add_argument("--port", type=int)
    bootstrap_parser.add_argument("--admin-password")
    bootstrap_parser.add_argument("--reconfigure", action="store_true")

    return parser


def _repo_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_ui_assets_or_exit(parser: argparse.ArgumentParser) -> None:
    repo_root = _repo_root_dir()
    web_dir = repo_root / "web"
    package_json = web_dir / "package.json"
    install_command = "cd web && npm install"
    retry_build_ui_command = "uv run workflows-mcp --build-ui"
    direct_build_command = "cd web && npm run build"

    if not package_json.is_file():
        parser.exit(
            2,
            "Cannot use --build-ui from this location: web/package.json was not found. "
            "Run workflows-mcp from the repository root/source checkout.\n",
        )

    try:
        build = subprocess.run(
            ["npm", "run", "build"],
            cwd=web_dir,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        parser.exit(
            2,
            "Failed to run frontend build because npm is unavailable. "
            f"Install frontend dependencies first with `{install_command}`, then retry "
            f"`{retry_build_ui_command}` or run `{direct_build_command}`. ({exc})\n",
        )

    if build.returncode != 0:
        parser.exit(
            2,
            "Frontend build failed before server startup. "
            f"Install frontend dependencies first with `{install_command}`, then retry "
            f"`{retry_build_ui_command}` or run `{direct_build_command}`.\n",
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Route CLI invocation to server startup or bootstrap."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None and args.build_ui:
        _build_ui_assets_or_exit(parser)

    if args.subcommand == "bootstrap":
        config_dir = server._resolve_base_dir(args.config_dir)
        has_existing_state = (config_dir / "server.db").exists()

        if has_existing_state and not args.reconfigure:
            print(
                "Bootstrap already initialized at "
                f"{config_dir}. Use --reconfigure to change existing state.",
                file=sys.stderr,
            )
            return

        if args.admin_password is not None and not args.admin_password.strip():
            parser.exit(
                2,
                "Invalid value for --admin-password: must not be empty or whitespace-only.\n",
            )

        defaults = read_bootstrap_defaults(config_dir)

        host = args.host
        if host is None:
            host_default = defaults.host if has_existing_state else DEFAULT_HOST
            host = _prompt_with_default("Host", host_default)

        port = args.port
        if port is None:
            port_default = defaults.port if has_existing_state else DEFAULT_PORT
            port = _prompt_port_with_default(port_default, parser)
        else:
            try:
                port = validate_port(port)
            except ValueError as exc:
                parser.exit(2, f"Invalid value for --port: {exc}.\n")

        admin_password = _resolve_password(
            parser=parser,
            has_existing_state=has_existing_state,
            explicit_password=args.admin_password,
        )

        try:
            result = bootstrap_if_needed(
                config_dir=config_dir,
                host=host,
                port=port,
                admin_password=admin_password,
                reconfigure=args.reconfigure,
            )
        except ValueError as exc:
            parser.exit(2, f"Bootstrap configuration error: {exc}.\n")

        if result.reconfigured:
            if admin_password is not None:
                print(
                    "Bootstrap reconfiguration completed for "
                    f"{config_dir}; admin password updated.",
                    file=sys.stderr,
                )
            else:
                print(f"Bootstrap reconfiguration enabled for {config_dir}.", file=sys.stderr)
        elif result.created:
            print(f"Bootstrap initialized at {config_dir}.", file=sys.stderr)
        return

    server.main()
