"""FastMCP server initialization for workflows-mcp.

This module initializes the MCP server and manages shared resources via lifespan context.
All tool implementations are in the tools module.

Following the official Anthropic Python SDK patterns:
- Lifespan context manager for resource initialization and cleanup
- Context injection for tool access to shared resources
- FastMCP server with stdio transport
"""

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .context import AppContext, AppContextType
from .engine import WorkflowRegistry
from .engine.executor_base import create_default_registry
from .engine.io_queue import IOQueue
from .engine.job_queue import JobQueue

logger = logging.getLogger(__name__)

# =============================================================================
# Shared Resources and Lifespan Management
# =============================================================================


def get_max_recursion_depth() -> int:
    """Get maximum workflow recursion depth from environment.

    Reads WORKFLOWS_MAX_RECURSION_DEPTH environment variable.
    Default: 50, Valid range: 1-10000 (clamped automatically)

    Returns:
        Maximum recursion depth (1-10000)
    """
    try:
        depth = int(os.getenv("WORKFLOWS_MAX_RECURSION_DEPTH", "50"))
        return max(1, min(10000, depth))
    except ValueError:
        return 50


def load_workflows(registry: WorkflowRegistry) -> None:
    """Load workflows from built-in templates and optional user-provided directories.

    This function:
    1. Parses WORKFLOWS_TEMPLATE_PATHS environment variable (comma-separated paths)
    2. Builds directory list: [built_in_templates, ...user_template_paths]
    3. Uses registry.load_from_directories() with on_duplicate="overwrite"
    4. Logs clearly which templates are built-in vs user-provided

    Priority: User templates OVERRIDE built-in templates by name.

    Environment Variables:
        WORKFLOWS_TEMPLATE_PATHS: Comma-separated list of additional template directories.
            Paths can use ~ for home directory. Empty or missing variable is handled gracefully.

    Example:
        WORKFLOWS_TEMPLATE_PATHS="~/my-workflows,/opt/company-workflows"
        # Load order:
        # 1. Built-in: src/workflows_mcp/templates/
        # 2. User: ~/my-workflows (overrides built-in by name)
        # 3. User: /opt/company-workflows (overrides both by name)

    Args:
        registry: Registry to load workflows into

    Note:
        Post ADR-008: WorkflowRunner is stateless, workflows only need to be
        loaded into registry. No executor loading step required.
    """
    # Built-in templates directory
    built_in_templates = Path(__file__).parent / "templates"

    # Validate built-in templates directory exists
    if not built_in_templates.exists():
        raise RuntimeError(
            f"Built-in templates directory not found: {built_in_templates}\n"
            "This indicates a broken installation. Please reinstall workflows-mcp."
        )
    if not built_in_templates.is_dir():
        raise RuntimeError(
            f"Built-in templates path is not a directory: {built_in_templates}\n"
            "This indicates a broken installation. Please reinstall workflows-mcp."
        )

    # Parse WORKFLOWS_TEMPLATE_PATHS environment variable
    env_paths_str = os.getenv("WORKFLOWS_TEMPLATE_PATHS", "")
    user_template_paths: list[Path] = []

    if env_paths_str.strip():
        # Split by comma, strip whitespace, expand ~, and convert to Path
        for path_str in env_paths_str.split(","):
            path_str = path_str.strip()
            if path_str:
                # Expand ~ for home directory
                expanded_path = Path(path_str).expanduser()

                # Validate path exists and is a directory
                if not expanded_path.exists():
                    logger.warning(f"Template path does not exist, skipping: {expanded_path}")
                    continue
                if not expanded_path.is_dir():
                    logger.warning(f"Template path is not a directory, skipping: {expanded_path}")
                    continue

                user_template_paths.append(expanded_path)

        if user_template_paths:
            logger.info(f"User template paths from WORKFLOWS_TEMPLATE_PATHS: {user_template_paths}")
        else:
            logger.warning("WORKFLOWS_TEMPLATE_PATHS provided but no valid directories found")

    # Build directory list: built-in first, then user paths (user paths override)
    # Cast to list[Path | str] for type compatibility with load_from_directories
    directories_to_load: list[Path | str] = [built_in_templates]
    directories_to_load.extend(user_template_paths)

    logger.info(f"Loading workflows from {len(directories_to_load)} directories")
    logger.info(f"  Built-in: {built_in_templates}")
    for idx, user_path in enumerate(user_template_paths, 1):
        logger.info(f"  User {idx}: {user_path}")

    # Load workflows from all directories with overwrite policy (user templates override)
    result = registry.load_from_directories(directories_to_load, on_duplicate="overwrite")

    if not result.is_success:
        error_msg = f"Failed to load workflows: {result.error}"
        logger.error(error_msg)
        raise RuntimeError(
            f"{error_msg}\n"
            "Server cannot start without workflows. Please check:\n"
            "1. Built-in templates directory is intact\n"
            "2. WORKFLOWS_TEMPLATE_PATHS (if set) contains valid workflow YAML files\n"
            "3. All workflow files follow the correct schema"
        )

    # Log loading results per directory
    # Type narrowing: result.is_success guarantees value is not None
    assert result.value is not None
    load_counts = result.value
    total_workflows = sum(load_counts.values())

    if load_counts:
        logger.info("Workflow loading summary:")
        built_in_count = load_counts.get(str(built_in_templates), 0)
        logger.info(f"  Built-in templates: {built_in_count} workflows")

        for user_path in user_template_paths:
            user_count = load_counts.get(str(user_path), 0)
            logger.info(f"  User templates ({user_path}): {user_count} workflows")

    logger.info(f"Successfully loaded {total_workflows} total workflows into registry")


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with resource initialization and cleanup.

    This lifespan context manager:
    1. Initializes shared resources (workflow registry, executor registry, checkpoint store)
    2. Loads workflows from built-in and user template directories
    3. Initializes secret management system
    4. Yields context to make resources available to tools
    5. Cleans up resources on shutdown

    Environment Variables:
        WORKFLOWS_MAX_RECURSION_DEPTH: Maximum workflow recursion depth
            (default: 50, range: 1-10000)
        WORKFLOW_SECRET_*: Secret environment variables for workflows

    Args:
        _server: FastMCP server instance (unused, required by FastMCP signature)

    Yields:
        AppContext with initialized resources (ADR-008 pattern)
    """
    # Startup: initialize resources
    logger.info("Initializing MCP server resources...")

    # Read max recursion depth from environment
    max_recursion_depth = get_max_recursion_depth()
    if max_recursion_depth != 50:
        logger.info(f"Using max recursion depth: {max_recursion_depth}")

    # Initialize secret provider and check for configured secrets
    from .engine.secrets import EnvVarSecretProvider

    secret_provider = EnvVarSecretProvider()
    secret_keys = await secret_provider.list_secret_keys()

    logger.info(f"Secret provider: {secret_provider.__class__.__name__}")
    logger.info(f"Available secrets: {len(secret_keys)}")

    if len(secret_keys) == 0:
        logger.warning(
            "No secrets configured. Use WORKFLOW_SECRET_* environment variables to provide secrets."
        )
    else:
        # Log secret keys (not values!) for debugging
        logger.debug(f"Secret keys: {', '.join(sorted(secret_keys))}")

    # Initialize LLM config loader
    from .engine.llm_config import LLMConfigLoader

    llm_config_loader = LLMConfigLoader()
    llm_config = llm_config_loader.load_config()

    logger.info(
        f"LLM config: {len(llm_config.providers)} providers, {len(llm_config.profiles)} profiles"
    )
    if llm_config.default_profile:
        logger.info(f"Default LLM profile: {llm_config.default_profile}")

    # Create executor registry with all built-in executors
    executor_registry = create_default_registry()

    # Create workflow registry
    registry = WorkflowRegistry()

    # Read queue configuration from environment variables
    io_queue_enabled = os.getenv("WORKFLOWS_IO_QUEUE_ENABLED", "true").lower() == "true"
    job_queue_enabled = os.getenv("WORKFLOWS_JOB_QUEUE_ENABLED", "true").lower() == "true"
    num_workers = int(os.getenv("WORKFLOWS_JOB_QUEUE_WORKERS", "3"))

    # Create IO queue if enabled (for serialized file operations)
    io_queue = IOQueue() if io_queue_enabled else None

    # Create AppContext first (needed by JobQueue)
    app_context = AppContext(
        registry=registry,
        executor_registry=executor_registry,
        llm_config_loader=llm_config_loader,
        io_queue=io_queue,
        job_queue=None,  # Will be set after JobQueue creation if enabled
        max_recursion_depth=max_recursion_depth,
    )

    # Create and initialize JobQueue if enabled
    job_queue = None
    if job_queue_enabled:
        job_queue = JobQueue(app_context, num_workers=num_workers)
        app_context.job_queue = job_queue

    # Load workflows into registry
    load_workflows(registry)

    # Start queues if enabled
    if io_queue:
        await io_queue.start()
        logger.info("IO queue started")
    else:
        logger.info("IO queue disabled")

    if job_queue:
        await job_queue.start()
        logger.info(f"Job queue started with {num_workers} workers")
    else:
        logger.info("Job queue disabled")

    try:
        # Make resources available to tools via AppContext
        yield app_context
    finally:
        # Shutdown: cleanup resources (reverse order)
        logger.info("Shutting down MCP server...")

        # Stop job queue if enabled (don't wait for completion on shutdown)
        if job_queue:
            await job_queue.stop(wait_for_completion=False)
            logger.info("Job queue stopped")

        # Stop IO queue if enabled
        if io_queue:
            await io_queue.stop()
            logger.info("IO queue stopped")

        # No other explicit cleanup required because:
        # - WorkflowRegistry: In-memory only, no persistent state
        # - ExecutorRegistry: Stateless executor instances
        # - No file handles, network connections, or external resources to close


# Initialize MCP server with lifespan management
# Following Python MCP naming convention: {service}_mcp
mcp = FastMCP("workflows_mcp", lifespan=app_lifespan)


# =============================================================================
# Server Entry Point
# =============================================================================


def main() -> None:
    """Entry point for running the MCP server.

    This function is called when the server is run directly via:
    - uv run python -m workflows_mcp
    - python -m workflows_mcp
    - uv run workflows-mcp (if entry point is configured in pyproject.toml)

    Defaults to stdio transport for MCP protocol communication.
    """
    # Get log level from environment variable, default to INFO
    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    log_level_str = os.getenv("WORKFLOWS_LOG_LEVEL", "INFO").upper()

    # Validate log level and provide feedback
    if log_level_str not in valid_log_levels:
        print(
            f"Warning: Invalid WORKFLOWS_LOG_LEVEL '{log_level_str}'. "
            f"Valid levels: {', '.join(sorted(valid_log_levels))}. "
            "Using INFO.",
            file=sys.stderr,
        )
        log_level_str = "INFO"

    log_level = getattr(logging, log_level_str)

    # Configure logging to stderr (MCP requirement)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    logger.info("Starting MCP server (press Ctrl+C to stop)...")

    try:
        # Run the MCP server with stdio transport
        # anyio.run() (used internally by mcp.run()) handles SIGINT gracefully
        # and raises KeyboardInterrupt for clean shutdown
        mcp.run()
    except KeyboardInterrupt:
        # Graceful shutdown via Ctrl+C (SIGINT)
        # anyio ensures proper cleanup of async resources and lifespan context
        logger.info("Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        # Unexpected errors
        logger.exception(f"Server error: {e}")
        sys.exit(1)

    logger.info("Server shutdown complete")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Server infrastructure
    "mcp",
    "main",
    "AppContext",
    "AppContextType",
    # Workflow loading (exposed for testing)
    "load_workflows",
]
