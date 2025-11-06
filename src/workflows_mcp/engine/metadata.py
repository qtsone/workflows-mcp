"""Universal node metadata for ADR-009 fractal for_each design.

This module implements the metadata namespace that unifies regular blocks
and for_each iterations under a single, recursive structure.

Key Concepts:
- Regular blocks are for_each with count=1 (fractal design)
- Every node (block or iteration) has identical metadata structure
- Supports recursive nesting (iterations can contain iterations)
- Executor-specific fields via Pydantic extra='allow'
- Computed properties for ADR-007 compliance (succeeded, failed, skipped, status)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """
    Universal metadata structure for blocks and iterations (ADR-009).

    This class implements the fractal design where:
    - Regular blocks = for_each with count=1, iterations={}, mode=None
    - For_each blocks have count>1, iterations={...}, mode="parallel"|"sequential"
    - Iterations are recursive - can contain their own iterations

    Core Fields (All Nodes):
        count: Number of sub-iterations (1 for leaf nodes)
        count_failed: Count of failed sub-iterations
        count_skipped: Count of skipped sub-iterations
        started_at: ISO 8601 timestamp when execution started
        duration_ms: Execution duration in milliseconds
        type: Executor type (e.g., "Shell", "LLMCall", "Workflow")
        id: Node identifier (block ID or iteration key)
        wave: Wave number in parallel execution (0-indexed)
        execution_order: Execution order within wave (0-indexed)
        index: Position in parent (0 for root blocks)
        value: Iteration data from parent (empty {} for root blocks)
        depth: Nesting depth (0 for root, 1+ for iterations)
        message: Informational message (error details, skip reason, etc.)
        mode: Execution mode (None for leaf, "parallel"/"sequential" for for_each)
        iterations: Child iterations (empty {} for leaf, populated for for_each)

    Executor-Specific Fields (Examples):
        exit_code: int (Shell)
        stdout_lines: int (Shell)
        tokens_used: int (LLMCall)
        model: str (LLMCall)
        file_size_bytes: int (ReadFile)
        status_code: int (HttpCall)

    Computed Properties (ADR-007):
        succeeded: bool - True if all sub-iterations succeeded
        failed: bool - True if any sub-iteration failed
        skipped: bool - True if all sub-iterations skipped
        status: str - Execution status ("completed", "failed", "skipped")

    Usage:
        # Leaf node (regular block)
        meta = Metadata.create_leaf_success(
            type="Shell",
            id="build",
            duration_ms=1234,
            start_time="2025-11-04T10:30:00Z",
            executor_fields={"exit_code": 0, "stdout_lines": 42}
        )
        # meta.count == 1, meta.succeeded == True, meta.mode == None

        # For_each parent node
        meta = Metadata.create_for_each_parent(
            type="LLMCall",
            id="analyze_files",
            iterations={"file1": {...}, "file2": {...}},
            child_metas=[child1_meta, child2_meta],
            mode="parallel"
        )
        # meta.count == 2, meta.mode == "parallel", meta.iterations != {}
    """

    # === Core Fields (All Nodes) ===
    count: int = Field(
        default=1,
        ge=1,
        description="Number of sub-iterations (1 for leaf nodes, N for for_each)",
    )
    count_failed: int = Field(
        default=0,
        ge=0,
        description="Count of failed sub-iterations",
    )
    count_skipped: int = Field(
        default=0,
        ge=0,
        description="Count of skipped sub-iterations",
    )

    # Timing
    started_at: str = Field(
        description="ISO 8601 timestamp when execution started",
    )
    duration_ms: int = Field(
        ge=0,
        description="Execution duration in milliseconds",
    )

    # Type & Identity
    type: str = Field(
        min_length=1,
        description="Executor type (e.g., 'Shell', 'LLMCall', 'Workflow')",
    )
    id: str = Field(
        min_length=1,
        description="Node identifier (block ID or iteration key)",
    )

    # Position & Context
    wave: int = Field(
        default=0,
        ge=0,
        description="Wave number in parallel execution (0-indexed)",
    )
    execution_order: int = Field(
        default=0,
        ge=0,
        description="Execution order within wave (0-indexed)",
    )
    index: int = Field(
        default=0,
        ge=0,
        description="Position in parent (0-based, 0 for root blocks)",
    )
    value: dict[str, Any] = Field(
        default_factory=dict,
        description="Iteration data from parent (empty {} for root blocks)",
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Nesting depth (0 for root, 1+ for iterations)",
    )

    # Message (failures, skips, pauses)
    message: str | None = Field(
        default=None,
        description="Informational message (error details, skip reason, etc.)",
    )

    # Outcome (operation result vs executor crash)
    outcome: Literal["success", "failure", "crash", "n/a"] = Field(
        default="success",
        description=(
            "Operation outcome - distinguishes operation failure from executor crash. "
            "'success': operation succeeded, "
            "'failure': operation failed (e.g., exit 1, test failure), "
            "'crash': executor crashed (exception), "
            "'n/a': never executed (skipped)"
        ),
    )

    # Recursion Support
    mode: Literal["parallel", "sequential"] | None = Field(
        default=None,
        description="Execution mode (None for leaf, 'parallel'/'sequential' for for_each)",
    )
    iterations: dict[str, Any] = Field(
        default_factory=dict,
        description="Child iterations (empty {} for leaf, populated for for_each)",
    )

    # Executor-specific fields accepted via extra='allow'
    model_config = {"extra": "allow"}

    # === Computed Properties (ADR-007) ===
    @property
    def succeeded(self) -> bool:
        """
        True if all sub-iterations succeeded (no failures or skips).

        For leaf nodes (count=1): succeeded if count_failed==0 and count_skipped==0
        For for_each nodes (count>1): succeeded if all children succeeded
        """
        return self.count_failed == 0 and self.count_skipped == 0 and self.count > 0

    @property
    def failed(self) -> bool:
        """
        True if any sub-iteration failed.

        For leaf nodes (count=1): failed if count_failed==1
        For for_each nodes (count>1): failed if any child failed
        """
        return self.count_failed > 0

    @property
    def skipped(self) -> bool:
        """
        True if all sub-iterations were skipped.

        For leaf nodes (count=1): skipped if count_skipped==1
        For for_each nodes (count>1): skipped if all children skipped
        """
        return self.count_skipped == self.count

    @property
    def status(self) -> str:
        """
        Execution status derived from counts (ADR-007 Tier 2).

        Returns:
            "completed" - All succeeded (count_failed==0, count_skipped==0)
            "failed" - Any failed (count_failed>0)
            "skipped" - All skipped (count_skipped==count)

        Note: For partial failures with continue_on_error=true,
              status is "completed" but succeeded is False.
        """
        if self.count_skipped == self.count:
            return "skipped"
        elif self.count_failed > 0:
            return "failed"
        else:
            return "completed"

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """
        Override model_dump to include computed properties.

        ADR-009 computed status fields (succeeded, failed, skipped, status)
        must be available in the dict for variable resolution.
        """
        data = super().model_dump(**kwargs)
        # Add computed properties to the dict (ADR-009 compliance)
        data["succeeded"] = self.succeeded
        data["failed"] = self.failed
        data["skipped"] = self.skipped
        data["status"] = self.status
        return data

    def requires_dependent_skip(self, required: bool) -> bool:
        """
        Determine if dependent blocks should be skipped based on this block's status.

        Used for dependency handling in workflow execution. If a dependency fails
        or is skipped, dependent blocks may need to skip execution.

        Args:
            required: Whether this is a required dependency

        Returns:
            True if dependent blocks should skip execution, False otherwise

        Logic:
            - Required dependencies: Skip dependents if this block failed or was skipped
            - Optional dependencies: Never skip dependents (they can proceed even if this fails)
        """
        if not required:
            # Optional dependencies never cause dependents to skip
            return False

        # Required dependencies: skip dependents if this block failed or was skipped
        return self.failed or self.skipped

    # === Factory Methods ===

    @classmethod
    def create_leaf_success(
        cls,
        type: str,
        id: str,
        duration_ms: int,
        started_at: str,
        wave: int = 0,
        execution_order: int = 0,
        index: int = 0,
        value: dict[str, Any] | None = None,
        depth: int = 0,
        **executor_fields: Any,
    ) -> Metadata:
        """
        Create metadata for a successfully executed leaf node.

        Args:
            type: Executor type (e.g., "Shell", "LLMCall")
            id: Block ID or iteration key
            duration_ms: Execution duration in milliseconds
            started_at: ISO 8601 timestamp when execution started
            wave: Wave number in parallel execution (default: 0)
            execution_order: Execution order within wave (default: 0)
            index: Position in parent (default: 0)
            value: Iteration data from parent (default: {})
            depth: Nesting depth (default: 0)
            **executor_fields: Executor-specific fields (e.g., exit_code=0)

        Returns:
            Metadata with count=1, count_failed=0, count_skipped=0, mode=None

        Example:
            meta = Metadata.create_leaf_success(
                type="Shell",
                id="build",
                duration_ms=1234,
                started_at="2025-11-04T10:30:00Z",
                wave=0,
                execution_order=0,
                exit_code=0,
                stdout_lines=42
            )
        """
        return cls(
            count=1,
            count_failed=0,
            count_skipped=0,
            started_at=started_at,
            duration_ms=duration_ms,
            type=type,
            id=id,
            wave=wave,
            execution_order=execution_order,
            index=index,
            value=value or {},
            depth=depth,
            outcome="success",  # Operation succeeded
            mode=None,  # Leaf node
            iterations={},  # No children
            **executor_fields,
        )

    @classmethod
    def create_leaf_failure(
        cls,
        type: str,
        id: str,
        duration_ms: int,
        started_at: str,
        wave: int = 0,
        execution_order: int = 0,
        index: int = 0,
        value: dict[str, Any] | None = None,
        depth: int = 0,
        outcome: Literal["failure", "crash"] = "failure",
        **executor_fields: Any,
    ) -> Metadata:
        """
        Create metadata for a failed leaf node.

        Args:
            type: Executor type
            id: Block ID or iteration key
            duration_ms: Execution duration in milliseconds
            started_at: ISO 8601 timestamp when execution started
            wave: Wave number in parallel execution (default: 0)
            execution_order: Execution order within wave (default: 0)
            index: Position in parent (default: 0)
            value: Iteration data from parent (default: {})
            depth: Nesting depth (default: 0)
            outcome: Failure type - "failure" (operation failed) or "crash" (executor error)
            **executor_fields: Executor-specific fields (e.g., exit_code=1, message="...")

        Returns:
            Metadata with count=1, count_failed=1, count_skipped=0, mode=None

        Example:
            # Operation failure (exit code 1)
            meta = Metadata.create_leaf_failure(
                type="Shell",
                id="test",
                duration_ms=567,
                started_at="2025-11-04T10:30:00Z",
                outcome="failure",
                exit_code=1,
                message="Command failed"
            )

            # Executor crash (exception)
            meta = Metadata.create_leaf_failure(
                type="Shell",
                id="test",
                duration_ms=123,
                started_at="2025-11-04T10:30:00Z",
                outcome="crash",
                message="KeyError: 'missing_key'"
            )
        """
        return cls(
            count=1,
            count_failed=1,  # This leaf failed
            count_skipped=0,
            started_at=started_at,
            duration_ms=duration_ms,
            type=type,
            id=id,
            wave=wave,
            execution_order=execution_order,
            index=index,
            value=value or {},
            depth=depth,
            outcome=outcome,  # "failure" or "crash"
            mode=None,  # Leaf node
            iterations={},  # No children
            **executor_fields,
        )

    @classmethod
    def create_leaf_skipped(
        cls,
        type: str,
        id: str,
        started_at: str,
        wave: int = 0,
        execution_order: int = 0,
        index: int = 0,
        value: dict[str, Any] | None = None,
        depth: int = 0,
        **executor_fields: Any,
    ) -> Metadata:
        """
        Create metadata for a skipped leaf node.

        Args:
            type: Executor type
            id: Block ID or iteration key
            started_at: ISO 8601 timestamp when skip decision was made
            wave: Wave number in parallel execution (default: 0)
            execution_order: Execution order within wave (default: 0)
            index: Position in parent (default: 0)
            value: Iteration data from parent (default: {})
            depth: Nesting depth (default: 0)
            **executor_fields: Executor-specific fields (e.g., message="Condition false")

        Returns:
            Metadata with count=1, count_failed=0, count_skipped=1, mode=None, duration_ms=0

        Example:
            meta = Metadata.create_leaf_skipped(
                type="Shell",
                id="deploy",
                started_at="2025-11-04T10:30:00Z",
                wave=0,
                execution_order=0,
                message="Condition false: tests failed"
            )
        """
        return cls(
            count=1,
            count_failed=0,
            count_skipped=1,  # This leaf was skipped
            started_at=started_at,
            duration_ms=0,  # No execution time for skipped
            type=type,
            id=id,
            wave=wave,
            execution_order=execution_order,
            index=index,
            value=value or {},
            depth=depth,
            outcome="n/a",  # Never executed
            mode=None,  # Leaf node
            iterations={},  # No children
            **executor_fields,
        )

    @classmethod
    def create_leaf_paused(
        cls,
        type: str,
        id: str,
        duration_ms: int,
        started_at: str,
        wave: int = 0,
        execution_order: int = 0,
        index: int = 0,
        value: dict[str, Any] | None = None,
        depth: int = 0,
        **executor_fields: Any,
    ) -> Metadata:
        """
        Create metadata for a paused leaf node (interactive blocks).

        Args:
            type: Executor type
            id: Block ID or iteration key
            duration_ms: Execution duration so far (partial)
            started_at: ISO 8601 timestamp when execution started
            wave: Wave number in parallel execution (default: 0)
            execution_order: Execution order within wave (default: 0)
            index: Position in parent (default: 0)
            value: Iteration data from parent (default: {})
            depth: Nesting depth (default: 0)
            **executor_fields: Executor-specific fields (e.g., message="Prompt text")

        Returns:
            Metadata with count=1, count_failed=0, count_skipped=0, mode=None

        Note:
            Paused state is represented in checkpoint data, not directly in metadata.
            The node shows succeeded=True (execution completed to pause point).
            Use executor-specific fields like 'message' to store pause details.

        Example:
            meta = Metadata.create_leaf_paused(
                type="Prompt",
                id="get_input",
                duration_ms=123,
                started_at="2025-11-04T10:30:00Z",
                wave=0,
                execution_order=0,
                message="Please provide API key"
            )
        """
        return cls(
            count=1,
            count_failed=0,
            count_skipped=0,
            started_at=started_at,
            duration_ms=duration_ms,
            type=type,
            id=id,
            wave=wave,
            execution_order=execution_order,
            index=index,
            value=value or {},
            depth=depth,
            outcome="success",  # Paused successfully (completed to pause point)
            mode=None,  # Leaf node
            iterations={},  # No children
            **executor_fields,
        )

    @classmethod
    def create_for_each_parent(
        cls,
        type: str,
        id: str,
        iterations: dict[str, Any],
        child_metas: list[Metadata],
        mode: Literal["parallel", "sequential"],
        depth: int = 0,
        index: int = 0,
        value: dict[str, Any] | None = None,
    ) -> Metadata:
        """
        Create metadata for a for_each parent node by aggregating child results.

        Args:
            type: Executor type
            id: Block ID
            iterations: Iteration data dict (keys → child iteration keys)
            child_metas: List of child Metadata objects
            mode: Execution mode ("parallel" or "sequential")
            depth: Nesting depth (default: 0)
            index: Position in parent (default: 0)
            value: Iteration data from parent (default: {})

        Returns:
            Metadata with aggregated counts from children

        Example:
            child1 = Metadata.create_leaf_success(...)
            child2 = Metadata.create_leaf_failure(...)
            parent = Metadata.create_for_each_parent(
                type="LLMCall",
                id="analyze_files",
                iterations={"file1": {...}, "file2": {...}},
                child_metas=[child1, child2],
                mode="parallel"
            )
            # parent.count == 2, parent.failed == 1
        """
        # Aggregate counts from children (fractal: consistent naming)
        count = len(child_metas)
        count_failed = sum(m.count_failed for m in child_metas)
        count_skipped = sum(m.count_skipped for m in child_metas)

        # Aggregate timing (total duration = max child duration for parallel,
        # sum for sequential, but we'll use max for simplicity)
        duration_ms = max((m.duration_ms for m in child_metas), default=0)
        started_at = min((m.started_at for m in child_metas), default="")

        # Aggregate wave and execution_order from first child (representative)
        wave = child_metas[0].wave if child_metas else 0
        execution_order = child_metas[0].execution_order if child_metas else 0

        # Compute aggregate outcome from children
        # - All skipped → "n/a"
        # - Any crash → "crash"
        # - Any failure (no crash) → "failure"
        # - All success → "success"
        outcome: Literal["success", "failure", "crash", "n/a"]
        if count_skipped == count:
            outcome = "n/a"  # All children skipped
        elif any(m.outcome == "crash" for m in child_metas):
            outcome = "crash"  # At least one child crashed
        elif count_failed > 0:
            outcome = "failure"  # At least one child failed (but no crashes)
        else:
            outcome = "success"  # All children succeeded

        return cls(
            count=count,
            count_failed=count_failed,
            count_skipped=count_skipped,
            started_at=started_at,
            duration_ms=duration_ms,
            type=type,
            id=id,
            wave=wave,
            execution_order=execution_order,
            index=index,
            value=value or {},
            depth=depth,
            outcome=outcome,  # Aggregated outcome
            mode=mode,  # for_each mode
            iterations=iterations,  # Child iteration data
        )
