"""
DAG dependency resolution for workflow execution order.

ARCHITECTURAL DECISION: This module is intentionally SYNCHRONOUS.

Rationale:
- Pure in-memory graph algorithms (Kahn's topological sort, wave computation)
- No I/O operations, external API calls, or database queries
- Fast execution: O(V + E) complexity, microseconds for typical workflows
- Used as planning step before async block execution

The async workflow executor calls these synchronous methods during the planning
phase, then executes blocks asynchronously based on the computed execution plan.

Design Pattern:
    1. DAGResolver.topological_sort() → synchronous planning
    2. WorkflowExecutor.execute() → async execution of planned blocks
"""

from collections import deque

from .load_result import LoadResult


class DAGResolver:
    """Resolves execution order for workflow blocks based on dependencies."""

    def __init__(self, blocks: list[str], dependencies: dict[str, list[str]]):
        """
        Initialize DAG resolver.

        Args:
            blocks: List of block names
            dependencies: Dict mapping block name to list of blocks it depends on
        """
        self.blocks = blocks
        self.dependencies = dependencies

    def topological_sort(self) -> LoadResult[list[str]]:
        """
        Perform topological sort to determine execution order.

        Returns:
            Result containing ordered list of block names or error if cyclic dependency
        """
        # Build adjacency list and in-degree count
        in_degree = {block: 0 for block in self.blocks}
        adj_list: dict[str, list[str]] = {block: [] for block in self.blocks}

        for block, deps in self.dependencies.items():
            if block not in self.blocks:
                return LoadResult.failure(f"Block '{block}' in dependencies but not in blocks list")

            for dep in deps:
                if dep not in self.blocks:
                    return LoadResult.failure(
                        f"Dependency '{dep}' for block '{block}' not found in blocks list"
                    )

                adj_list[dep].append(block)
                in_degree[block] += 1

        # Kahn's algorithm for topological sort
        queue = deque([block for block in self.blocks if in_degree[block] == 0])
        result = []

        while queue:
            # Process blocks with no dependencies
            current = queue.popleft()
            result.append(current)

            # Reduce in-degree for dependent blocks
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(self.blocks):
            return LoadResult.failure("Cyclic dependency detected in workflow")

        return LoadResult.success(result)

    def get_execution_waves(self) -> LoadResult[list[list[str]]]:
        """
        Group blocks into waves that can be executed in parallel.

        Returns:
            Result containing list of waves (each wave is a list of blocks that can run in parallel)
        """
        # Build reverse dependency graph
        reverse_deps: dict[str, set[str]] = {block: set() for block in self.blocks}
        for block, deps in self.dependencies.items():
            if block not in reverse_deps:
                reverse_deps[block] = set()
            for dep in deps:
                if dep not in self.blocks:
                    return LoadResult.failure(f"Dependency '{dep}' not found")
                reverse_deps[block].add(dep)

        waves = []
        completed = set()
        remaining = set(self.blocks)

        while remaining:
            # Find blocks whose dependencies are all completed
            ready = [
                block
                for block in remaining
                if all(dep in completed for dep in reverse_deps.get(block, []))
            ]

            if not ready:
                return LoadResult.failure("Cyclic dependency detected or isolated blocks")

            waves.append(ready)
            completed.update(ready)
            remaining.difference_update(ready)

        return LoadResult.success(waves)
