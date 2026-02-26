import asyncio, sys, os, json
from workflows_mcp.engine.workflow_runner import WorkflowRunner
from workflows_mcp.engine.registry import WorkflowRegistry
from workflows_mcp.engine.schema import WorkflowSchema
from workflows_mcp.engine.execution_context import ExecutionContext
from workflows_mcp.engine.executor_base import create_default_registry
from workflows_mcp.engine.llm_config import LLMConfigLoader


async def main():
    registry = WorkflowRegistry()
    registry.load_from_directory("../templates")
    runner = WorkflowRunner()

    workflow = registry.get("knowledge-document-ingest")
    if not workflow:
        print("Workflow 'knowledge-document-ingest' not found!")
        return

    context = ExecutionContext(
        workflow_registry=registry,
        executor_registry=create_default_registry(),
        llm_config_loader=LLMConfigLoader(),
        io_queue=None,
    )

    result = await runner.execute(
        workflow=workflow,
        runtime_inputs={
            "org_id": "org_1",
            "source_id": "test_src",
            "item_id": "test_item",
            "source_type": "text",
            "content": "Python is a great programming language. It is dynamically typed.",
        },
        context=context,
    )

    print("STATUS:", result.status)
    print("ERROR:", result.error)

    exec_state = result.execution
    print("BLOCKS:", list(exec_state.blocks.keys()))

    analyze_block = exec_state.blocks.get("analyze")
    if analyze_block:
        print(f"ANALYZE METADATA: {analyze_block.metadata.model_dump()}")
        print(f"ANALYZE OUTPUTS: {analyze_block.outputs}")
        print(
            "ANALYZE CHILD BLOCKS:",
            list(analyze_block.blocks.keys()) if hasattr(analyze_block, "blocks") else "NA",
        )

        propose_block = analyze_block.blocks.get("propose")
        if propose_block:
            print(f"PROPOSE METADATA: {propose_block.metadata.model_dump()}")
            print(f"PROPOSE OUTPUTS: {propose_block.outputs}")

        load_data_block = analyze_block.blocks.get("load_data")
        if load_data_block:
            print(f"LOAD_DATA METADATA: {load_data_block.metadata.model_dump()}")
            print(f"LOAD_DATA OUTPUTS: {load_data_block.outputs}")


if __name__ == "__main__":
    asyncio.run(main())
