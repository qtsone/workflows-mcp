import asyncio
from workflows_mcp.engine.resolver.unified_resolver import UnifiedVariableResolver


async def main():
    context = {
        "blocks": {
            "propose": {
                "metadata": {"succeeded": False, "failed": True},
                "outputs": {"response": {}, "success": False, "metadata": {}},
            }
        }
    }
    r = UnifiedVariableResolver(context)
    val = await r.resolve_async("{{blocks.propose.outputs.success}}")
    print(f"Type: {type(val)}, Value: {val}")

    val2 = await r.resolve_async("{{blocks.missing.outputs.success}}")
    print(f"Missing Type: {type(val2)}, Value: {val2}")

    from workflows_mcp.engine.executors_core import coerce_value_type

    try:
        c1 = coerce_value_type(val2, "bool")
        print(f"Coerced missing: Type: {type(c1)}, Value: {c1}")
    except ValueError as e:
        print(f"Coercion error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
