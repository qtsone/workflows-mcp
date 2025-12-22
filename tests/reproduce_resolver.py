import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from workflows_mcp.engine.resolver.unified_resolver import UnifiedVariableResolver
from workflows_mcp.engine.resolver.classifier import ExpressionClassifier, ExpressionType


async def run_test():
    print("reproduce_resolver.py: Starting test...")

    # Scenario: inputs.features is a dict
    features_dict = {
        "dual_track_enabled": True,
        "salience_enabled": False,
        "heuristics_enabled": True,
    }

    context = {"inputs": {"features": features_dict, "other_val": "some_string"}}

    resolver = UnifiedVariableResolver(context)

    # Test 1: Resolve {{ inputs.features }}
    # Expected: The dict object itself (preserved type)
    expression = "{{ inputs.features }}"
    print(f"\nTest 1: Resolving '{expression}'")

    classifier = ExpressionClassifier()
    expr_type = classifier.classify(expression)
    print(f"Classification: {expr_type}")

    try:
        result = await resolver.resolve_async(expression)
        print(f"Result type: {type(result)}")
        print(f"Result value: {result}")

        if isinstance(result, dict) and result == features_dict:
            print("SUCCESS: Resolved to dict correctly.")
        else:
            print("FAILURE: Did not resolve to dict correctly.")
            print(f"Expected: {features_dict}")

    except Exception as e:
        print(f"ERROR: {e}")

    # Test 2: Resolve {{ inputs }} (nested dict)
    expression2 = "{{ inputs }}"
    print(f"\nTest 2: Resolving '{expression2}'")
    try:
        result2 = await resolver.resolve_async(expression2)
        print(f"Result type: {type(result2)}")
        if isinstance(result2, dict) and result2.get("features") == features_dict:
            print("SUCCESS: Resolved parent dict correctly.")
        else:
            print("FAILURE: Did not resolve parent dict correctly.")
    except Exception as e:
        print(f"ERROR: {e}")

    # Test 3: Pass a dict that contains the expression (simulating Block Definition Inputs)
    # The WorkflowRunner iterates over inputs and resolves them.
    print(f"\nTest 3: Simulating block definition inputs")
    block_inputs = {
        "features": "{{ inputs.features }}",
        "nested": {"prop": "{{ inputs.features.dual_track_enabled }}"},
    }

    try:
        resolved_block_inputs = await resolver.resolve_async(block_inputs)
        print(f"Result: {resolved_block_inputs}")

        if resolved_block_inputs["features"] == features_dict:
            print("SUCCESS: Block input 'features' resolved to dict.")
        else:
            print("FAILURE: Block input 'features' failed.")

        if resolved_block_inputs["nested"]["prop"] is True:
            print("SUCCESS: Nested prop resolved to boolean.")
        else:
            print("FAILURE: Nested prop failed.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(run_test())
