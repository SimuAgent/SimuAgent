#!/usr/bin/env python3
"""
Test script for the simplified plan tool.

This script demonstrates how to use the plan tool with the simplified interface
that supports three interaction patterns:
- <tool>{"name": "plan", "args": {"plan_list": [...]}}</tool> - create/update plan
- <answer>Continue</answer> - execute next step  
- <answer>Finish</answer> - end the plan
"""

import sys
import os

# Add the parent directory to the path so we can import from tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.plan import plan, plan_continue, plan_finish, plan_status, set_plan_llm_executor


def mock_llm_executor(llm_path, prompt):
    """
    Mock LLM executor for testing purposes.
    
    Args:
        llm_path: Path to LLM model (ignored in mock)
        prompt: The prompt to execute
        
    Returns:
        Mock response based on the prompt content
    """
    if "analyze" in prompt.lower():
        return "<result>\nAnalysis completed: The task requires careful consideration of requirements, constraints, and available resources. Key factors identified include scope definition, timeline estimation, and resource allocation.\n</result>"
    elif "search" in prompt.lower():
        return "<result>\nSearch completed: Found relevant documentation and examples. Key resources include API documentation, code samples, and best practice guides.\n</result>"
    elif "design" in prompt.lower():
        return "<result>\nDesign phase completed: Created architecture diagrams, defined interfaces, and established design patterns. The modular approach will ensure maintainability and scalability.\n</result>"
    elif "implement" in prompt.lower():
        return "<result>\nImplementation completed: Core functionality has been developed following the established design patterns. Code is well-documented and follows coding standards.\n</result>"
    elif "test" in prompt.lower():
        return "<result>\nTesting completed: All unit tests pass, integration tests successful, and manual testing confirms expected behavior. Performance meets requirements.\n</result>"
    else:
        return f"<result>\nTask completed: {prompt}\n\nMock execution result for demonstration purposes.\n</result>"


def test_simplified_plan_tool():
    """Test the simplified plan tool functionality."""
    
    print("=== Simplified Plan Tool Test Demo ===\n")
    
    # Set up the mock LLM executor
    set_plan_llm_executor(mock_llm_executor, "/path/to/default/llm")
    
    # Test 1: Create a new plan
    print("1. Creating a new plan...")
    plan_steps = [
        "Analyze the project requirements and constraints",
        "Search for relevant documentation and examples", 
        "Design the system architecture",
        "Implement the core functionality",
        "Test and validate the implementation"
    ]
    
    result = plan(plan_steps)
    print(result)
    print()
    
    # Test 2: Check plan status
    print("2. Checking plan status...")
    status_result = plan_status()
    print(status_result)
    print()
    
    # Test 3: Execute first step (Continue)
    print("3. Executing first step (Continue)...")
    exec_result = plan_continue()
    print(exec_result)
    print()
    
    # Test 4: Execute second step (Continue)
    print("4. Executing second step (Continue)...")
    exec_result = plan_continue()
    print(exec_result)
    print()
    
    # Test 5: Update the plan (revise remaining steps)
    print("5. Updating the plan with revised remaining steps...")
    # Note: First 2 steps are already completed, so we only specify remaining steps
    updated_remaining_steps = [
        "Create detailed technical specifications",
        "Implement core modules with proper error handling", 
        "Conduct comprehensive testing and performance optimization",
        "Prepare documentation and deployment guide"
    ]
    
    update_result = plan(updated_remaining_steps)
    print(update_result)
    print()
    
    # Test 6: Check updated status
    print("6. Checking updated plan status...")
    status_result = plan_status()
    print(status_result)
    print()
    
    # Test 7: Execute next step after update
    print("7. Executing next step after update...")
    exec_result = plan_continue()
    print(exec_result)
    print()
    
    # Test 8: Finish the plan
    print("8. Finishing the plan...")
    finish_result = plan_finish()
    print(finish_result)
    print()
    
    print("=== Test completed successfully! ===")


def test_interaction_patterns():
    """Demonstrate the three interaction patterns."""
    
    print("\n=== Plan Tool Interaction Patterns ===\n")
    
    print("Pattern 1: Create/Update Plan")
    print('LLM outputs: <tool>{"name": "plan", "args": {"plan_list": ["Step 1", "Step 2"]}}</tool>')
    print("Result: Creates new plan or updates existing plan with remaining steps")
    print()
    
    print("Pattern 2: Execute Next Step")
    print("LLM outputs: <answer>Continue</answer>")
    print("Result: Executes the next step in the current plan")
    print()
    
    print("Pattern 3: Finish Plan")
    print("LLM outputs: <answer>Finish</answer>")
    print("Result: Completes and cleans up the current plan")
    print()


def test_plan_revision_workflow():
    """Test the plan revision workflow where completed steps are preserved."""
    
    print("\n=== Plan Revision Workflow Test ===\n")
    
    # Set up the mock LLM executor
    set_plan_llm_executor(mock_llm_executor, "/path/to/default/llm")
    
    # Create initial plan
    print("1. Create initial plan...")
    initial_plan = [
        "Research the problem domain",
        "Design preliminary solution",
        "Implement basic prototype"
    ]
    result = plan(initial_plan)
    print(result)
    print()
    
    # Execute first step
    print("2. Execute first step...")
    result = plan_continue()
    print(result)
    print()
    
    # Revise plan - referencing completed work
    print("3. Revise plan after first step completion...")
    print("(Completed: Research the problem domain <result>...research completed...</result>)")
    revised_remaining = [
        "Analyze competitive solutions and best practices",
        "Design comprehensive system architecture", 
        "Implement full solution with all features",
        "Perform thorough testing and optimization"
    ]
    result = plan(revised_remaining)
    print(result)
    print()
    
    # Check status to see preserved completed work
    print("4. Check status to verify completed work is preserved...")
    result = plan_status()
    print(result)
    print()


def test_api_usage_examples():
    """Show API usage examples."""
    
    print("\n=== API Usage Examples ===\n")
    
    print("Simplified Function Signature:")
    print("plan(plan_list: List[str]) -> str")
    print()
    
    print("Example Usage:")
    print("# Create a plan")
    print('plan(["Task 1", "Task 2", "Task 3"])')
    print()
    
    print("# Update plan with remaining steps (preserves completed work)")
    print('plan(["Revised task 2", "New task 3", "Additional task 4"])')
    print()
    
    print("# Execute next step")
    print('plan_continue()')
    print()
    
    print("# Check status")
    print('plan_status()')
    print()
    
    print("# Finish plan")
    print('plan_finish()')
    print()
    
    print("Tool Call Format for LLM:")
    print('{"name": "plan", "args": {"plan_list": ["Task 1", "Task 2"]}}')
    print()


if __name__ == "__main__":
    test_simplified_plan_tool()
    test_interaction_patterns()
    test_plan_revision_workflow()
    test_api_usage_examples() 