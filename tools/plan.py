"""
Plan tool for managing and executing sequential plans.

This tool allows an LLM to create and manage plans that consist of sequential sub-tasks.
The LLM controls plan execution through three fixed forms:
- <answer>Continue</answer> to run the next step
- <tool>{"name": "plan", "args": {"plan_list": [...]}}</tool> to rewrite/reorder steps  
- <answer>Finish</answer> to end the task
"""

import json
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import uuid


@dataclass
class PlanStep:
    """Represents a single step in a plan."""
    id: str
    description: str
    task: str
    completed: bool = False
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PlanState:
    """Represents the current state of a plan execution."""
    plan_id: str
    steps: List[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    completed: bool = False
    created_at: float = field(default_factory=lambda: __import__('time').time())


class PlanManager:
    """Manages plan execution and state."""
    
    def __init__(self, llm_executor: Optional[Callable] = None, default_llm_path: Optional[str] = None):
        """
        Initialize the plan manager.
        
        Args:
            llm_executor: Function to execute LLM calls (llm, prompt) -> result
            default_llm_path: Path to default LLM model to load
        """
        self.current_plan: Optional[PlanState] = None
        self.llm_executor = llm_executor
        self.default_llm_path = default_llm_path
    
    def create_plan(self, plan_list: List[str]) -> str:
        """
        Create a new plan from a list of tasks.
        
        Args:
            plan_list: List of task descriptions
            
        Returns:
            Plan creation message
        """
        plan_id = str(uuid.uuid4())
        steps = [
            PlanStep(
                id=f"{plan_id}_step_{i}",
                description=f"Step {i+1}",
                task=task
            )
            for i, task in enumerate(plan_list)
        ]
        
        self.current_plan = PlanState(plan_id=plan_id, steps=steps)
        
        return f"Plan created with {len(plan_list)} steps:\n" + \
               "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_list))
    
    def update_plan(self, plan_list: List[str]) -> str:
        """
        Update the current plan with new task list.
        The LLM must reference completed steps and only list remaining work.
        
        Args:
            plan_list: New list of task descriptions (remaining steps only)
            
        Returns:
            Update confirmation message
        """
        if not self.current_plan:
            return self.create_plan(plan_list)
        
        current_step = self.current_plan.current_step_index
        
        # Preserve completed steps
        updated_steps = self.current_plan.steps[:current_step]
        
        # Add new steps from the current position
        for i, task in enumerate(plan_list, start=current_step):
            step_id = f"{self.current_plan.plan_id}_step_{i}"
            updated_steps.append(PlanStep(
                id=step_id,
                description=f"Step {i+1}",
                task=task
            ))
        
        self.current_plan.steps = updated_steps
        self.current_plan.completed = False
        
        completed_count = current_step
        remaining_count = len(plan_list)
        
        response = f"Plan updated. {completed_count} steps completed, {remaining_count} remaining:\n"
        
        # Show completed steps
        for i in range(current_step):
            response += f"✓ {i+1}. {self.current_plan.steps[i].task}\n"
        
        # Show remaining steps
        for i, task in enumerate(plan_list, start=current_step):
            response += f"○ {i+1}. {task}\n"
        
        return response
    
    def execute_next_step(self) -> str:
        """
        Execute the next step in the current plan.
        
        Returns:
            Execution result message
        """
        if not self.current_plan:
            return "No active plan. Create a plan first."
        
        if self.current_plan.completed or self.current_plan.current_step_index >= len(self.current_plan.steps):
            return "Plan already completed."
        
        current_step = self.current_plan.steps[self.current_plan.current_step_index]
        
        try:
            # Execute the step
            if self.llm_executor:
                result = self.llm_executor(self.default_llm_path, current_step.task)
            else:
                # Fallback: simulate execution for testing
                result = f"Simulated execution of: {current_step.task}"
            
            # Parse result to extract content from <result> or <answer> tags
            parsed_result = self._parse_result(result)
            
            # Update step
            current_step.completed = True
            current_step.result = parsed_result
            
            # Move to next step
            self.current_plan.current_step_index += 1
            
            # Get updated plan status automatically
            plan_status_info = self.get_plan_status()
            
            # Check if plan is completed
            if self.current_plan.current_step_index >= len(self.current_plan.steps):
                self.current_plan.completed = True
                completion_msg = "\n\n🎉 All steps completed!"
            else:
                completion_msg = ""
            
            return f"Step {self.current_plan.current_step_index} completed:\n" + \
                   f"Task: {current_step.task}\n" + \
                   f"<result>\n{parsed_result}\n</result>\n\n" + \
                   f"Plan Status:\n{plan_status_info}" + \
                   completion_msg
            
        except Exception as e:
            current_step.error = str(e)
            return f"Step execution failed:\n" + \
                   f"Task: {current_step.task}\n" + \
                   f"Error: {str(e)}"
    
    def _parse_result(self, raw_result: str) -> str:
        """
        Parse result from LLM output, extracting content from <result> or <answer> tags.
        
        Args:
            raw_result: Raw output from LLM
            
        Returns:
            Parsed result content
        """
        # Try to extract from <result> tags first
        result_match = re.search(r'<result>(.*?)</result>', raw_result, re.DOTALL)
        if result_match:
            return result_match.group(1).strip()
        
        # Try to extract from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', raw_result, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # If no tags found, return the raw result
        return raw_result.strip()
    
    def get_plan_status(self) -> str:
        """
        Get the current status of the active plan.
        
        Returns:
            Plan status summary
        """
        if not self.current_plan:
            return "No active plan."
        
        total_steps = len(self.current_plan.steps)
        current_step = self.current_plan.current_step_index
        
        response = f"Plan Progress: {current_step}/{total_steps}\n"
        
        for i, step in enumerate(self.current_plan.steps):
            if step.completed:
                marker = "✓"
            elif i == current_step and not self.current_plan.completed:
                marker = "→"
            else:
                marker = "○"
            response += f"{marker} {i+1}. {step.task}\n"
        
        return response
    
    def finish_plan(self) -> str:
        """
        Finish the current plan and clean up.
        
        Returns:
            Completion message
        """
        if not self.current_plan:
            return "No active plan to finish."
        
        completed_steps = self.current_plan.current_step_index
        total_steps = len(self.current_plan.steps)
        
        self.current_plan = None
        
        return f"Plan finished. Completed {completed_steps}/{total_steps} steps."


# Global plan manager instance
_plan_manager = PlanManager()


def plan(plan_list: List[str]) -> str:
    """
    Create or update a plan with sequential sub-tasks.
    
    sArgs:
        plan_list: List of task descriptions/steps to execute
    
    Returns:
        Result string with plan status or execution results
    
    Examples:
        For complex multi-step tasks, you can create and manage plans:
        - Create a plan: <tool>{"name": "plan", "args": {"plan_list": ["Parse system configuration", "Validate electrical connections", "Generate power flow analysis"]}}</tool>
        - Execute next step: <answer>Continue</answer>
        - Complete the task: <answer>Finish</answer>
    """
    global _plan_manager
    
    try:
        if not plan_list:
            return "Error: plan_list cannot be empty. Provide a list of tasks."
        
        # If no current plan exists, create a new one
        if not _plan_manager.current_plan:
            return _plan_manager.create_plan(plan_list)
        
        # If current plan exists, update it with remaining steps
        return _plan_manager.update_plan(plan_list)
    
    except Exception as e:
        return f"Error in plan tool: {str(e)}"


def plan_continue() -> str:
    """
    Execute the next step in the current plan.
    This function is called when the LLM outputs <answer>Continue</answer>
    
    Returns:
        Step execution result
    """
    global _plan_manager
    return _plan_manager.execute_next_step()


def plan_finish() -> str:
    """
    Finish the current plan.
    This function is called when the LLM outputs <answer>Finish</answer>
    
    Returns:
        Plan completion message
    """
    global _plan_manager
    return _plan_manager.finish_plan()


def plan_status() -> str:
    """
    Get the current plan status.
    
    Returns:
        Plan status summary
    """
    global _plan_manager
    return _plan_manager.get_plan_status()


def set_plan_llm_executor(executor: Callable, default_llm_path: Optional[str] = None):
    """
    Set the LLM executor function for the plan manager.
    
    Args:
        executor: Function that takes (llm_path, prompt) and returns result
        default_llm_path: Path to default LLM model
    """
    global _plan_manager
    _plan_manager.llm_executor = executor
    _plan_manager.default_llm_path = default_llm_path


# For backward compatibility and easy access
__all__ = ["plan", "plan_continue", "plan_finish", "plan_status", "set_plan_llm_executor", "PlanManager", "PlanStep", "PlanState"] 