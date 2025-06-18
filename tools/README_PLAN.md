# Plan Tool Documentation

The Plan tool enables LLMs to create, manage, and execute sequential plans with a simplified interface using three fixed interaction forms.

## Overview

The Plan tool allows an LLM to:
- Generate structured plans represented as lists of tasks
- Execute sub-tasks sequentially using configurable LLM execution
- Revise and update plans during execution (preserving completed work)
- Control execution through three simple interaction patterns

## Key Features

### 1. Simplified Interface
- Single parameter: `plan_list` (List[str])
- No complex action parameters or plan IDs
- Automatic state management

### 2. Three Interaction Patterns
- `<tool>{"name": "plan", "args": {"plan_list": [...]}}</tool>` - create/update plan
- `<answer>Continue</answer>` - execute next step  
- `<answer>Finish</answer>` - end the plan

### 3. Smart Plan Management
- Automatic plan creation vs. update detection
- Preserves completed steps when revising
- No reruns of completed work
- Progress tracking with visual indicators

### 4. Sequential Execution
- Execute sub-tasks one by one in defined order
- Each sub-task runs as an external tool invocation
- Results are parsed from `<result>` or `<answer>` tags
- Conversations inside sub-tasks are not included in training content

## Usage

### Simplified Tool Call Format

The plan tool now has a minimal interface with only one parameter:

```json
{
  "name": "plan",
  "args": {
    "plan_list": ["Task 1", "Task 2", "Task 3"]
  }
}
```

### Three Control Patterns

#### 1. Create/Update Plan
```json
{
  "name": "plan", 
  "args": {
    "plan_list": ["Analyze requirements", "Design solution", "Implement code"]
  }
}
```

#### 2. Execute Next Step
```xml
<answer>Continue</answer>
```

#### 3. Finish Plan
```xml
<answer>Finish</answer>
```

## Example Workflow

### 1. Create a Plan
```json
{
  "name": "plan",
  "args": {
    "plan_list": [
      "Analyze system requirements",
      "Design architecture", 
      "Implement core features",
      "Test and validate"
    ]
  }
}
```
**Returns:** Plan creation confirmation with step listing

### 2. Execute Steps
```xml
<answer>Continue</answer>
```
**Returns:** Step result in `<result>` tags plus remaining step count

### 3. Revise Plan (if needed)
After step 1 completes with result `<result>Analysis shows need for additional security measures...</result>`:

```json
{
  "name": "plan",
  "args": {
    "plan_list": [
      "Research security best practices",
      "Design secure architecture with authentication", 
      "Implement core features with security",
      "Security testing and penetration testing",
      "Final validation and deployment"
    ]
  }
}
```
**Note:** Only specify remaining steps - completed work is automatically preserved

### 4. Continue or Finish
```xml
<answer>Continue</answer>
```
or
```xml
<answer>Finish</answer>
```

## Result Formats

### Plan Creation
```
Plan created with 4 steps:
1. Analyze system requirements
2. Design architecture
3. Implement core features
4. Test and validate
```

### Plan Update (Revision)
```
Plan updated. 1 steps completed, 5 remaining:
✓ 1. Analyze system requirements
○ 2. Research security best practices
○ 3. Design secure architecture with authentication
○ 4. Implement core features with security
○ 5. Security testing and penetration testing
○ 6. Final validation and deployment
```

### Step Execution
```
Step 2 completed:
Task: Research security best practices
<result>
Security research complete: Found industry standards for authentication,
authorization, encryption, and input validation. Key frameworks include...
</result>

Remaining steps: 4
```

### Plan Status
```
Plan Progress: 2/6
✓ 1. Analyze system requirements
✓ 2. Research security best practices
→ 3. Design secure architecture with authentication
○ 4. Implement core features with security
○ 5. Security testing and penetration testing
○ 6. Final validation and deployment
```

## API Functions

### Core Function
```python
plan(plan_list: List[str]) -> str
```
Creates new plan or updates existing plan with remaining steps.

### Helper Functions
```python
plan_continue() -> str      # Execute next step
plan_finish() -> str        # Finish current plan
plan_status() -> str        # Get plan status
```

## Integration with Environment

### Setting Up LLM Executor

```python
from tools.plan import set_plan_llm_executor

def my_llm_executor(llm_path, prompt):
    # Your LLM execution logic
    return result

set_plan_llm_executor(my_llm_executor, "/path/to/default/llm")
```

### Use in Environment

```python
from tools import plan
from envs import ToolEnvironment

env = ToolEnvironment(tools=[plan])
```

## Best Practices

### 1. Plan Revision Strategy
- **Reference completed work:** "Completed: Research phase `<result>findings show...</result>`"
- **Only list remaining steps:** Don't repeat completed tasks
- **Be specific:** Use insights from completed steps to refine remaining work

### 2. Granular Task Design
- Break complex tasks into manageable sub-tasks
- Each step should have a clear deliverable
- Avoid dependencies that can't be resolved sequentially

### 3. Result Utilization
- Use `<result>` content from previous steps to inform plan revisions
- Build on completed work rather than starting over
- Maintain context between related steps

### 4. Efficient Execution
- Use `Continue` to maintain momentum
- Use `Finish` when objectives are met or external input is needed
- Monitor progress with status checks when needed

## Configuration

### LLM Execution Mode
Whether subtasks run on the same LLM instance or load a new one is a human-side configuration, not a model argument. This is set through the `set_plan_llm_executor` function.

## Testing

Run the test script to see the simplified plan tool in action:

```bash
python tools/test_plan.py
```

This demonstrates:
- Plan creation and updating
- Step execution with `Continue` pattern
- Plan revision preserving completed work
- Finishing plans with `Finish` pattern

## Migration from Previous Version

### Old Interface (deprecated)
```python
plan(plan_list, action="create", plan_id="abc", use_same_llm=False)
plan([], action="execute", plan_id="abc")
plan(new_steps, action="update", plan_id="abc")
```

### New Interface
```python
plan(plan_list)              # Create or update
plan_continue()              # Execute next step
plan_finish()                # Finish plan
```

The new interface is much simpler and removes the need for managing plan IDs or specifying actions explicitly.

# Plan Tool Integration

## Overview

The plan tool has been successfully integrated into the SimuAgent training environment through `quick_start_plan.py`. This tool enables the LLM to create and manage sequential plans for complex tasks, providing better structured problem-solving capabilities.

## Integration Details

### Tools Included
- `plan`: Create or update a sequential plan with sub-tasks
- `search_blocks`: Search for Simulink blocks (existing tool)

Note: `plan_status` is automatically triggered after each step completion and doesn't need explicit invocation.

### Key Features

1. **Plan Creation**: LLMs can create structured plans using:
   ```json
   {"name": "plan", "args": {"plan_list": ["step1", "step2", "step3"]}}
   ```

2. **Plan Execution**: Special answer commands control plan flow:
   - `<answer>Continue</answer>` - Execute the next planned step
   - `<answer>Finish</answer>` - Complete and end the current plan

3. **Plan Monitoring**: Status is automatically shown after each step completion

### Environment Modifications

The `ToolEnvironment` has been enhanced to handle plan control answers:

1. **Answer Parsing**: The `_env_response` method now detects and handles "Continue" and "Finish" answers
2. **Completion Detection**: The `is_completed` method properly recognizes "Finish" answers
3. **Tool Integration**: Plan tools are registered alongside existing tools

### Training Configuration

Key changes in `quick_start_plan.py`:

- **Max Steps**: Increased from 3 to 10 to accommodate multi-step planning
- **Tools List**: Extended to include `[search_blocks, plan]` (plan_status is automatic)
- **System Prompt**: Enhanced with detailed plan tool usage instructions

### Usage Pattern

The recommended workflow for LLMs using the plan tool:

1. **Analysis**: Think about the task complexity
2. **Planning**: Create a plan if the task has multiple steps
3. **Execution**: Use "Continue" to execute steps one by one
4. **Monitoring**: Status is automatically displayed after each step
5. **Adaptation**: Update the plan if needed with new remaining steps
6. **Completion**: Use "Finish" when all steps are done

### System Prompt Template

The environment uses a comprehensive system prompt that explains:
- Available planning and search tools
- Tool usage syntax and patterns
- Step-by-step workflow recommendations
- Control flow with special answers

## Testing

The integration has been tested to ensure:
- ✅ All tools import correctly
- ✅ Plan creation and execution works
- ✅ Environment handles special answers
- ✅ Training configuration is valid

## Example Usage

```python
# Create plan
tools = [search_blocks, plan]
env = ToolEnvironment(
    dataset=dataset,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    max_steps=10,
    # ... other configs
)
```

This integration enables the SimuAgent to handle complex multi-step tasks with better structure and planning capabilities while maintaining compatibility with existing search functionality. 