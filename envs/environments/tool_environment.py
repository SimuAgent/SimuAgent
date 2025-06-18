"""
Tool-based environment implementation.

This module provides a clean, modular implementation of tool-based
environments using the new component architecture.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from datasets import Dataset
from vllm import LLM, SamplingParams

from ..core.interfaces import BaseEnvironment, ConversationState, EnvironmentState
from ..core.tool_manager import ToolManager
from ..core.execution_engine import ExecutionEngine
from ..core.state_manager import StateManager
from ..validation.system_validator import SystemValidator
from ..validation.config import ValidationConfig

from utils.xml_parser import XMLParser
from utils.data_utils import format_dataset
from rewards.tool_reward import ToolReward


class ToolEnvironment(BaseEnvironment):
    """
    Modern, clean implementation of a tool-based environment.
    
    This environment provides tool execution capabilities with proper
    separation of concerns and modular architecture.
    """
    
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tools: List[Callable] = None,
        system_prompt: str = "",
        few_shot: List[Dict[str, str]] = None,
        validation_config: Optional[ValidationConfig] = None,
        max_steps: int = 10,
        max_workers: int = 10,
        mask_env_response: bool = True,
        eot_id: int = 151643,
        message_end_id: int = 151645,
        reward: Any = None,
        sampling_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the tool environment.
        
        Args:
            dataset: Training dataset
            eval_dataset: Evaluation dataset  
            tools: List of callable tools
            system_prompt: System prompt template
            few_shot: Few-shot examples
            validation_config: Validation configuration
            max_steps: Maximum steps per conversation
            max_workers: Maximum worker threads
            mask_env_response: Whether to mask environment responses
            eot_id: End of text token ID
            message_end_id: Message end token ID
            reward: Reward function
            sampling_args: Additional sampling arguments
        """
        # Initialize base attributes directly (BaseEnvironment doesn't have __init__)
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.few_shot = few_shot or []
        self.reward = reward
        
        # Configuration
        self.validation_config = validation_config or ValidationConfig()
        self.max_workers = max_workers
        self.eot_id = eot_id
        self.message_end_id = message_end_id
        
        # Sampling configuration
        default_sampling_args = {
            "stop": ["</tool>\n", "</answer>\n"],
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        if sampling_args:
            default_sampling_args.update(sampling_args)
        self.sampling_args = default_sampling_args
        
        # Component initialization
        self.tool_manager = ToolManager()
        self.execution_engine = ExecutionEngine(
            validator=SystemValidator(self.validation_config)
        )
        self.state_manager = StateManager(max_steps=max_steps)
        
        # Environment configuration
        self.env_mask = 0 if mask_env_response else 1
        
        # Parsers
        self.llm_parser = XMLParser(fields=["think", ("tool", "answer")])
        self.env_parser = XMLParser(fields=["result"])
        
        # Setup components
        self._setup_components()
    
    def _setup_components(self) -> None:
        """Initialize environment-specific components."""
        # Register tools
        if self.tools:
            self.tool_manager.register_tools(self.tools)
        
        # Setup system prompt with tool descriptions
        if "{tool_descriptions}" in self.system_prompt:
            tool_descriptions = self.tool_manager.get_tool_descriptions()
            self.system_prompt = self.system_prompt.format(
                tool_descriptions=tool_descriptions
            )
        
        # Format datasets
        self._format_datasets()
        
        # Initialize reward
        if self.reward is None:
            self.reward = ToolReward(tools=self.tools)
    
    def _format_datasets(self) -> None:
        """Format datasets with system prompt and few-shot examples."""
        if self.dataset is not None:
            self.dataset = format_dataset(
                dataset=self.dataset,
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
        
        if self.eval_dataset is not None:
            self.eval_dataset = format_dataset(
                dataset=self.eval_dataset,
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
    
    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        init_codes: List[str],
        llm: LLM,
        sampling_params: SamplingParams,
        lora_request: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: List of conversation prompts
            init_codes: Initialization codes for each prompt
            llm: Language model instance
            sampling_params: Sampling parameters
            lora_request: LoRA request configuration
            
        Returns:
            Dictionary with generation results
        """
        # Create custom sampling parameters
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)
        
        # Initialize conversation states
        states = self.state_manager.batch_create_initial_states(prompts, init_codes)
        
        # Main generation loop
        while not all(state.completed for state in states):
            states = self.step(states, llm, custom_sp, lora_request)
        
        # Extract results
        return self._extract_results(states)
    
    def step(
        self,
        states: List[ConversationState],
        llm: LLM,
        sampling_params: SamplingParams,
        lora_request: Any
    ) -> List[ConversationState]:
        """
        Execute a single step for all conversation states.
        
        Args:
            states: List of conversation states
            llm: Language model instance
            sampling_params: Sampling parameters
            lora_request: LoRA request configuration
            
        Returns:
            Updated conversation states
        """
        # Find states that need LLM generation
        active_indices = [
            i for i, state in enumerate(states)
            if not state.completed and state.state != EnvironmentState.ERROR
        ]
        
        if not active_indices:
            return states
        
        # Prepare messages for LLM
        messages_to_generate = [states[i].messages for i in active_indices]
        
        # Generate LLM responses
        llm_responses = llm.chat(
            messages_to_generate,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request
        )
        
        # Process responses in parallel
        def update_state(i, llm_response):
            return self._update_single_state(states[active_indices[i]], llm_response)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            updated_states = list(executor.map(
                lambda args: update_state(*args),
                enumerate(llm_responses)
            ))
        
        # Update the original states list
        for i, updated_state in enumerate(updated_states):
            states[active_indices[i]] = updated_state
        
        return states
    
    def _update_single_state(
        self,
        state: ConversationState,
        llm_response: Any
    ) -> ConversationState:
        """Update a single conversation state with LLM response."""
        # Update state with completion data
        new_state = self.state_manager.update_completion_data(
            state, llm_response, self.env_mask
        )
        
        # Check for completion using StateManager's logic
        if self.state_manager.is_state_completed(new_state) or len(new_state.completion_ids) > 4000:
            new_state.completed = True
            new_state.state = EnvironmentState.COMPLETED
            # Ensure consistent completion data
            new_state = self.state_manager.ensure_completion_consistency(
                new_state, self.message_end_id
            )
        else:
            # Generate environment response
            env_message, updated_globals, executed_init, updated_distances = self._env_response(
                new_state.messages,
                new_state.init_code,
                new_state.gen_globals,
                new_state.executed_init,
                new_state.distances
            )
            
            # Update state with environment response
            new_state = self.state_manager.update_state(new_state, {
                'messages': new_state.messages + [env_message],
                'gen_globals': updated_globals,
                'executed_init': executed_init,
                'distances': updated_distances
            })
        
        return new_state
    
    def _env_response(
        self,
        messages: List[Dict[str, str]],
        init_code: str,
        gen_globals: Dict[str, Any],
        executed_init: bool,
        distances: Dict[str, Any]
    ) -> tuple[Dict[str, str], Dict[str, Any], bool, Dict[str, Any]]:
        """
        Generate environment response to the last message.
        
        Returns:
            Tuple of (env_message, updated_globals, executed_init, updated_distances)
        """
        # Parse the LLM response
        parsed = self.llm_parser.parse(messages[-1]["content"])
        
        # Handle plan control answers (Continue/Finish)
        if getattr(parsed, "answer", None):
            answer_text = parsed.answer.strip().lower()
            
            if answer_text == "continue":
                # Execute plan_continue function
                from tools.plan import plan_continue
                result = plan_continue()
                content = self.env_parser.format(result=result)
                return (
                    {"role": "user", "content": content},
                    gen_globals,
                    executed_init,
                    distances
                )
            elif answer_text == "finish":
                # Execute plan_finish function
                from tools.plan import plan_finish
                result = plan_finish()
                content = self.env_parser.format(result=result)
                return (
                    {"role": "user", "content": content},
                    gen_globals,
                    executed_init,
                    distances
                )
        
        # Try tool execution
        if getattr(parsed, "tool", None):
            result = self.tool_manager.execute_tool_from_json(parsed.tool)
            
            if result.success:
                content = self.env_parser.format(result=result.content)
            else:
                content = f"Error: {result.error}"
            
            return (
                {"role": "user", "content": content},
                gen_globals,
                executed_init,
                distances
            )
        
        # Try Python code execution
        from rewards.reward_helpers import extract_python_code
        python_code = extract_python_code(messages[-1]["content"])
        
        if not python_code:
            return (
                {"role": "user", "content": "Error: No valid tool command or Python code found."},
                gen_globals,
                executed_init,
                distances
            )
        
        # Execute initialization code if needed
        if not executed_init and init_code:
            init_py = extract_python_code(init_code)
            if init_py:
                gen_globals, distances = self.execution_engine.execute_init_code(
                    init_py, gen_globals
                )
                executed_init = True
        
        # Execute user code
        execution_result = self.execution_engine.execute_code(
            python_code, gen_globals, distances, validate_changes=True
        )
        
        # Format response based on execution result
        if execution_result.success:
            if execution_result.execution_log:
                content = "\n".join(execution_result.execution_log)
            else:
                content = "Code executed successfully."
        else:
            content = execution_result.error or "Code execution failed."
        
        return (
            {"role": "user", "content": content},
            execution_result.globals_dict,
            executed_init,
            execution_result.distances
                )

    def is_completed(self, messages: List[Dict[str, str]], **kwargs) -> bool:
        """Check if a conversation is completed. Required by BaseEnvironment."""
        # This method is required by the abstract base class but we use StateManager's logic
        # Create a minimal state to check completion
        if not messages:
            return False
        
        # For compatibility with the abstract interface, we'll create a temporary state
        # In practice, the main completion logic uses state_manager.is_state_completed()
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field with "finish" value
            return (hasattr(parsed, 'answer') and parsed.answer is not None and 
                   parsed.answer.strip().lower() == "finish")
        except Exception:
            return False
    
    def _extract_results(self, states: List[ConversationState]) -> Dict[str, Any]:
        """Extract final results from conversation states."""
        return {
            "ids": [state.completion_ids for state in states],
            "messages": [
                state.messages[state.prompt_messages:] for state in states
            ],
            "mask": [state.completion_mask for state in states],
            "gen_globals": [state.gen_globals for state in states]
        }
    
    def get_reward_funcs(self, **kwargs) -> List[Callable]:
        """Get reward functions from the reward system."""
        if self.reward is None:
            return []
        
        if hasattr(self.reward, 'get_reward_funcs'):
            return self.reward.get_reward_funcs()
        
        # If reward doesn't have get_reward_funcs, return empty list
        return []
    
    def get_reward_weights(self, **kwargs) -> List[float]:
        """Get reward weights from the reward system."""
        if self.reward is None:
            return [1.0]
        
        if hasattr(self.reward, 'get_reward_weights'):
            return self.reward.get_reward_weights()
        
        # If reward doesn't have get_reward_weights, return default weight
        return [1.0]
    
    def get_dataset(self, n: int = -1, seed: int = 0, **kwargs):
        """Get training dataset with optional sampling."""
        return self.dataset
    
    def get_eval_dataset(self, n: int = -1, seed: int = 0, **kwargs):
        """Get evaluation dataset with optional sampling."""
        return self.eval_dataset 