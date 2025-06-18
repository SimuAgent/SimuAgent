"""
State management system for environment conversations.

This module provides clean state management for conversation threads,
handling state transitions and validation.
"""

from typing import Any, Dict, List, Optional
from copy import deepcopy

from .interfaces import ConversationState, EnvironmentState, StateManagerProtocol


class StateManager:
    """
    Manages conversation states and their transitions.
    
    This class provides a centralized way to handle conversation state
    management, ensuring consistent state transitions and validation.
    """
    
    def __init__(self, max_steps: int = 10):
        """
        Initialize the state manager.
        
        Args:
            max_steps: Maximum number of steps allowed per conversation
        """
        self.max_steps = max_steps
    
    def create_initial_state(
        self, 
        prompt: List[Dict[str, Any]], 
        init_code: str
    ) -> ConversationState:
        """
        Create initial conversation state.
        
        Args:
            prompt: Initial conversation prompt
            init_code: Initialization code for the conversation
            
        Returns:
            New ConversationState instance
        """
        return ConversationState(
            messages=prompt.copy(),
            prompt_messages=len(prompt),
            prompt_ids=[],
            completion_ids=[],
            completion_mask=[],
            init_code=init_code,
            completed=False,
            executed_init=False,
            gen_globals={},
            distances={},
            state=EnvironmentState.INITIALIZING,
            step_count=0
        )
    
    def update_state(
        self, 
        state: ConversationState, 
        update_data: Dict[str, Any]
    ) -> ConversationState:
        """
        Update conversation state with new data.
        
        Args:
            state: Current conversation state
            update_data: Dictionary of updates to apply
            
        Returns:
            Updated ConversationState instance
        """
        # Create a safe copy of the state
        new_state = self._create_safe_copy(state)
        
        # Apply updates
        for key, value in update_data.items():
            if hasattr(new_state, key):
                setattr(new_state, key, value)
        
        # Update step count if we added a new assistant message
        if 'messages' in update_data:
            new_state.step_count = self._count_assistant_steps(new_state.messages)
        
        # Update state status based on conditions
        new_state.state = self._determine_state(new_state)
        
        return new_state
    
    def _create_safe_copy(self, state: ConversationState) -> ConversationState:
        """Create a safe copy of the conversation state."""
        return ConversationState(
            messages=deepcopy(state.messages),
            prompt_messages=state.prompt_messages,
            prompt_ids=deepcopy(state.prompt_ids),
            completion_ids=deepcopy(state.completion_ids),
            completion_mask=deepcopy(state.completion_mask),
            init_code=state.init_code,
            completed=state.completed,
            executed_init=state.executed_init,
            gen_globals=state.gen_globals,  # Shallow copy to avoid unpicklable objects
            distances=deepcopy(state.distances),
            state=state.state,
            step_count=state.step_count
        )
    
    def _count_assistant_steps(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of assistant steps in the conversation."""
        return sum(1 for msg in messages if msg.get('role') == 'assistant')
    
    def _determine_state(self, state: ConversationState) -> EnvironmentState:
        """Determine the current state based on conversation conditions."""
        if state.completed:
            return EnvironmentState.COMPLETED
        elif state.step_count >= self.max_steps:
            return EnvironmentState.COMPLETED
        elif state.step_count == 0:
            return EnvironmentState.INITIALIZING
        else:
            return EnvironmentState.RUNNING
    
    def is_state_completed(self, state: ConversationState) -> bool:
        """
        Check if a conversation state is completed.
        
        Args:
            state: Conversation state to check
            
        Returns:
            True if the conversation is completed
        """
        # Check explicit completion flag
        if state.completed:
            return True
        
        # Check if we've hit max steps
        if state.step_count >= self.max_steps:
            return True
        
        # Check for explicit "Finish" answer in the last message
        if state.messages:
            last_message = state.messages[-1]
            if last_message.get('role') == 'assistant':
                content = last_message.get('content', '')
                try:
                    # Import here to avoid circular imports
                    from rewards.reward_helpers import LLMOutputParser
                    parser = LLMOutputParser()
                    parsed = parser.parse(content)
                    # Check if we got a valid answer field with "finish" value
                    if hasattr(parsed, 'answer') and parsed.answer is not None and parsed.answer.lower() == "finish":
                        return True
                except Exception:
                    pass
        
        return False
    
    def update_completion_data(
        self,
        state: ConversationState,
        llm_response: Any,
        env_mask: int = 0
    ) -> ConversationState:
        """
        Update state with completion data from LLM response.
        
        Args:
            state: Current conversation state
            llm_response: Response from the LLM
            env_mask: Mask value for environment responses
            
        Returns:
            Updated conversation state
        """
        new_state = self._create_safe_copy(state)
        
        # Initialize prompt ids if not set
        if len(new_state.prompt_ids) == 0:
            new_state.prompt_ids = llm_response.prompt_token_ids
        
        # Add assistant message
        new_state.messages.append({
            "role": "assistant", 
            "content": llm_response.outputs[0].text
        })
        
        # Calculate token lengths
        total_prev_len = len(new_state.prompt_ids) + len(new_state.completion_ids)
        env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len
        new_completion_len = len(llm_response.outputs[0].token_ids)
        
        # Update completion masks and ids
        new_state.completion_mask.extend([env_mask] * env_response_len)
        new_state.completion_mask.extend([1] * new_completion_len)
        
        new_state.completion_ids = list(llm_response.prompt_token_ids)
        new_state.completion_ids.extend(list(llm_response.outputs[0].token_ids))
        new_state.completion_ids = new_state.completion_ids[len(new_state.prompt_ids):]
        
        # Update step count and state
        new_state.step_count = self._count_assistant_steps(new_state.messages)
        new_state.state = self._determine_state(new_state)
        
        return new_state
    
    def ensure_completion_consistency(
        self,
        state: ConversationState,
        message_end_id: int = 151645
    ) -> ConversationState:
        """
        Ensure completion IDs and masks are consistent.
        
        Args:
            state: Conversation state to fix
            message_end_id: ID for message end token
            
        Returns:
            State with consistent completion data
        """
        new_state = self._create_safe_copy(state)
        
        # Add message end tokens if needed
        if (len(new_state.completion_ids) >= 2 and 
            new_state.completion_ids[-1] != 198 and 
            new_state.completion_ids[-2] != message_end_id):
            new_state.completion_ids.append(message_end_id)
            new_state.completion_ids.append(198)
            new_state.completion_mask.append(1)
            new_state.completion_mask.append(1)
        
        # Ensure lengths match
        if len(new_state.completion_ids) > len(new_state.completion_mask):
            diff = len(new_state.completion_ids) - len(new_state.completion_mask)
            new_state.completion_mask.extend([1] * diff)
        elif len(new_state.completion_mask) > len(new_state.completion_ids):
            new_state.completion_mask = new_state.completion_mask[:len(new_state.completion_ids)]
        
        return new_state
    
    def batch_create_initial_states(
        self,
        prompts: List[List[Dict[str, Any]]],
        init_codes: List[str]
    ) -> List[ConversationState]:
        """
        Create initial states for multiple conversations.
        
        Args:
            prompts: List of conversation prompts
            init_codes: List of initialization codes
            
        Returns:
            List of initial conversation states
        """
        return [
            self.create_initial_state(prompt, init_code)
            for prompt, init_code in zip(prompts, init_codes)
        ]
    
    def get_state_summary(self, state: ConversationState) -> Dict[str, Any]:
        """
        Get a summary of the conversation state.
        
        Args:
            state: Conversation state to summarize
            
        Returns:
            Dictionary with state summary information
        """
        return {
            "state": state.state.value,
            "step_count": state.step_count,
            "completed": state.completed,
            "message_count": len(state.messages),
            "prompt_messages": state.prompt_messages,
            "executed_init": state.executed_init,
            "completion_length": len(state.completion_ids),
            "has_globals": bool(state.gen_globals),
            "has_distances": bool(state.distances)
        } 