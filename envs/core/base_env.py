"""
Base environment infrastructure for tool-based environments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datasets import Dataset


@dataclass
class EnvironmentConfig:
    """Configuration for environment behavior."""
    max_steps: int = 10
    max_workers: int = 10
    mask_env_response: bool = True
    eot_id: int = 151643
    message_end_id: int = 151645
    sampling_args: Dict[str, Any] = field(default_factory=lambda: {
        "stop": ["</tool>\n", "</answer>\n"],
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "n": 1
    })


class BaseEnvironment(ABC):
    """
    Base class for tool-based environments.
    
    Provides common infrastructure for environments that use tools
    to interact with external systems or perform computations.
    """
    
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tools: List[Callable] = None,
        system_prompt: str = "",
        few_shot: List[Dict[str, str]] = None,
        config: Optional[EnvironmentConfig] = None,
        reward: Any = None,
        **kwargs
    ):
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.few_shot = few_shot or []
        self.config = config or EnvironmentConfig()
        self.reward = reward
        
        # Environment state
        self.env_mask = 0 if self.config.mask_env_response else 1
        
        # Initialize components
        self._setup_components()
    
    @abstractmethod
    def _setup_components(self) -> None:
        """Initialize environment-specific components."""
        pass
    
    @abstractmethod
    def generate(self, prompts: List[List[Dict[str, Any]]],
                init_codes: List[str],
                llm: Any,
                sampling_params: Any,
                lora_request: Any,
                **kwargs) -> Dict[str, Any]:
        """Generate responses using the environment."""
        pass
    
    @abstractmethod
    def step(self, states: List[Dict[str, Any]], 
            llm: Any, sampling_params: Any, 
            lora_request: Any) -> List[Dict[str, Any]]:
        """Execute a single step in the environment."""
        pass
    
    def get_dataset(self, n: int = -1, seed: int = 0, **kwargs) -> Optional[Dataset]:
        """Get training dataset with optional sampling."""
        if n > 0 and self.dataset is not None:
            return self.dataset.shuffle(seed=seed).select(range(n))
        return self.dataset
    
    def get_eval_dataset(self, n: int = -1, seed: int = 0, **kwargs) -> Optional[Dataset]:
        """Get evaluation dataset with optional sampling."""
        if n > 0 and self.eval_dataset is not None:
            return self.eval_dataset.shuffle(seed=seed).select(range(n))
        return self.eval_dataset
    
    def get_reward_funcs(self, **kwargs) -> Any:
        """Get reward functions for the environment."""
        return self.reward.get_reward_funcs() if self.reward else None
    
    def get_reward_weights(self, **kwargs) -> List[float]:
        """Get reward weights for the environment."""
        return self.reward.get_reward_weights() if self.reward else [1.0]
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs) -> bool:
        """Check if the current conversation is completed."""
        if not messages:
            return False
        
        last_message = messages[-1]
        content = last_message.get('content', '')
        
        # Check for completion markers
        completion_markers = ['</answer>', '</tool>']
        return any(marker in content for marker in completion_markers)
    
    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of assistant steps in the conversation."""
        return sum(1 for msg in messages if msg.get('role') == 'assistant')
    
    def _create_initial_state(self, prompt: List[Dict[str, Any]], 
                            init_code: str) -> Dict[str, Any]:
        """Create initial state for a conversation."""
        return {
            "messages": prompt,
            "prompt_messages": len(prompt),
            "prompt_ids": [],
            "init_code": init_code,
            "completed": False,
            "completion_ids": [],
            "completion_mask": [],
            "gen_globals": {},
            "executed_init": False,
            "distances": {},
        } 