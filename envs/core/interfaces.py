"""
Core interfaces and protocols for the environment system.

This module defines the contracts that all environment components must follow,
ensuring consistency and enabling easy extension of the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union, Callable, Sequence
from dataclasses import dataclass
from enum import Enum


class EnvironmentState(Enum):
    """Possible states of an environment conversation."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ConversationState:
    """State information for a single conversation thread."""
    messages: List[Dict[str, Any]]
    prompt_messages: int
    prompt_ids: List[int]
    completion_ids: List[int]
    completion_mask: List[int]
    init_code: str
    completed: bool
    executed_init: bool
    gen_globals: Dict[str, Any]
    distances: Dict[str, Any]
    state: EnvironmentState = EnvironmentState.INITIALIZING
    step_count: int = 0


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    content: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result from validation operations."""
    valid: bool
    reason: str = ""
    suggestions: List[str] = None
    error_details: Optional[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ToolProtocol(Protocol):
    """Protocol that all tools must implement."""
    
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        ...
    
    @property
    def name(self) -> str:
        """Get the tool name."""
        ...
    
    @property
    def description(self) -> str:
        """Get the tool description."""
        ...


class ValidatorProtocol(Protocol):
    """Protocol for validation components."""
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data with optional context."""
        ...


class ExecutorProtocol(Protocol):
    """Protocol for code execution components."""
    
    def execute(
        self, 
        code: str, 
        globals_dict: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute code in the given globals context."""
        ...


class RewardProtocol(Protocol):
    """Protocol for reward functions."""
    
    def get_reward_funcs(self) -> List[Callable]:
        """Get list of reward functions."""
        ...
    
    def get_reward_weights(self) -> List[float]:
        """Get weights for reward functions."""
        ...


class BaseEnvironment(ABC):
    """
    Abstract base class for all environments.
    
    Defines the core interface that all environment implementations must follow.
    """
    
    @abstractmethod
    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        init_codes: List[str],
        llm: Any,
        sampling_params: Any,
        lora_request: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate responses for the given prompts."""
        ...
    
    @abstractmethod
    def step(
        self,
        states: List[ConversationState],
        llm: Any,
        sampling_params: Any,
        lora_request: Any
    ) -> List[ConversationState]:
        """Execute a single step for all conversation states."""
        ...
    
    @abstractmethod
    def is_completed(self, messages: List[Dict[str, str]], **kwargs) -> bool:
        """Check if a conversation is completed."""
        ...
    
    def get_dataset(self, n: int = -1, seed: int = 0, **kwargs) -> Optional[Any]:
        """Get training dataset with optional sampling."""
        return None
    
    def get_eval_dataset(self, n: int = -1, seed: int = 0, **kwargs) -> Optional[Any]:
        """Get evaluation dataset with optional sampling."""
        return None
    
    def get_reward_funcs(self, **kwargs) -> Optional[List[Callable]]:
        """Get reward functions."""
        return None
    
    def get_reward_weights(self, **kwargs) -> List[float]:
        """Get reward weights."""
        return [1.0]


class ToolManagerProtocol(Protocol):
    """Protocol for tool management systems."""
    
    def register_tool(self, tool: ToolProtocol) -> None:
        """Register a new tool."""
        ...
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given arguments."""
        ...
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools."""
        ...
    
    def list_tools(self) -> List[str]:
        """Get list of available tool names."""
        ...


class StateManagerProtocol(Protocol):
    """Protocol for state management systems."""
    
    def create_initial_state(self, prompt: List[Dict[str, Any]], init_code: str) -> ConversationState:
        """Create initial conversation state."""
        ...
    
    def update_state(self, state: ConversationState, update_data: Dict[str, Any]) -> ConversationState:
        """Update conversation state with new data."""
        ...
    
    def is_state_completed(self, state: ConversationState) -> bool:
        """Check if a conversation state is completed."""
        ... 