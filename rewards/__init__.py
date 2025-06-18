"""
Rewards Package - Comprehensive Reward System for AI Agents

A modular, extensible reward system for evaluating AI agent performance across
multiple domains including mathematics, code execution, tool usage, and power systems.

Key Features:
- Modular reward components with clear separation of concerns
- Extensible base classes for custom reward implementations
- Specialized evaluators for different problem types
- Comprehensive power system evaluation capabilities
- Support for tool-based interactions and validations

Architecture:
- core/: Base classes and interfaces
- evaluators/: Specialized evaluation logic
- components/: Individual reward components
- power_system/: Power system specific rewards
- math/: Mathematical problem evaluation
- tools/: Tool execution and validation rewards

Example Usage:
    from rewards import PowerSystemReward, ToolReward, MathReward
    
    # Power system evaluation
    ps_reward = PowerSystemReward(
        weights={'load_satisfaction': 1.0, 'connectivity': 0.5}
    )
    
    # Tool-based evaluation
    tool_reward = ToolReward(
        tools=[my_tool1, my_tool2],
        weights={'execution': 0.8, 'format': 0.2}
    )
    
    # Math problem evaluation
    math_reward = MathReward(
        grading_mode='symbolic',
        timeout=10.0
    )
"""

# Core interfaces and base classes
from .core.base_reward import BaseReward
from .core.interfaces import (
    RewardComponent,
    RewardEvaluator,
    RewardCalculator,
    RewardConfiguration
)
from .core.exceptions import (
    RewardError,
    EvaluationError,
    ConfigurationError
)

# Reward implementations (using correct paths)
from .tool_reward import ToolReward
from .power_system_reward import PowerSystemReward
from .complete_reward import CompleteReward

# Math evaluator (commented out as it doesn't exist)
# from .math_grader import MathEvaluator

# Components
from .components.reward_components import RewardComponents

# Utilities
from .reward_helpers import (
    calculate_weighted_score
)

# Legacy compatibility (with deprecation warnings)
from .legacy_compatibility import (
    get_legacy_power_system_reward,
    get_legacy_tool_reward
)

# Base reward from file
from .base_reward import BaseReward as FileBaseReward

__version__ = "2.0.0"
__author__ = "Rewards Package Team"

__all__ = [
    # Core
    'BaseReward',
    'RewardComponent',
    'RewardEvaluator', 
    'RewardCalculator',
    'RewardConfiguration',
    
    # Exceptions
    'RewardError',
    'EvaluationError',
    'ConfigurationError',
    
    # Main reward implementations
    'ToolReward',
    'PowerSystemReward', 
    'CompleteReward',
    
    # Components
    'RewardComponents',
    
    # Utilities
    'calculate_weighted_score',
    
    # Legacy
    'get_legacy_power_system_reward',
    'get_legacy_tool_reward',
    
    # Base reward from file
    'FileBaseReward',
]

# Module configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
