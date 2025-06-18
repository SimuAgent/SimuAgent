"""
Core Module for Rewards Package

Contains base classes, interfaces, and fundamental components
for the reward system architecture.
"""

from .base_reward import BaseReward
from .interfaces import (
    RewardComponent,
    RewardEvaluator,
    RewardCalculator,
    RewardConfiguration
)
from .exceptions import (
    RewardError,
    EvaluationError,
    ConfigurationError
)

__all__ = [
    'BaseReward',
    'RewardComponent',
    'RewardEvaluator',
    'RewardCalculator', 
    'RewardConfiguration',
    'RewardError',
    'EvaluationError',
    'ConfigurationError',
] 