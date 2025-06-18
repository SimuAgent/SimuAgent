"""
Base reward class providing common functionality for all reward implementations.

This module contains the abstract base class that defines the interface
and common behavior for all reward system implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union

from .interfaces import (
    RewardType, 
    RewardResult, 
    RewardConfiguration,
    RewardEvaluator,
    RewardCalculator
)
from .exceptions import RewardError, ConfigurationError, EvaluationError


class DefaultRewardConfiguration:
    """Default implementation of RewardConfiguration protocol."""
    
    def __init__(self, weights: Optional[Dict[Union[str, RewardType], float]] = None):
        self._weights = {}
        if weights:
            for key, value in weights.items():
                if isinstance(key, str):
                    # Try to convert string to RewardType
                    try:
                        key = RewardType(key)
                    except ValueError:
                        # If not a valid RewardType, skip
                        continue
                self._weights[key] = value
    
    def get_weight(self, reward_type: RewardType) -> float:
        """Get weight for a specific reward type."""
        return self._weights.get(reward_type, 0.0)
    
    def is_enabled(self, reward_type: RewardType) -> bool:
        """Check if a reward type is enabled (has non-zero weight)."""
        return self.get_weight(reward_type) > 0.0
    
    def get_parameters(self, reward_type: RewardType) -> Dict[str, Any]:
        """Get parameters for a specific reward type."""
        # Default implementation returns empty dict
        return {}


class DefaultRewardCalculator(RewardCalculator):
    """Default implementation of RewardCalculator."""
    
    def calculate_total_reward(
        self, 
        results: Dict[RewardType, RewardResult],
        weights: Optional[Dict[RewardType, float]] = None
    ) -> float:
        """Calculate the total weighted reward score."""
        if not results:
            return 0.0
        
        if weights is None:
            weights = {}
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for reward_type, result in results.items():
            if result.is_valid:
                weight = weights.get(reward_type, 1.0 / len(results))
                total_weighted_score += result.normalized_score * weight
                total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def calculate_component_scores(
        self,
        results: Dict[RewardType, RewardResult]
    ) -> Dict[RewardType, float]:
        """Calculate individual component scores."""
        return {
            reward_type: result.normalized_score if result.is_valid else 0.0
            for reward_type, result in results.items()
        }


class BaseReward(ABC):
    """
    Abstract base class for all reward implementations.
    
    This class provides common functionality and defines the interface
    that all reward implementations must follow.
    """
    
    def __init__(
        self,
        config: Optional[RewardConfiguration] = None,
        calculator: Optional[RewardCalculator] = None,
        parser: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the base reward system.
        
        Args:
            config: Configuration for reward weights and parameters
            calculator: Calculator for combining reward scores
            parser: Parser for extracting answers from completions
            **kwargs: Additional configuration parameters
        """
        self.logger = logging.getLogger(f"rewards.{self.__class__.__name__}")
        self.config = config or DefaultRewardConfiguration()
        self.calculator = calculator or DefaultRewardCalculator()
        self.parser = parser
        
        # Store additional configuration
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize reward function registry
        self._reward_funcs: List[Callable] = []
        self._reward_weights: List[float] = []
        
        # Initialize the reward system
        self._initialize_reward_system()
    
    @abstractmethod
    def _initialize_reward_system(self) -> None:
        """Initialize the specific reward system components."""
        pass
    
    def add_reward_function(
        self, 
        func: Callable, 
        weight: float = 1.0
    ) -> None:
        """Add a reward function with associated weight."""
        self._reward_funcs.append(func)
        self._reward_weights.append(weight)
    
    def get_reward_funcs(self) -> List[Callable]:
        """Get all registered reward functions."""
        return self._reward_funcs.copy()
    
    def get_reward_weights(self) -> List[float]:
        """Get weights for all reward functions."""
        return self._reward_weights.copy()
    
    def get_assistant_messages(self, trajectory: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Helper function to extract assistant messages from a trajectory."""
        return [msg for msg in trajectory if msg.get('role') == 'assistant']
    
    def get_last_answer(self, trajectory: List[Dict[str, str]]) -> Optional[str]:
        """
        Extract the last answer from a trajectory.
        
        Args:
            trajectory: List of conversation messages
            
        Returns:
            The last answer found, or None if no answer
            
        Raises:
            RewardError: If parser is not configured
        """
        if self.parser is None:
            raise RewardError("Parser is not configured")
        
        for msg in reversed(trajectory):
            if msg.get('role') == 'assistant':
                try:
                    parsed = self.parser.parse(msg.get('content', ''))
                    if hasattr(parsed, 'answer') and parsed.answer is not None:
                        return parsed.answer
                except Exception as e:
                    self.logger.debug(f"Failed to parse message: {e}")
                    continue
        
        return None
    
    def exact_answer_reward_func(
        self, 
        completions: List[List[Dict[str, str]]], 
        answers: List[str], 
        **kwargs
    ) -> List[float]:
        """Reward function that checks if the final answer matches exactly."""
        try:
            responses = [self.get_last_answer(c) for c in completions]
            return [
                1.0 if str(r) == str(a) else 0.0 
                for r, a in zip(responses, answers)
            ]
        except Exception as e:
            self.logger.error(f"Error in exact_answer_reward_func: {e}")
            return [0.0] * len(completions)
    
    def int_answer_reward_func(
        self, 
        completions: List[List[Dict[str, str]]], 
        **kwargs
    ) -> List[float]:
        """Reward function that checks if the final answer is an integer."""
        try:
            responses = [self.get_last_answer(c) for c in completions]
            return [
                1.0 if str(r).strip().isdigit() else 0.0 
                for r in responses
            ]
        except Exception as e:
            self.logger.error(f"Error in int_answer_reward_func: {e}")
            return [0.0] * len(completions)
    
    @abstractmethod
    def evaluate(
        self, 
        completions: List[List[Dict[str, str]]], 
        **kwargs
    ) -> List[float]:
        """
        Evaluate completions and return reward scores.
        
        This is the main interface that must be implemented by all
        reward system implementations.
        
        Args:
            completions: List of conversation trajectories
            **kwargs: Additional evaluation parameters
            
        Returns:
            List of reward scores (one per completion)
        """
        pass
    
    def batch_evaluate(
        self,
        batch_completions: List[List[List[Dict[str, str]]]],
        **kwargs
    ) -> List[List[float]]:
        """
        Evaluate multiple batches of completions.
        
        Args:
            batch_completions: List of completion batches
            **kwargs: Additional evaluation parameters
            
        Returns:
            List of reward score lists (one per batch)
        """
        results = []
        for completions in batch_completions:
            try:
                scores = self.evaluate(completions, **kwargs)
                results.append(scores)
            except Exception as e:
                self.logger.error(f"Error evaluating batch: {e}")
                results.append([0.0] * len(completions))
        
        return results
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not self._reward_funcs:
            errors.append("No reward functions configured")
        
        if len(self._reward_funcs) != len(self._reward_weights):
            errors.append("Mismatch between reward functions and weights")
        
        if any(w < 0 for w in self._reward_weights):
            errors.append("Negative weights are not allowed")
        
        return errors
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            'reward_functions': len(self._reward_funcs),
            'total_weight': sum(self._reward_weights),
            'average_weight': (
                sum(self._reward_weights) / len(self._reward_weights)
                if self._reward_weights else 0.0
            ),
            'parser_configured': self.parser is not None,
            'config_type': type(self.config).__name__,
            'calculator_type': type(self.calculator).__name__,
        }


def equals_reward_func(
    completions: List[List[Dict[str, str]]], 
    answers: List[str], 
    **kwargs
) -> List[float]:
    """
    Simple reward function that checks exact string equality.
    
    This is a utility function for simple reward scenarios.
    """
    try:
        responses = [c[0]['content'] if c else '' for c in completions]
        return [1.0 if r == a else 0.0 for r, a in zip(responses, answers)]
    except Exception:
        return [0.0] * len(completions) 