"""
Core interfaces for the system parser package.

This module defines the abstract base classes and protocols that define
the contract for various components in the system parser.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

class SystemAnalyzer(ABC):
    """
    Abstract base class for system analyzers.
    
    System analyzers are responsible for analyzing different aspects
    of power system configurations.
    """
    
    @abstractmethod
    def analyze(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the given system data.
        
        Args:
            system_data: The system data to analyze
            
        Returns:
            Analysis results dictionary
        """
        pass

class GraphBuilder(ABC):
    """
    Abstract base class for graph builders.
    
    Graph builders are responsible for constructing system graphs
    from various input formats.
    """
    
    @abstractmethod
    def build_from_dict(self, data: Dict[str, Any]) -> Any:
        """
        Build a system graph from dictionary data.
        
        Args:
            data: Dictionary containing system configuration
            
        Returns:
            Built system graph object
        """
        pass
    
    @abstractmethod
    def build_from_file(self, file_path: str) -> Any:
        """
        Build a system graph from a file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Built system graph object
        """
        pass

class RewardEvaluator(ABC):
    """
    Abstract base class for reward evaluators.
    """
    
    @abstractmethod
    def evaluate(self, response: str, ground_truth: Optional[str] = None) -> float:
        """
        Evaluate a response and return a reward score.
        
        Args:
            response: The response to evaluate
            ground_truth: Optional ground truth for comparison
            
        Returns:
            Reward score between 0.0 and 1.0
        """
        pass

class RewardCalculator(ABC):
    """
    Abstract base class for reward calculators.
    """
    
    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """
        Calculate a reward score based on input parameters.
        
        Returns:
            Calculated reward score
        """
        pass

class RewardComponent(ABC):
    """
    Abstract base class for reward components.
    """
    
    @abstractmethod
    def compute_reward(self, **kwargs) -> float:
        """
        Compute the reward for this component.
        
        Returns:
            Component reward score
        """
        pass

class RewardConfiguration(ABC):
    """
    Abstract base class for reward configurations.
    """
    
    @abstractmethod
    def get_weights(self) -> Dict[str, float]:
        """
        Get the weight configuration for different reward components.
        
        Returns:
            Dictionary mapping component names to weights
        """
        pass 