"""
Core interfaces for the reward system.

This module defines the abstract base classes and protocols that establish
the contract for reward system components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass
from enum import Enum


class RewardType(Enum):
    """Types of rewards supported by the system."""
    CONNECTIVITY = "connectivity"
    VALIDATION = "validation"
    PARAMETER = "parameter"
    CONVERSION = "conversion"
    DIAGNOSTIC = "diagnostic"
    LOAD_SATISFACTION = "load_satisfaction"
    STRUCTURE = "structure"
    TOOL_EXECUTION = "tool_execution"
    FORMAT = "format"
    XML = "xml"
    CONNECTION_ADDITION = "connection_addition"
    BLOCK_ADDITION = "block_addition"
    FREQUENCY_COHERENCE = "frequency_coherence"
    VOLTAGE_COHERENCE = "voltage_coherence"
    MATH = "math"
    CODE = "code"
    MULTIPLE_CHOICE = "mc"


@dataclass
class RewardResult:
    """Result of a reward calculation."""
    score: float
    max_score: float = 1.0
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    @property
    def normalized_score(self) -> float:
        """Get normalized score (0.0 to 1.0)."""
        if self.max_score == 0:
            return 0.0
        return max(0.0, min(1.0, self.score / self.max_score))
    
    @property
    def is_valid(self) -> bool:
        """Check if the reward result is valid."""
        return self.error_message is None


class RewardConfiguration(Protocol):
    """Protocol for reward configuration objects."""
    
    def get_weight(self, reward_type: RewardType) -> float:
        """Get weight for a specific reward type."""
        ...
    
    def is_enabled(self, reward_type: RewardType) -> bool:
        """Check if a reward type is enabled."""
        ...
    
    def get_parameters(self, reward_type: RewardType) -> Dict[str, Any]:
        """Get parameters for a specific reward type."""
        ...


class RewardComponent(ABC):
    """Abstract base class for individual reward components."""
    
    def __init__(self, config: Optional[RewardConfiguration] = None):
        self.config = config
        self._cache = {}
    
    @property
    @abstractmethod
    def reward_type(self) -> RewardType:
        """The type of reward this component calculates."""
        pass
    
    @abstractmethod
    def calculate(self, context: Dict[str, Any]) -> RewardResult:
        """Calculate the reward for given context."""
        pass
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this component is applicable to the given context."""
        return True
    
    def clear_cache(self) -> None:
        """Clear any cached results."""
        self._cache.clear()


class RewardEvaluator(ABC):
    """Abstract base class for reward evaluators."""
    
    def __init__(self, components: Optional[List[RewardComponent]] = None):
        self.components = components or []
        self._results_cache = {}
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Dict[RewardType, RewardResult]:
        """Evaluate all applicable reward components."""
        pass
    
    def add_component(self, component: RewardComponent) -> None:
        """Add a reward component to the evaluator."""
        self.components.append(component)
    
    def remove_component(self, reward_type: RewardType) -> None:
        """Remove a reward component by type."""
        self.components = [c for c in self.components if c.reward_type != reward_type]
    
    def get_component(self, reward_type: RewardType) -> Optional[RewardComponent]:
        """Get a component by reward type."""
        for component in self.components:
            if component.reward_type == reward_type:
                return component
        return None
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._results_cache.clear()
        for component in self.components:
            component.clear_cache()


class RewardCalculator(ABC):
    """Abstract base class for calculating final reward scores."""
    
    @abstractmethod
    def calculate_total_reward(
        self, 
        results: Dict[RewardType, RewardResult],
        weights: Optional[Dict[RewardType, float]] = None
    ) -> float:
        """Calculate the total weighted reward score."""
        pass
    
    @abstractmethod
    def calculate_component_scores(
        self,
        results: Dict[RewardType, RewardResult]
    ) -> Dict[RewardType, float]:
        """Calculate individual component scores."""
        pass


class RewardAggregator(Protocol):
    """Protocol for aggregating multiple reward results."""
    
    def aggregate(
        self,
        results: List[Dict[RewardType, RewardResult]]
    ) -> Dict[RewardType, RewardResult]:
        """Aggregate multiple reward result sets."""
        ...


class RewardValidator(Protocol):
    """Protocol for validating reward configurations and results."""
    
    def validate_config(self, config: RewardConfiguration) -> List[str]:
        """Validate a reward configuration. Returns list of error messages."""
        ...
    
    def validate_result(self, result: RewardResult) -> List[str]:
        """Validate a reward result. Returns list of error messages."""
        ...


class RewardContext(Protocol):
    """Protocol for reward evaluation context."""
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from the context."""
        ...
    
    def set_data(self, key: str, value: Any) -> None:
        """Set data in the context."""
        ...
    
    def has_data(self, key: str) -> bool:
        """Check if context has specific data."""
        ...


class RewardMetrics(Protocol):
    """Protocol for reward system metrics and monitoring."""
    
    def record_evaluation_time(self, component_type: RewardType, duration: float) -> None:
        """Record evaluation time for a component."""
        ...
    
    def record_error(self, component_type: RewardType, error: Exception) -> None:
        """Record an error that occurred during evaluation."""
        ...
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics."""
        ... 