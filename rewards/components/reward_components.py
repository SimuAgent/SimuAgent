"""
Reward Components Data Structure

Contains the main RewardComponents dataclass that holds individual
reward component scores and provides methods for calculating totals.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from ..core.interfaces import RewardType


@dataclass
class RewardComponents:
    """Container for individual reward components with improved structure."""
    
    # Core power system rewards
    connectivity_reward: float = 0.0
    validation_reward: float = 0.0
    parameter_reward: float = 0.0
    conversion_reward: float = 0.0
    diagnostic_reward: float = 0.0
    load_satisfaction_reward: float = 0.0
    structure_reward: float = 0.0
    frequency_coherence_reward: float = 0.0
    voltage_coherence_reward: float = 0.0
    
    # Connection and block rewards
    connection_addition_reward: float = 0.0
    block_addition_reward: float = 0.0
    
    # Tool and format rewards
    tool_execution_reward: float = 0.0
    format_reward: float = 0.0
    xml_reward: float = 0.0
    
    # Additional metadata
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Default weights for different components
    _default_weights: Dict[str, float] = field(default_factory=lambda: {
        'connectivity': 0.0,
        'validation': 0.0,
        'parameter': 0.0,
        'conversion': 0.0,
        'diagnostic': 0.0,
        'load_satisfaction': 1.0,
        'structure': 0.0,
        'tool_execution': 0.10,
        'format': 0.05,
        'xml': 0.05,
        'connection_addition': 0.1,
        'block_addition': 0.1,
        'frequency_coherence': 0.0,
        'voltage_coherence': 0.0
    })
    
    def get_total_reward(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted total reward.
        
        Args:
            weights: Optional custom weights for components
            
        Returns:
            Weighted total reward score (0.0 to 1.0)
        """
        if weights is None:
            weights = self._default_weights
        
        total = (
            weights.get('connectivity', 0.0) * self.connectivity_reward +
            weights.get('validation', 0.0) * self.validation_reward +
            weights.get('parameter', 0.0) * self.parameter_reward +
            weights.get('conversion', 0.0) * self.conversion_reward +
            weights.get('diagnostic', 0.0) * self.diagnostic_reward +
            weights.get('load_satisfaction', 0.0) * self.load_satisfaction_reward +
            weights.get('structure', 0.0) * self.structure_reward +
            weights.get('tool_execution', 0.0) * self.tool_execution_reward +
            weights.get('format', 0.0) * self.format_reward +
            weights.get('xml', 0.0) * self.xml_reward +
            weights.get('connection_addition', 0.0) * self.connection_addition_reward +
            weights.get('block_addition', 0.0) * self.block_addition_reward +
            weights.get('frequency_coherence', 0.0) * self.frequency_coherence_reward +
            weights.get('voltage_coherence', 0.0) * self.voltage_coherence_reward
        )
        
        return max(0.0, min(1.0, total))
    
    def get_component_dict(self) -> Dict[str, float]:
        """Get all reward components as a dictionary."""
        return {
            'connectivity': self.connectivity_reward,
            'validation': self.validation_reward,
            'parameter': self.parameter_reward,
            'conversion': self.conversion_reward,
            'diagnostic': self.diagnostic_reward,
            'load_satisfaction': self.load_satisfaction_reward,
            'structure': self.structure_reward,
            'tool_execution': self.tool_execution_reward,
            'format': self.format_reward,
            'xml': self.xml_reward,
            'connection_addition': self.connection_addition_reward,
            'block_addition': self.block_addition_reward,
            'frequency_coherence': self.frequency_coherence_reward,
            'voltage_coherence': self.voltage_coherence_reward,
        }
    
    def get_enabled_components(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Get only the enabled components (non-zero weights)."""
        if weights is None:
            weights = self._default_weights
        
        components = self.get_component_dict()
        return {
            name: score for name, score in components.items()
            if weights.get(name, 0.0) > 0.0
        }
    
    def get_weighted_components(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Get components multiplied by their weights."""
        if weights is None:
            weights = self._default_weights
        
        components = self.get_component_dict()
        return {
            name: score * weights.get(name, 0.0)
            for name, score in components.items()
        }
    
    def set_component(self, component_name: str, value: float) -> None:
        """Set a component value by name."""
        if hasattr(self, f"{component_name}_reward"):
            setattr(self, f"{component_name}_reward", value)
        else:
            raise ValueError(f"Unknown component: {component_name}")
    
    def get_component(self, component_name: str) -> float:
        """Get a component value by name."""
        if hasattr(self, f"{component_name}_reward"):
            return getattr(self, f"{component_name}_reward")
        else:
            raise ValueError(f"Unknown component: {component_name}")
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update the default weights."""
        self._default_weights.update(new_weights)
    
    def add_detail(self, key: str, value: Any) -> None:
        """Add a detail to the metadata."""
        self.details[key] = value
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the reward components."""
        return {
            'total_reward': self.get_total_reward(),
            'enabled_components': len(self.get_enabled_components()),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'details_count': len(self.details),
            'top_components': self._get_top_components(),
        }
    
    def _get_top_components(self, n: int = 3) -> List[tuple]:
        """Get the top N components by score."""
        components = self.get_component_dict()
        sorted_components = sorted(
            components.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_components[:n]
    
    def __str__(self) -> str:
        """String representation of the reward components."""
        summary = self.get_summary()
        return (
            f"RewardComponents(total={summary['total_reward']:.3f}, "
            f"enabled={summary['enabled_components']}, "
            f"errors={summary['error_count']}, "
            f"warnings={summary['warning_count']})"
        ) 