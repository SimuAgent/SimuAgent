"""
Legacy Compatibility Module

Provides backward compatibility for the old reward system interfaces
while guiding users to migrate to the new modular structure.
"""

import warnings
from typing import Any, Dict, List, Optional, Callable, Union

# Import new implementations
from .power_system_reward import PowerSystemReward
from .tool_reward import ToolReward  
# from .math_reward import MathReward  # Commented out as it doesn't exist
from .components.reward_components import RewardComponents


def get_legacy_power_system_reward(*args, **kwargs):
    """
    Create a PowerSystemReward instance with legacy parameter mapping.
    
    This function provides backward compatibility for the old PowerSystemReward
    constructor while warning users about deprecation.
    """
    warnings.warn(
        "Direct instantiation via get_legacy_power_system_reward is deprecated. "
        "Use 'from rewards import PowerSystemReward' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Map old parameter names to new ones
    if 'reward_weights' in kwargs:
        kwargs['weights'] = kwargs.pop('reward_weights')
    
    if 'power_system_weights' in kwargs:
        kwargs['weights'] = kwargs.pop('power_system_weights')
    
    return PowerSystemReward(*args, **kwargs)


def get_legacy_tool_reward(*args, **kwargs):
    """
    Create a ToolReward instance with legacy parameter mapping.
    
    This function provides backward compatibility for the old ToolReward
    constructor while warning users about deprecation.
    """
    warnings.warn(
        "Direct instantiation via get_legacy_tool_reward is deprecated. "
        "Use 'from rewards import ToolReward' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return ToolReward(*args, **kwargs)


class LegacyCompleteReward:
    """
    Legacy wrapper for CompleteReward functionality.
    
    This class maintains backward compatibility while encouraging
    migration to the new PowerSystemReward class.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "CompleteReward is deprecated. Use PowerSystemReward instead. "
            "from rewards import PowerSystemReward",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Map old parameter names to new ones
        if 'reward_weights' in kwargs:
            kwargs['weights'] = kwargs.pop('reward_weights')
        
        # Create the new implementation
        self._reward_system = PowerSystemReward(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the new implementation."""
        # Handle renamed methods
        method_mapping = {
            'update_reward_weights': 'update_weights',
            'get_reward_weights': 'get_weights',
            'correct_answer_reward_func': 'evaluate',
        }
        
        if name in method_mapping:
            warnings.warn(
                f"{name} is deprecated. Use {method_mapping[name]} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return getattr(self._reward_system, method_mapping[name])
        
        return getattr(self._reward_system, name)


class LegacyRewardComponents(RewardComponents):
    """
    Legacy wrapper for RewardComponents.
    
    Provides backward compatibility for the old RewardComponents interface.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Using legacy RewardComponents interface. "
            "Consider migrating to the new components structure.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
    
    def get_total_reward(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Legacy method with deprecation warning."""
        if weights is None:
            # Use old default weights for compatibility
            weights = {
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
            }
        
        return super().get_total_reward(weights)


def create_legacy_reward_system(
    reward_type: str = "power_system",
    **kwargs
) -> Union[PowerSystemReward, ToolReward]:
    """
    Factory function for creating legacy reward systems.
    
    Args:
        reward_type: Type of reward system to create
        **kwargs: Configuration parameters
        
    Returns:
        Appropriate reward system instance
    """
    warnings.warn(
        "create_legacy_reward_system is deprecated. "
        "Import and instantiate reward classes directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if reward_type == "power_system":
        return PowerSystemReward(**kwargs)
    elif reward_type == "tool":
        return ToolReward(**kwargs)
    elif reward_type == "math":
        raise ValueError("MathReward is not available in this version")
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


# Legacy imports for backward compatibility
def import_legacy_modules():
    """Import legacy modules and set up aliases."""
    warnings.warn(
        "Legacy module imports are deprecated. "
        "Use the new modular import structure.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create global aliases for old imports
    import sys
    current_module = sys.modules[__name__]
    
    # Set up legacy aliases
    setattr(current_module, 'CompleteReward', LegacyCompleteReward)
    setattr(current_module, 'RewardComponents', LegacyRewardComponents)
    setattr(current_module, 'PowerSystemReward', PowerSystemReward)
    setattr(current_module, 'ToolReward', ToolReward)


# Migration guide
MIGRATION_GUIDE = """
Migration Guide for Rewards Package
===================================

The rewards package has been refactored for better modularity and maintainability.

OLD USAGE:
    from rewards.power_system_reward import PowerSystemReward
    from rewards.tool_reward import ToolReward
    from rewards.complete_reward import CompleteReward

NEW USAGE:
    from rewards import PowerSystemReward, ToolReward, MathReward
    
    # PowerSystemReward replaces CompleteReward
    ps_reward = PowerSystemReward(weights={'load_satisfaction': 1.0})
    
    # Enhanced tool evaluation
    tool_reward = ToolReward(tools=[my_tool], weights={'execution': 0.8})
    
    # New math evaluation capabilities
    math_reward = MathReward(grading_mode='symbolic')

KEY CHANGES:
1. Modular architecture with clear separation of concerns
2. Enhanced configuration system with typed interfaces
3. Better error handling and logging
4. Improved documentation and examples
5. Comprehensive test coverage

DEPRECATED FEATURES:
- CompleteReward class (use PowerSystemReward)
- Direct file imports (use package imports)
- Old configuration parameter names
- Legacy method names

For detailed migration instructions, see the documentation.
"""


def print_migration_guide():
    """Print the migration guide for users."""
    print(MIGRATION_GUIDE) 