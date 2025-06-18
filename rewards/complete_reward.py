"""
DEPRECATED: This module has been moved and renamed.

Please use:
    from rewards.power_system_reward import PowerSystemReward

The PowerSystemReward class provides all the functionality of CompleteReward
plus integration with ToolReward functionality.
"""

import warnings
from rewards.power_system_reward import PowerSystemReward, RewardComponents

# Backward compatibility alias
class CompleteReward(PowerSystemReward):
    """
    DEPRECATED: Use PowerSystemReward instead.
    
    This class is provided for backward compatibility only.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "CompleteReward is deprecated. Use PowerSystemReward instead. "
            "from rewards.power_system_reward import PowerSystemReward",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Map old parameter names to new ones
        if 'reward_weights' in kwargs:
            kwargs['power_system_weights'] = kwargs.pop('reward_weights')
        
        super().__init__(*args, **kwargs)
    
    # Map old method names to new ones for backward compatibility
    def update_reward_weights(self, new_weights):
        """DEPRECATED: Use update_power_system_weights instead."""
        warnings.warn(
            "update_reward_weights is deprecated. Use update_power_system_weights instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.update_power_system_weights(new_weights)
    
    def get_reward_weights(self):
        """DEPRECATED: Use get_power_system_weights instead."""
        warnings.warn(
            "get_reward_weights is deprecated. Use get_power_system_weights instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_power_system_weights()


# Re-export for backward compatibility
__all__ = ['CompleteReward', 'RewardComponents', 'PowerSystemReward']
