"""
Validation configuration management.
"""

from typing import Dict, Any


class ValidationConfiguration:
    """
    Manages validation configuration settings.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Default validation settings
        self.defaults = {
            'strict_mode': False,
            'check_electrical': True,
            'check_distances': True,
            'check_parameters': True,
            'voltage_threshold': 0.2,
            'max_errors': 100
        }
    
    def get(self, key: str, default=None):
        """Get a configuration value."""
        return self.config.get(key, self.defaults.get(key, default))
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
    
    def is_strict_mode(self) -> bool:
        """Check if strict validation mode is enabled."""
        return self.get('strict_mode', False) 