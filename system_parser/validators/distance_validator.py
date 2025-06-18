"""
Distance validator for validating system distances.
"""

from typing import Dict, Any, List
from .validation_result import ValidationResult


class DistanceValidator:
    """
    Validates distances in the system.
    """
    
    def __init__(self, validation_config=None):
        self.validation_config = validation_config
    
    def validate(self, connections: List[Dict[str, Any]], blocks: Dict[str, Any], distances: Dict[Any, float]) -> ValidationResult:
        """
        Validate system distances.
        
        Args:
            connections: List of connection definitions
            blocks: Dictionary of block objects
            distances: Dictionary of distance values
        
        Returns:
            ValidationResult containing validation outcome
        """
        result = ValidationResult()
        
        try:
            # Basic distance validation
            for distance_key, distance_value in distances.items():
                if not isinstance(distance_value, (int, float)):
                    result.add_error(f"Distance value must be numeric, got {type(distance_value)} for {distance_key}")
                elif distance_value < 0:
                    result.add_error(f"Distance value cannot be negative: {distance_value} for {distance_key}")
        
        except Exception as e:
            result.add_error(f"Error validating distances: {e}")
        
        return result 