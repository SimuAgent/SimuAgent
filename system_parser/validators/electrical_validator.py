"""
Electrical validator for validating electrical properties.
"""

from typing import Dict, Any, List
from .validation_result import ValidationResult


class ElectricalValidator:
    """
    Validates electrical properties in the system.
    """
    
    def __init__(self, validation_config=None):
        self.validation_config = validation_config
    
    def validate(self, blocks: Dict[str, Any], connections: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate electrical properties.
        
        Args:
            blocks: Dictionary of block objects
            connections: List of connection definitions
        
        Returns:
            ValidationResult containing validation outcome
        """
        result = ValidationResult()
        
        try:
            # Basic electrical validation
            for block_name, block in blocks.items():
                if hasattr(block, 'parameters'):
                    parameters = block.parameters
                    
                    # Check for common electrical parameters
                    if 'voltage' in parameters:
                        voltage = parameters['voltage']
                        if isinstance(voltage, (int, float)) and voltage < 0:
                            result.add_error(f"Block {block_name} has negative voltage: {voltage}")
                    
                    if 'power' in parameters:
                        power = parameters['power']
                        if isinstance(power, (int, float)) and power < 0:
                            result.add_warning(f"Block {block_name} has negative power: {power}")
        
        except Exception as e:
            result.add_error(f"Error validating electrical properties: {e}")
        
        return result 