"""
Parameter validator for validating system parameters.
"""

from typing import Dict, Any
from .validation_result import ValidationResult


class ParameterValidator:
    """
    Validates parameters in the system.
    """
    
    def __init__(self, validation_config=None):
        self.validation_config = validation_config
    
    def validate(self, blocks_data: Dict[str, Any], block_config=None) -> ValidationResult:
        """
        Validate system parameters.
        
        Args:
            blocks_data: Dictionary of block data
            block_config: Block configuration object
        
        Returns:
            ValidationResult containing validation outcome
        """
        result = ValidationResult()
        
        for block_name, block_data in blocks_data.items():
            try:
                parameters = block_data.get('parameters', {})
                
                # Basic parameter validation
                if not isinstance(parameters, dict):
                    result.add_error(f"Block {block_name} parameters must be a dictionary")
                    continue
                
                # Check for required parameters (if config is available)
                if block_config and hasattr(block_config, 'get_required_parameters'):
                    block_type = block_data.get('type', 'Unknown')
                    required_params = block_config.get_required_parameters(block_type)
                    
                    for param in required_params:
                        if param not in parameters:
                            result.add_error(f"Block {block_name} missing required parameter: {param}")
                
            except Exception as e:
                result.add_error(f"Error validating parameters for block {block_name}: {e}")
        
        return result 