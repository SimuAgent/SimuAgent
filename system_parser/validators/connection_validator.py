"""
Connection validator for validating system connections.
"""

from typing import Dict, Any, List
from .validation_result import ValidationResult


class ConnectionValidator:
    """
    Validates connections in the system.
    """
    
    def __init__(self, validation_config=None):
        self.validation_config = validation_config
    
    def validate(self, connections: List[Dict[str, Any]], blocks: Dict[str, Any]) -> ValidationResult:
        """
        Validate system connections.
        
        Args:
            connections: List of connection definitions
            blocks: Dictionary of block objects
        
        Returns:
            ValidationResult containing validation outcome
        """
        result = ValidationResult()
        
        for connection in connections:
            try:
                from_block = connection.get('from_block')
                to_block = connection.get('to_block')
                
                # Basic validation
                if not from_block:
                    result.add_error("Connection missing 'from_block'")
                if not to_block:
                    result.add_error("Connection missing 'to_block'")
                    
                # Check if referenced blocks exist
                if from_block and from_block not in blocks:
                    result.add_error(f"Connection references non-existent block: {from_block}")
                if to_block and to_block not in blocks:
                    result.add_error(f"Connection references non-existent block: {to_block}")
                    
            except Exception as e:
                result.add_error(f"Error validating connection: {e}")
        
        return result 