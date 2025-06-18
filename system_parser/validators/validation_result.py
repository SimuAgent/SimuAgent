"""
Validation result classes for storing validation outcomes.
"""

from typing import List, Dict, Any


class ValidationResult:
    """
    Stores validation results including errors, warnings, and info messages.
    """
    
    def __init__(self, is_valid: bool = True):
        self.is_valid = is_valid
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.infos: List[str] = []
        self.details: Dict[str, Any] = {}
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.infos.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.infos.extend(other.infos)
        self.details.update(other.details)
        
        if not other.is_valid:
            self.is_valid = False
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of validation results."""
        return {
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'infos': len(self.infos)
        }
    
    def __str__(self) -> str:
        """String representation of validation result."""
        summary = self.get_summary()
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, errors={summary['errors']}, warnings={summary['warnings']}, infos={summary['infos']})" 