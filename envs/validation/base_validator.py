"""
Base validator class for the validation system.

This module provides the abstract base class for all validators,
ensuring consistent interfaces and behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .validation_result import ValidationResult
from .config import ValidationConfig


class BaseValidator(ABC):
    """
    Abstract base class for all validators.
    
    This class defines the common interface that all validators
    must implement, ensuring consistency across the validation system.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the validator with configuration.
        
        Args:
            config: ValidationConfig instance. If None, creates default configuration.
        """
        self.config = config or ValidationConfig()
    
    @abstractmethod
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate the given data.
        
        Args:
            data: Data to validate
            context: Optional context information for validation
            
        Returns:
            ValidationResult indicating success or failure with details
        """
        pass
    
    def _create_success_result(self, message: Optional[str] = None) -> ValidationResult:
        """Create a successful validation result."""
        return ValidationResult.success(message)
    
    def _create_failure_result(self, message: str, **kwargs) -> ValidationResult:
        """Create a failed validation result."""
        return ValidationResult.failure(message, **kwargs) 