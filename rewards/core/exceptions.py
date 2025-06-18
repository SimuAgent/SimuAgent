"""
Exception classes for the reward system.

This module defines custom exceptions used throughout the reward system
to provide clear error handling and debugging information.
"""

from typing import Any, Dict, List, Optional


class RewardError(Exception):
    """Base exception for reward system errors."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        component: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.component = component
    
    def __str__(self) -> str:
        base_msg = self.message
        if self.component:
            base_msg = f"[{self.component}] {base_msg}"
        
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg = f"{base_msg} (Details: {details_str})"
        
        return base_msg


class EvaluationError(RewardError):
    """Error that occurs during reward evaluation."""
    
    def __init__(
        self,
        message: str,
        reward_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if context:
            details['context_keys'] = list(context.keys())
        
        super().__init__(
            message, 
            details=details,
            component=reward_type,
            **kwargs
        )
        self.reward_type = reward_type
        self.context = context


class ConfigurationError(RewardError):
    """Error in reward system configuration."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[type] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if expected_type:
            details['expected_type'] = expected_type.__name__
        if actual_value is not None:
            details['actual_value'] = str(actual_value)
            details['actual_type'] = type(actual_value).__name__
        
        super().__init__(
            message,
            details=details,
            component=config_key,
            **kwargs
        )
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value


class ValidationError(RewardError):
    """Error in data validation."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        field_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if validation_errors:
            details['validation_errors'] = validation_errors
        
        super().__init__(
            message,
            details=details,
            component=field_name,
            **kwargs
        )
        self.validation_errors = validation_errors or []
        self.field_name = field_name


class CalculationError(RewardError):
    """Error during reward calculation."""
    
    def __init__(
        self,
        message: str,
        calculation_step: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if input_data:
            details['input_data_keys'] = list(input_data.keys())
        
        super().__init__(
            message,
            details=details,
            component=calculation_step,
            **kwargs
        )
        self.calculation_step = calculation_step
        self.input_data = input_data


class TimeoutError(RewardError):
    """Error when evaluation times out."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if timeout_duration:
            details['timeout_duration'] = timeout_duration
        
        super().__init__(
            message,
            details=details,
            component=operation,
            **kwargs
        )
        self.timeout_duration = timeout_duration
        self.operation = operation


class ComponentNotFoundError(RewardError):
    """Error when a required reward component is not found."""
    
    def __init__(
        self,
        message: str,
        component_type: Optional[str] = None,
        available_components: Optional[List[str]] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if available_components:
            details['available_components'] = available_components
        
        super().__init__(
            message,
            details=details,
            component=component_type,
            **kwargs
        )
        self.component_type = component_type
        self.available_components = available_components or []


class DataError(RewardError):
    """Error related to data processing or format."""
    
    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        expected_format: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if expected_format:
            details['expected_format'] = expected_format
        
        super().__init__(
            message,
            details=details,
            component=data_source,
            **kwargs
        )
        self.data_source = data_source
        self.expected_format = expected_format 