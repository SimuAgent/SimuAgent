"""
Custom exceptions for the system parser module.
"""


class SystemParserError(Exception):
    """Base exception class for system parser errors."""
    pass


class ValidationError(SystemParserError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(SystemParserError):
    """Raised when configuration is invalid or missing."""
    pass


class ConnectionError(ValidationError):
    """Raised when connection validation fails."""
    pass


class ParameterError(ValidationError):
    """Raised when parameter validation fails."""
    pass 