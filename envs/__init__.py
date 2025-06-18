"""
Environment Package - Refactored and Modular.

A clean, object-oriented package for environment management, validation,
and tool-based interactions with improved modularity and maintainability.

Example usage:
    from envs import ToolEnvironment, SystemValidator
    
    # Create environment with tools
    env = ToolEnvironment(
        tools=[my_tool1, my_tool2],
        system_prompt="You are a helpful assistant..."
    )
    
    # Validate system configurations
    validator = SystemValidator()
    result = validator.validate_system_dict(old_dict, new_dict)
    
    if result.valid:
        print("System update is valid!")
    else:
        print(f"Validation failed: {result.reason}")
"""

# Core interfaces and base classes
from .core.interfaces import (
    BaseEnvironment,
    ConversationState,
    EnvironmentState,
    ToolResult,
    ValidationResult as CoreValidationResult
)

# Core components
from .core.tool_manager import ToolManager
from .core.execution_engine import ExecutionEngine, ExecutionResult
from .core.state_manager import StateManager

# Validation system
from .validation.validation_result import ValidationResult, ValidationIssue, ValidationSeverity
from .validation.config import ValidationConfig
from .validation.system_validator import SystemValidator

# Environment implementations
from .environments.tool_environment import ToolEnvironment

# Legacy compatibility - import old classes and create aliases
try:
    from .tool_env import ToolEnv
    # Create aliases for backward compatibility
    ToolEnvironment_Legacy = ToolEnv
except ImportError:
    # If old file doesn't exist, use new implementation
    ToolEnvironment_Legacy = ToolEnvironment
    ToolEnv = ToolEnvironment

# For backward compatibility with old ValidationConfig location
try:
    from .validation_config import ValidationConfig as LegacyValidationConfig
except ImportError:
    LegacyValidationConfig = ValidationConfig

__version__ = "2.0.0"
__author__ = "Environment Package Team"

__all__ = [
    # Core interfaces
    'BaseEnvironment',
    'ConversationState', 
    'EnvironmentState',
    'ToolResult',
    
    # Core components
    'ToolManager',
    'ExecutionEngine',
    'ExecutionResult',
    'StateManager',
    
    # Validation system
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity',
    'ValidationConfig',
    'SystemValidator',
    
    # Environment implementations
    'ToolEnvironment',
    
    # Backward compatibility
    'ToolEnv',
    'ToolEnvironment_Legacy',
    'LegacyValidationConfig',
]

# Deprecation warnings for old imports
import warnings

def __getattr__(name):
    """Handle deprecated imports with warnings."""
    if name == "ToolEnv":
        warnings.warn(
            "ToolEnv is deprecated. Use ToolEnvironment instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return ToolEnvironment
    
    if name in ["ValidationConfig"] and hasattr(LegacyValidationConfig, name):
        warnings.warn(
            f"Importing {name} from envs root is deprecated. "
            f"Use 'from envs.validation import {name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return getattr(LegacyValidationConfig, name)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
