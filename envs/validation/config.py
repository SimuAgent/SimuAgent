"""
Configuration system for validation components.

This module provides centralized configuration management for all
validation-related settings and behaviors.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ValidationConfig:
    """
    Configuration settings for the validation system.
    
    This class centralizes all validation-related configuration,
    providing easy customization and consistent behavior across
    validation components.
    """
    
    # File paths
    config_base_path: Optional[str] = None
    config_subdir: str = 'system_parser'
    allowed_params_filename: str = 'allowed_block_parameters.json'
    block_config_filename: str = 'block_config.json'
    
    # Validation behavior settings
    strict_parameter_validation: bool = True
    suggest_similar_types: bool = True
    suggest_similar_parameters: bool = True
    max_suggestions: int = 6
    suggestion_cutoff: float = 0.4
    validate_connections: bool = True
    validate_port_counts: bool = True
    strict_length_validation: bool = False
    
    # Message length limits
    max_error_message_length: int = 500
    max_execution_log_length: int = 300
    max_validation_reason_length: int = 400
    max_total_log_length: int = 2000  # Total accumulated length of all log messages
    truncation_suffix: str = "..."
    
    # Performance settings
    cache_configuration: bool = True
    max_cache_size: int = 100
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.config_base_path is None:
            # Default to parent directory of the envs package
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.config_base_path = os.path.dirname(current_dir)
    
    @property
    def allowed_params_path(self) -> str:
        """Full path to allowed parameters configuration file."""
        return os.path.join(
            self.config_base_path, 
            self.config_subdir, 
            self.allowed_params_filename
        )
    
    @property
    def block_config_path(self) -> str:
        """Full path to block configuration file.""" 
        return os.path.join(
            self.config_base_path,
            self.config_subdir,
            self.block_config_filename
        )
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a validation setting value.
        
        Args:
            key: Setting key to retrieve
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        return getattr(self, key, default)
    
    def update_settings(self, **kwargs) -> None:
        """
        Update multiple validation settings.
        
        Args:
            **kwargs: Settings to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_lenient_mode(self) -> None:
        """Configure for lenient validation mode."""
        self.update_settings(
            strict_parameter_validation=False,
            strict_length_validation=False,
            suggest_similar_types=True,
            suggest_similar_parameters=True,
            max_suggestions=3
        )
    
    def set_strict_mode(self) -> None:
        """Configure for strict validation mode."""
        self.update_settings(
            strict_parameter_validation=True,
            strict_length_validation=True,
            suggest_similar_types=False,
            suggest_similar_parameters=False,
            max_suggestions=1
        )
    
    def set_development_mode(self) -> None:
        """Configure for development-friendly validation."""
        self.update_settings(
            strict_parameter_validation=False,
            strict_length_validation=False,
            suggest_similar_types=True,
            suggest_similar_parameters=True,
            max_suggestions=6,
            validate_connections=True,
            validate_port_counts=False
        )
    
    def set_production_mode(self) -> None:
        """Configure for production validation."""
        self.update_settings(
            strict_parameter_validation=True,
            strict_length_validation=True,
            suggest_similar_types=False,
            suggest_similar_parameters=True,
            max_suggestions=3,
            validate_connections=True,
            validate_port_counts=True
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            'config_base_path': self.config_base_path,
            'config_subdir': self.config_subdir,
            'allowed_params_filename': self.allowed_params_filename,
            'block_config_filename': self.block_config_filename,
            'strict_parameter_validation': self.strict_parameter_validation,
            'suggest_similar_types': self.suggest_similar_types,
            'suggest_similar_parameters': self.suggest_similar_parameters,
            'max_suggestions': self.max_suggestions,
            'suggestion_cutoff': self.suggestion_cutoff,
            'validate_connections': self.validate_connections,
            'validate_port_counts': self.validate_port_counts,
            'strict_length_validation': self.strict_length_validation,
            'max_error_message_length': self.max_error_message_length,
            'max_execution_log_length': self.max_execution_log_length,
            'max_validation_reason_length': self.max_validation_reason_length,
            'max_total_log_length': self.max_total_log_length,
            'truncation_suffix': self.truncation_suffix,
            'cache_configuration': self.cache_configuration,
            'max_cache_size': self.max_cache_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ValidationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def validate_paths(self) -> bool:
        """
        Validate that configuration file paths exist.
        
        Returns:
            True if all required paths exist
        """
        paths_to_check = [
            self.allowed_params_path,
            self.block_config_path
        ]
        
        return all(os.path.exists(path) for path in paths_to_check)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"ValidationConfig(\n"
            f"  allowed_params_path='{self.allowed_params_path}',\n"
            f"  block_config_path='{self.block_config_path}',\n"
            f"  strict_parameter_validation={self.strict_parameter_validation},\n"
            f"  strict_length_validation={self.strict_length_validation}\n"
            f")"
        ) 