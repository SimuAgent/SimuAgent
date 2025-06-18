from typing import Dict, Any, Optional
import os


class ValidationConfig:
    """
    Configuration helper for system validation.
    
    This class centralizes validation configuration settings and provides
    easy customization options for different use cases.
    """
    
    def __init__(self, 
                 config_base_path: Optional[str] = None,
                 allowed_params_filename: str = 'allowed_block_parameters.json',
                 block_config_filename: str = 'block_config.json',
                 config_subdir: str = 'system_parser'):
        """
        Initialize validation configuration.
        
        Args:
            config_base_path: Base path for configuration files. If None, uses relative path.
            allowed_params_filename: Name of the allowed parameters JSON file
            block_config_filename: Name of the block configuration JSON file
            config_subdir: Subdirectory containing config files
        """
        if config_base_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_base_path = os.path.dirname(current_dir)
        
        self.config_base_path = config_base_path
        self.config_subdir = config_subdir
        self.allowed_params_filename = allowed_params_filename
        self.block_config_filename = block_config_filename
        
        # Validation settings
        self.validation_settings = {
            'strict_parameter_validation': True,
            'suggest_similar_types': True,
            'suggest_similar_parameters': True,
            'max_suggestions': 6,
            'suggestion_cutoff': 0.4,  # For difflib matching
            'validate_connections': True,
            'validate_port_counts': True,
            'strict_length_validation': False,  # Default is False - only check if block has length capability, not actual length value
        }
    
    @property
    def allowed_params_path(self) -> str:
        """Full path to allowed parameters configuration file."""
        return os.path.join(self.config_base_path, self.config_subdir, self.allowed_params_filename)
    
    @property
    def block_config_path(self) -> str:
        """Full path to block configuration file."""
        return os.path.join(self.config_base_path, self.config_subdir, self.block_config_filename)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a validation setting value."""
        return self.validation_settings.get(key, default)
    
    def update_settings(self, **kwargs) -> None:
        """Update validation settings."""
        self.validation_settings.update(kwargs)
    
    def set_lenient_mode(self) -> None:
        """Configure for lenient validation mode."""
        self.update_settings(
            strict_parameter_validation=False,
            suggest_similar_types=True,
            suggest_similar_parameters=True,
            max_suggestions=3
        )
    
    def set_strict_mode(self) -> None:
        """Configure for strict validation mode."""
        self.update_settings(
            strict_parameter_validation=True,
            suggest_similar_types=False,
            suggest_similar_parameters=False,
            max_suggestions=1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            'config_base_path': self.config_base_path,
            'config_subdir': self.config_subdir,
            'allowed_params_filename': self.allowed_params_filename,
            'block_config_filename': self.block_config_filename,
            'validation_settings': self.validation_settings.copy()
        } 