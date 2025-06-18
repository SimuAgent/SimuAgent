"""
Block configuration management with improved caching and error handling.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class BlockConfiguration:
    """
    Manages block configuration data with improved caching and error handling.
    """
    
    def __init__(self, config_filepath: str):
        self.config_filepath = Path(config_filepath)
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file with proper error handling."""
        try:
            if not self.config_filepath.exists():
                raise ConfigurationError(f"Configuration file not found: {self.config_filepath}")
            
            with open(self.config_filepath, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
                
            logger.info(f"Successfully loaded block configuration from: {self.config_filepath}")
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file {self.config_filepath}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {self.config_filepath}: {e}")

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration data."""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        return self._config

    @lru_cache(maxsize=128)
    def get_block_template(self, block_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the template for a given block type with caching.
        
        Args:
            block_type: The type of block to get template for
            
        Returns:
            Block template dictionary or None if not found
        """
        return self.config.get(block_type)

    def get_valid_block_types(self) -> List[str]:
        """Returns a list of all valid block types."""
        return list(self.config.keys())

    @lru_cache(maxsize=256)
    def get_port_definition(self, block_type: str, port_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the definition for a specific port of a block type with caching.
        
        Args:
            block_type: The type of block
            port_name: The name of the port
            
        Returns:
            Port definition dictionary or None if not found
        """
        block_template = self.get_block_template(block_type)
        if block_template and 'Ports' in block_template:
            return block_template['Ports'].get(port_name)
        return None

    @lru_cache(maxsize=128)
    def get_block_parameters(self, block_type: str) -> Dict[str, Any]:
        """
        Get the basic parameters for a block type with caching.
        
        Args:
            block_type: The type of block
            
        Returns:
            Dictionary of basic parameters
        """
        template = self.get_block_template(block_type)
        if template and 'Basic Parameters' in template:
            return template['Basic Parameters'].copy()
        return {}

    @lru_cache(maxsize=128)
    def get_block_ports(self, block_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get the ports definition for a block type with caching.
        
        Args:
            block_type: The type of block
            
        Returns:
            Dictionary of port definitions
        """
        template = self.get_block_template(block_type)
        if template and 'Ports' in template:
            return template['Ports'].copy()
        return {}

    def validate_block_type(self, block_type: str) -> bool:
        """
        Check if a block type is valid.
        
        Args:
            block_type: The type to validate
            
        Returns:
            True if valid, False otherwise
        """
        return block_type in self.config

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.get_block_template.cache_clear()
        self.get_port_definition.cache_clear()
        self.get_block_parameters.cache_clear()
        self.get_block_ports.cache_clear()

    def reload_config(self) -> None:
        """Reload configuration from file and clear cache."""
        self.clear_cache()
        self._load_config()

    def get_config_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded configuration."""
        total_types = len(self.config)
        total_ports = sum(
            len(template.get('Ports', {})) 
            for template in self.config.values()
        )
        
        return {
            'total_block_types': total_types,
            'total_ports': total_ports,
            'config_file': str(self.config_filepath),
            'cache_stats': {
                'get_block_template': self.get_block_template.cache_info(),
                'get_port_definition': self.get_port_definition.cache_info(),
                'get_block_parameters': self.get_block_parameters.cache_info(),
                'get_block_ports': self.get_block_ports.cache_info(),
            }
        }

    def get_required_parameters(self, block_type: str) -> List[str]:
        """Get required parameters for a block type."""
        template = self.get_block_template(block_type)
        return template.get('required_parameters', []) 