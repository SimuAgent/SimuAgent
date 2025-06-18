"""
DEPRECATED: This file has been refactored into system_parser.config.block_config

Please update your imports:
    OLD: from system_parser.block_config_loader import BlockConfig
    NEW: from system_parser.config.block_config import BlockConfiguration
    OR:  from system_parser import BlockConfiguration

This file will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "system_parser.block_config_loader is deprecated. "
    "Use system_parser.config.block_config.BlockConfiguration or import from system_parser directly. "
    "This module will be removed in version 3.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
try:
    from .config.block_config import BlockConfiguration as BlockConfig
except ImportError:
    # Fallback if new structure not available
    import json

    class BlockConfig:
        """
        DEPRECATED: Legacy BlockConfig class.
        Use BlockConfiguration instead.
        """
        def __init__(self, config_filepath):
            warnings.warn(
                "BlockConfig is deprecated. Use BlockConfiguration instead.",
                DeprecationWarning,
                stacklevel=2
            )
            try:
                with open(config_filepath, 'r') as f:
                    self._config = json.load(f)
            except FileNotFoundError:
                print(f"Error: Block configuration file not found at {config_filepath}")
                self._config = {}
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {config_filepath}")
                self._config = {}

        def get_block_template(self, block_type):
            """Retrieves the template for a given block type."""
            return self._config.get(block_type)

        def get_valid_block_types(self):
            """Returns a list of all valid block types."""
            return list(self._config.keys())

        def get_port_definition(self, block_type, port_name):
            """Retrieves the definition for a specific port of a block type."""
            block_template = self.get_block_template(block_type)
            if block_template and 'Ports' in block_template:
                return block_template['Ports'].get(port_name)
            return None

__all__ = ['BlockConfig']
    