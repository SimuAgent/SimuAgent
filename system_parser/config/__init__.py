"""
Configuration package for system parser.

This package handles loading and managing configuration data
for block types, parameters, and validation rules.
"""

from .block_config import BlockConfiguration
from .validation_config import ValidationConfiguration

__all__ = [
    'BlockConfiguration',
    'ValidationConfiguration',
] 