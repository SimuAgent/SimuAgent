"""
DEPRECATED: This file has been refactored into system_parser.core.models

Please update your imports:
    OLD: from system_parser.system_components import Block, Port
    NEW: from system_parser.core.models import Block, Port
    OR:  from system_parser import Block, Port

This file will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "system_parser.system_components is deprecated. "
    "Use system_parser.core.models or import from system_parser directly. "
    "This module will be removed in version 3.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from .core.models import (
    Block, Port, ParameterValidationIssue, UnconnectedPort,
    ParameterValidator
)

__all__ = [
    'Block', 'Port', 'ParameterValidationIssue', 'UnconnectedPort',
    'ParameterValidator', 'find_closest_parameter_key'
]

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import difflib


@dataclass
class Port:
    """
    Represents a port on a block.
    """
    name: str  # e.g., 'A', 'B', 'a'
    type: str  # e.g., 'Conserving'
    block_name: str  # e.g., 'Gen1'
    
    @property
    def global_id(self) -> str:
        """Unique ID for networkx node."""
        return f"{self.block_name}_{self.name}"

    def __str__(self) -> str:
        return f"{self.block_name}.{self.name} ({self.type})"


@dataclass
class ParameterValidationIssue:
    """Represents a parameter validation issue."""
    block_name: str
    block_type: str
    invalid_key: str
    suggested_key: Optional[str] = None
    similarity_score: float = 0.0

    def __str__(self) -> str:
        suggestion_text = f" (suggested: '{self.suggested_key}')" if self.suggested_key else ""
        return f"Block '{self.block_name}' ({self.block_type}): invalid parameter key '{self.invalid_key}'{suggestion_text}"


def find_closest_parameter_key(invalid_key: str, valid_keys: List[str], min_similarity: float = 0.4) -> Optional[Tuple[str, float]]:
    """
    Find the closest matching parameter key using fuzzy string matching.
    
    Args:
        invalid_key: The invalid parameter key
        valid_keys: List of valid parameter keys from block config
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
    
    Returns:
        Tuple of (suggested_key, similarity_score) if a good match is found, None otherwise
    """
    return ParameterValidator.find_closest_parameter_key(invalid_key, valid_keys, min_similarity)


class Block:
    """
    Represents a system block with its parameters and ports.
    """
    def __init__(self, name: str, block_type: str, input_params: Dict[str, Any], block_config_instance, validate_param_keys: bool = False) -> None:
        self.name = name
        self.type = block_type
        self.params: Dict[str, Any] = {}  # Final parameters for the block
        self.ports: Dict[str, Port] = {}   # Dict of Port objects, keyed by port name
        self.param_validation_issues: List[ParameterValidationIssue] = []  # Track parameter validation issues

        template = block_config_instance.get_block_template(block_type)
        if not template:
            raise ValueError(f"Unknown block type: '{block_type}' for block '{name}'. It's not defined in block_config.json.")

        # Load default parameters from template
        if "Basic Parameters" in template:
            self.params.update(template["Basic Parameters"])

        # Override with parameters from system_input.json
        # Remove "Type" from input_params before merging
        specific_params = {k: v for k, v in input_params.items() if k != "Type"}
        
        # Validate parameter keys if requested
        if validate_param_keys and specific_params:
            self._validate_parameter_keys(specific_params, template.get("Basic Parameters", {}))
        
        self.params.update(specific_params)

        # Create ports from template
        if "Ports" in template:
            for port_name, port_info in template["Ports"].items():
                port_type = port_info.get("Type", "Unknown")
                self.ports[port_name] = Port(
                    name=port_name,
                    type=port_type,
                    block_name=self.name
                )
        # If a block type in config genuinely has no ports, this will remain empty, which is fine.

    def _validate_parameter_keys(self, input_params: Dict[str, Any], valid_params: Dict[str, Any]) -> None:
        """
        Validate that parameter keys in input_params exist in valid_params.
        Store validation issues for later reporting.
        """
        valid_keys = list(valid_params.keys())
        
        for param_key in input_params.keys():
            if param_key not in valid_keys:
                # Find a suggested alternative
                suggestion_result = find_closest_parameter_key(param_key, valid_keys)
                
                issue = ParameterValidationIssue(
                    block_name=self.name,
                    block_type=self.type,
                    invalid_key=param_key
                )
                
                if suggestion_result:
                    issue.suggested_key = suggestion_result[0]
                    issue.similarity_score = suggestion_result[1]
                
                self.param_validation_issues.append(issue)

    def get_port(self, port_name: str) -> Optional[Port]:
        """Retrieves a port by its name."""
        return self.ports.get(port_name)
    
    @property
    def port_count(self) -> int:
        """Number of ports in this block."""
        return len(self.ports)
    
    @property
    def param_count(self) -> int:
        """Number of parameters in this block."""
        return len(self.params)

    def __str__(self) -> str:
        return f"Block(Name: {self.name}, Type: {self.type}, Params: {self.param_count}, Ports: {self.port_count})"
    
    def __repr__(self) -> str:
        return self.__str__()