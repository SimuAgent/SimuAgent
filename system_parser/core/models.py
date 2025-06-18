"""
Core data models for power system components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import difflib

from .exceptions import ParameterError


@dataclass
class Port:
    """Represents a port on a block."""
    name: str
    type: str
    block_name: str

    @property
    def global_id(self) -> str:
        """Unique ID for networkx node."""
        return f"{self.block_name}_{self.name}"

    def __str__(self) -> str:
        return f"{self.block_name}.{self.name} ({self.type})"


@dataclass
class Connection:
    """Represents a connection between two ports."""
    from_block: str
    from_port: str
    to_block: str
    to_port: str
    
    @property
    def from_port_id(self) -> str:
        """Global ID of the source port."""
        return f"{self.from_block}_{self.from_port}"
    
    @property
    def to_port_id(self) -> str:
        """Global ID of the destination port."""
        return f"{self.to_block}_{self.to_port}"
    
    def __str__(self) -> str:
        return f"{self.from_block}.{self.from_port} -> {self.to_block}.{self.to_port}"


@dataclass
class UnconnectedPort:
    """Represents an unconnected port with its details."""
    block_name: str
    port_name: str
    port_type: str
    global_id: str

    def __str__(self) -> str:
        return f"{self.block_name}.{self.port_name} ({self.port_type})"


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


class ParameterValidator:
    """Handles parameter validation and suggestions."""
    
    @staticmethod
    def find_closest_parameter_key(
        invalid_key: str, 
        valid_keys: List[str], 
        min_similarity: float = 0.4
    ) -> Optional[Tuple[str, float]]:
        """Find the closest matching parameter key using fuzzy string matching."""
        if not valid_keys:
            return None
        
        best_match = None
        best_score = 0.0
        
        for valid_key in valid_keys:
            # Strategy 1: Overall sequence similarity
            seq_score = difflib.SequenceMatcher(None, invalid_key.lower(), valid_key.lower()).ratio()
            
            # Strategy 2: Check if invalid key is contained in valid key (partial match)
            if invalid_key.lower() in valid_key.lower():
                partial_score = len(invalid_key) / len(valid_key) * 0.8
            else:
                partial_score = 0
            
            # Strategy 3: Check for common words
            invalid_words = set(invalid_key.lower().replace('_', ' ').replace('-', ' ').split())
            valid_words = set(valid_key.lower().replace('(', ' ').replace(')', ' ').replace('-', ' ').split())
            if invalid_words and valid_words:
                common_words = invalid_words.intersection(valid_words)
                word_score = len(common_words) / len(invalid_words) * 0.7
            else:
                word_score = 0
            
            # Take the maximum of all strategies
            final_score = max(seq_score, partial_score, word_score)
            
            if final_score > best_score:
                best_score = final_score
                best_match = valid_key
        
        if best_score >= min_similarity:
            return (best_match, best_score)
        
        return None


class Block:
    """Represents a system block with its parameters and ports."""
    
    def __init__(
        self, 
        name: str, 
        block_type: str, 
        input_params: Dict[str, Any], 
        block_config_instance,
        validate_param_keys: bool = False
    ) -> None:
        self.name = name
        self.type = block_type
        self.params: Dict[str, Any] = {}
        self.ports: Dict[str, Port] = {}
        self.param_validation_issues: List[ParameterValidationIssue] = []

        template = block_config_instance.get_block_template(block_type)
        if not template:
            raise ParameterError(f"Unknown block type: '{block_type}' for block '{name}'")

        # Load default parameters from template
        if "Basic Parameters" in template:
            self.params.update(template["Basic Parameters"])

        # Override with parameters from input (excluding "Type")
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

    def _validate_parameter_keys(self, input_params: Dict[str, Any], valid_params: Dict[str, Any]) -> None:
        """Validate that parameter keys in input_params exist in valid_params."""
        valid_keys = list(valid_params.keys())
        
        for param_key in input_params.keys():
            if param_key not in valid_keys:
                # Find a suggested alternative
                suggestion_result = ParameterValidator.find_closest_parameter_key(param_key, valid_keys)
                
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