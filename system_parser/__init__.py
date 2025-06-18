"""
System Parser Package - System Graph Analysis and Validation

A comprehensive, object-oriented library for parsing, validating, and analyzing
system configurations with improved modularity and maintainability.

Key Features:
- Modular system graph representation
- Comprehensive validation framework
- Electrical parameter analysis
- Connectivity and topology analysis
- Pandapower integration
- Visualization capabilities

Architecture:
- core/: Base models, interfaces, and exceptions
- graph/: Graph construction and analysis
- validation/: Validation components and rules
- analysis/: Analysis tools (connectivity, electrical, etc.)
- config/: Configuration management
- visualization/: Graph rendering and plotting
- pandapower/: Pandapower converter integration

Example Usage:
    from system_parser import SystemGraph, GraphBuilder, SystemValidator
    
    # Create and validate system
    builder = GraphBuilder(config_file="config/block_config.json")
    system = builder.build_from_dict(system_dict)
    
    # Validate system
    validator = SystemValidator()
    validation_result = validator.validate(system)
    
    if validation_result.is_valid:
        print("System is valid!")
        # Perform analysis
        connectivity = system.analyze_connectivity()
        print(f"Connectivity ratio: {connectivity.connectivity_ratio:.2%}")
    else:
        print(f"Validation failed: {validation_result.summary}")
"""

# Core models and interfaces
from .core.models import Block, Port, Connection, UnconnectedPort
from .core.interfaces import SystemAnalyzer, GraphBuilder as IGraphBuilder
from .core.exceptions import SystemParserError, ValidationError, ConfigurationError

# Main system graph implementation
from .graph.system_graph import SystemGraph

# Legacy system graph (available as fallback)
from .system_graph import SystemGraph as LegacySystemGraph

__version__ = "2.0.0"
__author__ = "System Parser Team"

# For backward compatibility, expose the main classes at the module level
PowerSystemGraph = SystemGraph  # Backward compatibility alias

__all__ = [
    # Core models
    'Block',
    'Port', 
    'Connection',
    'UnconnectedPort',
    
    # Interfaces
    'SystemAnalyzer',
    'IGraphBuilder',
    
    # Exceptions
    'SystemParserError',
    'ValidationError',
    'ConfigurationError',
    
    # Main classes
    'SystemGraph',
    'PowerSystemGraph',  # Legacy alias
    'LegacySystemGraph',
]

# Module configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())