"""
Refactored SystemGraph class with improved modularity and organization.
"""

import logging
from typing import Dict, List, Any, Optional
import networkx as nx

from ..core.models import Block, UnconnectedPort
from ..core.exceptions import ValidationError, ConfigurationError
from ..config.block_config import BlockConfiguration
from ..config.validation_config import ValidationConfiguration
from ..validators import (
    ConnectionValidator, ParameterValidator, 
    DistanceValidator, ElectricalValidator,
    ValidationResult
)
from .graph_builder import GraphBuilder
from .connectivity_analyzer import ConnectivityAnalyzer
from .graph_visualizer import GraphVisualizer

logger = logging.getLogger(__name__)


class SystemGraph:
    """
    Refactored SystemGraph with improved modularity and separation of concerns.
    
    This class orchestrates the building and validation of a power system graph
    by delegating responsibilities to specialized components.
    """
    
    def __init__(
        self,
        system_dict: Dict[str, Any],
        block_config_instance: Optional[BlockConfiguration] = None,
        validation_config: Optional[ValidationConfiguration] = None,
        validate_param_keys: bool = False,
        distances: Optional[Dict[Any, float]] = None,
    ):
        self.system_dict = system_dict
        self.distances = distances or {}
        self.validate_param_keys = validate_param_keys
        
        # Configuration
        self.block_config = block_config_instance or self._load_default_config()
        self.validation_config = validation_config or ValidationConfiguration()
        
        # Core components
        self.graph: nx.DiGraph = nx.DiGraph(name="System Graph")
        self.blocks: Dict[str, Block] = {}
        
        # Specialized components
        self._graph_builder = GraphBuilder(self.block_config)
        self._connectivity_analyzer = ConnectivityAnalyzer()
        self._graph_visualizer = GraphVisualizer()
        
        # Validators
        self._connection_validator = ConnectionValidator(self.validation_config)
        self._parameter_validator = ParameterValidator(self.validation_config)
        self._distance_validator = DistanceValidator(self.validation_config)
        self._electrical_validator = ElectricalValidator(self.validation_config)
        
        # Validation results
        self._validation_result = ValidationResult(is_valid=True)
        
        # Build the graph
        self._build_system()
    
    def _load_default_config(self) -> BlockConfiguration:
        """Load default block configuration."""
        try:
            config_file = "system_parser/block_config.json"
            return BlockConfiguration(config_file)
        except Exception as e:
            raise ConfigurationError(f"Failed to load default configuration: {e}")
    
    def _build_system(self) -> None:
        """Build the complete system graph with validation."""
        try:
            # Validate configuration
            if not self.block_config or not self.block_config.config:
                raise ConfigurationError("Block configuration is required")
            
            # Build blocks
            self._build_blocks()
            
            # Build graph structure
            self._build_graph_structure()
            
            # Perform validations
            self._validate_system()
            
        except Exception as e:
            logger.error(f"Failed to build system: {e}")
            self._validation_result.add_error(f"System build failed: {e}")
    
    def _build_blocks(self) -> None:
        """Build all blocks in the system."""
        blocks_data = self.system_dict.get("Blocks", {})
        
        if not blocks_data:
            self._validation_result.add_error("No blocks found in system definition")
            return
        
        # Use graph builder to create blocks
        try:
            self.blocks = self._graph_builder.build_blocks(
                blocks_data, 
                validate_param_keys=self.validate_param_keys
            )
            
            # Collect parameter validation issues
            for block in self.blocks.values():
                for issue in block.param_validation_issues:
                    self._validation_result.add_error(str(issue))
                    
        except Exception as e:
            self._validation_result.add_error(f"Failed to build blocks: {e}")
    
    def _build_graph_structure(self) -> None:
        """Build the NetworkX graph structure."""
        try:
            # Use graph builder to create the graph
            self.graph = self._graph_builder.build_graph(
                self.blocks, 
                self.system_dict.get("Connections", [])
            )
            
        except Exception as e:
            self._validation_result.add_error(f"Failed to build graph structure: {e}")
    
    def _validate_system(self) -> None:
        """Perform comprehensive system validation."""
        connections = self.system_dict.get("Connections", [])
        
        # Validate connections
        conn_result = self._connection_validator.validate(connections, self.blocks)
        self._validation_result.merge(conn_result)
        
        # Validate parameters
        param_result = self._parameter_validator.validate(
            self.system_dict.get("Blocks", {}), 
            self.block_config
        )
        self._validation_result.merge(param_result)
        
        # Validate distances if available
        if self.distances:
            dist_result = self._distance_validator.validate(
                connections, self.blocks, self.distances
            )
            self._validation_result.merge(dist_result)
        
        # Validate electrical properties
        elec_result = self._electrical_validator.validate(self.blocks, connections)
        self._validation_result.merge(elec_result)
    
    def get_validation_report(
        self,
        max_error_items: int = 5,
        max_warning_items: int = 5,
        include_unconnected_summary: bool = True,
        include_connectivity_analysis: bool = True,
        include_param_validation: bool = True,
    ) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            max_error_items: Maximum number of errors to include
            max_warning_items: Maximum number of warnings to include
            include_unconnected_summary: Include unconnected ports summary
            include_connectivity_analysis: Include connectivity analysis
            include_param_validation: Include parameter validation details
            
        Returns:
            Formatted validation report string
        """
        report_lines = []
        
        # Summary
        summary = self._validation_result.get_summary()
        report_lines.append("=== SYSTEM VALIDATION REPORT ===")
        report_lines.append(f"Overall Status: {'VALID' if self._validation_result.is_valid else 'INVALID'}")
        report_lines.append(f"Errors: {summary['errors']}, Warnings: {summary['warnings']}, Info: {summary['infos']}")
        report_lines.append("")
        
        # Errors
        if self._validation_result.errors:
            report_lines.append("ERRORS:")
            for i, error in enumerate(self._validation_result.errors[:max_error_items]):
                report_lines.append(f"  {i+1}. {error}")
            if len(self._validation_result.errors) > max_error_items:
                report_lines.append(f"  ... and {len(self._validation_result.errors) - max_error_items} more errors")
            report_lines.append("")
        
        # Warnings
        if self._validation_result.warnings:
            report_lines.append("WARNINGS:")
            for i, warning in enumerate(self._validation_result.warnings[:max_warning_items]):
                report_lines.append(f"  {i+1}. {warning}")
            if len(self._validation_result.warnings) > max_warning_items:
                report_lines.append(f"  ... and {len(self._validation_result.warnings) - max_warning_items} more warnings")
            report_lines.append("")
        
        # Additional analysis
        if include_connectivity_analysis:
            try:
                connectivity_result = self.analyze_connectivity()
                report_lines.append("CONNECTIVITY ANALYSIS:")
                report_lines.append(f"  Generators: {connectivity_result.total_generators}")
                report_lines.append(f"  Loads: {connectivity_result.total_loads}")
                report_lines.append(f"  Connected Loads: {connectivity_result.connected_loads}")
                report_lines.append(f"  Connectivity Ratio: {connectivity_result.connectivity_ratio:.2%}")
                report_lines.append("")
            except Exception as e:
                report_lines.append(f"Connectivity analysis failed: {e}")
                report_lines.append("")
        
        if include_unconnected_summary:
            unconnected = self.get_unconnected_ports()
            if unconnected:
                report_lines.append(f"UNCONNECTED PORTS ({len(unconnected)}):")
                for port in unconnected[:5]:  # Show first 5
                    report_lines.append(f"  - {port}")
                if len(unconnected) > 5:
                    report_lines.append(f"  ... and {len(unconnected) - 5} more")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def get_unconnected_ports(self) -> List[UnconnectedPort]:
        """Get list of unconnected ports in the system."""
        return self._graph_builder.get_unconnected_ports(self.graph, self.blocks)
    
    def analyze_connectivity(self):
        """Analyze connectivity between generators and loads."""
        return self._connectivity_analyzer.analyze_connectivity(
            self.graph, self.blocks
        )
    
    def draw_graph(self, filepath: Optional[str] = None, **kwargs) -> Optional[str]:
        """Draw the system graph using the graph visualizer."""
        return self._graph_visualizer.draw_graph(
            self.graph, self.blocks, filepath=filepath, **kwargs
        )
    
    def get_reward(self) -> float:
        """Calculate system reward based on validation results."""
        if not self._validation_result.is_valid:
            return 0.0
        
        # Base reward for valid system
        base_reward = 1.0
        
        # Penalty for warnings
        warning_penalty = len(self._validation_result.warnings) * 0.1
        
        # Connectivity bonus
        try:
            connectivity = self.analyze_connectivity()
            connectivity_bonus = connectivity.connectivity_ratio * 0.5
        except Exception:
            connectivity_bonus = 0.0
        
        return max(0.0, base_reward - warning_penalty + connectivity_bonus)
    
    @property
    def is_valid(self) -> bool:
        """Check if the system is valid."""
        return self._validation_result.is_valid
    
    @property
    def validation_errors(self) -> List[str]:
        """Get validation error messages."""
        return [str(error) for error in self._validation_result.errors]
    
    @property
    def validation_warnings(self) -> List[str]:
        """Get validation warning messages."""
        return [str(warning) for warning in self._validation_result.warnings] 