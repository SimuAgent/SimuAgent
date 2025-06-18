import ast
import json
import os
import traceback
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional
import difflib

from .config import ValidationConfig
from .validation_result import ValidationResult, ValidationSeverity
from .base_validator import BaseValidator


class SystemValidator(BaseValidator):
    """
    Handles validation and execution of system configuration changes.
    
    This class provides a clean interface for:
    - Executing Python code line by line with validation
    - Validating system_dict changes against configuration rules
    - Validating connections between blocks
    - Providing helpful suggestions for invalid configurations
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the validator with configuration.
        
        Args:
            config: ValidationConfig instance. If None, creates default configuration.
        """
        super().__init__(config)
        
        self.allowed_params_path = self.config.allowed_params_path
        self.block_config_path = self.config.block_config_path
        
        # Cache configuration data
        self._allowed_params = None
        self._block_config = None
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate the given data.
        
        Args:
            data: Data to validate (typically system_dict)
            context: Optional context information for validation
            
        Returns:
            ValidationResult indicating success or failure with details
        """
        # For now, we'll just return success since this is a wrapper
        # The actual validation is done through the specific methods
        return self._create_success_result("General validation passed")
    
    def _truncate_message(self, message: str, max_length: Optional[int] = None) -> str:
        """
        Truncate a message to the specified maximum length.
        
        Args:
            message: Message to truncate
            max_length: Maximum length. If None, uses config setting
            
        Returns:
            Truncated message with suffix if needed
        """
        if max_length is None:
            max_length = self.config.get_setting('max_validation_reason_length', 400)
        
        if len(message) <= max_length:
            return message
        
        suffix = self.config.get_setting('truncation_suffix', '...')
        truncated_length = max_length - len(suffix)
        
        if truncated_length <= 0:
            return suffix
        
        return message[:truncated_length] + suffix
    
    @property
    def allowed_params(self) -> Dict[str, Any]:
        """Lazy load allowed parameters configuration."""
        if self._allowed_params is None:
            with open(self.allowed_params_path, 'r') as f:
                self._allowed_params = json.load(f)
        return self._allowed_params
    
    @property
    def block_config(self) -> Dict[str, Any]:
        """Lazy load block configuration."""
        if self._block_config is None:
            with open(self.block_config_path, 'r') as f:
                self._block_config = json.load(f)
        return self._block_config
    
    def execute_and_validate_code(
        self,
        python_code: str,
        gen_globals: Dict[str, Any],
        distances: Dict[str, Any],
        execution_log: List[str]
    ) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        """
        Execute Python code line by line, validating each step and correcting errors.
        
        Args:
            python_code: Python code to execute
            gen_globals: Global variables dictionary (will be modified in place)
            distances: Distance constraints dictionary
            execution_log: List to append execution logs to
            
        Returns:
            tuple: (corrected_lines, final_system_dict, final_distances)
        """
        corrected_lines = []
        current_system_dict = deepcopy(gen_globals.get("system_dict", {}))
        current_distances = distances.copy()
        
        # Parse the code into statements
        try:
            tree = ast.parse(python_code)
        except SyntaxError as e:
            execution_log.append(f"Syntax error in code: {e}")
            return corrected_lines, current_system_dict, current_distances
        
        for node in tree.body:
            try:
                # Convert the AST node back to source code
                line_code = ast.unparse(node)
                
                # Execute the line
                exec(line_code, gen_globals)
                
                # Get updated system_dict
                updated_system_dict = gen_globals.get("system_dict", {})
                
                # Validate the line if it modifies system_dict
                if updated_system_dict != current_system_dict:
                    # Always validate system_dict changes
                    validation_result = self.validate_system_dict_change(
                        current_system_dict, updated_system_dict, current_distances, line_code
                    )
                    
                    if validation_result["valid"]:
                        current_system_dict = deepcopy(updated_system_dict)
                        corrected_lines.append(line_code)
                        execution_log.append(f"✓ Executed: {line_code}")
                    else:
                        # Revert the change - this is critical! Use deepcopy to avoid object aliasing
                        gen_globals["system_dict"] = deepcopy(current_system_dict)
                        execution_log.append(f"✗ Failed: {line_code}. {validation_result['reason']}")
                else:
                    # Line didn't change system_dict, so it's safe to execute
                    corrected_lines.append(line_code)
                    execution_log.append(f"✓ Executed: {line_code}")
                    
            except Exception as e:
                line_code_str = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                execution_log.append(f"Failed: {line_code_str}. Error: {str(e)}")
                continue
        
        return corrected_lines, current_system_dict, current_distances
    
    def validate_system_dict_change(
        self,
        old_system_dict: Dict[str, Any],
        new_system_dict: Dict[str, Any],
        distances: Dict[str, Any],
        line_code: str
    ) -> Dict[str, Any]:
        """
        Validate a change to system_dict against distance constraints and other rules.
        
        Args:
            old_system_dict: Previous state of system_dict
            new_system_dict: New state of system_dict
            distances: Distance constraints
            line_code: The code line that caused the change
            
        Returns:
            dict: {"valid": bool, "reason": str}
        """
        try:
            # First, do explicit validation for blocks that changed
            if "Blocks" in new_system_dict:
                for block_name, block_data in new_system_dict["Blocks"].items():
                    old_block_data = old_system_dict.get("Blocks", {}).get(block_name, {})
                    
                    # Check if this is a new block or a modified block
                    if block_data != old_block_data:
                        validation_result = self._validate_block_data(block_name, block_data)
                        if not validation_result["valid"]:
                            return validation_result

            # Validate connections changes
            if "Connections" in new_system_dict:
                old_connections = old_system_dict.get("Connections", [])
                new_connections = new_system_dict.get("Connections", [])
                
                # Check if connections have changed
                if new_connections != old_connections:
                    # Get the current blocks (after any block changes)
                    current_blocks = new_system_dict.get("Blocks", {})
                    
                    # Check for duplicate connections first
                    duplicate_check = self._check_duplicate_connections(new_connections)
                    if not duplicate_check["valid"]:
                        return duplicate_check
                    
                    # Validate each connection
                    for connection in new_connections:
                        if connection not in old_connections:  # This is a new connection
                            validation_result = self.validate_connection(connection, current_blocks)
                            if not validation_result["valid"]:
                                return validation_result
                    
                    # Validate distance constraints for all connections
                    distance_validation = self._validate_distance_constraints(new_connections, current_blocks, distances)
                    if not distance_validation["valid"]:
                        return distance_validation

            return {"valid": True, "reason": ""}
            
        except Exception as e:
            error_details = traceback.format_exc()
            return {"valid": False, "reason": f"Validation failed: {str(e)}. Details: {error_details}"}
    
    def _validate_block_data(self, block_name: str, block_data: Any) -> Dict[str, Any]:
        """Validate a single block's data structure and parameters."""
        if not isinstance(block_data, dict):
            return {
                "valid": False,
                "reason": f"Block '{block_name}' data must be a dictionary"
            }
        
        block_type = block_data.get("Type")
        if not block_type:
            return {
                "valid": False,
                "reason": f"Block '{block_name}' is missing 'Type' field"
            }
        
        # Check block type validity FIRST
        if block_type not in self.allowed_params:
            if self.config.get_setting('suggest_similar_types', True):
                suggestions = self.get_block_type_suggestions(block_type, self.allowed_params.keys())
                max_suggestions = self.config.get_setting('max_suggestions', 6)
                if suggestions:
                    suggestion_text = f"Did you mean: {', '.join(suggestions[:max_suggestions])}?"
                else:
                    available_types = sorted(self.allowed_params.keys())
                    suggestion_text = f"Available types: {', '.join(available_types[:max_suggestions])}"
            else:
                suggestion_text = "Please check available block types in configuration."
            
            error_msg = f"Block type '{block_type}' for block '{block_name}' not found in configuration. {suggestion_text}"
            return {
                "valid": False,
                "reason": error_msg
            }
        
        # Check parameter keys
        allowed_keys = set(self.allowed_params[block_type]["allowed_keys"])
        invalid_keys = []
        for key in block_data.keys():
            if key not in allowed_keys:
                invalid_keys.append(key)
        
        if invalid_keys:
            # Get suggestions for invalid keys
            suggestions = []
            if self.config.get_setting('suggest_similar_parameters', True):
                for invalid_key in invalid_keys:
                    suggestion = self.get_parameter_key_suggestion(invalid_key, allowed_keys)
                    if suggestion:
                        suggestions.append(f"'{invalid_key}' → '{suggestion}'")
                    else:
                        suggestions.append(f"'{invalid_key}' (invalid)")
            else:
                suggestions = [f"'{key}' (invalid)" for key in invalid_keys]
            
            allowed_list = sorted([k for k in allowed_keys if k != "Type"])
            if allowed_list:
                allowed_keys_str = "', '".join(allowed_list)
                error_msg = (f"Block '{block_name}' (type '{block_type}') has invalid parameter keys: {', '.join(suggestions)}. "
                            f"Allowed keys: 'Type', '{allowed_keys_str}'")
            else:
                error_msg = (f"Block '{block_name}' (type '{block_type}') has invalid parameter keys: {', '.join(suggestions)}. "
                            f"Only 'Type' is allowed for this block type")
            
            return {"valid": False, "reason": error_msg}
        
        return {"valid": True, "reason": ""}
    
    def validate_connection(self, connection: List[str], current_blocks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single connection [source, target] format.
        
        Args:
            connection: Connection in format ["source_block/(port1,port2)", "target_block/(port1,port2)"]
            current_blocks: Dictionary of current blocks
            
        Returns:
            dict: {"valid": bool, "reason": str}
        """
        try:
            if not isinstance(connection, list) or len(connection) != 2:
                return {"valid": False, "reason": f"Connection must be a list of exactly 2 elements, got: {connection}"}
            
            source_spec, target_spec = connection
            
            # Parse and validate both source and target, collecting port counts
            source_ports = []
            target_ports = []
            
            # Validate source and target
            for spec, spec_type, port_list in [(source_spec, "source", source_ports), (target_spec, "target", target_ports)]:
                if not isinstance(spec, str) or "/" not in spec:
                    return {"valid": False, "reason": f"Connection {spec_type} must be in format 'block_name/(port1,port2,...)' but got: {spec}"}
                
                block_name, port_spec = spec.split("/", 1)
                
                # Check if block exists
                if block_name not in current_blocks:
                    return {"valid": False, "reason": f"Connection {spec_type} block '{block_name}' does not exist"}
                
                # Get block type and validate ports
                block_data = current_blocks[block_name]
                block_type = block_data.get("Type")
                
                if block_type not in self.block_config:
                    return {"valid": False, "reason": f"Block type '{block_type}' not found in block configuration"}
                
                # Get valid ports for this block type
                valid_ports = set(self.block_config[block_type].get("Ports", {}).keys())
                
                # Parse port specification (remove parentheses and split by comma)
                if not port_spec.startswith("(") or not port_spec.endswith(")"):
                    return {"valid": False, "reason": f"Port specification must be in format '(port1,port2,...)' but got: {port_spec}"}
                
                port_list_str = port_spec[1:-1]  # Remove parentheses
                if not port_list_str:
                    return {"valid": False, "reason": f"Empty port specification: {port_spec}"}
                
                specified_ports = [p.strip() for p in port_list_str.split(",")]
                
                # Check each specified port
                for port in specified_ports:
                    if port not in valid_ports:
                        valid_ports_str = "', '".join(sorted(valid_ports))
                        return {"valid": False, "reason": f"Port '{port}' does not exist on block '{block_name}' (type '{block_type}'). Valid ports: '{valid_ports_str}'"}
                
                # Store ports for count validation
                port_list.extend(specified_ports)
            
            # Validate port count match between source and target
            if len(source_ports) != len(target_ports):
                return {"valid": False, "reason": f"Port count mismatch: source has {len(source_ports)} ports {source_ports} but target has {len(target_ports)} ports {target_ports}"}
            
            return {"valid": True, "reason": ""}
            
        except Exception as e:
            return {"valid": False, "reason": f"Error validating connection {connection}: {str(e)}"}
    
    def get_block_type_suggestions(self, invalid_type: str, available_types) -> List[str]:
        """
        Get suggestions for similar block types using simple string matching.
        
        Args:
            invalid_type: The invalid block type entered
            available_types: Iterable of available block types
            
        Returns:
            List of suggested block types
        """
        suggestions = []
        invalid_lower = invalid_type.lower()
        
        # Special case handling for common mistakes
        if invalid_lower == 'three-winding transformer':
            # Return transformer types + other related blocks
            transformer_suggestions = [t for t in available_types if 'transformer' in t.lower()]
            other_suggestions = [t for t in available_types if any(word in t.lower() for word in ['source', 'load']) and 'transformer' not in t.lower()]
            return transformer_suggestions + other_suggestions[:3]
        elif invalid_lower == 'three-phase transformer':
            return ["Three-Phase Transformer (Two Windings)", "Three-Phase Transformer (Three Windings)"]
        
        # Simple matching: find types that contain similar words
        invalid_words = set(invalid_type.lower().split())
        
        for available_type in available_types:
            available_words = set(available_type.lower().split())
            # Check for word overlap - require at least 1 common word
            if invalid_words.intersection(available_words):
                suggestions.append(available_type)
        
        # Sort by relevance: exact word matches first, then by length
        def relevance_score(suggestion):
            suggestion_words = set(suggestion.lower().split())
            common_words = invalid_words.intersection(suggestion_words)
            # More common words = higher relevance (lower score for sorting)
            return -len(common_words), len(suggestion)
        
        suggestions.sort(key=relevance_score)
        return suggestions[:3]  # Return top suggestions for better user experience
    
    def get_parameter_key_suggestion(self, invalid_key: str, allowed_keys: set) -> str:
        """
        Get a suggestion for an invalid parameter key using string matching and common patterns.
        
        Args:
            invalid_key: The invalid parameter key
            allowed_keys: Set of allowed parameter keys
            
        Returns:
            Best suggestion or empty string if no good match
        """
        invalid_lower = invalid_key.lower()
        
        # Special case mappings for common mistakes
        common_mappings = {
            'distance (km)': 'Line length (km)',
            'length (km)': 'Line length (km)', 
            'line distance (km)': 'Line length (km)',
            'distance': 'Line length (km)',
            'length': 'Line length (km)',
            'frequency (hz)': 'Frequency used for rlc specification (Hz)',
            'frequency': 'Frequency used for rlc specification (Hz)',
            'freq (hz)': 'Frequency used for rlc specification (Hz)',
            'nominal power (va)': 'Nominal power and frequency [Pn(VA), fn(Hz)]',
            'power (va)': 'Nominal power and frequency [Pn(VA), fn(Hz)]',
            'nominal power': 'Nominal power and frequency [Pn(VA), fn(Hz)]',
            'voltage (vrms)': 'Phase-to-phase voltage (Vrms)',
            'voltage': 'Phase-to-phase voltage (Vrms)',
            'primary voltage': 'Winding 1 parameters [V1 Ph-Ph(Vrms), R1(pu), L1(pu)]',
            'secondary voltage': 'Winding 2 parameters [V2 Ph-Ph(Vrms), R2(pu), L2(pu)]',
        }
        
        # Check for direct mappings
        for pattern, suggestion in common_mappings.items():
            if pattern in invalid_lower and suggestion in allowed_keys:
                return suggestion
        
        # Use difflib to find the closest match
        cutoff = self.config.get_setting('suggestion_cutoff', 0.4)
        matches = difflib.get_close_matches(invalid_key, allowed_keys, n=1, cutoff=cutoff)
        return matches[0] if matches else ""
    
    def _check_duplicate_connections(self, connections: List[List[str]]) -> Dict[str, Any]:
        """
        Check for duplicate connections between the same two ports (regardless of direction).
        
        Args:
            connections: List of connections in format [["source_block/(port1,port2)", "target_block/(port1,port2)"], ...]
            
        Returns:
            dict: {"valid": bool, "reason": str}
        """
        try:
            # Track all port pairs that have been connected
            connected_port_pairs = set()
            
            for i, connection in enumerate(connections):
                if not isinstance(connection, list) or len(connection) != 2:
                    continue  # This will be caught by individual connection validation
                
                source_spec, target_spec = connection
                
                # Parse the port specifications to get individual port connections
                try:
                    # Parse source
                    if "/" not in source_spec:
                        continue  # Will be caught by individual validation
                    src_block, src_port_spec = source_spec.split("/", 1)
                    if not src_port_spec.startswith("(") or not src_port_spec.endswith(")"):
                        continue  # Will be caught by individual validation
                    src_ports = [p.strip() for p in src_port_spec[1:-1].split(",")]
                    
                    # Parse target
                    if "/" not in target_spec:
                        continue  # Will be caught by individual validation
                    tgt_block, tgt_port_spec = target_spec.split("/", 1)
                    if not tgt_port_spec.startswith("(") or not tgt_port_spec.endswith(")"):
                        continue  # Will be caught by individual validation
                    tgt_ports = [p.strip() for p in tgt_port_spec[1:-1].split(",")]
                    
                    # Check each port-to-port connection
                    if len(src_ports) == len(tgt_ports):
                        for src_port, tgt_port in zip(src_ports, tgt_ports):
                            # Create normalized port identifiers
                            port1 = f"{src_block}/{src_port}"
                            port2 = f"{tgt_block}/{tgt_port}"
                            
                            # Create a normalized connection (lexicographically sorted to be direction-independent)
                            normalized_connection = tuple(sorted([port1, port2]))
                            
                            # Check if this port pair is already connected
                            if normalized_connection in connected_port_pairs:
                                return {
                                    "valid": False,
                                    "reason": f"Duplicate connection detected: ports {port1} and {port2} are already connected. Each pair of ports can only be connected once, regardless of direction."
                                }
                            
                            connected_port_pairs.add(normalized_connection)
                            
                except Exception:
                    continue  # Skip malformed connections, they'll be caught by individual validation
            
            return {"valid": True, "reason": ""}
            
        except Exception as e:
            return {"valid": False, "reason": f"Error checking for duplicate connections: {str(e)}"}
    
    def _validate_distance_constraints(
        self, 
        connections: List[List[str]], 
        current_blocks: Dict[str, Any], 
        distances: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that connections respect distance constraints.
        
        Components cannot be connected to different locations unless they have physical length
        and their ports are on different ends.
        
        Args:
            connections: List of connections
            current_blocks: Dictionary of current blocks
            distances: Distance constraints between component clusters
            
        Returns:
            dict: {"valid": bool, "reason": str}
        """
        try:
            if not distances:
                return {"valid": True, "reason": ""}
            
            # Create a mapping from blocks to their locations (clusters)
            block_to_cluster = {}
            cluster_distances = {}
            
            # Process distances to extract block locations
            for cluster_pair, distance_km in distances.items():
                if isinstance(cluster_pair, tuple) and len(cluster_pair) == 2:
                    cluster1, cluster2 = cluster_pair
                    
                    # Each cluster is a tuple of block names at the same location
                    if isinstance(cluster1, tuple):
                        for block_name in cluster1:
                            block_to_cluster[block_name] = cluster1
                    if isinstance(cluster2, tuple):
                        for block_name in cluster2:
                            block_to_cluster[block_name] = cluster2
                    
                    # Store distance between clusters
                    cluster_distances[(cluster1, cluster2)] = distance_km
                    cluster_distances[(cluster2, cluster1)] = distance_km  # Make it bidirectional
            
            # First check direct connections between distant blocks
            for connection in connections:
                if not isinstance(connection, list) or len(connection) != 2:
                    continue
                
                source_spec, target_spec = connection
                
                # Parse block names and ports
                if "/" not in source_spec or "/" not in target_spec:
                    continue
                
                source_block, source_port_spec = source_spec.split("/", 1)
                target_block, target_port_spec = target_spec.split("/", 1)
                
                # Skip if blocks don't exist
                if source_block not in current_blocks or target_block not in current_blocks:
                    continue
                
                # Get block locations
                source_cluster = block_to_cluster.get(source_block)
                target_cluster = block_to_cluster.get(target_block)
                
                # Check if blocks are in different clusters (different locations)
                if source_cluster and target_cluster and source_cluster != target_cluster:
                    # Get the distance between clusters
                    distance_between = cluster_distances.get((source_cluster, target_cluster), 
                                                           cluster_distances.get((target_cluster, source_cluster), None))
                    
                    if distance_between is not None and distance_between > 0:
                        # Check if either block can bridge this distance
                        source_block_type = current_blocks.get(source_block, {}).get("Type", "")
                        target_block_type = current_blocks.get(target_block, {}).get("Type", "")
                        
                        # Check if this is a valid transmission line connection
                        if source_block_type == "Three-Phase PI Section Line":
                            source_can_bridge = self._can_block_bridge_distance(source_block, current_blocks, distance_between, source_port_spec, target_cluster)
                            if not source_can_bridge:
                                return {
                                    "valid": False,
                                    "reason": f"Transmission line '{source_block}' cannot bridge the distance between its connected components. Increase the line length parameter."
                                }
                        elif target_block_type == "Three-Phase PI Section Line":
                            target_can_bridge = self._can_block_bridge_distance(target_block, current_blocks, distance_between, target_port_spec, source_cluster)
                            if not target_can_bridge:
                                return {
                                    "valid": False,
                                    "reason": f"Transmission line '{target_block}' cannot bridge the distance between its connected components. Increase the line length parameter."
                                }
                        else:
                            # Neither block is a transmission line - direct connection not allowed
                            return {
                                "valid": False,
                                "reason": f"'{source_block}' (type '{source_block_type}') and '{target_block}' (type '{target_block_type}') are not at the same location and cannot be directly connected. Use a transmission line with sufficient length to bridge the distance between different locations."
                            }
            
            # Analyze all connections to detect constraint violations
            # Create a comprehensive view of which ports connect to which clusters
            port_to_cluster = {}
            
            for connection in connections:
                if not isinstance(connection, list) or len(connection) != 2:
                    continue
                
                source_spec, target_spec = connection
                
                # Parse block names and ports
                if "/" not in source_spec or "/" not in target_spec:
                    continue
                
                source_block, source_port_spec = source_spec.split("/", 1)
                target_block, target_port_spec = target_spec.split("/", 1)
                
                # Skip if blocks don't exist
                if source_block not in current_blocks or target_block not in current_blocks:
                    continue
                
                # Parse individual ports from the specifications
                def parse_ports(port_spec):
                    if not port_spec.startswith("(") or not port_spec.endswith(")"):
                        return []
                    port_list_str = port_spec[1:-1]
                    return [p.strip() for p in port_list_str.split(",") if p.strip()]
                
                source_ports = parse_ports(source_port_spec)
                target_ports = parse_ports(target_port_spec)
                
                # Map each port to its cluster
                source_cluster = block_to_cluster.get(source_block)
                target_cluster = block_to_cluster.get(target_block)
                
                # Record port connections
                for src_port in source_ports:
                    port_id = f"{source_block}/{src_port}"
                    if port_id not in port_to_cluster:
                        port_to_cluster[port_id] = set()
                    if source_cluster:
                        port_to_cluster[port_id].add(source_cluster)
                    if target_cluster:
                        port_to_cluster[port_id].add(target_cluster)
                
                for tgt_port in target_ports:
                    port_id = f"{target_block}/{tgt_port}"
                    if port_id not in port_to_cluster:
                        port_to_cluster[port_id] = set()
                    if source_cluster:
                        port_to_cluster[port_id].add(source_cluster)
                    if target_cluster:
                        port_to_cluster[port_id].add(target_cluster)
            
            # Check for transmission line port direction constraints
            port_direction_validation = self._validate_transmission_line_port_directions(connections, current_blocks, block_to_cluster)
            if not port_direction_validation["valid"]:
                return port_direction_validation

            # Special check for transmission lines: ensure they don't have both ends connected to the same block
            for connection in connections:
                if not isinstance(connection, list) or len(connection) != 2:
                    continue
                    
                source_spec, target_spec = connection
                
                # Parse block names and ports
                if "/" not in source_spec or "/" not in target_spec:
                    continue
                
                source_block, source_port_spec = source_spec.split("/", 1)
                target_block, target_port_spec = target_spec.split("/", 1)
                
                # Check if this is a transmission line with invalid connections
                source_block_type = current_blocks.get(source_block, {}).get("Type", "")
                target_block_type = current_blocks.get(target_block, {}).get("Type", "")
                
                if source_block_type == "Three-Phase PI Section Line":
                    # Check if both ends of the transmission line connect to the same block
                    line_connections = self._get_transmission_line_connections(source_block, connections)
                    invalid_connection = self._check_transmission_line_same_block_connection(source_block, line_connections)
                    if invalid_connection:
                        return {
                            "valid": False,
                            "reason": f"Transmission line '{source_block}' has both ends connected to the same block '{invalid_connection}'. Transmission lines must connect different locations."
                        }
                
                if target_block_type == "Three-Phase PI Section Line":
                    # Check if both ends of the transmission line connect to the same block
                    line_connections = self._get_transmission_line_connections(target_block, connections)
                    invalid_connection = self._check_transmission_line_same_block_connection(target_block, line_connections)
                    if invalid_connection:
                        return {
                            "valid": False,
                            "reason": f"Transmission line '{target_block}' has both ends connected to the same block '{invalid_connection}'. Transmission lines must connect different locations."
                        }

            # Now check each block to see if it violates distance constraints
            for block_name, block_data in current_blocks.items():
                block_type = block_data.get("Type", "")
                
                # Get all ports of this block that are connected
                block_ports = []
                for port_id, clusters in port_to_cluster.items():
                    if port_id.startswith(f"{block_name}/"):
                        port_name = port_id.split("/", 1)[1]
                        block_ports.append((port_name, clusters))
                
                if not block_ports:
                    continue
                
                # Check if this block connects to multiple distant locations
                all_clusters = set()
                for port_name, clusters in block_ports:
                    all_clusters.update(clusters)
                
                if len(all_clusters) > 1:
                    # Block connects to multiple clusters - check if this is valid
                    if block_type == "Three-Phase PI Section Line":
                        # Transmission lines can connect to multiple clusters, but only if ports are on different ends
                        validation_result = self._validate_transmission_line_endpoints(
                            block_name, block_data, block_ports, all_clusters, cluster_distances
                        )
                        if not validation_result["valid"]:
                            return validation_result
                    else:
                        # Other blocks cannot span multiple distant locations
                        cluster_list = list(all_clusters)
                        distances_between = []
                        for i in range(len(cluster_list)):
                            for j in range(i + 1, len(cluster_list)):
                                dist = cluster_distances.get((cluster_list[i], cluster_list[j]), 0)
                                if dist > 0:
                                    distances_between.append(dist)
                        
                        if distances_between:
                            return {
                                "valid": False,
                                "reason": f"Block '{block_name}' (type '{block_type}') cannot connect to components at different locations. Only transmission lines with sufficient length can bridge distances between different locations."
                            }
            
            return {"valid": True, "reason": ""}
            
        except Exception as e:
            return {"valid": False, "reason": f"Error validating distance constraints: {str(e)}"}
    
    def _can_block_bridge_distance(
        self,
        block_name: str,
        current_blocks: Dict[str, Any],
        required_distance: float,
        port_spec: str,
        target_cluster: tuple
    ) -> bool:
        """
        Check if a block can bridge the required distance.
        
        Args:
            block_name: Name of the block
            current_blocks: Dictionary of current blocks
            required_distance: Required distance to bridge in km
            port_spec: Port specification for this connection
            target_cluster: Target cluster this block is connecting to
            
        Returns:
            bool: True if block can bridge the distance
        """
        try:
            block_data = current_blocks.get(block_name, {})
            block_type = block_data.get("Type", "")
            
            # Check if block has configurable physical length
            block_config = self.block_config.get(block_type, {})
            physical_length_config = block_config.get("Physical_Length", {})
            
            if physical_length_config.get("Type") != "configurable":
                return False
            
            # If strict length validation is disabled, just check if block has length capability
            if not self.config.get_setting('strict_length_validation', False):
                # Only verify that this is a transmission line with proper port usage
                if block_type == "Three-Phase PI Section Line":
                    # Parse the ports to see if they're on the correct ends
                    if not port_spec.startswith("(") or not port_spec.endswith(")"):
                        return False
                    
                    port_list_str = port_spec[1:-1]
                    specified_ports = [p.strip() for p in port_list_str.split(",")]
                    
                    # Get port positions from block config
                    ports_config = block_config.get("Ports", {})
                    
                    # Check if all specified ports are on the same side (Left or Right)
                    port_positions = set()
                    for port in specified_ports:
                        if port in ports_config:
                            position = ports_config[port].get("Position")
                            if position:
                                port_positions.add(position)
                    
                    # If all ports are on the same side, this connection can work
                    return len(port_positions) == 1
                return True
            
            # Strict mode: check actual length
            # Get the block's configured length
            length_param = physical_length_config.get("Parameter")
            if not length_param:
                return False
            
            block_length = 0.0
            if length_param in block_data:
                try:
                    length_value = block_data[length_param]
                    if isinstance(length_value, (int, float)):
                        block_length = float(length_value)
                    elif isinstance(length_value, str):
                        block_length = float(length_value)
                except (ValueError, TypeError):
                    block_length = physical_length_config.get("Default", 0.0)
            else:
                block_length = physical_length_config.get("Default", 0.0)
            
            # Check if block length is sufficient
            if block_length < required_distance:
                return False
            
            # For blocks with physical length (like transmission lines), 
            # check if ports are on different ends
            if block_type == "Three-Phase PI Section Line":
                # Parse the ports to see if they're on the correct ends
                if not port_spec.startswith("(") or not port_spec.endswith(")"):
                    return False
                
                port_list_str = port_spec[1:-1]
                specified_ports = [p.strip() for p in port_list_str.split(",")]
                
                # Get port positions from block config
                ports_config = block_config.get("Ports", {})
                
                # Check if all specified ports are on the same side (Left or Right)
                port_positions = set()
                for port in specified_ports:
                    if port in ports_config:
                        position = ports_config[port].get("Position")
                        if position:
                            port_positions.add(position)
                
                # If all ports are on the same side, this connection can work
                # (the other end of the transmission line can connect to the target cluster)
                return len(port_positions) == 1
            
            return True
            
        except Exception:
            return False
    
    def _validate_transmission_line_endpoints(
        self,
        block_name: str,
        block_data: Dict[str, Any],
        block_ports: List[Tuple[str, set]],
        all_clusters: set,
        cluster_distances: Dict[Tuple, float]
    ) -> Dict[str, Any]:
        """
        Validate that transmission line endpoints connect to different locations properly.
        
        Args:
            block_name: Name of the transmission line block
            block_data: Block data dictionary
            block_ports: List of (port_name, clusters) tuples
            all_clusters: Set of all clusters this block connects to
            cluster_distances: Distance mapping between clusters
            
        Returns:
            dict: {"valid": bool, "reason": str}
        """
        try:
            block_type = block_data.get("Type", "")
            
            if block_type != "Three-Phase PI Section Line":
                return {"valid": True, "reason": ""}  # Not a transmission line
            
            # Get the transmission line's configured length
            line_length = 0.0
            length_param = "Line length (km)"
            if length_param in block_data:
                try:
                    length_value = block_data[length_param]
                    if isinstance(length_value, (int, float)):
                        line_length = float(length_value)
                    elif isinstance(length_value, str):
                        line_length = float(length_value)
                except (ValueError, TypeError):
                    line_length = 100.0  # Default value
            else:
                line_length = 100.0  # Default value
            
            # Get port positions from block config
            block_config = self.block_config.get(block_type, {})
            ports_config = block_config.get("Ports", {})
            
            # Group ports by their position (Left or Right)
            left_ports = set()
            right_ports = set()
            
            for port_name, clusters in block_ports:
                if port_name in ports_config:
                    position = ports_config[port_name].get("Position")
                    if position == "Left":
                        left_ports.update(clusters)
                    elif position == "Right":
                        right_ports.update(clusters)
            
            # Check if line connects to multiple clusters
            if len(all_clusters) > 1:
                # Only check actual length in strict mode
                if self.config.get_setting('strict_length_validation', False):
                    # Calculate the maximum distance the line needs to bridge
                    cluster_list = list(all_clusters)
                    max_distance = 0.0
                    for i in range(len(cluster_list)):
                        for j in range(i + 1, len(cluster_list)):
                            dist = cluster_distances.get((cluster_list[i], cluster_list[j]), 0)
                            max_distance = max(max_distance, dist)
                    
                    # Check if line length is sufficient
                    if line_length < max_distance:
                        return {
                            "valid": False,
                            "reason": f"Transmission line '{block_name}' is not long enough to bridge the distance between its connected components. Increase the line length parameter."
                        }
                
                # Check if left and right sides connect to different locations
                if left_ports and right_ports:
                    # If both sides have connections, they should connect to different clusters
                    if left_ports == right_ports:
                        return {
                            "valid": False,
                            "reason": f"Transmission line '{block_name}' has both ends (A,B,C and a,b,c ports) connected to the same location. Transmission lines must connect different locations."
                        }
                else:
                    # Only one side is connected - this might be valid for an incomplete system
                    pass
            else:
                # Line connects to only one cluster - check if both ends are connected to the same cluster
                # This is invalid for transmission lines since they should span distances
                left_connected = any(port_name in ['A', 'B', 'C'] for port_name, _ in block_ports)
                right_connected = any(port_name in ['a', 'b', 'c'] for port_name, _ in block_ports)
                
                if left_connected and right_connected:
                    # Both ends connected to the same single cluster - this is invalid
                    cluster_name = list(all_clusters)[0] if all_clusters else "unknown"
                    return {
                        "valid": False,
                        "reason": f"Transmission line '{block_name}' has both ends (A,B,C and a,b,c ports) connected to the same location. Transmission lines must connect different locations."
                    }
            
            return {"valid": True, "reason": ""}
            
        except Exception as e:
            return {"valid": False, "reason": f"Error validating transmission line endpoints: {str(e)}"}
    
    def _get_transmission_line_connections(self, line_name: str, connections: List[List[str]]) -> List[Tuple[str, str, str]]:
        """
        Get all connections involving a transmission line.
        
        Args:
            line_name: Name of the transmission line
            connections: List of all connections
            
        Returns:
            List of (connected_block, line_ports, block_ports) tuples
        """
        line_connections = []
        
        for connection in connections:
            if not isinstance(connection, list) or len(connection) != 2:
                continue
            
            source_spec, target_spec = connection
            
            if "/" not in source_spec or "/" not in target_spec:
                continue
            
            source_block, source_port_spec = source_spec.split("/", 1)
            target_block, target_port_spec = target_spec.split("/", 1)
            
            if source_block == line_name:
                line_connections.append((target_block, source_port_spec, target_port_spec))
            elif target_block == line_name:
                line_connections.append((source_block, target_port_spec, source_port_spec))
        
        return line_connections
    
    def _check_transmission_line_same_block_connection(self, line_name: str, line_connections: List[Tuple[str, str, str]]) -> str:
        """
        Check if transmission line has both ends connected to the same block.
        
        Args:
            line_name: Name of the transmission line
            line_connections: List of (connected_block, line_ports, block_ports) tuples
            
        Returns:
            str: Name of the block if both ends connect to same block, empty string otherwise
        """
        try:
            # Group connections by the ports on the transmission line
            left_side_connections = []  # A, B, C ports
            right_side_connections = []  # a, b, c ports
            
            for connected_block, line_ports, block_ports in line_connections:
                # Parse line ports
                if line_ports.startswith("(") and line_ports.endswith(")"):
                    port_list_str = line_ports[1:-1]
                    ports = [p.strip() for p in port_list_str.split(",")]
                    
                    # Check if these are left side (A,B,C) or right side (a,b,c) ports
                    if any(port in ['A', 'B', 'C'] for port in ports):
                        left_side_connections.append(connected_block)
                    if any(port in ['a', 'b', 'c'] for port in ports):
                        right_side_connections.append(connected_block)
            
            # Check if both sides connect to the same block
            if left_side_connections and right_side_connections:
                for left_block in left_side_connections:
                    for right_block in right_side_connections:
                        if left_block == right_block:
                            return left_block
            
            return ""
            
        except Exception:
            return ""

    def _validate_transmission_line_port_directions(
        self,
        connections: List[List[str]],
        current_blocks: Dict[str, Any],
        block_to_cluster: Dict[str, tuple]
    ) -> Dict[str, Any]:
        """
        Validate that transmission line ports are used correctly with respect to directions and distances.
        
        A transmission line's left side ports (A,B,C) that are connected to one location 
        cannot also connect to blocks in different locations.
        
        Args:
            connections: List of connections
            current_blocks: Dictionary of current blocks
            block_to_cluster: Mapping from block names to their clusters
            
        Returns:
            dict: {"valid": bool, "reason": str}
        """
        try:
            # Find all transmission lines
            transmission_lines = []
            for block_name, block_data in current_blocks.items():
                if block_data.get("Type") == "Three-Phase PI Section Line":
                    transmission_lines.append(block_name)
            
            # For each transmission line, check its port usage
            for line_name in transmission_lines:
                # Get all connections involving this transmission line
                line_connections = []
                
                for connection in connections:
                    if not isinstance(connection, list) or len(connection) != 2:
                        continue
                    
                    source_spec, target_spec = connection
                    
                    if "/" not in source_spec or "/" not in target_spec:
                        continue
                    
                    source_block, source_port_spec = source_spec.split("/", 1)
                    target_block, target_port_spec = target_spec.split("/", 1)
                    
                    if source_block == line_name:
                        line_connections.append((target_block, source_port_spec, target_port_spec, "source"))
                    elif target_block == line_name:
                        line_connections.append((source_block, target_port_spec, source_port_spec, "target"))
                
                # Group connections by port side (Left/Right)
                left_side_blocks = []  # Blocks connected to A,B,C ports
                right_side_blocks = []  # Blocks connected to a,b,c ports
                
                for connected_block, line_port_spec, block_port_spec, direction in line_connections:
                    # Parse line ports
                    if line_port_spec.startswith("(") and line_port_spec.endswith(")"):
                        port_list_str = line_port_spec[1:-1]
                        ports = [p.strip() for p in port_list_str.split(",")]
                        
                        # Check if these are left side (A,B,C) or right side (a,b,c) ports
                        has_left_ports = any(port in ['A', 'B', 'C'] for port in ports)
                        has_right_ports = any(port in ['a', 'b', 'c'] for port in ports)
                        
                        if has_left_ports:
                            left_side_blocks.append(connected_block)
                        if has_right_ports:
                            right_side_blocks.append(connected_block)
                
                # Check if left side ports are used to connect to multiple different locations
                if len(left_side_blocks) > 1:
                    # Get clusters for these blocks
                    left_clusters = set()
                    for block in left_side_blocks:
                        cluster = block_to_cluster.get(block)
                        if cluster:
                            left_clusters.add(cluster)
                    
                    if len(left_clusters) > 1:
                        return {
                            "valid": False,
                            "reason": f"Transmission line '{line_name}' left side ports (A,B,C) are connected to blocks in different locations: {left_side_blocks}. If connecting to different locations, use right side ports (a,b,c) for the distant connection."
                        }
                
                # Check if right side ports are used to connect to multiple different locations
                if len(right_side_blocks) > 1:
                    # Get clusters for these blocks
                    right_clusters = set()
                    for block in right_side_blocks:
                        cluster = block_to_cluster.get(block)
                        if cluster:
                            right_clusters.add(cluster)
                    
                    if len(right_clusters) > 1:
                        return {
                            "valid": False,
                            "reason": f"Transmission line '{line_name}' right side ports (a,b,c) are connected to blocks in different locations: {right_side_blocks}. Each side of the transmission line should connect to only one location."
                        }
                
                # Check for the specific constraint: same ports connecting to different locations
                for connected_block, line_port_spec, block_port_spec, direction in line_connections:
                    if line_port_spec.startswith("(") and line_port_spec.endswith(")"):
                        port_list_str = line_port_spec[1:-1]
                        ports = [p.strip() for p in port_list_str.split(",")]
                        
                        # For each port, check if it's used in multiple connections to different locations
                        for port in ports:
                            # Find all connections using this specific port
                            port_connections = []
                            for other_connection in line_connections:
                                other_block, other_line_ports, other_block_ports, other_dir = other_connection
                                if other_line_ports.startswith("(") and other_line_ports.endswith(")"):
                                    other_port_list = other_line_ports[1:-1].split(",")
                                    other_ports = [p.strip() for p in other_port_list]
                                    if port in other_ports:
                                        port_connections.append(other_block)
                            
                            # Check if this port connects to multiple blocks in different locations
                            if len(port_connections) > 1:
                                port_clusters = set()
                                for block in port_connections:
                                    cluster = block_to_cluster.get(block)
                                    if cluster:
                                        port_clusters.add(cluster)
                                
                                if len(port_clusters) > 1:
                                    return {
                                        "valid": False,
                                        "reason": f"Transmission line '{line_name}' port '{port}' is connected to blocks in different locations: {port_connections}. Each port can only connect to one location."
                                    }
            
            return {"valid": True, "reason": ""}
            
        except Exception as e:
            return {"valid": False, "reason": f"Error validating transmission line port directions: {str(e)}"} 