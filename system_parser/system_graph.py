# power_system_graph.py
"""
Builds and validates a directed graph representation of a power system.

Changes in this version
-----------------------
* get_validation_report(): switched from json.dumps() → pprint / black formatting
  so any Python object (set, tuple, datetime, …) can be printed safely.
* Added connectivity analysis for generator-to-load transmission assessment
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import pprint
import io
import base64
import graphviz

import networkx as nx

from .system_components import Block, ParameterValidationIssue
from .block_config_loader import BlockConfig

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONN_RE = re.compile(
    r"([^/]+)/\(?([^)]+)\)?"
)  # e.g. "Block/(P1,P2)" or "Block/P1"
_MAX_ERROR_ITEMS = 5  # Hard-limit when returning validation feedback
_MAX_WARNING_ITEMS = 0  # Hard-limit when returning validation feedback

# Block type categorization for connectivity analysis
GENERATOR_TYPES = {
    "Three-Phase Source",  # External grid or generator
    "Synchronous Machine",  # Sync generator/motor
    "AC Voltage Source",   # AC source
}

LOAD_TYPES = {
    "Three-Phase Parallel RLC Load",
    "Three-Phase Series RLC Load",
}

TRANSMISSION_TYPES = {
    "Three-Phase PI Section Line",
    "Single-Phase Transmission Line",
    "Three-Phase Transformer (Two Windings)",
    "Three-Phase Transformer (Three Windings)",
    "Single-Phase Transformer",
}

BUS_TYPES = {
    "Three-Phase V-I Measurement",  # Acts as a bus/node
}


@dataclass
class UnconnectedPort:
    """Represents an unconnected port with its details."""

    block_name: str
    port_name: str
    port_type: str
    global_id: str

    def __str__(self) -> str:  # noqa: D401
        return f"{self.block_name}.{self.port_name} ({self.port_type})"


@dataclass
class ConnectivityResult:
    """Results of connectivity analysis between generators and loads."""
    
    total_generators: int
    total_loads: int
    connected_loads: int
    connectivity_ratio: float  # 0.0 to 1.0
    connected_load_names: List[str]
    isolated_load_names: List[str]
    generator_names: List[str]
    paths_found: Dict[str, List[str]]  # load_name -> list of generator names that can reach it


class SystemGraph:
    """Builds and validates a directed graph representation of a system.

    Parameters
    ----------
    system_dict : dict
        The raw system definition (typically loaded from JSON/YAML).
    block_config_instance : BlockConfig | None, optional
        If *None*, a default configuration will be loaded from
        ``power_system_parser/block_config.json``.
    distances : dict, optional
        Distance information between connected components. Format:
        {(("Block1", "Port1"), ("Block2", "Port2")): distance_value}

    Notes
    -----
    • Validation issues (errors & warnings) are collected during construction
      and can later be retrieved with :pymeth:`get_validation_report`.
    • The public :pyattr:`graph` attribute is a *networkx* ``DiGraph`` whose
      nodes are both *blocks* and *ports*; ownership edges (block→port) are
      tagged ``edge_type='has_port'`` while inter-port connections are tagged
      ``edge_type='connects_to'``.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        system_dict: Dict[str, Any],
        block_config_instance: Optional[BlockConfig] = None,
        validate_param_keys: bool = False,
        distances: Optional[Dict[Any, float]] = None,
    ):
        self.system_dict = system_dict
        self.distances = distances or {}

        # Configuration ---------------------------------------------------
        if block_config_instance is None:
            config_file = "system_parser/block_config.json"
            self.block_config = BlockConfig(config_file)
        else:
            self.block_config = block_config_instance

        # Runtime state ----------------------------------------------------
        self.graph: nx.DiGraph = nx.DiGraph(name="System Graph")
        self.blocks: Dict[str, Block] = {}
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        self.unconnected_ports: List[UnconnectedPort] = []
        self.validate_param_keys = validate_param_keys
        self.param_validation_issues: List[ParameterValidationIssue] = []
        # Distance validation tracking
        self.distance_violations: List[str] = []
        # Frequency and voltage coherence tracking
        self.frequency_violations: List[str] = []
        self.voltage_violations: List[str] = []

        # Build ------------------------------------------------------------
        self._build_graph()

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _add_validation_error(self, message: str) -> None:
        self.validation_errors.append(message)
        logger.error(message)

    def _add_validation_warning(self, message: str) -> None:
        self.validation_warnings.append(message)
        logger.warning(message)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_connection_string(conn_str: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Parse a connection spec of the form ``"Block/Port"`` or
        ``"Block/(P1,P2)"``.

        Returns
        -------
        (block_name, [port_names]) or (None, None) on failure.
        """
        match = _DEFAULT_CONN_RE.match(conn_str)
        if not match:
            return None, None
        block_name = match.group(1)
        ports_str = match.group(2)
        port_names = [p.strip() for p in ports_str.split(",")]
        return block_name, port_names

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        """Top-level builder routine."""
        if not self.block_config or not self.block_config._config:
            self._add_validation_error(
                "Block configuration is empty or failed to load. Cannot build graph."
            )
            return

        self._add_blocks()
        if self.blocks:  # Only attempt connections if blocks were valid
            self._add_connections()
            self._detect_unconnected_ports()
            self._validate_distance_constraints()
            self._validate_frequency_coherence()
            self._validate_voltage_coherence()
        logger.debug(
            "Graph construction complete: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # ..................................................................
    # Blocks
    # ..................................................................

    def _add_blocks(self) -> None:
        if not isinstance(self.system_dict, dict):
            self._add_validation_error("System dictionary is not a dictionary.")
            return

        raw_blocks = self.system_dict.get("Blocks")
        if not raw_blocks:
            self._add_validation_error("No blocks defined in the system dictionary.")
            return

        if not isinstance(raw_blocks, dict):
            self._add_validation_error("Blocks is not a dictionary.")
            return

        for blk_name, blk_data in raw_blocks.items():
            if not isinstance(blk_data, dict):
                self._add_validation_error(f"Block '{blk_name}' is not a dictionary.")
                continue

            blk_type = blk_data.get("Type")
            if not blk_type:
                self._add_validation_error(f"Block '{blk_name}' is missing 'Type' information. Skipping.")
                continue

            if not self.block_config.get_block_template(blk_type):
                self._add_validation_error(
                    f"Block type '{blk_type}' for block '{blk_name}' not found in configuration. Skipping this block."
                )
                continue

            try:
                block_obj = Block(blk_name, blk_type, blk_data, self.block_config, self.validate_param_keys)
            except ValueError as exc:
                self._add_validation_error(str(exc))
                continue

            # Register --------------------------------------------------
            self.blocks[blk_name] = block_obj
            
            # Collect parameter validation issues
            if block_obj.param_validation_issues:
                self.param_validation_issues.extend(block_obj.param_validation_issues)
            
            self.graph.add_node(
                blk_name,
                label=f"{blk_name}\n({blk_type})",
                node_type="block",
                block_type=blk_type,
                params=block_obj.params,
            )

            # Ports -----------------------------------------------------
            for port_name, port_obj in block_obj.ports.items():
                self.graph.add_node(
                    port_obj.global_id,
                    label=port_obj.global_id,
                    node_type="port",
                    block_name=blk_name,
                    port_name=port_name,
                    port_type=port_obj.type,
                )
                self.graph.add_edge(blk_name, port_obj.global_id, edge_type="has_port")

            logger.debug(
                "Added block '%s' (Type: %s) with %d ports", blk_name, blk_type, len(block_obj.ports)
            )

        if not self.blocks:
            self._add_validation_warning(
                "No valid blocks were added to the system based on the input and configuration."
            )

    # ..................................................................
    # Connections
    # ..................................................................

    def _add_connections(self) -> None:
        raw_conns = self.system_dict.get("Connections", [])
        if not raw_conns:
            self._add_validation_warning("No connections defined in the system input.")
            return

        for idx, conn in enumerate(raw_conns, start=1):
            if not (
                isinstance(conn, list)
                and len(conn) == 2
                and all(isinstance(c, str) for c in conn)
            ):
                self._add_validation_error(
                    f"Connection entry #{idx} is malformed: {conn!r}. Expected a list of two strings. Skipping."
                )
                continue

            src_str, tgt_str = conn
            src_blk, src_ports = self._parse_connection_string(src_str)
            tgt_blk, tgt_ports = self._parse_connection_string(tgt_str)

            if not src_blk or not tgt_blk:
                self._add_validation_error(f"Connection {conn}: Failed to parse source or target spec. Skipping.")
                continue

            if src_blk not in self.blocks:
                self._add_validation_error(f"Connection {conn}: Source block '{src_blk}' not defined. Skipping.")
                continue
            if tgt_blk not in self.blocks:
                self._add_validation_error(f"Connection {conn}: Target block '{tgt_blk}' not defined. Skipping.")
                continue

            if len(src_ports) != len(tgt_ports):
                self._add_validation_error(
                    f"Connection {conn}: Port count mismatch (source {len(src_ports)} vs target {len(tgt_ports)}). Skipping."
                )
                continue

            src_block_obj = self.blocks[src_blk]
            tgt_block_obj = self.blocks[tgt_blk]

            for s_name, t_name in zip(src_ports, tgt_ports):
                s_port = src_block_obj.get_port(s_name)
                t_port = tgt_block_obj.get_port(t_name)

                if not s_port or not t_port:
                    if not s_port:
                        self._add_validation_error(
                            f"Connection {conn}: Source port '{s_name}' not found on '{src_blk}'."
                        )
                    if not t_port:
                        self._add_validation_error(
                            f"Connection {conn}: Target port '{t_name}' not found on '{tgt_blk}'."
                        )
                    continue

                if self.graph.has_edge(s_port.global_id, t_port.global_id):
                    # Duplicate connection – silently ignore to avoid clutter.
                    continue

                self.graph.add_edge(s_port.global_id, t_port.global_id, edge_type="connects_to")
                logger.debug("Connected %s → %s", s_port.global_id, t_port.global_id)

    # ..................................................................
    # Unconnected ports detection
    # ..................................................................

    def _detect_unconnected_ports(self) -> None:
        """Detect and store information about unconnected ports."""
        port_nodes = [
            node for node, data in self.graph.nodes(data=True) if data.get("node_type") == "port"
        ]

        connected_ports: Set[str] = set()

        # Find all ports that have connections (either incoming or outgoing)
        for edge in self.graph.edges(data=True):
            if edge[2].get("edge_type") == "connects_to":
                connected_ports.add(edge[0])  # source port
                connected_ports.add(edge[1])  # target port

        # Identify unconnected ports
        unconnected_port_ids = set(port_nodes) - connected_ports

        self.unconnected_ports = [
            UnconnectedPort(
                block_name=self.graph.nodes[port_id]["block_name"],
                port_name=self.graph.nodes[port_id]["port_name"],
                port_type=self.graph.nodes[port_id]["port_type"],
                global_id=port_id,
            )
            for port_id in unconnected_port_ids
        ]

        # Add warnings for unconnected ports
        if self.unconnected_ports:
            for port in self.unconnected_ports:
                self._add_validation_warning(f"Unconnected port: {port}")

    # ..................................................................
    # Distance validation
    # ..................................................................

    def _get_block_distance(self, block_name: str) -> float:
        """Get the distance/length of a block based on its configuration."""
        if block_name not in self.blocks:
            return 0.0
        
        block = self.blocks[block_name]
        block_config = self.block_config.get_block_template(block.type)
        
        if not block_config or "Physical_Length" not in block_config:
            return 0.0
        
        distance_config = block_config["Physical_Length"]
        distance_type = distance_config.get("Type", "negligible")
        
        if distance_type == "negligible":
            return distance_config.get("Value", 0.0)
        elif distance_type == "configurable":
            # Get distance from block parameters
            param_name = distance_config.get("Parameter")
            if param_name and param_name in block.params:
                try:
                    # Handle different parameter formats
                    param_value = block.params[param_name]
                    if isinstance(param_value, (int, float)):
                        return float(param_value)
                    elif isinstance(param_value, str):
                        # Try to parse as number
                        return float(param_value)
                except (ValueError, TypeError):
                    return distance_config.get("Default", 0.0)
            else:
                return distance_config.get("Default", 0.0)
        
        return 0.0

    def _validate_distance_constraints(self) -> None:
        """Validate that connections respect distance constraints."""
        if not self.distances:
            return
        
        # Check each connection against distance constraints
        for edge in self.graph.edges(data=True):
            if edge[2].get("edge_type") != "connects_to":
                continue
            
            src_port_id = edge[0]
            tgt_port_id = edge[1]
            
            # Get block names for source and target ports
            src_data = self.graph.nodes[src_port_id]
            tgt_data = self.graph.nodes[tgt_port_id]
            
            src_block = src_data.get("block_name")
            tgt_block = tgt_data.get("block_name")
            
            if not src_block or not tgt_block:
                continue
            
            # Check if this connection violates distance constraints
            self._check_connection_distance(src_block, tgt_block)

    def _check_connection_distance(self, block1: str, block2: str) -> bool:
        """Check if a connection between two blocks violates distance constraints."""
        # Get distances from the distances dict
        # The distances dict contains positions of connected components
        
        # Find positions of both blocks
        block1_positions = set()
        block2_positions = set()
        
        for position_tuple, distance in self.distances.items():
            for pos_pair in position_tuple:
                if isinstance(pos_pair, tuple) and len(pos_pair) == 2:
                    block_name, port_or_bus = pos_pair
                    if block_name == block1:
                        block1_positions.add(position_tuple)
                    elif block_name == block2:
                        block2_positions.add(position_tuple)
        
        # If blocks are in the same position tuple, they can connect (distance = 0)
        common_positions = block1_positions.intersection(block2_positions)
        if common_positions:
            return True  # Connection is valid
        
        # Check if blocks can be connected through transmission lines
        # Find the shortest path considering distances
        min_distance = float('inf')
        
        for pos1 in block1_positions:
            for pos2 in block2_positions:
                # Check direct distance between positions
                if (pos1, pos2) in self.distances:
                    min_distance = min(min_distance, self.distances[(pos1, pos2)])
                elif (pos2, pos1) in self.distances:
                    min_distance = min(min_distance, self.distances[(pos2, pos1)])
        
        # Check if any transmission line can bridge this distance
        bridge_block = None
        for block_name, block in self.blocks.items():
            if block.type == "Three-Phase PI Section Line":
                block_distance = self._get_block_distance(block_name)
                if block_distance >= min_distance and min_distance != float('inf'):
                    bridge_block = block_name
                    break
        
        if min_distance == float('inf'):
            # No distance information available - assume it's a new connection
            return True
        
        if not bridge_block:
            self.distance_violations.append(
                f"Connection between '{block1}' and '{block2}' requires distance {min_distance} km, "
                f"but no transmission line of sufficient length is available."
            )
            return False
        
        return True

    def _validate_frequency_coherence(self) -> None:
        """Validate that all blocks have coherent frequency values."""
        frequencies = {}  # block_name -> frequency_value
        
        for block_name, block in self.blocks.items():
            frequency = self._extract_frequency(block)
            if frequency is not None:
                frequencies[block_name] = frequency
        
        if not frequencies:
            return  # No frequency information found
        
        # Find the most common frequency
        freq_values = list(frequencies.values())
        unique_frequencies = list(set(freq_values))
        
        if len(unique_frequencies) > 1:
            # Multiple different frequencies found
            freq_counts = {freq: freq_values.count(freq) for freq in unique_frequencies}
            most_common_freq = max(freq_counts, key=freq_counts.get)
            
            for block_name, freq in frequencies.items():
                if freq != most_common_freq:
                    self.frequency_violations.append(
                        f"Block '{block_name}' has frequency {freq} Hz, but system frequency should be {most_common_freq} Hz"
                    )
                    self._add_validation_error(
                        f"Frequency incoherence: Block '{block_name}' has frequency {freq} Hz, "
                        f"but most blocks use {most_common_freq} Hz. All blocks should have the same frequency."
                    )

    def _extract_frequency(self, block: 'Block') -> Optional[float]:
        """Extract frequency value from a block's parameters."""
        params = block.params
        
        # Different blocks have different frequency parameter names
        frequency_keys = [
            "Frequency (Hz)",
            "Nominal frequency fn (Hz)", 
            "Frequency used for rlc specification (Hz)"
        ]
        
        for key in frequency_keys:
            if key in params:
                try:
                    return float(params[key])
                except (ValueError, TypeError):
                    continue
        
        # Handle transformer frequency (in array format)
        if "Nominal power and frequency [Pn(VA), fn(Hz)]" in params:
            try:
                power_freq_str = params["Nominal power and frequency [Pn(VA), fn(Hz)]"]
                # Parse array format like "[250e6, 60]"
                import ast
                power_freq_list = ast.literal_eval(power_freq_str)
                if isinstance(power_freq_list, list) and len(power_freq_list) >= 2:
                    return float(power_freq_list[1])  # frequency is second element
            except (ValueError, TypeError, SyntaxError):
                pass
        
        return None

    def _validate_voltage_coherence(self, voltage_threshold: float = 0.2) -> None:
        """Validate that adjacent blocks have coherent voltage levels.
        
        Args:
            voltage_threshold: Maximum allowed relative voltage difference (e.g., 0.2 = 20%)
        """
        # Get voltage information for each block
        block_voltages = {}  # block_name -> voltage_value
        
        for block_name, block in self.blocks.items():
            voltage = self._extract_voltage(block)
            if voltage is not None:
                block_voltages[block_name] = voltage
        
        if len(block_voltages) < 2:
            return  # Not enough voltage information for comparison
        
        # Track checked block pairs to avoid duplicates
        checked_pairs = set()
        
        # Check voltage coherence for connected blocks
        for edge in self.graph.edges(data=True):
            if edge[2].get("edge_type") != "connects_to":
                continue
            
            src_port_id = edge[0]
            tgt_port_id = edge[1]
            
            # Get block names for source and target ports
            src_data = self.graph.nodes[src_port_id]
            tgt_data = self.graph.nodes[tgt_port_id]
            
            src_block = src_data.get("block_name")
            tgt_block = tgt_data.get("block_name")
            
            if not src_block or not tgt_block:
                continue
            
            # Skip Three-Phase V-I Measurement blocks (they don't have voltage)
            src_block_obj = self.blocks.get(src_block)
            tgt_block_obj = self.blocks.get(tgt_block)
            
            if (src_block_obj and src_block_obj.type == "Three-Phase V-I Measurement" or
                tgt_block_obj and tgt_block_obj.type == "Three-Phase V-I Measurement"):
                continue
            
            # Create a normalized pair key (alphabetically sorted to avoid duplicates)
            pair_key = tuple(sorted([src_block, tgt_block]))
            if pair_key in checked_pairs:
                continue  # Already checked this block pair
            
            checked_pairs.add(pair_key)
            
            # Check voltage compatibility
            src_voltage = block_voltages.get(src_block)
            tgt_voltage = block_voltages.get(tgt_block)
            
            if src_voltage is not None and tgt_voltage is not None:
                self._check_voltage_compatibility(src_block, tgt_block, src_voltage, tgt_voltage, voltage_threshold)

    def _extract_voltage(self, block: 'Block') -> Optional[float]:
        """Extract voltage value from a block's parameters."""
        params = block.params
        
        # Different blocks have different voltage parameter names
        voltage_keys = [
            "Phase-to-phase voltage (Vrms)",
            "Nominal phase-to-phase voltage Vn (Vrms)"
        ]
        
        for key in voltage_keys:
            if key in params:
                try:
                    return float(params[key])
                except (ValueError, TypeError):
                    continue
        
        # Handle transformer voltages (take primary winding voltage)
        winding_keys = [
            "Winding 1 parameters [V1 Ph-Ph(Vrms), R1(pu), L1(pu)]",
            "Winding 2 parameters [V2 Ph-Ph(Vrms), R2(pu), L2(pu)]"
        ]
        
        for key in winding_keys:
            if key in params:
                try:
                    winding_str = params[key]
                    # Parse array format like "[735e3, 0.002, 0.08]"
                    import ast
                    winding_list = ast.literal_eval(winding_str)
                    if isinstance(winding_list, list) and len(winding_list) >= 1:
                        return float(winding_list[0])  # voltage is first element
                except (ValueError, TypeError, SyntaxError):
                    continue
        
        return None

    def _check_voltage_compatibility(self, block1: str, block2: str, voltage1: float, voltage2: float, threshold: float) -> None:
        """Check if two voltages are compatible within the given threshold."""
        # Calculate relative difference
        max_voltage = max(voltage1, voltage2)
        min_voltage = min(voltage1, voltage2)
        
        if max_voltage == 0:
            return  # Avoid division by zero
        
        relative_diff = abs(voltage1 - voltage2) / max_voltage
        
        # For transformers, large voltage differences are expected
        block1_obj = self.blocks.get(block1)
        block2_obj = self.blocks.get(block2)
        
        is_transformer_connection = (
            (block1_obj and "Transformer" in block1_obj.type) or
            (block2_obj and "Transformer" in block2_obj.type)
        )
        
        # Use different threshold for transformer connections
        actual_threshold = threshold * 5 if is_transformer_connection else threshold
        
        if relative_diff > actual_threshold:
            voltage_diff_percent = relative_diff * 100
            self.voltage_violations.append(
                f"Voltage mismatch between '{block1}' ({voltage1:.0f} V) and '{block2}' ({voltage2:.0f} V): {voltage_diff_percent:.1f}% difference"
            )
            
            severity = "warning" if is_transformer_connection else "error"
            message = (
                f"Voltage incoherence: Block '{block1}' ({voltage1:.0f} V) and '{block2}' ({voltage2:.0f} V) "
                f"have {voltage_diff_percent:.1f}% voltage difference, which exceeds the {actual_threshold*100:.0f}% threshold."
            )
            
            if severity == "error":
                self._add_validation_error(message)
            else:
                self._add_validation_warning(message)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_unconnected_ports(self) -> List[UnconnectedPort]:
        """Return a list of unconnected ports."""
        return self.unconnected_ports.copy()

    def get_unconnected_ports_by_block(self) -> Dict[str, List[UnconnectedPort]]:
        """Return unconnected ports grouped by block."""
        result: Dict[str, List[UnconnectedPort]] = {}
        for port in self.unconnected_ports:
            result.setdefault(port.block_name, []).append(port)
        
        # Sort ports alphabetically by port name within each block
        for block_name in result:
            result[block_name].sort(key=lambda port: port.port_name)
        
        return result

    def get_connectivity_details(self) -> ConnectivityResult:
        """Get detailed connectivity analysis results."""
        return self.analyze_connectivity()

    def get_validation_report(
        self,
        max_error_items: Optional[int] = _MAX_ERROR_ITEMS,
        max_warning_items: Optional[int] = _MAX_WARNING_ITEMS,
        include_unconnected_summary: bool = True,
        include_connectivity_analysis: bool = True,
        include_param_validation: bool = True,
    ) -> str:
        """Return a detailed validation report including errors, warnings, unconnected ports, and connectivity analysis."""
        report_parts: List[str] = []

        # ----------------------------------------------------------------
        # 1) Print *current* system_dict as pretty Python code
        # ----------------------------------------------------------------
        raw_code = f"system_dict = {pprint.pformat(self.system_dict, width=88, indent=4, compact=False, sort_dicts=False)}"
        try:
            # Re-format with Black if available for nicer line breaks
            import black

            system_dict_str = black.format_str(raw_code, mode=black.Mode())
        except Exception:  # Black not installed or formatting failed
            system_dict_str = raw_code

        report_parts.append("CURRENT SYSTEM DICTIONARY:")
        report_parts.append("```python")
        report_parts.append(system_dict_str.strip())
        report_parts.append("```")
        report_parts.append("")

        # ----------------------------------------------------------------
        # 2) Errors
        # ----------------------------------------------------------------
        errors = (
            self.validation_errors[:max_error_items]
            if max_error_items is not None
            else list(self.validation_errors)
        )
        if errors:
            report_parts.append("ERRORS:")
            report_parts.extend(f"  • {error}" for error in errors)
            if max_error_items is not None and len(self.validation_errors) > max_error_items:
                report_parts.append(
                    f"  ... and {len(self.validation_errors) - max_error_items} more errors"
                )

        # ----------------------------------------------------------------
        # 3) Warnings
        # ----------------------------------------------------------------
        warnings = (
            self.validation_warnings[:max_warning_items]
            if max_warning_items is not None
            else list(self.validation_warnings)
        )
        if warnings:
            if errors:  # only add blank line separator if error section exists
                report_parts.append("")
            report_parts.append("WARNINGS:")
            report_parts.extend(f"  • {warning}" for warning in warnings)
            if max_warning_items is not None and len(self.validation_warnings) > max_warning_items:
                report_parts.append(
                    f"  ... and {len(self.validation_warnings) - max_warning_items} more warnings"
                )

        # ----------------------------------------------------------------
        # 4) Distance Violations
        # ----------------------------------------------------------------
        if self.distance_violations:
            if errors or warnings:
                report_parts.append("")
            report_parts.append("DISTANCE CONSTRAINT VIOLATIONS:")
            report_parts.extend(f"  • {violation}" for violation in self.distance_violations)

        # ----------------------------------------------------------------
        # 5) Frequency Coherence Violations
        # ----------------------------------------------------------------
        if self.frequency_violations:
            if errors or warnings or self.distance_violations:
                report_parts.append("")
            report_parts.append("FREQUENCY COHERENCE VIOLATIONS:")
            report_parts.extend(f"  • {violation}" for violation in self.frequency_violations)

        # ----------------------------------------------------------------
        # 6) Voltage Coherence Violations
        # ----------------------------------------------------------------
        if self.voltage_violations:
            if errors or warnings or self.distance_violations or self.frequency_violations:
                report_parts.append("")
            report_parts.append("VOLTAGE COHERENCE VIOLATIONS:")
            report_parts.extend(f"  • {violation}" for violation in self.voltage_violations)

        # ----------------------------------------------------------------
        # 7) Parameter Validation Issues
        # ----------------------------------------------------------------
        if include_param_validation and self.param_validation_issues:
            if errors or warnings or self.distance_violations or self.frequency_violations or self.voltage_violations:
                report_parts.append("")
            report_parts.append("PARAMETER VALIDATION ISSUES:")
            
            # Group issues by block for better readability
            issues_by_block = {}
            for issue in self.param_validation_issues:
                if issue.block_name not in issues_by_block:
                    issues_by_block[issue.block_name] = []
                issues_by_block[issue.block_name].append(issue)
            
            for block_name, block_issues in issues_by_block.items():
                block_type = block_issues[0].block_type  # All issues for same block have same type
                report_parts.append(f"  • Block '{block_name}' ({block_type}):")
                for issue in block_issues:
                    if issue.suggested_key:
                        report_parts.append(f"    - Invalid key '{issue.invalid_key}' → suggested: '{issue.suggested_key}' (similarity: {issue.similarity_score:.2f})")
                    else:
                        report_parts.append(f"    - Invalid key '{issue.invalid_key}' (no close match found)")
            
            total_issues = len(self.param_validation_issues)
            affected_blocks = len(issues_by_block)
            report_parts.append(f"  Total: {total_issues} invalid parameter keys across {affected_blocks} blocks")

        # ----------------------------------------------------------------
        # 6) Connectivity Analysis (skip if there are critical errors)
        # ----------------------------------------------------------------
        if include_connectivity_analysis and not errors and not self.distance_violations:
            connectivity = self.analyze_connectivity()
            if warnings or (include_param_validation and self.param_validation_issues):
                report_parts.append("")
            report_parts.append("CONNECTIVITY ANALYSIS:")
            report_parts.append(f"  • Total Generators: {connectivity.total_generators}")
            report_parts.append(f"  • Total Loads: {connectivity.total_loads}")
            report_parts.append(f"  • Connected Loads: {connectivity.connected_loads}")
            report_parts.append(f"  • Connectivity Ratio: {connectivity.connectivity_ratio:.2%}")
            
            if connectivity.connected_load_names:
                report_parts.append(f"  • Connected Loads: {', '.join(connectivity.connected_load_names)}")
            
            if connectivity.isolated_load_names:
                report_parts.append(f"  • Isolated Loads: {', '.join(connectivity.isolated_load_names)}")
            
            if connectivity.paths_found:
                report_parts.append("  • Generator-Load Paths:")
                for load, generators in connectivity.paths_found.items():
                    report_parts.append(f"    - {load} ← {', '.join(generators)}")

        # ----------------------------------------------------------------
        # 7) Unconnected ports summary (skip if there are critical errors)
        # ----------------------------------------------------------------
        if include_unconnected_summary and self.unconnected_ports and not errors and not self.distance_violations:
            if warnings or (include_param_validation and self.param_validation_issues) or (include_connectivity_analysis and not errors and not self.distance_violations):
                report_parts.append("")
            report_parts.append("UNCONNECTED PORTS SUMMARY:")

            unconnected_by_block = self.get_unconnected_ports_by_block()
            for block_name, ports in unconnected_by_block.items():
                port_list = ", ".join(port.port_name for port in ports)
                report_parts.append(f"  • {block_name}: {port_list}")

            total_unconnected = len(self.unconnected_ports)
            total_ports = len(
                [
                    node
                    for node, data in self.graph.nodes(data=True)
                    if data.get("node_type") == "port"
                ]
            )
            report_parts.append(
                f"  Total: {total_unconnected} unconnected out of {total_ports} total ports"
            )

        return "\n".join(report_parts) if report_parts else "No validation issues found."

    def draw_graph(self,
                   filepath: Optional[str] = None,
                   show_type: bool = False,
                   engine: str = "dot",
                   rankdir: str = "LR",
                   splines: str = "true",
                   directed: bool = True,
                   initial_graph: Optional["SystemGraph"] = None,
                   highlight_changes: bool = False) -> Optional[str]:
        """
        Use Graphviz to draw power system graph with proper port positioning.

        Parameters
        ----------
        filepath : str | None
            If provided, save PNG to this path (without extension, function will append `.png`).
        engine : str
            Graphviz layout engine, such as 'dot' / 'sfdp' / 'neato', etc.
        rankdir : str
            'LR'→left to right, 'TB'→top to bottom.
        splines : str
            Edge routing style. Options:
            - "true" or "line": straight lines (default)
            - "curved": curved splines
            - "ortho": orthogonal lines (right angles)
            - "polyline": polyline routing
            - "spline": curved splines
            - "false": no edges drawn
            Note: Some spline types work better with certain engines (e.g., "ortho" with "dot")
        directed : bool
            Whether to draw directional arrows on edges. Default is False (undirected).
        initial_graph : SystemGraph | None
            If provided, use this graph to highlight changes against it.
        highlight_changes : bool
            Whether to highlight changes in the graph.

        Returns
        -------
        str | None
            Base64 string of image for wandb/HTML embedding; returns None on failure.
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty. Nothing to draw.")
            return None

        # ------------------------------------------------------------------
        # Prepare diff information for change highlighting
        # ------------------------------------------------------------------
        nodes_added: set[str] = set()
        nodes_removed: set[str] = set()
        edges_added: set[tuple] = set()
        edges_removed: set[tuple] = set()

        if highlight_changes and initial_graph is not None:
            initial_nodes = set(initial_graph.graph.nodes())
            final_nodes = set(self.graph.nodes())
            nodes_added = final_nodes - initial_nodes
            nodes_removed = initial_nodes - final_nodes

            initial_edges = set(initial_graph.graph.edges())
            final_edges = set(self.graph.edges())
            edges_added = final_edges - initial_edges
            edges_removed = initial_edges - final_edges

        # ---------- 1) Parse has_port relationships: port → block ----------
        port_to_block: dict[str, str] = {}
        for u, v, edata in self.graph.edges(data=True):
            if edata.get("edge_type") != "has_port":
                continue
            # Handle both directions: u might be block or port
            if self.graph.nodes[u].get("node_type") == "block":
                port_to_block[v] = u
            else:
                port_to_block[u] = v

        # Include mappings from the initial graph so removed elements are still
        # rendered when highlighting changes
        if initial_graph is not None:
            for u, v, edata in initial_graph.graph.edges(data=True):
                if edata.get("edge_type") != "has_port":
                    continue
                if initial_graph.graph.nodes[u].get("node_type") == "block":
                    port_to_block.setdefault(v, u)
                else:
                    port_to_block.setdefault(u, v)

        # ---------- 2) Generate block nodes (record shape with embedded ports) ----------
        g = graphviz.Digraph("PowerGraph",
                            engine=engine,
                            format="png",
                            graph_attr={"rankdir": rankdir, "splines": splines})

        blocks_final = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "block"]
        if highlight_changes and initial_graph is not None:
            blocks_initial = [n for n, d in initial_graph.graph.nodes(data=True) if d.get("node_type") == "block"]
            blocks = list(set(blocks_final) | set(blocks_initial))
        else:
            blocks = blocks_final

        for blk in blocks:
            # Find all ports belonging to this block
            block_ports = []
            for port_global_id, block_name in port_to_block.items():
                if block_name == blk:
                    if port_global_id in self.graph.nodes:
                        port_name = self.graph.nodes[port_global_id].get("port_name", port_global_id)
                    elif initial_graph is not None and port_global_id in initial_graph.graph.nodes:
                        port_name = initial_graph.graph.nodes[port_global_id].get("port_name", port_global_id)
                    else:
                        port_name = port_global_id

                    if blk in self.graph.nodes:
                        block_type = self.graph.nodes[blk].get("block_type", "")
                    elif initial_graph is not None and blk in initial_graph.graph.nodes:
                        block_type = initial_graph.graph.nodes[blk].get("block_type", "")
                    else:
                        block_type = ""
                    
                    # Get port position from block config
                    port_position = "Right"  # default
                    if block_type and self.block_config:
                        block_template = self.block_config.get_block_template(block_type)
                        if block_template and "Ports" in block_template:
                            port_config = block_template["Ports"].get(port_name, {})
                            port_position = port_config.get("Position", "Right")
                    
                    block_ports.append((port_global_id, port_name, port_position))
            
            # Sort ports for consistent ordering
            block_ports.sort(key=lambda x: x[1])  # Sort by port name
            
            if block_ports:
                # Separate ports by position
                left_ports = [p for p in block_ports if p[2] == "Left"]
                right_ports = [p for p in block_ports if p[2] == "Right"]
                upper_ports = [p for p in block_ports if p[2] in ["Upper", "Top"]]
                lower_ports = [p for p in block_ports if p[2] in ["Lower", "Bottom"]]
                
                # Build label with proper port positioning
                block_name = blk
                # Get block type from current graph first, then initial graph if not found
                if blk in self.graph.nodes:
                    block_type = self.graph.nodes[blk].get("block_type", "")
                elif initial_graph is not None and blk in initial_graph.graph.nodes:
                    block_type = initial_graph.graph.nodes[blk].get("block_type", "")
                else:
                    block_type = ""
                
                # Format block label as: BlockName\n(BlockType)
                if block_type and show_type:
                    block_label = f"{block_name}\\n{block_type}".replace("Three-Phase ", "")
                else:
                    block_label = block_name
                
                # Create port sections - each port with proper anchor syntax
                left_port_items = [f"<{p[0]}> {p[1]}" for p in left_ports]
                right_port_items = [f"<{p[0]}> {p[1]}" for p in right_ports]
                upper_port_items = [f"<{p[0]}> {p[1]}" for p in upper_ports]
                lower_port_items = [f"<{p[0]}> {p[1]}" for p in lower_ports]
                
                # Build the complete record label with proper structure for port positioning
                label_parts = []
                
                # Top row: upper ports (if any)
                if upper_port_items:
                    label_parts.append("{ " + " | ".join(upper_port_items) + " }")
                
                # Middle row: left ports | block label | right ports
                middle_parts = []
                if left_port_items:
                    # For left ports, create a left-aligned section
                    middle_parts.append("{ " + " | ".join(left_port_items) + " }")
                    
                # Add the main block label in the center
                middle_parts.append(block_label)
                
                if right_port_items:
                    # For right ports, create a right-aligned section  
                    middle_parts.append("{ " + " | ".join(right_port_items) + " }")
                
                # Join middle parts horizontally
                if len(middle_parts) == 1:
                    # Only block label
                    label_parts.append(middle_parts[0])
                else:
                    # Multiple elements - arrange horizontally with proper record syntax
                    label_parts.append("{ " + " | ".join(middle_parts) + " }")
                
                # Bottom row: lower ports (if any)
                if lower_port_items:
                    label_parts.append("{ " + " | ".join(lower_port_items) + " }")
                
                # Combine all parts vertically with proper record syntax
                if len(label_parts) == 1:
                    # Single row (just middle)
                    label = label_parts[0]
                else:
                    # Multiple rows - arrange vertically
                    label = "{ " + " | ".join(label_parts) + " }"
                
                node_kwargs = {}
                if highlight_changes and initial_graph is not None:
                    if blk in nodes_added:
                        node_kwargs.update({"color": "blue", "penwidth": "2"})
                    elif blk in nodes_removed:
                        node_kwargs.update({"color": "red", "style": "dashed"})
                g.node(blk, label=label, shape="record", **node_kwargs)
            else:
                # No ports - use simple box
                block_name = blk
                # Get block type from current graph first, then initial graph if not found
                if blk in self.graph.nodes:
                    block_type = self.graph.nodes[blk].get("block_type", "")
                elif initial_graph is not None and blk in initial_graph.graph.nodes:
                    block_type = initial_graph.graph.nodes[blk].get("block_type", "")
                else:
                    block_type = ""
                if block_type:
                    label = f"{block_name}\\n({block_type})"
                else:
                    label = block_name
                node_kwargs = {}
                if highlight_changes and initial_graph is not None:
                    if blk in nodes_added:
                        node_kwargs.update({"color": "blue", "penwidth": "2"})
                    elif blk in nodes_removed:
                        node_kwargs.update({"color": "red", "style": "dashed"})
                g.node(blk, label=label, shape="box", **node_kwargs)

        # ---------- 3) Draw connects_to: port → port ----------
        if highlight_changes and initial_graph is not None:
            combined_edges = []
            combined_edges.extend(self.graph.edges(data=True))
            combined_edges.extend(initial_graph.graph.edges(data=True))

            seen_edges: set[tuple] = set()
            for u, v, edata in combined_edges:
                if edata.get("edge_type") != "connects_to":
                    continue
                key = (u, v)
                if key in seen_edges:
                    continue
                seen_edges.add(key)

                bu = port_to_block.get(u)
                bv = port_to_block.get(v)
                if bu is None or bv is None:
                    continue

                if key in edges_added:
                    edge_attrs = {"color": "blue", "style": "solid"}
                elif key in edges_removed:
                    edge_attrs = {"color": "red", "style": "dashed"}
                else:
                    edge_attrs = {"color": "black", "style": "solid"}

                if directed:
                    edge_attrs.update({"arrowsize": "1.0", "dir": "forward"})
                else:
                    edge_attrs["dir"] = "none"

                g.edge(f"{bu}:{u}", f"{bv}:{v}", **edge_attrs)
        else:
            for u, v, edata in self.graph.edges(data=True):
                if edata.get("edge_type") != "connects_to":
                    continue
                bu = port_to_block.get(u)   # Source port's block
                bv = port_to_block.get(v)   # Target port's block
                if bu is None or bv is None:
                    logger.warning("Edge %s→%s missing has_port information, skipping.", u, v)
                    continue

                # Graphviz edge format: block:portname
                edge_attrs = {
                    "color": "black",
                    "style": "solid"
                }
                if directed:
                    edge_attrs.update({"arrowsize": "1.0", "dir": "forward"})
                else:
                    edge_attrs["dir"] = "none"

                g.edge(f"{bu}:{u}", f"{bv}:{v}", **edge_attrs)

        # ---------- 4) Output ----------
        try:
            if filepath:
                g.render(filename=filepath, cleanup=True)   # Save PNG
                logger.info("Graph visualization saved to %s.png", filepath)

            png_bytes = g.pipe()   # Get binary PNG directly
            return base64.b64encode(png_bytes).decode()

        except graphviz.ExecutableNotFound as e:
            logger.error("Graphviz executable not found: %s", e)
            return None
        except Exception as e:
            logger.error("Graphviz drawing failed: %s", e)
            return None

    def get_reward(self) -> float:  # noqa: D401
        """Compute a normalised reward based on validation outcomes and system connectivity."""
        err = len(self.validation_errors)
        warn = len(self.validation_warnings)
        unconnected = len(self.unconnected_ports)
        
        # Analyze connectivity between generators and loads
        connectivity_result = self.analyze_connectivity()
        
        # Connectivity reward component (0.0 to 1.0)
        connectivity_reward = connectivity_result.connectivity_ratio
        
        # Weight factors: errors are most severe, warnings moderate, unconnected ports mild
        base_penalty = err + 0.5 * warn + 0.2 * unconnected
        base_reward = max(0.0, min(1.0, 1.0 / (1.0 + base_penalty)))
        
        # Combine base reward with connectivity reward (weighted average)
        # Give more weight to connectivity if there are generators and loads
        if connectivity_result.total_generators > 0 and connectivity_result.total_loads > 0:
            # 60% weight to connectivity, 40% to basic validation
            final_reward = 0.6 * connectivity_reward + 0.4 * base_reward
        else:
            # If no generators or loads, fall back to basic validation only
            final_reward = base_reward
            
        return max(0.0, min(1.0, final_reward))

    def get_combined_reward(self, conversion_reward: Optional[float] = None, diagnostic_reward: Optional[float] = None) -> float:
        """
        Compute a combined reward that includes graph validation, connectivity, conversion success, and diagnostics.
        
        Args:
            conversion_reward (float, optional): Conversion success reward from PandapowerConverter
            diagnostic_reward (float, optional): Diagnostic reward from PandapowerConverter
        
        Returns:
            float: Combined reward score (0.0 to 1.0)
        """
        # Base graph validation and connectivity reward
        base_reward = self.get_reward()
        
        # If only base reward is available
        if conversion_reward is None and diagnostic_reward is None:
            return base_reward
        
        # Calculate weighted combination
        weights = []
        rewards = []
        
        # Base graph validation and connectivity (always included)
        weights.append(0.3)
        rewards.append(base_reward)
        
        # Conversion success (if available)
        if conversion_reward is not None:
            weights.append(0.4)
            rewards.append(conversion_reward)
        
        # Diagnostic results (if available)  
        if diagnostic_reward is not None:
            weights.append(0.3)
            rewards.append(diagnostic_reward)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            combined_reward = sum(w * r for w, r in zip(normalized_weights, rewards))
        else:
            combined_reward = base_reward
            
        return max(0.0, min(1.0, combined_reward))

    def draw_graph_networkx(self, filepath: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> Optional[str]:
        """
        Draw the power system graph and optionally save it to a file.
        
        Args:
            filepath: Optional path to save the figure. If None, the figure is not saved to disk.
            figsize: Tuple specifying the figure size (width, height) in inches.
            
        Returns:
            Base64 encoded string of the figure image for embedding in wandb, or None if plotting fails.
        """
        try:
            import matplotlib
            # Set backend before importing pyplot if not already set
            if matplotlib.get_backend() != 'Agg':
                matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            if not self.graph or self.graph.number_of_nodes() == 0:
                logger.warning("Graph is empty. Nothing to draw.")
                return None

            fig, ax = plt.subplots(figsize=figsize)
            
            # Separate block nodes and port nodes
            block_nodes = [node for node, data in self.graph.nodes(data=True) if data.get("node_type") == "block"]
            port_nodes = [node for node, data in self.graph.nodes(data=True) if data.get("node_type") == "port"]
            
            # Create layout
            pos = nx.kamada_kawai_layout(self.graph)
            # pos = nx.spring_layout(self.graph, k=2, iterations=50)
            
            # Draw block nodes
            if block_nodes:
                block_labels = {n: f"{n}\n({self.graph.nodes[n].get('block_type', 'Unknown')})" for n in block_nodes}
                nx.draw_networkx_nodes(
                    self.graph, pos, 
                    nodelist=block_nodes,
                    node_color='lightblue', 
                    node_size=3000, 
                    alpha=0.8, 
                    ax=ax
                )
                nx.draw_networkx_labels(
                    self.graph, pos, 
                    labels=block_labels, 
                    font_size=8, 
                    font_weight='bold', 
                    ax=ax
                )
            
            # Draw port nodes (smaller)
            if port_nodes:
                port_colors = []
                for port in port_nodes:
                    port_type = self.graph.nodes[port].get('port_type', '')
                    if 'input' in port_type.lower():
                        port_colors.append('lightgreen')
                    elif 'output' in port_type.lower():
                        port_colors.append('lightcoral')
                    else:
                        port_colors.append('lightgray')
                
                nx.draw_networkx_nodes(
                    self.graph, pos,
                    nodelist=port_nodes,
                    node_color=port_colors,
                    node_size=500,
                    alpha=0.7,
                    ax=ax
                )
                
                # Port labels (just the port name, not the full global_id)
                port_labels = {}
                for port in port_nodes:
                    port_name = self.graph.nodes[port].get('port_name', port)
                    port_labels[port] = port_name
                
                nx.draw_networkx_labels(
                    self.graph, pos,
                    labels=port_labels,
                    font_size=6,
                    ax=ax
                )
            
            # Draw edges
            # has_port edges (block to port)
            has_port_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == 'has_port']
            if has_port_edges:
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=has_port_edges,
                    edge_color='gray',
                    width=1,
                    alpha=0.5,
                    style='dashed',
                    ax=ax
                )
            
            # connects_to edges (port to port)
            connects_to_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == 'connects_to']
            if connects_to_edges:
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=connects_to_edges,
                    edge_color='blue',
                    width=2,
                    alpha=0.7,
                    arrows=True,
                    arrowstyle='->',
                    arrowsize=20,
                    ax=ax
                )
            
            # Add title and legend
            ax.set_title(f"Power System Graph\n{len(block_nodes)} blocks, {len(port_nodes)} ports, {len(connects_to_edges)} connections", 
                        fontsize=14, fontweight='bold')
            
            # Create legend
            legend_elements = [
                patches.Patch(color='lightblue', label='Blocks'),
                patches.Patch(color='lightgreen', label='Input Ports'),
                patches.Patch(color='lightcoral', label='Output Ports'),
                patches.Patch(color='lightgray', label='Other Ports'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            ax.axis('off')
            plt.tight_layout()
            
            # Save to file if requested
            if filepath:
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                logger.info(f"Graph visualization saved to {filepath}")
            
            # Convert to base64 for wandb
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot draw graph. Please install it: pip install matplotlib")
            return None
        except Exception as e:
            logger.error(f"An error occurred during graph drawing: {e}")
            return None

    def draw_graph_networkx_2(self, filepath: Optional[str] = None, figsize: Tuple[int, int] = (18, 12)) -> Optional[str]:
        """
        Draw the power system graph with edge labels and enhanced styling.
        
        Args:
            filepath: Optional path to save the figure. If None, the figure is not saved to disk.
            figsize: Tuple specifying the figure size (width, height) in inches.
            
        Returns:
            Base64 encoded string of the figure image for embedding in wandb, or None if plotting fails.
        """
        try:
            import matplotlib
            # Set backend before importing pyplot if not already set
            if matplotlib.get_backend() != 'Agg':
                matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot draw graph. Please install it: pip install matplotlib")
            return None

        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty. Nothing to draw.")
            return None

        plt.figure(figsize=figsize)
        
        # Use a layout that tries to respect subgraphs if we were using them, or just a good general layout
        # For this structure, spring_layout or kamada_kawai_layout can be good.
        pos = nx.kamada_kawai_layout(self.graph)
        # pos = nx.spring_layout(self.graph)

        node_colors = []
        node_sizes = []
        node_labels = {n: d.get('label', n) for n, d in self.graph.nodes(data=True)}

        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'block':
                node_colors.append('skyblue')
                node_sizes.append(5000)
            elif data.get('node_type') == 'port':
                node_colors.append('lightgreen')
                node_sizes.append(2500)
            else:  # Should not happen with current logic
                node_colors.append('grey')
                node_sizes.append(1000)
        
        edge_colors = []
        style_map = {
            'has_port': 'dashed',
            'connects_to': 'solid'
        }
        edge_styles = []

        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'has_port':
                edge_colors.append('grey')
                edge_styles.append(style_map['has_port'])
            elif data.get('edge_type') == 'connects_to':
                edge_colors.append('red')
                edge_styles.append(style_map['connects_to'])
            else:
                edge_colors.append('black')
                edge_styles.append('solid')

        nx.draw(self.graph, pos,
                labels=node_labels,
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                edge_color=edge_colors,
                style=edge_styles,
                font_size=8,
                font_weight='bold',
                width=1.5,  # Edge width
                arrows=True,
                arrowstyle='-|>',
                arrowsize=15
               )

        edge_labels = {(u, v): data.get('edge_type', '') for u, v, data in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='purple', font_size=7)
        
        plt.title("Power System Network Graph", size=15)
        
        # Save to file if requested
        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {filepath}")
        
        # Convert to base64 for wandb
        try:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()  # Close the figure to free memory
            
            return image_base64
            
        except ImportError:
            logger.warning("Matplotlib is not installed. Cannot draw graph. Please install it: pip install matplotlib")
            return None
        except Exception as e:
            logger.error(f"An error occurred during graph drawing: {e}")
            return None

    def analyze_connectivity(self) -> ConnectivityResult:
        """Analyze the connectivity between generators and loads in the power system."""
        generators = [blk for blk, data in self.graph.nodes(data=True) if data.get('block_type') in GENERATOR_TYPES]
        loads = [blk for blk, data in self.graph.nodes(data=True) if data.get('block_type') in LOAD_TYPES]
        
        if not generators or not loads:
            self._add_validation_warning("No generators or loads found in the system.")
            return ConnectivityResult(0, 0, 0, 0.0, [], [], [], {})
        
        connected_loads = 0
        connected_load_names = []
        isolated_load_names = []
        generator_names = [gen for gen in generators]  # Store all generator names
        paths_found = {}
        
        # Create an undirected version of the graph for connectivity analysis
        # Power can flow in both directions through transmission equipment
        undirected_graph = self.graph.to_undirected()
        
        for load in loads:
            reachable_generators = []
            for generator in generators:
                try:
                    # Check if there's a path between generator and load in the undirected graph
                    if nx.has_path(undirected_graph, generator, load):
                        reachable_generators.append(generator)
                except nx.NetworkXNoPath:
                    continue
            
            if reachable_generators:
                connected_loads += 1
                connected_load_names.append(load)
                paths_found[load] = reachable_generators
            else:
                isolated_load_names.append(load)
        
        total_generators = len(generators)
        total_loads = len(loads)
        connectivity_ratio = connected_loads / total_loads if total_loads > 0 else 0.0
        
        return ConnectivityResult(
            total_generators,
            total_loads,
            connected_loads,
            connectivity_ratio,
            connected_load_names,
            isolated_load_names,
            generator_names,
            paths_found
        )
