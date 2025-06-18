import pandapower as pp
import json
import logging
import io
import math # For potential use with other normalization like sigmoid, though linear is used here

# Configure logging - Assuming this is already done as in the original snippet
# If not, ensure it's configured appropriately.
# Example:
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Or logging.DEBUG for more detailed reward calculation steps

# If logger is not already defined globally for the module, define it:
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if this script/module is reloaded
    logging.basicConfig(level=logging.INFO) # Default level


class PandapowerConverter:
    """
    Converts a SystemGraph object to a Pandapower network
    using a mapping configuration file.
    """
    # Define constants for reward normalization
    # PERFECT_SCORE_RAW is the score given for a network with no diagnostic issues.
    PERFECT_SCORE_RAW = 10.0
    # TARGET_MIN_REWARD_FOR_NORMALIZATION is an estimated score for a network with significant issues.
    # This value can be tuned based on observed raw reward ranges.
    TARGET_MIN_REWARD_FOR_NORMALIZATION = -1000.0


    def __init__(self, ps_graph_obj, mapping_config_path=None):
        self.ps_graph = ps_graph_obj
        
        # Use default mapping config if none provided
        if mapping_config_path is None:
            mapping_config_path = "system_parser/pandapower/simulink_to_pandapower_map.json"
            
        try:
            with open(mapping_config_path, 'r') as f:
                self.mapping_config = json.load(f)
            logger.info(f"Successfully loaded Pandapower mapping from: {mapping_config_path}")
        except FileNotFoundError:
            logger.error(f"Pandapower mapping file not found: {mapping_config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from Pandapower mapping file: {mapping_config_path}")
            raise
        
        self.net = pp.create_empty_network(name=self.ps_graph.graph.name if self.ps_graph.graph else "ConvertedNet")
        self.simulink_block_to_pp_bus_idx = {} # Maps Simulink block name to Pandapower bus index
        self.created_elements_count = {} # To count created pandapower elements
        
        # Track connections used during conversion
        self.used_connections = set()  # Set of (source_port_global_id, target_port_global_id) tuples
        self.all_connections = set()  # Set of all available connections
        self._initialize_connection_tracking()

    def _initialize_connection_tracking(self):
        """Initialize tracking of all available connections in the graph."""
        # Track only the actual directed connections from the input, not reverse directions
        for u, v, data in self.ps_graph.graph.edges(data=True):
            if data.get('edge_type') == 'connects_to':
                self.all_connections.add((u, v))

    def _get_mapped_params(self, block_obj, pp_element_map_info):
        """Helper to get and transform parameters for a pandapower element."""
        params = {}
        params.update(pp_element_map_info.get("default_params", {}))

        for sim_param_key, transform_info in pp_element_map_info.get("value_transformations", {}).items():
            if sim_param_key in block_obj.params:
                value = block_obj.params[sim_param_key]
                
                transform_list = transform_info if isinstance(transform_info, list) else [transform_info]
                
                for transform in transform_list:
                    try:
                        extracted_value = value
                        if transform.get("parse_array", False):
                            extracted_value = value.strip('[]').split(',')[transform.get("extract_index", 0)].strip()
                        
                        num_value = float(extracted_value)
                        if "scale" in transform:
                            num_value *= transform["scale"]
                        
                        target_param = transform.get("target_param")
                        if target_param:
                            params[target_param] = num_value
                    except (ValueError, IndexError, TypeError) as e:
                        logger.warning(f"Could not process parameter '{sim_param_key}' transformation for block '{block_obj.name}': {e}")
                        continue
        
        for sim_param_key, pp_param_key in pp_element_map_info.get("parameter_mapping", {}).items():
            if sim_param_key == "_BLOCK_NAME_":
                params[pp_param_key] = block_obj.name
                continue

            if sim_param_key in block_obj.params and pp_param_key not in params:
                value = block_obj.params[sim_param_key]
                try:
                    params[pp_param_key] = float(value)
                except ValueError:
                    params[pp_param_key] = value 
        return params

    def _find_connected_bus_idx(self, block_name, connection_port_group_sim):
        if not connection_port_group_sim:
            logger.warning(f"Block {block_name} has no connection_port_group defined for bus lookup.")
            return None
        
        # Track all ports in the connection group, not just the representative one
        bus_idx = None
        for sim_port_name in connection_port_group_sim:
            block_port_global_id = f"{block_name}_{sim_port_name}"
            
            if block_port_global_id not in self.ps_graph.graph: # type: ignore
                continue
                
            for neighbor_direction in [self.ps_graph.graph.successors, self.ps_graph.graph.predecessors]: # type: ignore
                for neighbor_port_gid in neighbor_direction(block_port_global_id):
                    neighbor_block_name = self.ps_graph.graph.nodes[neighbor_port_gid].get('block_name') # type: ignore
                    if neighbor_block_name in self.simulink_block_to_pp_bus_idx:
                        # Track this connection as used - check both possible directions in original input
                        connection_forward = (block_port_global_id, neighbor_port_gid)
                        connection_reverse = (neighbor_port_gid, block_port_global_id)
                        
                        # Add whichever direction exists in the original connections
                        if connection_forward in self.all_connections:
                            self.used_connections.add(connection_forward)
                        elif connection_reverse in self.all_connections:
                            self.used_connections.add(connection_reverse)
                        
                        # Store the bus index to return (all ports should connect to the same bus)
                        if bus_idx is None:
                            bus_idx = self.simulink_block_to_pp_bus_idx[neighbor_block_name]
        
        if bus_idx is None:
            logger.warning(f"Could not find connected Pandapower bus for '{block_name}' via any port in group {connection_port_group_sim}.")
        
        return bus_idx

    def _create_buses(self):
        logger.info("--- Creating Pandapower Buses ---")
        for block_name, block_obj in self.ps_graph.blocks.items(): # type: ignore
            pp_element_map_info = self.mapping_config.get(block_obj.type)
            if pp_element_map_info and pp_element_map_info["pandapower_element"] == "bus":
                params = self._get_mapped_params(block_obj, pp_element_map_info)
                if "vn_kv" not in params:
                    logger.error(f"Bus '{block_name}' is missing 'vn_kv'. Skipping.")
                    continue
                try:
                    bus_idx = pp.create_bus(self.net, **params) # type: ignore
                    self.simulink_block_to_pp_bus_idx[block_name] = bus_idx
                    logger.info(f"Created BUS '{params.get('name', block_name)}' (vn_kv={params['vn_kv']}) -> pp_idx:{bus_idx}")
                    self.created_elements_count['bus'] = self.created_elements_count.get('bus', 0) + 1
                except Exception as e:
                    logger.error(f"Failed to create Pandapower bus for '{block_name}': {e}. Params: {params}")

    def _create_ext_grids(self):
        logger.info("--- Creating Pandapower External Grids ---")
        for block_name, block_obj in self.ps_graph.blocks.items(): # type: ignore
            pp_element_map_info = self.mapping_config.get(block_obj.type)
            if pp_element_map_info and pp_element_map_info["pandapower_element"] == "ext_grid":
                sim_conn_ports = pp_element_map_info.get("connection_port_group")
                bus_idx = self._find_connected_bus_idx(block_name, sim_conn_ports)
                if bus_idx is None:
                    logger.warning(f"Skipping EXT_GRID '{block_name}' as connected bus not found.")
                    continue
                
                params = self._get_mapped_params(block_obj, pp_element_map_info)
                params.setdefault("vm_pu", 1.0) # Default if not in mapping or block

                try:
                    pp.create_ext_grid(self.net, bus_idx, **params) # type: ignore
                    logger.info(f"Created EXT_GRID '{params.get('name', block_name)}' at bus_idx:{bus_idx} (vm_pu={params['vm_pu']})")
                    self.created_elements_count['ext_grid'] = self.created_elements_count.get('ext_grid', 0) + 1
                except Exception as e:
                    logger.error(f"Failed to create Pandapower ext_grid for '{block_name}': {e}. Params: {params}")

    def _create_gens(self):
        logger.info("--- Creating Pandapower Generators ---")
        for block_name, block_obj in self.ps_graph.blocks.items(): # type: ignore
            pp_element_map_info = self.mapping_config.get(block_obj.type)
            if pp_element_map_info and pp_element_map_info["pandapower_element"] == "gen": # Corrected from "Gen" to "gen"
                sim_conn_ports = pp_element_map_info.get("connection_port_group")
                bus_idx = self._find_connected_bus_idx(block_name, sim_conn_ports)
                if bus_idx is None:
                    logger.warning(f"Skipping GEN '{block_name}' as connected bus not found.")
                    continue
                
                params = self._get_mapped_params(block_obj, pp_element_map_info)
                params.setdefault("p_mw", 1.0) # Default if not provided
                params.setdefault("vm_pu", 1.0) # Default if not provided
                # Slack is often important, make it configurable via mapping or a default
                is_slack = params.pop("slack", pp_element_map_info.get("default_params", {}).get("slack", True)) # Default to True if not specified


                try:
                    # Pandapower uses 'gen' for the element type key
                    pp.create_gen(self.net, bus_idx, slack=is_slack, **params) # type: ignore
                    logger.info(f"Created GEN '{params.get('name', block_name)}' at bus_idx:{bus_idx} (p_mw={params['p_mw']}, vm_pu={params['vm_pu']}, slack={is_slack})")
                    self.created_elements_count['gen'] = self.created_elements_count.get('gen', 0) + 1
                except Exception as e:
                    logger.error(f"Failed to create Pandapower generator for '{block_name}': {e}. Params: {params}")

    def _create_loads(self):
        logger.info("--- Creating Pandapower Loads ---")
        for block_name, block_obj in self.ps_graph.blocks.items(): # type: ignore
            pp_element_map_info = self.mapping_config.get(block_obj.type)
            if pp_element_map_info and pp_element_map_info["pandapower_element"] == "load":
                sim_conn_ports = pp_element_map_info.get("connection_port_group")
                bus_idx = self._find_connected_bus_idx(block_name, sim_conn_ports)
                if bus_idx is None:
                    logger.warning(f"Skipping LOAD '{block_name}' as connected bus not found.")
                    continue

                params = self._get_mapped_params(block_obj, pp_element_map_info)
                if "p_mw" not in params:
                    logger.warning(f"LOAD '{block_name}' is missing 'p_mw'. Skipping.")
                    continue
                
                try:
                    pp.create_load(self.net, bus_idx, **params) # type: ignore
                    logger.info(f"Created LOAD '{params.get('name', block_name)}' at bus_idx:{bus_idx} (p_mw={params['p_mw']})")
                    self.created_elements_count['load'] = self.created_elements_count.get('load', 0) + 1
                except Exception as e:
                    logger.error(f"Failed to create Pandapower load for '{block_name}': {e}. Params: {params}")

    def _find_connected_bus_idx_for_line_terminal(self, line_block_name, line_terminal_sim_ports, terminal_label):
        if not line_terminal_sim_ports:
            logger.warning(f"Line '{line_block_name}' terminal '{terminal_label}' no ports in mapping.")
            return None
        
        # Check ALL ports in the terminal group to ensure they connect to the same bus
        connected_bus_idx = None
        all_ports_connected = True
        used_connections_for_terminal = []
        
        for line_port_name in line_terminal_sim_ports:
            line_port_global_id = f"{line_block_name}_{line_port_name}"
            
            if line_port_global_id not in self.ps_graph.graph: # type: ignore
                logger.warning(f"Port '{line_port_global_id}' for line '{line_block_name}' terminal '{terminal_label}' not in graph.")
                all_ports_connected = False
                continue
            
            port_bus_idx = None
            port_used_connections = []
            
            # Check predecessors
            for pred_port_gid in self.ps_graph.graph.predecessors(line_port_global_id): # type: ignore
                pred_block_name = self.ps_graph.graph.nodes[pred_port_gid].get('block_name') # type: ignore
                if pred_block_name in self.simulink_block_to_pp_bus_idx:
                    # Check which direction exists in original connections
                    connection_forward = (pred_port_gid, line_port_global_id)
                    connection_reverse = (line_port_global_id, pred_port_gid)
                    
                    if connection_forward in self.all_connections:
                        port_used_connections.append(connection_forward)
                    elif connection_reverse in self.all_connections:
                        port_used_connections.append(connection_reverse)
                    
                    port_bus_idx = self.simulink_block_to_pp_bus_idx[pred_block_name]
                    break
                    
            # Check successors if no predecessor found
            if port_bus_idx is None:
                for succ_port_gid in self.ps_graph.graph.successors(line_port_global_id): # type: ignore
                    succ_block_name = self.ps_graph.graph.nodes[succ_port_gid].get('block_name') # type: ignore
                    if succ_block_name in self.simulink_block_to_pp_bus_idx:
                        # Check which direction exists in original connections
                        connection_forward = (line_port_global_id, succ_port_gid)
                        connection_reverse = (succ_port_gid, line_port_global_id)
                        
                        if connection_forward in self.all_connections:
                            port_used_connections.append(connection_forward)
                        elif connection_reverse in self.all_connections:
                            port_used_connections.append(connection_reverse)
                        
                        port_bus_idx = self.simulink_block_to_pp_bus_idx[succ_block_name]
                        break
            
            if port_bus_idx is None:
                logger.warning(f"No connected bus for line '{line_block_name}' terminal '{terminal_label}' port '{line_port_name}'.")
                all_ports_connected = False
                continue
            
            # Check if all ports connect to the same bus
            if connected_bus_idx is None:
                connected_bus_idx = port_bus_idx
            elif connected_bus_idx != port_bus_idx:
                logger.warning(f"Line '{line_block_name}' terminal '{terminal_label}' has ports connected to different buses: {connected_bus_idx} vs {port_bus_idx}. This is not a valid three-phase line configuration.")
                return None
            
            # Store connections for this port
            used_connections_for_terminal.extend(port_used_connections)
        
        if not all_ports_connected or connected_bus_idx is None:
            logger.warning(f"Line '{line_block_name}' terminal '{terminal_label}' does not have all ports properly connected to a single bus.")
            return None
        
        # Track all used connections for this terminal
        for connection in used_connections_for_terminal:
            self.used_connections.add(connection)
        
        logger.debug(f"Line '{line_block_name}' terminal '{terminal_label}' successfully connected to bus {connected_bus_idx}")
        return connected_bus_idx

    def _create_lines(self):
        logger.info("--- Creating Pandapower Lines ---")
        for block_name, block_obj in self.ps_graph.blocks.items(): # type: ignore
            pp_element_map_info = self.mapping_config.get(block_obj.type)
            if pp_element_map_info and pp_element_map_info["pandapower_element"] == "line":
                terminal_map = pp_element_map_info.get("terminal_ports")
                if not terminal_map:
                    logger.warning(f"LINE '{block_name}' missing 'terminal_ports' in mapping. Skipping.")
                    continue

                from_sim_ports = terminal_map.get("from_terminal_ports")
                to_sim_ports = terminal_map.get("to_terminal_ports")

                from_bus_idx = self._find_connected_bus_idx_for_line_terminal(block_name, from_sim_ports, "from")
                to_bus_idx = self._find_connected_bus_idx_for_line_terminal(block_name, to_sim_ports, "to")

                if from_bus_idx is None or to_bus_idx is None:
                    logger.warning(f"Skipping LINE '{block_name}' as buses not found (From: {from_bus_idx}, To: {to_bus_idx}).")
                    continue
                
                params = self._get_mapped_params(block_obj, pp_element_map_info)
                if "length_km" not in params or "std_type" not in params:
                     logger.warning(f"LINE '{block_name}' missing 'length_km' or 'std_type'. Skipping.")
                     continue
                
                try:
                    pp.create_line(self.net, from_bus_idx, to_bus_idx, **params) # type: ignore
                    logger.info(f"Created LINE '{params.get('name', block_name)}' from bus:{from_bus_idx} to bus:{to_bus_idx} (len={params['length_km']})")
                    self.created_elements_count['line'] = self.created_elements_count.get('line', 0) + 1
                except Exception as e:
                    logger.error(f"Failed to create Pandapower line for '{block_name}': {e}. Params: {params}")

    def _create_transformers(self):
        logger.info("--- Creating Pandapower Transformers ---")
        for block_name, block_obj in self.ps_graph.blocks.items(): # type: ignore
            pp_element_map_info = self.mapping_config.get(block_obj.type)
            if pp_element_map_info and pp_element_map_info["pandapower_element"] == "trafo":
                terminal_map = pp_element_map_info.get("terminal_ports")
                if not terminal_map:
                    logger.warning(f"TRANSFORMER '{block_name}' missing 'terminal_ports' in mapping. Skipping.")
                    continue

                hv_sim_ports = terminal_map.get("hv_terminal_ports")
                lv_sim_ports = terminal_map.get("lv_terminal_ports")

                hv_bus_idx = self._find_connected_bus_idx_for_line_terminal(block_name, hv_sim_ports, "hv")
                lv_bus_idx = self._find_connected_bus_idx_for_line_terminal(block_name, lv_sim_ports, "lv")
                
                params = self._get_mapped_params(block_obj, pp_element_map_info)
                
                required_params = ["sn_mva", "vn_hv_kv", "vn_lv_kv"] 
                missing_params = [p for p in required_params if p not in params and p not in pp_element_map_info.get("default_params", {})]
                if missing_params and "std_type" not in params : 
                    logger.warning(f"TRANSFORMER '{block_name}' may be missing required parameters if not using std_type: {missing_params}. Proceeding with available params.")
                
                try:
                    if hv_bus_idx is None or lv_bus_idx is None:
                        logger.error(f"Cannot create TRANSFORMER '{params.get('name', block_name)}' due to missing bus connections (HV: {hv_bus_idx}, LV: {lv_bus_idx}).")
                        continue

                    pp.create_transformer_from_parameters(self.net, hv_bus_idx, lv_bus_idx, **params) # type: ignore
                    logger.info(f"Created TRANSFORMER '{params.get('name', block_name)}' from bus:{hv_bus_idx} to bus:{lv_bus_idx} (sn_mva={params.get('sn_mva','N/A')})")
                    self.created_elements_count['trafo'] = self.created_elements_count.get('trafo', 0) + 1
                except Exception as e:
                    logger.error(f"Failed to create Pandapower transformer for '{block_name}': {e}. HV bus: {hv_bus_idx}, LV bus: {lv_bus_idx}, Params: {params}")

    def _fix_geo_coordinates(self):
        """
        Fix None values in geo columns of the bus dataframe with reasonable default coordinates.
        This prevents JSON serialization errors during pandapower plotting.
        """
        if self.net.bus.shape[0] == 0:
            return
        
        import numpy as np
        
        # Check if geo column exists and has None values
        if 'geo' in self.net.bus.columns:
            # Check for None values in the geo column
            null_mask = self.net.bus['geo'].isnull()
            if null_mask.any():
                num_buses = self.net.bus.shape[0]
                
                # Create reasonable default coordinates in a simple grid pattern
                # Each geo entry should be a tuple/list of (x, y) coordinates
                for i, bus_idx in enumerate(self.net.bus.index[null_mask]):
                    if num_buses <= 10:
                        # Arrange buses horizontally for small networks
                        x = i * 100.0
                        y = 0.0
                    else:
                        # Arrange buses in a grid pattern for larger networks
                        grid_width = int(np.ceil(np.sqrt(num_buses)))
                        x = (i % grid_width) * 100.0
                        y = (i // grid_width) * 100.0
                    
                    # Set the geo coordinate as a GeoJSON Point string
                    # Pandapower plotting expects valid GeoJSON format
                    import json
                    geojson_point = {
                        "type": "Point",
                        "coordinates": [x, y]
                    }
                    self.net.bus.at[bus_idx, 'geo'] = json.dumps(geojson_point)
                
                logger.info(f"Fixed {null_mask.sum()} None values in bus.geo column with default coordinates")

    def create_network_plot(self, output_path="system_parser/examples/pandapower_network_plot.png"):
        """
        Create a pandapower network plot, fixing geo coordinates only if initial plotting fails.
        
        Args:
            output_path (str): Path to save the plot image
            
        Returns:
            bool: True if plot was created successfully, False otherwise
        """
        if self.net.bus.shape[0] == 0:
            logger.warning("Network has no buses - skipping pandapower plot")
            return False
        
        import matplotlib.pyplot as plt
        import pandapower.plotting as pplot
        
        # First attempt: try plotting without fixing geo coordinates
        try:
            plt.figure(figsize=(6, 4))
            pplot.simple_plot(self.net, plot_loads=True, plot_gens=True, 
                              plot_sgens=True, show_plot=False)
            # plt.title("Pandapower Network Layout")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved pandapower network plot to: {output_path}")
            return True
            
        except Exception as e:
            plt.close()  # Ensure figure is closed even if plotting failed
            logger.warning(f"Initial plotting attempt failed: {e}")
            logger.info("Attempting to fix geo coordinates and retry plotting...")
            
            # Second attempt: fix geo coordinates and try again
            try:
                self._fix_geo_coordinates()
                
                plt.figure(figsize=(6, 4))
                pplot.simple_plot(self.net, plot_loads=True, plot_gens=True, 
                                  plot_sgens=True, show_plot=False)
                # plt.title("Pandapower Network Layout")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved pandapower network plot to: {output_path} (after fixing geo coordinates)")
                return True
                
            except Exception as e2:
                plt.close()  # Ensure figure is closed
                logger.warning(f"Could not create pandapower network plot even after fixing geo coordinates: {e2}")
                return False

    def convert_to_pandapower_net(self, run_diagnostics=True):
        """
        Main conversion method. Creates elements in a specific order.
        Now returns a tuple: (pandapower_network, diagnostic_reward, conversion_reward)
        
        Args:
            run_diagnostics (bool): Whether to run diagnostics after conversion
        
        Returns:
            tuple: (pandapower.pandapowerNet, float or None, float)
                   The pandapower network object, the normalized diagnostic reward (0-1), 
                   and the conversion success reward (0-1).
                   Diagnostic reward is None if diagnostics are not run.
        """
        if not self.ps_graph or not self.ps_graph.blocks: # type: ignore
            logger.error("Cannot convert: PowerSystemGraph has no blocks.")
            return self.net, None, 0.0 # Return empty net, None for diagnostic reward, 0.0 for conversion reward

        logger.info("Starting Pandapower network conversion...")
        self._create_buses()
        self._create_ext_grids()
        self._create_gens()
        self._create_loads()
        self._create_lines()
        self._create_transformers()

        logger.info("Pandapower network conversion finished.")
        logger.info(f"Created elements: {self.created_elements_count}")
        if not self.net.bus.shape[0] and not self.net.line.shape[0] and not self.net.ext_grid.shape[0]: # type: ignore
            logger.warning("The resulting Pandapower network is empty.")

        # Calculate conversion success reward
        conversion_reward = self.get_conversion_success_reward()
        logger.info(f"Conversion success reward: {conversion_reward:.4f} (0=worst, 1=best)")

        # Calculate diagnostic reward if requested
        diagnostic_reward = None
        if run_diagnostics:
            diag_results = self.run_diagnostics()
            if diag_results is not None: 
                diagnostic_reward = self.get_diagnostic_reward(diag_results)
                logger.info(f"Normalized diagnostic reward score: {diagnostic_reward:.4f} (0=worst, 1=best)")
            else:
                logger.error("Diagnostics failed to produce results, cannot calculate reward.")
                diagnostic_reward = 0.0 # Assign worst reward if diagnostics failed to produce results for reward calculation
            
        return self.net, diagnostic_reward, conversion_reward

    def run_diagnostics(self, return_result_dict=True):
        """
        Runs pandapower's built-in diagnostic function, provides concise feedback,
        and augments results with critical failure flags from logs.
        
        Args:
            return_result_dict (bool): If True, returns diagnostic results dictionary
            
        Returns:
            dict: Diagnostic results if return_result_dict=True, None otherwise.
                  The dictionary may be augmented with keys like 'main_power_flow_convergence_failure'.
        """
        logger.info("--- Running Pandapower Built-in Diagnostics ---")
        
        diag_logger_pandapower = logging.getLogger('pandapower.diagnostic_reports')
        original_level = diag_logger_pandapower.level
        original_handlers = diag_logger_pandapower.handlers[:]
        
        log_capture_string = io.StringIO()
        stream_handler = logging.StreamHandler(log_capture_string)
        formatter = logging.Formatter('%(message)s') 
        stream_handler.setFormatter(formatter)
        
        diag_logger_pandapower.handlers = [stream_handler]
        diag_logger_pandapower.setLevel(logging.WARNING) 
        diag_logger_pandapower.propagate = False

        diag_results_dict = {}
        raw_log_output = ""

        try:
            diag_results_dict = pp.diagnostic( # type: ignore
                self.net, 
                report_style=None, 
                warnings_only=True, 
                return_result_dict=True 
            )
            raw_log_output = log_capture_string.getvalue()
            
            critical_messages_found = False
            if raw_log_output:
                print("\n--- Concise Diagnostic Feedback (from captured logs) ---")
                problem_indicators = [
                    "deviate more than +/-", 
                    "Overload check failed: Power flow still does not converge",
                    "Power flow still does not converge with all switches closed",
                    "Power Flow nr did not converge", 
                    "numba_comparison failed", 
                    "Disconnected bus", 
                    "Multiple connections to bus",
                    "implausible impedance values", 
                    "No external grid found", 
                    "multiple gens and/or ext_grids",
                    "zero impedance",
                    "different voltage levels connected"
                ]
                
                for line in raw_log_output.splitlines():
                    line_lower = line.lower() 
                    if any(indicator.lower() in line_lower for indicator in problem_indicators):
                        print(line) 
                        critical_messages_found = True
                
                if not critical_messages_found:
                    # This message might be too verbose if there are minor, non-critical warnings.
                    # Consider if it should only print if `problem_indicators` are not found but log is non-empty.
                    # print("Pandapower diagnostics reported some warnings/errors. Review full output if necessary.")
                    pass # Decided to keep it less verbose if only non-critical warnings are present
            
            if not critical_messages_found and not raw_log_output:
                 print("Pandapower diagnostics reported no warnings or errors via logs.")
            print("-------------------------------------------------------\n")

            # Ensure diag_results_dict is a dictionary before augmenting
            if not isinstance(diag_results_dict, dict):
                diag_results_dict = {}


            if "Power Flow nr did not converge".lower() in raw_log_output.lower() or \
               ("numba_comparison failed".lower() in raw_log_output.lower() and "converge" in raw_log_output.lower()):
                diag_results_dict['main_power_flow_convergence_failure'] = True
            
            if "Overload check failed: Power flow still does not converge".lower() in raw_log_output.lower():
                diag_results_dict['overload_check_convergence_failure'] = True
            
            if "Power flow still does not converge with all switches closed".lower() in raw_log_output.lower():
                diag_results_dict['switch_check_convergence_failure'] = True

        except Exception as e:
            logger.error(f"Failed to run pandapower diagnostics or process logs: {e}", exc_info=True)
            print(f"ERROR: Failed during pandapower diagnostics: {e}")
            if not isinstance(diag_results_dict, dict): diag_results_dict = {} # Ensure it's a dict
            diag_results_dict["error"] = str(e) 
        finally:
            log_capture_string.close()
            diag_logger_pandapower.handlers = original_handlers
            diag_logger_pandapower.setLevel(original_level)
            diag_logger_pandapower.propagate = True
            
        if not diag_results_dict and not raw_log_output : 
            logger.info("Pandapower diagnostic tool returned no structured results and no log warnings/errors.")
        elif not diag_results_dict: # diag_results_dict might be None or empty if pp.diagnostic has issues.
            logger.info("Pandapower diagnostic tool returned no/empty structured results, but logs were captured.")
            if not isinstance(diag_results_dict, dict): # Ensure it's a dict for return
                 diag_results_dict = {}


        if return_result_dict:
            return diag_results_dict
        return None


    def get_diagnostic_reward(self, diag_results):
        """
        Calculates a raw reward score based on diagnostic results and then normalizes it
        to a 0-1 range (0 = worst, 1 = best).

        Args:
            diag_results (dict): The dictionary from run_diagnostics (potentially augmented).

        Returns:
            float: The normalized reward score (0.0 to 1.0).
        """
        if not isinstance(diag_results, dict):
            logger.error("Invalid diag_results for reward calculation. Expected dict, got %s.", type(diag_results))
            return 0.0 # Worst possible normalized score for invalid input

        raw_reward = 0.0
        
        weights = {
            "main_power_flow_convergence_failure": -200.0,
            "overload_check_convergence_failure": -150.0,
            "switch_check_convergence_failure": -120.0,
            "error": -250.0, 
            "no_ext_grid": -100.0, 
            "disconnected_elements": -50.0, 
            "nominal_voltages_dont_match": -20.0, 
            "implausible_impedance_values": -25.0, 
            "multiple_voltage_controlling_elements_per_island": -30.0, 
            "different_voltage_levels_connected": -20.0, 
            "invalid_values": -15.0, 
            "overload": -10.0, 
            "deviations_from_std_type": -5.0, 
            "parallel_switches": -2.0,      
            "problematic_switches": -10.0,  
            "missing_bus_indices": -40.0,   
        }
        
        has_issues = False

        if diag_results.get("error"):
            raw_reward += weights["error"]
            has_issues = True
            logger.debug(f"Raw reward penalty: Diagnostic error ({weights['error']})")
        if diag_results.get("main_power_flow_convergence_failure"):
            raw_reward += weights["main_power_flow_convergence_failure"]
            has_issues = True
            logger.debug(f"Raw reward penalty: Main PF convergence failure ({weights['main_power_flow_convergence_failure']})")
        if diag_results.get("overload_check_convergence_failure"):
            raw_reward += weights["overload_check_convergence_failure"]
            has_issues = True
            logger.debug(f"Raw reward penalty: Overload check convergence failure ({weights['overload_check_convergence_failure']})")
        if diag_results.get("switch_check_convergence_failure"):
            raw_reward += weights["switch_check_convergence_failure"]
            has_issues = True
            logger.debug(f"Raw reward penalty: Switch check convergence failure ({weights['switch_check_convergence_failure']})")

        if diag_results.get("no_ext_grid", False): 
            raw_reward += weights["no_ext_grid"]
            has_issues = True
            logger.debug(f"Raw reward penalty: No external grid ({weights['no_ext_grid']})")

        nom_volt_issues = diag_results.get("nominal_voltages_dont_match", {})
        count = 0
        if isinstance(nom_volt_issues, dict):
            for _el_type, issues_by_terminal in nom_volt_issues.items():
                if isinstance(issues_by_terminal, dict):
                    for _terminal, elements in issues_by_terminal.items():
                        count += len(elements) if isinstance(elements, list) else 0
        if count > 0:
            raw_reward += weights["nominal_voltages_dont_match"] * count
            has_issues = True
            logger.debug(f"Raw reward penalty: {count} nominal_voltages_dont_match issues ({weights['nominal_voltages_dont_match'] * count})")
        
        impl_imp_issues = diag_results.get("implausible_impedance_values", [])
        count = 0
        if isinstance(impl_imp_issues, list):
            for item in impl_imp_issues:
                if isinstance(item, dict):
                    for _el_type, elements in item.items():
                        count += len(elements) if isinstance(elements, list) else 0
        if count > 0:
            raw_reward += weights["implausible_impedance_values"] * count
            has_issues = True
            logger.debug(f"Raw reward penalty: {count} implausible_impedance_values issues ({weights['implausible_impedance_values'] * count})")

        disconnected = diag_results.get("disconnected_elements", [])
        count = 0
        if isinstance(disconnected, list):
            for group in disconnected: 
                if isinstance(group, dict):
                    for _el_type, el_indices in group.items():
                        count += len(el_indices) if isinstance(el_indices, list) else 0
        if count > 0:
            raw_reward += weights["disconnected_elements"] * count
            has_issues = True
            logger.debug(f"Raw reward penalty: {count} disconnected_elements issues ({weights['disconnected_elements'] * count})")
        
        invalid_vals = diag_results.get("invalid_values", {})
        count = 0
        if isinstance(invalid_vals, dict):
            for _el_type, params_dict in invalid_vals.items():
                if isinstance(params_dict, dict):
                    for _param, indices in params_dict.items():
                        count += len(indices) if isinstance(indices, list) else 0
        if count > 0:
            raw_reward += weights["invalid_values"] * count
            has_issues = True
            logger.debug(f"Raw reward penalty: {count} invalid_values issues ({weights['invalid_values'] * count})")

        overload_findings = diag_results.get("overload", {})
        count = 0
        if not diag_results.get("overload_check_convergence_failure") and isinstance(overload_findings, dict):
            for _el_type, elements in overload_findings.items():
                if isinstance(elements, list):
                    count += len(elements)
        if count > 0:
            raw_reward += weights["overload"] * count
            has_issues = True
            logger.debug(f"Raw reward penalty: {count} elements found overloaded ({weights['overload'] * count})")
        
        is_perfect = not has_issues 
        if is_perfect:
            # Check for other potential issues that might not set has_issues but are present in diag_results
            for key_check in weights:
                 # These are already handled by has_issues or are general flags
                if key_check in ["main_power_flow_convergence_failure", "overload_check_convergence_failure", 
                                 "switch_check_convergence_failure", "error"]:
                    continue
                
                val_check = diag_results.get(key_check)
                if isinstance(val_check, bool) and val_check is True: # e.g. no_ext_grid: True
                    is_perfect = False; break
                if (isinstance(val_check, list) or isinstance(val_check, dict)) and val_check: # Non-empty list/dict
                    is_perfect = False; break
        
        if is_perfect:
            raw_reward += self.PERFECT_SCORE_RAW 
            logger.debug(f"Raw reward base for clean diagnostics: +{self.PERFECT_SCORE_RAW}")

        logger.info(f"Calculated raw diagnostic reward: {raw_reward}")

        # --- Normalize the raw_reward to 0-1 range ---
        norm_denominator = self.PERFECT_SCORE_RAW - self.TARGET_MIN_REWARD_FOR_NORMALIZATION
        
        if norm_denominator <= 1e-6: # Avoid division by zero or very small denominator
            logger.warning("Normalization range is invalid or too small (perfect score approx. min target). Clamping reward.")
            if raw_reward >= self.PERFECT_SCORE_RAW:
                normalized_reward = 1.0
            elif raw_reward <= self.TARGET_MIN_REWARD_FOR_NORMALIZATION:
                normalized_reward = 0.0
            else: 
                # If denominator is ~0, and raw_reward is between min and max, it implies min and max are almost equal.
                # This case is tricky. If they are indeed equal, any value is valid if raw_reward also equals them.
                # Let's assume for safety, if raw_reward is close to PERFECT_SCORE_RAW, make it 1.
                if abs(raw_reward - self.PERFECT_SCORE_RAW) < 1e-6 :
                    normalized_reward = 1.0
                else: # Otherwise, if raw_reward is different, and range is zero, it's an undefined state for simple linear.
                    normalized_reward = 0.0 # Default to worst if range is problematic and value isn't perfect
        else:
            normalized_reward = (raw_reward - self.TARGET_MIN_REWARD_FOR_NORMALIZATION) / norm_denominator
        
        final_normalized_reward = max(0.0, min(1.0, normalized_reward))
        
        logger.info(f"Final normalized diagnostic reward: {final_normalized_reward:.4f}")
        return final_normalized_reward

    def _format_connection_readable(self, src_port, tgt_port):
        """
        Convert port global IDs to readable format.
        
        Args:
            src_port (str): Source port global ID
            tgt_port (str): Target port global ID
            
        Returns:
            tuple: (readable_connection_string, src_block, src_port_name, tgt_block, tgt_port_name)
        """
        # Extract block and port names from global IDs
        src_parts = src_port.split('_')
        tgt_parts = tgt_port.split('_')
        
        if len(src_parts) >= 2 and len(tgt_parts) >= 2:
            src_block = '_'.join(src_parts[:-1])
            src_port_name = src_parts[-1]
            tgt_block = '_'.join(tgt_parts[:-1])
            tgt_port_name = tgt_parts[-1]
            
            readable_connection = f"{src_block}/{src_port_name} -> {tgt_block}/{tgt_port_name}"
            return readable_connection, src_block, src_port_name, tgt_block, tgt_port_name
        
        return f"{src_port} -> {tgt_port}", "", "", "", ""

    def get_connection_analysis(self):
        """
        Analyze and return both used and unused connections after conversion.
        
        Returns:
            dict: Dictionary containing comprehensive connection information with keys:
                  - 'used_connections': List of used (source_port, target_port) tuples
                  - 'used_connections_readable': List of human-readable used connection strings
                  - 'used_connections_organized': List of organized used connections
                  - 'unused_connections': List of unused (source_port, target_port) tuples
                  - 'unused_connections_readable': List of human-readable unused connection strings
                  - 'unused_connections_organized': List of organized unused connections
                  - 'total_connections': Total number of available connections
                  - 'used_count': Number of connections used during conversion
                  - 'unused_count': Number of unused connections
                  - 'usage_ratio': Ratio of used to total connections (0.0 to 1.0)
        """
        # Find connections that were available but not used
        unused_connections = self.all_connections - self.used_connections
        
        # Convert used connections to readable format and organize them
        used_connections_data = []
        for src_port, tgt_port in self.used_connections:
            readable, src_block, src_port_name, tgt_block, tgt_port_name = self._format_connection_readable(src_port, tgt_port)
            used_connections_data.append({
                'readable': readable,
                'src_block': src_block,
                'src_port': src_port_name,
                'tgt_block': tgt_block,
                'tgt_port': tgt_port_name,
                'raw': (src_port, tgt_port)
            })
        
        # Sort used connections: by source block, source port, target block, target port
        used_connections_data.sort(key=lambda x: (x['src_block'], x['src_port'], x['tgt_block'], x['tgt_port']))
        
        # Convert unused connections to readable format and organize them
        unused_connections_data = []
        for src_port, tgt_port in unused_connections:
            readable, src_block, src_port_name, tgt_block, tgt_port_name = self._format_connection_readable(src_port, tgt_port)
            unused_connections_data.append({
                'readable': readable,
                'src_block': src_block,
                'src_port': src_port_name,
                'tgt_block': tgt_block,
                'tgt_port': tgt_port_name,
                'raw': (src_port, tgt_port)
            })
        
        # Sort unused connections: by source block, source port, target block, target port
        unused_connections_data.sort(key=lambda x: (x['src_block'], x['src_port'], x['tgt_block'], x['tgt_port']))
        
        total_connections = len(self.all_connections)
        used_count = len(self.used_connections)
        unused_count = len(unused_connections)
        usage_ratio = used_count / total_connections if total_connections > 0 else 0.0
        
        return {
            'used_connections': list(self.used_connections),
            'used_connections_readable': [item['readable'] for item in used_connections_data],
            'used_connections_organized': used_connections_data,
            'unused_connections': list(unused_connections),
            'unused_connections_readable': [item['readable'] for item in unused_connections_data],
            'unused_connections_organized': unused_connections_data,
            'total_connections': total_connections,
            'used_count': used_count,
            'unused_count': unused_count,
            'usage_ratio': usage_ratio
        }

    def get_unused_connections(self):
        """
        Identify and return unused connections after conversion.
        
        Returns:
            dict: Dictionary containing unused connection information with keys:
                  - 'unused_connections': List of unused (source_port, target_port) tuples
                  - 'unused_connections_readable': List of human-readable unused connection strings
                  - 'total_connections': Total number of available connections
                  - 'used_connections': Number of connections used during conversion
                  - 'unused_count': Number of unused connections
                  - 'usage_ratio': Ratio of used to total connections (0.0 to 1.0)
        """
        # Use the comprehensive analysis and extract just unused connection info
        analysis = self.get_connection_analysis()
        return {
            'unused_connections': analysis['unused_connections'],
            'unused_connections_readable': analysis['unused_connections_readable'],
            'total_connections': analysis['total_connections'],
            'used_connections': analysis['used_count'],
            'unused_count': analysis['unused_count'],
            'usage_ratio': analysis['usage_ratio']
        }

    def get_conversion_success_reward(self):
        """
        Calculate a conversion success reward based on multiple factors:
        - Connection usage ratio (how many available connections were actually used)
        - Element creation success (how many blocks were successfully converted)
        - Penalty for unused connections (as they are probably wrong)
        - Network structure validation
        
        Returns:
            float: Conversion success reward score (0.0 to 1.0)
        """
        connection_info = self.get_unused_connections()
        
        # Connection usage component (0.0 to 1.0)
        connection_usage_score = connection_info['usage_ratio']
        
        # Penalty for unused connections (they are probably wrong)
        unused_connections_count = connection_info['unused_count']
        total_connections = connection_info['total_connections']
        
        # Calculate penalty: more unused connections = higher penalty
        if total_connections > 0:
            unused_ratio = unused_connections_count / total_connections
            # Penalty scales with unused ratio: 0% unused = no penalty, 100% unused = max penalty
            unused_penalty = unused_ratio * 0.3  # Maximum penalty of 0.3 for all connections unused
        else:
            unused_penalty = 0.0
        
        # Element creation success component
        total_blocks = len(self.ps_graph.blocks) if self.ps_graph and self.ps_graph.blocks else 1
        total_elements_created = sum(self.created_elements_count.values())
        element_creation_score = min(1.0, total_elements_created / total_blocks)
        
        # Network structure component (check if network has basic elements)
        has_buses = self.created_elements_count.get('bus', 0) > 0
        has_power_elements = (self.created_elements_count.get('ext_grid', 0) + 
                             self.created_elements_count.get('gen', 0) + 
                             self.created_elements_count.get('load', 0)) > 0
        
        structure_score = 0.0
        if has_buses:
            structure_score += 0.5
        if has_power_elements:
            structure_score += 0.5
            
        # Weighted combination of factors with penalty applied
        # Give more weight to successful element creation and structure
        base_score = (0.3 * connection_usage_score + 
                     0.4 * element_creation_score + 
                     0.3 * structure_score)
        
        # Apply penalty for unused connections
        final_score = base_score - unused_penalty
        
        logger.debug(f"Conversion reward components: usage={connection_usage_score:.3f}, "
                    f"creation={element_creation_score:.3f}, structure={structure_score:.3f}, "
                    f"unused_penalty={unused_penalty:.3f}, final={final_score:.3f}")
        
        return max(0.0, min(1.0, final_score))

    def print_connection_analysis(self):
        """Print a comprehensive analysis of both used and unused connections after conversion."""
        analysis = self.get_connection_analysis()
        
        print("\n" + "="*80)
        print("CONNECTION ANALYSIS")
        print("="*80)
        print(f"Total connections available: {analysis['total_connections']}")
        print(f"Connections used during conversion: {analysis['used_count']}")
        print(f"Unused connections: {analysis['unused_count']}")
        print(f"Connection usage ratio: {analysis['usage_ratio']:.2%}")
        
        # Print used connections in organized order
        if analysis['used_connections_readable']:
            print(f"\nUSED CONNECTIONS (in organized order):")
            print("-" * 50)
            for i, connection in enumerate(analysis['used_connections_readable'], 1):
                print(f"  {i:2d}. {connection}")
            
            # Also print them in a more compact format grouped by source block
            print(f"\nUSED CONNECTIONS (compact grouped format):")
            print("-" * 50)
            current_block = None
            block_connections = []
            
            for conn_data in analysis['used_connections_organized']:
                src_block = conn_data['src_block']
                if current_block != src_block:
                    if current_block is not None and block_connections:
                        # Print the previous block's connections
                        print(f"  {current_block}: {', '.join(block_connections)}")
                    current_block = src_block
                    block_connections = []
                
                # Format as X/A -> Y/B
                short_format = f"{conn_data['src_port']} -> {conn_data['tgt_block']}/{conn_data['tgt_port']}"
                block_connections.append(short_format)
            
            # Print the last block's connections
            if current_block is not None and block_connections:
                print(f"  {current_block}: {', '.join(block_connections)}")
        else:
            print("\nNo connections were used during conversion!")
        
        # Print unused connections
        if analysis['unused_connections_readable']:
            print(f"\nUNUSED CONNECTIONS (probably wrong):")
            print("-" * 50)
            for i, connection in enumerate(analysis['unused_connections_readable'], 1):
                print(f"  {i:2d}. {connection}")
        else:
            print("\nNo unused connections found - all connections were utilized!")
        
        print("="*80)

    def print_unused_connections(self):
        """Print a summary of unused connections after conversion. (Legacy method - use print_connection_analysis for full details)"""
        self.print_connection_analysis()

    def get_organized_connections_for_storage(self):
        """
        Get organized connections in a format suitable for storage or further analysis.
        
        Returns:
            dict: Dictionary with organized connection data including:
                  - 'summary': Connection count summary
                  - 'used_connections_by_block': Used connections grouped by source block
                  - 'unused_connections_by_block': Unused connections grouped by source block
                  - 'all_used_connections': All used connections in readable format
                  - 'all_unused_connections': All unused connections in readable format
        """
        analysis = self.get_connection_analysis()
        
        # Group used connections by source block
        used_by_block = {}
        for conn_data in analysis['used_connections_organized']:
            src_block = conn_data['src_block']
            if src_block not in used_by_block:
                used_by_block[src_block] = []
            used_by_block[src_block].append(conn_data)
        
        # Group unused connections by source block
        unused_by_block = {}
        for conn_data in analysis['unused_connections_organized']:
            src_block = conn_data['src_block']
            if src_block not in unused_by_block:
                unused_by_block[src_block] = []
            unused_by_block[src_block].append(conn_data)
        
        return {
            'summary': {
                'total_connections': analysis['total_connections'],
                'used_count': analysis['used_count'],
                'unused_count': analysis['unused_count'],
                'usage_ratio': analysis['usage_ratio']
            },
            'used_connections_by_block': used_by_block,
            'unused_connections_by_block': unused_by_block,
            'all_used_connections': analysis['used_connections_readable'],
            'all_unused_connections': analysis['unused_connections_readable']
        }

    def save_connection_analysis(self, filepath):
        """
        Save comprehensive connection analysis to a JSON file.
        
        Args:
            filepath (str): Path to save the analysis file
        """
        import json
        
        organized_data = self.get_organized_connections_for_storage()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(organized_data, f, indent=2)
            logger.info(f"Connection analysis saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save connection analysis to {filepath}: {e}")

    def save_connection_analysis_text(self, filepath):
        """
        Save connection analysis in a human-readable text format.
        
        Args:
            filepath (str): Path to save the text file
        """
        import io
        from contextlib import redirect_stdout
        
        try:
            # Capture the print output
            f = io.StringIO()
            with redirect_stdout(f):
                self.print_connection_analysis()
            
            output = f.getvalue()
            
            # Write to file
            with open(filepath, 'w') as file:
                file.write(output)
            logger.info(f"Connection analysis text saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save connection analysis text to {filepath}: {e}")