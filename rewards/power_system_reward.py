import json
import logging
import copy
from typing import List, Dict, Callable, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass

from utils.xml_parser import XMLParser
from rewards.base_reward import BaseReward
from rewards.math_grader import grade
from rewards.reward_helpers import extract_python_code
from system_parser.pandapower import PandapowerConverter
from system_parser.system_graph import SystemGraph


@dataclass
class RewardComponents:
    """Container for individual reward components."""
    connectivity_reward: float = 0.0
    validation_reward: float = 0.0
    parameter_reward: float = 0.0
    conversion_reward: float = 0.0
    diagnostic_reward: float = 0.0
    load_satisfaction_reward: float = 0.0
    structure_reward: float = 0.0
    tool_execution_reward: float = 0.0
    format_reward: float = 0.0
    xml_reward: float = 0.0
    connection_addition_reward: float = 0.0
    block_addition_reward: float = 0.0
    frequency_coherence_reward: float = 0.0
    voltage_coherence_reward: float = 0.0
    
    def get_total_reward(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted total reward."""
        if weights is None:
            # Default weights
            weights = {
                'connectivity': 0.0,
                'validation': 0.0,
                'parameter': 0.0,
                'conversion': 0.0,
                'diagnostic': 0.0,
                'load_satisfaction': 1.0,
                'structure': 0.0,
                'tool_execution': 0.10,
                'format': 0.05,
                'xml': 0.05,
                'connection_addition': 0.1,
                'block_addition': 0.1,
                'frequency_coherence': 0.0,
                'voltage_coherence': 0.0
            }
        
        total = (
            weights.get('connectivity', 0.0) * self.connectivity_reward +
            weights.get('validation', 0.0) * self.validation_reward +
            weights.get('parameter', 0.0) * self.parameter_reward +
            weights.get('conversion', 0.0) * self.conversion_reward +
            weights.get('diagnostic', 0.0) * self.diagnostic_reward +
            weights.get('load_satisfaction', 0.0) * self.load_satisfaction_reward +
            weights.get('structure', 0.0) * self.structure_reward +
            weights.get('tool_execution', 0.0) * self.tool_execution_reward +
            weights.get('format', 0.0) * self.format_reward +
            weights.get('xml', 0.0) * self.xml_reward +
            weights.get('connection_addition', 0.0) * self.connection_addition_reward +
            weights.get('block_addition', 0.0) * self.block_addition_reward +
            weights.get('frequency_coherence', 0.0) * self.frequency_coherence_reward +
            weights.get('voltage_coherence', 0.0) * self.voltage_coherence_reward
        )
        return max(0.0, min(1.0, total))


class PowerSystemReward(BaseReward):
    """
    A comprehensive reward system that evaluates power system designs across multiple dimensions.
    
    This class extends BaseReward to include power system evaluation components as well as
    standard tool execution and formatting rewards.
    
    Performance Optimization:
    - Calculations are skipped for reward components with weight = 0, improving performance
    - Only necessary computations (e.g., PandapowerConverter) are performed based on active weights
    
    Reward Components:
    - Power System Components:
      - Connectivity: How well generators can reach loads
      - Validation: Basic graph validation (errors, warnings, unconnected ports)
      - Parameters: Correctness of block parameters
      - Conversion: Success of converting to Pandapower format
      - Diagnostics: Power flow and electrical validity
      - Load Satisfaction: Whether loads are adequately supplied
      - Structure: Overall network structure quality
      - Connection Addition: Reward for successfully adding connections (max reward: 3)
      - Block Addition: Reward for successfully adding blocks (max reward: 2)
      - Frequency Coherence: Consistency of frequency values across all blocks
      - Voltage Coherence: Compatibility of voltage levels between connected blocks
    
    - Tool & Format Components:
      - Tool Execution: Success rate of tool calls
      - Format: XML formatting correctness
      - XML: XML structure validity
    """
    
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"]),
                 tools: List[Callable] = [],
                 power_system_weights: Optional[Dict[str, float]] = None):
        # Initialize base class
        super().__init__()
        
        # Store parsers and tools
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {tool.__name__: tool for tool in tools}
        
        # Store power system specific weights
        self.power_system_weights = power_system_weights or {
            'connectivity': 0.0,
            'validation': 0.0,
            'parameter': 0.0,
            'conversion': 0.0,
            'diagnostic': 0.0,
            'load_satisfaction': 1.0,
            'structure': 0.0,
            'tool_execution': 0.10,
            'format': 0.05,
            'xml': 0.05,
            'connection_addition': 0.1,
            'block_addition': 0.1,
            'frequency_coherence': 0.0,
            'voltage_coherence': 0.0
        }
        
        # Set up reward functions - power system evaluation serves as the main reward
        self.reward_funcs = [
            self.power_system_reward_func,  # This replaces correct_answer_reward_func
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ]
        self.reward_weights = [
            2.0,  # Higher weight for power system evaluation
            0.5,
            0.25,
            0.25,
        ]
        
        # Add tool-specific reward functions
        for tool_name in self.tools.keys():
            self.reward_funcs.append(self.get_named_tool_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_count_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_attempt_reward_func(tool_name))
            self.reward_weights.append(0.0)

    @contextmanager
    def _suppress_logging(self):
        """Context manager to temporarily suppress logging from power system modules."""
        logger = logging.getLogger('power_system_parser.system_graph')
        pp_logger = logging.getLogger('power_system_parser.pandapower_converter')
        
        original_level = logger.level
        pp_original_level = pp_logger.level
        
        logger.setLevel(logging.CRITICAL)
        pp_logger.setLevel(logging.CRITICAL)
        
        try:
            yield
        finally:
            logger.setLevel(original_level)
            pp_logger.setLevel(pp_original_level)
    
    def _calculate_connectivity_reward(self, system_graph: SystemGraph) -> float:
        """Calculate reward based on generator-to-load connectivity."""
        try:
            connectivity_result = system_graph.analyze_connectivity()
            return connectivity_result.connectivity_ratio
        except Exception as e:
            print(f"Error calculating connectivity reward: {e}")
            return 0.0
    
    def _calculate_validation_reward(self, system_graph: SystemGraph) -> float:
        """Calculate reward based on basic graph validation (errors, warnings, unconnected ports)."""
        try:
            errors = len(system_graph.validation_errors)
            warnings = len(system_graph.validation_warnings)
            unconnected = len(system_graph.unconnected_ports)
            
            # Weight factors: errors are most severe, warnings moderate, unconnected ports mild
            penalty = errors + 0.5 * warnings + 0.2 * unconnected
            reward = max(0.0, min(1.0, 1.0 / (1.0 + penalty)))
            return reward
        except Exception as e:
            print(f"Error calculating validation reward: {e}")
            return 0.0
    
    def _calculate_parameter_reward(self, system_graph: SystemGraph) -> float:
        """Calculate reward based on parameter validation issues."""
        try:
            param_issues = len(system_graph.param_validation_issues)
            # Convert parameter issues to a 0-1 reward
            reward = max(0.0, min(1.0, 1.0 / (1.0 + param_issues)))
            return reward
        except Exception as e:
            print(f"Error calculating parameter reward: {e}")
            return 0.0
    
    def _calculate_conversion_reward(self, converter: PandapowerConverter) -> float:
        """Calculate reward based on conversion success to Pandapower format."""
        try:
            return converter.get_conversion_success_reward()
        except Exception as e:
            print(f"Error calculating conversion reward: {e}")
            return 0.0
    
    def _calculate_diagnostic_reward(self, converter: PandapowerConverter) -> Tuple[float, dict]:
        """Calculate reward based on Pandapower diagnostics."""
        try:
            _, diagnostic_results, _ = converter.convert_to_pandapower_net(run_diagnostics=True)
            diagnostic_reward = converter.get_diagnostic_reward(diagnostic_results)
            return diagnostic_reward, diagnostic_results
        except Exception as e:
            print(f"Error calculating diagnostic reward: {e}")
            return 0.0, {}
    
    def _calculate_load_satisfaction_reward(self, system_graph: SystemGraph, diagnostic_results) -> float:
        """Calculate reward based on whether loads are being adequately satisfied."""
        try:
            # Ensure diagnostic_results is always a dict
            if not isinstance(diagnostic_results, dict):
                print(f"Warning: diagnostic_results is not a dict (type: {type(diagnostic_results)}), defaulting to empty dict")
                diagnostic_results = {}
            
            # Basic approach: check if loads are connected and no critical power flow issues
            connectivity_result = system_graph.analyze_connectivity()
            base_satisfaction = connectivity_result.connectivity_ratio
            
            # Penalize if there are power flow convergence failures
            convergence_penalty = 0.0
            if diagnostic_results.get("main_power_flow_convergence_failure", False):
                convergence_penalty = 0.5
            
            # Penalize if there are overload issues
            overload_penalty = 0.0
            overload_count = 0
            overload_findings = diagnostic_results.get("overload", {})
            if isinstance(overload_findings, dict):
                for elements in overload_findings.values():
                    if isinstance(elements, list):
                        overload_count += len(elements)
            if overload_count > 0:
                overload_penalty = min(0.3, overload_count * 0.1)
            
            reward = max(0.0, base_satisfaction - convergence_penalty - overload_penalty)
            return reward
        except Exception as e:
            print(f"Error calculating load satisfaction reward: {e}")
            return 0.0
    
    def _calculate_structure_reward(self, system_graph: SystemGraph, converter: PandapowerConverter) -> float:
        """Calculate reward based on overall network structure quality."""
        try:
            # Check for basic structural requirements
            structure_score = 0.0
            
            # Check if we have essential element types
            has_generators = any(block.type in {'Three-Phase Source', 'Synchronous Machine', 'AC Voltage Source'} 
                               for block in system_graph.blocks.values())
            has_loads = any(block.type in {'Three-Phase Parallel RLC Load', 'Three-Phase Series RLC Load'} 
                          for block in system_graph.blocks.values())
            has_transmission = any(block.type in {'Three-Phase PI Section Line', 'Three-Phase Transformer (Two Windings)'} 
                                 for block in system_graph.blocks.values())
            
            if has_generators:
                structure_score += 0.4
            if has_loads:
                structure_score += 0.4
            if has_transmission:
                structure_score += 0.2
            
            # Check if conversion created the right elements
            created_elements = converter.created_elements_count
            if created_elements.get('bus', 0) > 0:
                structure_score = min(1.0, structure_score + 0.2)
            
            return min(1.0, structure_score)
        except Exception as e:
            print(f"Error calculating structure reward: {e}")
            return 0.0
    
    def _calculate_connection_addition_reward(self, system_graph: SystemGraph, initial_system_graph: SystemGraph, max_reward: float = 3.0) -> float:
        """Calculate reward based on the number of connections successfully added.
        
        Args:
            system_graph: Final system graph
            initial_system_graph: Initial system graph (can be None)
            max_reward: Maximum reward value (default: 3.0)
            
        Returns:
            float: Reward value between 0.0 and 1.0
        """
        try:
            if initial_system_graph is None:
                # If no initial graph, count all connections in final graph
                initial_connections = 0
            else:
                # Count connections in initial graph
                initial_connections = sum(1 for _, _, data in initial_system_graph.graph.edges(data=True) 
                                        if data.get('edge_type') == 'connects_to')
            
            # Count connections in final graph
            final_connections = sum(1 for _, _, data in system_graph.graph.edges(data=True) 
                                  if data.get('edge_type') == 'connects_to')
            
            # Calculate connections added
            connections_added = max(0, final_connections - initial_connections)
            
            # Normalize to 0-1 scale with max_reward as the cap
            reward = min(connections_added / max_reward, 1.0) if max_reward > 0 else 0.0
            
            return reward
        except Exception as e:
            print(f"Error calculating connection addition reward: {e}")
            return 0.0
    
    def _calculate_block_addition_reward(self, system_graph: SystemGraph, initial_system_graph: SystemGraph, max_reward: float = 2.0) -> float:
        """Calculate reward based on the number of blocks successfully added.
        
        Args:
            system_graph: Final system graph
            initial_system_graph: Initial system graph (can be None)
            max_reward: Maximum reward value (default: 2.0)
            
        Returns:
            float: Reward value between 0.0 and 1.0
        """
        try:
            if initial_system_graph is None:
                # If no initial graph, count all blocks in final graph
                initial_blocks = 0
            else:
                # Count blocks in initial graph
                initial_blocks = len(initial_system_graph.blocks)
            
            # Count blocks in final graph
            final_blocks = len(system_graph.blocks)
            
            # Calculate blocks added
            blocks_added = max(0, final_blocks - initial_blocks)
            
            # Normalize to 0-1 scale with max_reward as the cap
            reward = min(blocks_added / max_reward, 1.0) if max_reward > 0 else 0.0
            
            return reward
        except Exception as e:
            print(f"Error calculating block addition reward: {e}")
            return 0.0
    
    def _calculate_frequency_coherence_reward(self, system_graph: SystemGraph) -> float:
        """Calculate reward based on frequency coherence across all blocks.
        
        Args:
            system_graph: The system graph to evaluate
            
        Returns:
            float: Reward value between 0.0 and 1.0, where 1.0 means perfect frequency coherence
        """
        try:
            # Check if there are any frequency violations
            frequency_violations = getattr(system_graph, 'frequency_violations', [])
            
            # If no violations, perfect score
            if not frequency_violations:
                return 1.0
            
            # Calculate penalty based on number of violations
            # Each violation reduces the reward
            num_violations = len(frequency_violations)
            penalty = min(1.0, num_violations * 0.5)  # Each violation costs 0.5, max penalty is 1.0
            
            reward = max(0.0, 1.0 - penalty)
            return reward
        except Exception as e:
            print(f"Error calculating frequency coherence reward: {e}")
            return 0.0
    
    def _calculate_voltage_coherence_reward(self, system_graph: SystemGraph) -> float:
        """Calculate reward based on voltage coherence between connected blocks.
        
        Args:
            system_graph: The system graph to evaluate
            
        Returns:
            float: Reward value between 0.0 and 1.0, where 1.0 means perfect voltage coherence
        """
        try:
            # Check if there are any voltage violations
            voltage_violations = getattr(system_graph, 'voltage_violations', [])
            
            # If no violations, perfect score
            if not voltage_violations:
                return 1.0
            
            # Calculate penalty based on number of violations
            # Each violation reduces the reward
            num_violations = len(voltage_violations)
            penalty = min(1.0, num_violations * 0.3)  # Each violation costs 0.3, max penalty is 1.0
            
            reward = max(0.0, 1.0 - penalty)
            return reward
        except Exception as e:
            print(f"Error calculating voltage coherence reward: {e}")
            return 0.0
    
    def _calculate_sample_reward_components(self, system_dict: dict, system_dict_initial: dict, sample_idx: int, completion: List[Dict[str, str]] = None, weights: Dict[str, float] = None) -> Tuple[RewardComponents, str, str, str]:
        """Calculate individual reward components for a single sample.
        
        Args:
            weights: Dictionary of weights for each component. If a weight is 0, the calculation is skipped.
        
        Returns:
            tuple: (RewardComponents, graph_viz, network_plot_b64, connectivity_report)
        """
        if not system_dict:
            error_msg = f"No system_dict found for sample {sample_idx}."
            print(error_msg)
            return RewardComponents(), None, None, error_msg
        
        if weights is None:
            weights = self.power_system_weights
        
        with self._suppress_logging():
            try:
                initial_system_graph = SystemGraph(system_dict_initial) if system_dict_initial else None
                power_system_graph = SystemGraph(system_dict)
                
                # Calculate individual reward components
                components = RewardComponents()
                
                # Initialize converter and diagnostic_results as None - will be created if needed
                converter = None
                diagnostic_results = {}  # Always initialize as empty dict
                
                # 1. Connectivity reward
                if weights.get('connectivity', 0.0) > 0:
                    components.connectivity_reward = self._calculate_connectivity_reward(power_system_graph)
                
                # 2. Validation reward
                if weights.get('validation', 0.0) > 0:
                    components.validation_reward = self._calculate_validation_reward(power_system_graph)
                
                # 3. Parameter reward
                if weights.get('parameter', 0.0) > 0:
                    components.parameter_reward = self._calculate_parameter_reward(power_system_graph)
                
                # 4. Conversion and diagnostic rewards (create converter only if needed)
                if weights.get('conversion', 0.0) > 0 or weights.get('diagnostic', 0.0) > 0 or weights.get('load_satisfaction', 0.0) > 0 or weights.get('structure', 0.0) > 0:
                    converter = PandapowerConverter(power_system_graph)
                    
                    if weights.get('conversion', 0.0) > 0:
                        components.conversion_reward = self._calculate_conversion_reward(converter)
                    
                    if weights.get('diagnostic', 0.0) > 0 or weights.get('load_satisfaction', 0.0) > 0:
                        components.diagnostic_reward, diagnostic_results = self._calculate_diagnostic_reward(converter)
                        # If only load_satisfaction needs diagnostic_results but diagnostic weight is 0, still calculate but don't store diagnostic reward
                        if weights.get('diagnostic', 0.0) == 0:
                            components.diagnostic_reward = 0.0
                    
                    # 5. Load satisfaction reward (needs diagnostic_results)
                    if weights.get('load_satisfaction', 0.0) > 0:
                        components.load_satisfaction_reward = self._calculate_load_satisfaction_reward(power_system_graph, diagnostic_results)
                    
                    # 6. Structure reward
                    if weights.get('structure', 0.0) > 0:
                        components.structure_reward = self._calculate_structure_reward(power_system_graph, converter)
                
                # 7. Tool execution reward (if completion is provided)
                if completion is not None:
                    if weights.get('tool_execution', 0.0) > 0:
                        components.tool_execution_reward = self.tool_execution_reward_func([completion])[0]
                    if weights.get('format', 0.0) > 0:
                        components.format_reward = self.parser.get_format_reward_func()([completion])[0]
                    if weights.get('xml', 0.0) > 0:
                        components.xml_reward = self.parser.get_xml_reward_func()([completion])[0]
                
                # 8. Connection addition reward
                if weights.get('connection_addition', 0.0) > 0:
                    components.connection_addition_reward = self._calculate_connection_addition_reward(power_system_graph, initial_system_graph)
                
                # 9. Block addition reward
                if weights.get('block_addition', 0.0) > 0:
                    components.block_addition_reward = self._calculate_block_addition_reward(power_system_graph, initial_system_graph)
                
                # 10. Frequency coherence reward
                if weights.get('frequency_coherence', 0.0) > 0:
                    components.frequency_coherence_reward = self._calculate_frequency_coherence_reward(power_system_graph)
                
                # 11. Voltage coherence reward
                if weights.get('voltage_coherence', 0.0) > 0:
                    components.voltage_coherence_reward = self._calculate_voltage_coherence_reward(power_system_graph)
                
                # Generate visualizations
                if initial_system_graph is not None:
                    graph_viz = power_system_graph.draw_graph(initial_graph=initial_system_graph, highlight_changes=True)
                else:
                    graph_viz = power_system_graph.draw_graph()
                
                # Get connectivity report
                connectivity_details = power_system_graph.get_connectivity_details()
                connectivity_report = f"Connectivity: {connectivity_details.connected_loads}/{connectivity_details.total_loads} loads connected ({connectivity_details.connectivity_ratio:.2%})"
                if connectivity_details.isolated_load_names:
                    connectivity_report += f" | Isolated: {', '.join(connectivity_details.isolated_load_names)}"
                
                # Create network plot (only if converter was created)
                network_plot_b64 = None
                if converter is not None:
                    network_plot_b64 = self._create_network_plot(converter, sample_idx)
                
                # Print detailed reward breakdown (only for calculated components)
                print(f"Sample {sample_idx} power system reward breakdown:")
                if weights.get('connectivity', 0.0) > 0:
                    print(f"  Connectivity: {components.connectivity_reward:.4f}")
                if weights.get('validation', 0.0) > 0:
                    print(f"  Validation: {components.validation_reward:.4f}")
                if weights.get('parameter', 0.0) > 0:
                    print(f"  Parameters: {components.parameter_reward:.4f}")
                if weights.get('conversion', 0.0) > 0:
                    print(f"  Conversion: {components.conversion_reward:.4f}")
                if weights.get('diagnostic', 0.0) > 0:
                    print(f"  Diagnostics: {components.diagnostic_reward:.4f}")
                if weights.get('load_satisfaction', 0.0) > 0:
                    print(f"  Load Satisfaction: {components.load_satisfaction_reward:.4f}")
                if weights.get('structure', 0.0) > 0:
                    print(f"  Structure: {components.structure_reward:.4f}")
                if weights.get('tool_execution', 0.0) > 0:
                    print(f"  Tool Execution: {components.tool_execution_reward:.4f}")
                if weights.get('format', 0.0) > 0:
                    print(f"  Format: {components.format_reward:.4f}")
                if weights.get('xml', 0.0) > 0:
                    print(f"  XML: {components.xml_reward:.4f}")
                if weights.get('connection_addition', 0.0) > 0:
                    print(f"  Connection Addition: {components.connection_addition_reward:.4f}")
                if weights.get('block_addition', 0.0) > 0:
                    print(f"  Block Addition: {components.block_addition_reward:.4f}")
                if weights.get('frequency_coherence', 0.0) > 0:
                    print(f"  Frequency Coherence: {components.frequency_coherence_reward:.4f}")
                if weights.get('voltage_coherence', 0.0) > 0:
                    print(f"  Voltage Coherence: {components.voltage_coherence_reward:.4f}")
                print(f"  Total: {components.get_total_reward(weights):.4f}")
                print(f"  {connectivity_report}")
                
                return components, graph_viz, network_plot_b64, connectivity_report
                
            except Exception as e:
                error_msg = f"Error processing sample {sample_idx}: {e}"
                print(error_msg)
                return RewardComponents(), None, None, error_msg
    
    def _create_network_plot(self, converter: PandapowerConverter, sample_idx: int) -> Optional[str]:
        """Create network plot and return base64 encoded image."""
        network_plot_path = f"temp_network_plot_{sample_idx}.png"
        plot_success = converter.create_network_plot(network_plot_path)
        
        if not plot_success:
            return None
        
        try:
            import base64
            import os
            with open(network_plot_path, "rb") as img_file:
                network_plot_b64 = base64.b64encode(img_file.read()).decode('utf-8')
            os.remove(network_plot_path)  # Clean up temporary file
            return network_plot_b64
        except Exception as e:
            print(f"Error encoding network plot for sample {sample_idx}: {e}")
            return None
    
    def _process_with_gen_globals(self, gen_globals_list: list, init_code: list, completions: list = None) -> tuple:
        """Process samples when gen_globals_list is provided."""
        rewards, graph_visualizations, network_plots, connectivity_reports = [], [], [], []
        all_reward_components = []
        
        for i, (gen_globals, each_init_code) in enumerate(zip(gen_globals_list, init_code)):
            system_dict = gen_globals.get('system_dict', {})
            
            # Extract and execute init_code to get initial system state
            system_dict_initial = self._get_initial_system_dict(each_init_code, i)
            
            # Get completion data if available
            completion = completions[i] if completions and i < len(completions) else None
            
            components, graph_viz, network_plot, connectivity_report = self._calculate_sample_reward_components(
                system_dict, system_dict_initial, i, completion, self.power_system_weights
            )
            
            # Calculate total reward using the components
            total_reward = components.get_total_reward(self.power_system_weights)
            
            rewards.append(total_reward)
            all_reward_components.append(components)
            graph_visualizations.append(graph_viz)
            network_plots.append(network_plot)
            connectivity_reports.append(connectivity_report)
        
        # Store components for later retrieval
        self._last_reward_components = all_reward_components
        
        return rewards, graph_visualizations, network_plots, connectivity_reports
    
    def _process_with_completions(self, completions: list, init_code: list) -> tuple:
        """Process samples when only completions are provided (fallback method)."""
        responses = [completion[-1]['content'] for completion in completions]
        rewards, graph_visualizations, network_plots, connectivity_reports = [], [], [], []
        all_reward_components = []
        
        for i, (response, each_init_code) in enumerate(zip(responses, init_code)):
            print(f"Response: {response}")
            python_code = extract_python_code(response)
            
            # Get initial system state
            gen_globals = {}
            system_dict_initial = self._get_initial_system_dict(each_init_code, i, gen_globals)
            
            # Execute the response code
            try:
                exec(python_code, gen_globals)
                print("\033[94m" + python_code + "\033[0m")  # Print in blue color
            except Exception as e:
                print(f"Error executing code: {e}")
                print("\033[91m" + python_code + "\033[0m")  # Print in red color
                rewards.append(0.0)
                all_reward_components.append(RewardComponents())
                graph_visualizations.append(None)
                network_plots.append(None)
                connectivity_reports.append("Code execution failed.")
                continue
            
            system_dict = gen_globals.get('system_dict', {})
            components, graph_viz, network_plot, connectivity_report = self._calculate_sample_reward_components(
                system_dict, system_dict_initial, i, completions[i], self.power_system_weights
            )
            
            # Calculate total reward using the components
            total_reward = components.get_total_reward(self.power_system_weights)
            
            rewards.append(total_reward)
            all_reward_components.append(components)
            graph_visualizations.append(graph_viz)
            network_plots.append(network_plot)
            connectivity_reports.append(connectivity_report)
        
        # Store components for later retrieval
        self._last_reward_components = all_reward_components
        
        return rewards, graph_visualizations, network_plots, connectivity_reports
    
    def _get_initial_system_dict(self, init_code_raw: str, sample_idx: int, existing_globals: dict = None) -> dict:
        """Extract and execute init_code to get initial system state."""
        init_code = extract_python_code(init_code_raw)
        if not init_code:
            return {}
        
        init_globals = existing_globals if existing_globals is not None else {}
        try:
            exec(init_code, init_globals)
            return copy.deepcopy(init_globals.get('system_dict', {}))
        except Exception as e:
            print(f"Error executing init_code for sample {sample_idx}: {e}")
            return {}
    
    def _store_results(self, graph_visualizations: list, network_plots: list, connectivity_reports: list):
        """Store results as instance attributes for later retrieval."""
        self._last_graph_visualizations = graph_visualizations
        self._last_network_plots = network_plots
        self._last_connectivity_reports = connectivity_reports
        
    def correct_answer_reward_func(self, completions, init_code, gen_globals_list=None, **kwargs) -> list[float]:
        """
        Override BaseReward's method with sophisticated power system evaluation.
        
        This method evaluates power system designs using component-based rewards.
        """
        return self.power_system_reward_func(completions, init_code, gen_globals_list, **kwargs)
    
    def power_system_reward_func(self, completions, init_code, gen_globals_list=None, **kwargs) -> list[float]:
        """
        Main power system evaluation function that replaces correct_answer_reward_func.
        
        This method evaluates power system designs using component-based rewards.
        """
        
        if gen_globals_list is not None:
            rewards, graph_visualizations, network_plots, connectivity_reports = self._process_with_gen_globals(
                gen_globals_list, init_code, completions
            )
        else:
            rewards, graph_visualizations, network_plots, connectivity_reports = self._process_with_completions(
                completions, init_code
            )
        
        self._store_results(graph_visualizations, network_plots, connectivity_reports)
        return rewards
    
    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                            tool_attempts += 1
                            # Check response with env_parser
                            multiplier = 1.0 
                            response = str(parsed.tool)
                            if (("sympy" in response) or ("numpy" in response)) and len(response) > 100:
                                multiplier = 1.5
                            else:
                                multiplier = 0.5
                            parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                            if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                successful_executions += 1 * multiplier
            
            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return (successful_executions / tool_attempts)
        
        return [check_execution(c) for c in completions]
    
    def get_named_tool_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that checks tool execution success for a specific tool.

        Uses XMLParser to identify proper tool calls.
        """
        def tool_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that checks execution success for the {tool_name} tool.
            
            Uses XMLParser to identify proper tool calls for the specified tool.
            """
            import json
            
            def check_tool_execution(trajectory: List[Dict[str, str]]) -> float:
                tool_attempts = 0
                successful_executions = 0
                
                # Find assistant messages with the specific tool and their responses
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        # Use parser to check for tool tag
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    # Found a properly formatted tool message for the specific tool
                                    if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                        tool_attempts += 1
                                        # Check response with env_parser
                                        parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                                        if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                            successful_executions += 1
                            except json.JSONDecodeError:
                                pass
                
                # Calculate reward
                if tool_attempts == 0:
                    return 0.0
                return (successful_executions / tool_attempts)
            
            return [check_tool_execution(c) for c in completions]
        
        # Create a function with the dynamic name based on tool_name
        tool_reward_func.__name__ = f"{tool_name}_reward_func"
        return tool_reward_func
    
    def get_named_tool_count_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_count_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                successful_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    # Found a properly formatted tool message for the specific tool
                                    if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                        parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                                        if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                            successful_executions += 1
                            except json.JSONDecodeError:
                                pass
                return successful_executions
            
            return [count_tool_executions(c) for c in completions]
        
        tool_count_reward_func.__name__ = f"{tool_name}_count_reward_func"
        return tool_count_reward_func

    def get_named_tool_attempt_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_attempt_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                attempted_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    attempted_executions += 1
                            except json.JSONDecodeError:
                                pass
                return attempted_executions
            
            return [count_tool_executions(c) for c in completions]
            
        tool_attempt_reward_func.__name__ = f"{tool_name}_attempt_reward_func"
        return tool_attempt_reward_func
    
    # Power System Specific Methods
    def get_last_reward_components(self) -> Optional[List[RewardComponents]]:
        """Get the detailed reward components from the last computation."""
        return getattr(self, '_last_reward_components', None)
    
    def get_component_rewards_summary(self) -> Optional[Dict[str, List[float]]]:
        """Get a summary of all component rewards from the last computation."""
        components = self.get_last_reward_components()
        if not components:
            return None
        
        return {
            'connectivity': [c.connectivity_reward for c in components],
            'validation': [c.validation_reward for c in components],
            'parameter': [c.parameter_reward for c in components],
            'conversion': [c.conversion_reward for c in components],
            'diagnostic': [c.diagnostic_reward for c in components],
            'load_satisfaction': [c.load_satisfaction_reward for c in components],
            'structure': [c.structure_reward for c in components],
            'tool_execution': [c.tool_execution_reward for c in components],
            'format': [c.format_reward for c in components],
            'xml': [c.xml_reward for c in components],
            'connection_addition': [c.connection_addition_reward for c in components],
            'block_addition': [c.block_addition_reward for c in components],
            'frequency_coherence': [c.frequency_coherence_reward for c in components],
            'voltage_coherence': [c.voltage_coherence_reward for c in components],
            'total': [c.get_total_reward(self.power_system_weights) for c in components]
        }
    
    def get_formatted_reward_strings(self) -> Optional[List[str]]:
        """Get formatted reward strings for concise logging."""
        components = self.get_last_reward_components()
        if not components:
            return None
        
        formatted_strings = []
        for c in components:
            parts = [f"Total: {c.get_total_reward(self.power_system_weights):.3f}"]
            
            # Only include components with non-zero weights
            if self.power_system_weights.get('tool_execution', 0.0) > 0:
                parts.append(f"tool_exec: {c.tool_execution_reward:.3f}")
            if self.power_system_weights.get('format', 0.0) > 0:
                parts.append(f"format: {c.format_reward:.3f}")
            if self.power_system_weights.get('xml', 0.0) > 0:
                parts.append(f"xml: {c.xml_reward:.3f}")
            if self.power_system_weights.get('connectivity', 0.0) > 0:
                parts.append(f"conn: {c.connectivity_reward:.3f}")
            if self.power_system_weights.get('validation', 0.0) > 0:
                parts.append(f"valid: {c.validation_reward:.3f}")
            if self.power_system_weights.get('parameter', 0.0) > 0:
                parts.append(f"param: {c.parameter_reward:.3f}")
            if self.power_system_weights.get('conversion', 0.0) > 0:
                parts.append(f"conv: {c.conversion_reward:.3f}")
            if self.power_system_weights.get('diagnostic', 0.0) > 0:
                parts.append(f"diag: {c.diagnostic_reward:.3f}")
            if self.power_system_weights.get('load_satisfaction', 0.0) > 0:
                parts.append(f"load: {c.load_satisfaction_reward:.3f}")
            if self.power_system_weights.get('structure', 0.0) > 0:
                parts.append(f"struct: {c.structure_reward:.3f}")
            if self.power_system_weights.get('connection_addition', 0.0) > 0:
                parts.append(f"conn_add: {c.connection_addition_reward:.3f}")
            if self.power_system_weights.get('block_addition', 0.0) > 0:
                parts.append(f"block_add: {c.block_addition_reward:.3f}")
            if self.power_system_weights.get('frequency_coherence', 0.0) > 0:
                parts.append(f"freq_coh: {c.frequency_coherence_reward:.3f}")
            if self.power_system_weights.get('voltage_coherence', 0.0) > 0:
                parts.append(f"volt_coh: {c.voltage_coherence_reward:.3f}")
            
            formatted_strings.append("\n".join(parts))
        
        return formatted_strings
    
    def update_power_system_weights(self, new_weights: Dict[str, float]):
        """Update the power system reward weights for different components."""
        self.power_system_weights.update(new_weights)
    
    def get_power_system_weights(self) -> Dict[str, float]:
        """Get the current power system reward weights."""
        return self.power_system_weights.copy()
    
    # Legacy getter methods (for backward compatibility)
    def get_last_graph_visualizations(self) -> Optional[List[Optional[str]]]:
        """Get the graph visualizations from the last reward computation."""
        return getattr(self, '_last_graph_visualizations', None)

    def get_last_network_plots(self) -> Optional[List[Optional[str]]]:
        """Get the network plots from the last reward computation."""
        return getattr(self, '_last_network_plots', None)

    def get_last_connectivity_reports(self) -> Optional[List[str]]:
        """Get the connectivity reports from the last reward computation."""
        return getattr(self, '_last_connectivity_reports', None) 