"""
Connectivity analyzer for system graphs.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConnectivityResult:
    """Results from connectivity analysis."""
    
    def __init__(self):
        self.connectivity_ratio = 0.0
        self.connected_components = []
        self.isolated_nodes = []
        self.analysis_details = {}


class ConnectivityAnalyzer:
    """
    Analyzes connectivity properties of system graphs.
    """
    
    def __init__(self):
        pass
    
    def analyze(self, graph) -> ConnectivityResult:
        """
        Analyze connectivity of the given graph.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            ConnectivityResult with analysis details
        """
        result = ConnectivityResult()
        
        try:
            if graph and graph.number_of_nodes() > 0:
                # Basic connectivity analysis
                total_nodes = graph.number_of_nodes()
                total_edges = graph.number_of_edges()
                
                # Calculate basic connectivity ratio
                if total_nodes > 1:
                    max_possible_edges = total_nodes * (total_nodes - 1)
                    result.connectivity_ratio = total_edges / max_possible_edges if max_possible_edges > 0 else 0.0
                else:
                    result.connectivity_ratio = 1.0 if total_nodes == 1 else 0.0
                
                # Find isolated nodes (nodes with no connections)
                result.isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]
                
                result.analysis_details = {
                    'total_nodes': total_nodes,
                    'total_edges': total_edges,
                    'isolated_count': len(result.isolated_nodes)
                }
                
        except Exception as e:
            logger.error(f"Connectivity analysis failed: {e}")
            
        return result 