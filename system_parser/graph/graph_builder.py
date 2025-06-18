"""
Graph builder module for constructing system graphs.
"""

import logging
from typing import Dict, List, Any
import networkx as nx
from ..core.models import Block

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds system graphs from block data.
    """
    
    def __init__(self, block_config=None):
        self.block_config = block_config
    
    def build_blocks(self, blocks_data: Dict[str, Any], validate_param_keys: bool = False) -> Dict[str, Block]:
        """
        Build blocks from data dictionary.
        
        Args:
            blocks_data: Dictionary containing block definitions
            validate_param_keys: Whether to validate parameter keys
            
        Returns:
            Dictionary mapping block names to Block objects
        """
        blocks = {}
        
        for block_name, block_data in blocks_data.items():
            try:
                # Create a simple block for now
                block = Block(
                    name=block_name,
                    block_type=block_data.get('type', 'Unknown'),
                    parameters=block_data.get('parameters', {}),
                    position=block_data.get('position', (0, 0))
                )
                blocks[block_name] = block
                
            except Exception as e:
                logger.error(f"Failed to create block {block_name}: {e}")
                
        return blocks
    
    def build_graph(self, blocks: Dict[str, Block], connections: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a NetworkX graph from blocks and connections.
        
        Args:
            blocks: Dictionary of Block objects
            connections: List of connection definitions
            
        Returns:
            NetworkX DiGraph representing the system
        """
        graph = nx.DiGraph()
        
        # Add nodes for blocks
        for block_name, block in blocks.items():
            graph.add_node(block_name, block=block)
        
        # Add edges for connections
        for connection in connections:
            try:
                from_block = connection.get('from_block')
                to_block = connection.get('to_block')
                
                if from_block and to_block and from_block in blocks and to_block in blocks:
                    graph.add_edge(from_block, to_block, connection=connection)
                    
            except Exception as e:
                logger.error(f"Failed to add connection: {e}")
                
        return graph 