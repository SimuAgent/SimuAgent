import json
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

def load_block_config():
    """Load block configuration from block_config.json file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'system_parser', 'block_config.json')
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to current directory if not found in system_parser
        config_path = os.path.join(os.path.dirname(__file__), 'block_config.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: block_config.json not found. Using empty configuration.")
            return {}

class SemanticBlockSearcher:
    """
    Uses SentenceTransformer to perform semantic search for Simulink Blocks based on block_config.json.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic block searcher.

        Args:
            model_name: Name of the SentenceTransformers model to use.
                      Defaults to 'all-MiniLM-L6-v2' as a lightweight example.
        """
        self.model = SentenceTransformer(model_name)
        
        # Load block configuration from JSON file
        self.block_config = load_block_config()
        
        # Transform block config into searchable format
        self.blocks_data = []
        self.blocks_texts = []
        
        for block_name, block_info in self.block_config.items():
            # Create searchable text combining name, description, and parameters
            search_text_parts = [block_name, block_info.get('Description', '')]
            
            # Add basic parameters to search text if available
            if 'Basic Parameters' in block_info:
                for param_name, param_value in block_info['Basic Parameters'].items():
                    search_text_parts.append(f"{param_name}: {param_value}")
            
            # Add port information if available
            if 'Ports' in block_info:
                port_info = ', '.join([f"{port_name} ({port_data.get('Type', 'Unknown')} port)" 
                                     for port_name, port_data in block_info['Ports'].items()])
                search_text_parts.append(f"Ports: {port_info}")
            
            search_text = ' '.join(search_text_parts)
            
            self.blocks_data.append({
                'name': block_name,
                'info': block_info
            })
            self.blocks_texts.append(search_text)
        
        # Encode (vectorize) all concatenated block texts and cache the results
        if self.blocks_texts:
            self.blocks_embeddings = self.model.encode(
                self.blocks_texts,
                convert_to_tensor=True
            )
        else:
            self.blocks_embeddings = None
    
    def search_blocks(self, query_list: List[str], num_results: int = 5, 
                     include_parameters: bool = False, include_ports: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform semantic search for the given query list and return the most matching Block information.

        Args:
            query_list: List of queries to search for
            num_results: Number of top results to return for each query
            include_parameters: Whether to include basic parameters in results (default: False)
            include_ports: Whether to include port information in results (default: False)

        Returns:
            Dictionary mapping each query to a list of matched blocks with their details
        """
        results = {}
        
        if not self.blocks_embeddings or not self.blocks_data:
            print("Warning: No block data available for search.")
            return {query: [] for query in query_list}
        
        for query in query_list:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            
            hits = util.semantic_search(query_embedding, self.blocks_embeddings, top_k=num_results)[0]
            
            matched_blocks = []
            for hit in hits:
                block_index = hit['corpus_id']
                score = hit['score']
                block_data = self.blocks_data[block_index]
                
                block_info = {
                    "name": block_data["name"],
                    "description": block_data["info"].get("Description", ""),
                    "score": round(float(score), 3)
                }
                
                # Optionally include basic parameters
                if include_parameters:
                    block_info["basic_parameters"] = block_data["info"].get("Basic Parameters", {})
                
                # Optionally include ports
                if include_ports:
                    block_info["ports"] = block_data["info"].get("Ports", {})
                
                matched_blocks.append(block_info)

            results[query] = matched_blocks
        
        return results


def search_blocks(query_list: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search for Simulink blocks based on a list of queries.

    Args:
        query_list: List of queries to search for
    
    Returns:
        A dictionary with queries as keys and lists of matched blocks as values.
        Each block contains its name, description, and similarity score.
    """
    
    # TODO: Optionally includes basic parameters and ports if requested. 
    include_parameters = False # include_parameters: Whether to include basic parameters in results (default: False)
    include_ports = False  # include_ports: Whether to include port information in results (default: False)
    static_searcher = SemanticBlockSearcher()
    return static_searcher.search_blocks(query_list, num_results=3, 
                                       include_parameters=include_parameters, 
                                       include_ports=include_ports)
