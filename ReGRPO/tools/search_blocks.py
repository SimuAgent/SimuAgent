from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

SIMULINK_BLOCKS = [
    {
        "name": "Three-Phase V-I Measurement",
        "description": (
            "Measures voltage and current in a three-phase system for analysis, "
            "and can also represent a bus element."
        )
    },
    {
        "name": "Synchronous Machine",
        "description": "Simulates a synchronous generator or motor for three-phase AC systems."
    },
    {
        "name": "AC Voltage Source",
        "description": "Generates a sinusoidal voltage supply for AC power simulations."
    },
    {
        "name": "Three-Phase PI Section Line",
        "description": "Simulates a three-phase transmission line using a PI-section model."
    },
    {
        "name": "Single-Phase Transmission Line",
        "description": "Simulates a basic single-phase line in power systems."
    },
    {
        "name": "Three-Phase Transformer (Two Windings)",
        "description": "Represents a three-phase transformer for stepping voltage up or down."
    },
    {
        "name": "Single-Phase Transformer",
        "description": "Represents a single-phase transformer for power conversion."
    },
    {
        "name": "Three-Phase Source",
        "description": (
            "Provides a balanced three-phase supply for testing and simulations, "
            "and can serve as an external grid model."
        )
    },
    {
        "name": "Three-Phase Series RLC Load",
        "description": "Simulates a series RLC load in three-phase applications."
    },
    {
        "name": "Three-Phase Parallel RLC Load",
        "description": "Simulates a parallel RLC load in three-phase applications."
    }
]

class SemanticBlockSearcher:
    """
    Uses SentenceTransformer to perform semantic search for Simulink Blocks.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic block searcher.

        Args:
            model_name: Name of the SentenceTransformers model to use.
                      Defaults to 'all-MiniLM-L6-v2' as a lightweight example.
        """
        self.model = SentenceTransformer(model_name)
        
        # Pre-compute vectors by concatenating block names and descriptions
        self.blocks_data = SIMULINK_BLOCKS
        self.blocks_texts = [
            f"{block['name']} - {block['description']}"
            for block in self.blocks_data
        ]
        
        # Encode (vectorize) all concatenated block texts and cache the results
        self.blocks_embeddings = self.model.encode(
            self.blocks_texts,
            convert_to_tensor=True
        )
    
    def search_blocks(self, query_list: List[str], num_results: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform semantic search for the given query list and return the most matching Block information.

        Args:
            query_list: List of queries to search for
            num_results: Number of top results to return for each query

        Returns:
            Dictionary mapping each query to a list of matched blocks with their details
        """
        results = {}
        
        for query in query_list:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            
            hits = util.semantic_search(query_embedding, self.blocks_embeddings, top_k=num_results)[0]
            
            matched_blocks = []
            for hit in hits:
                block_index = hit['corpus_id']
                score = hit['score']
                block_info = {
                    "name": self.blocks_data[block_index]["name"],
                    "description": self.blocks_data[block_index]["description"],
                    "score": round(float(score), 1)
                }
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
    static_searcher = SemanticBlockSearcher()
    return static_searcher.search_blocks(query_list, num_results=2)
