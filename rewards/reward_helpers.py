import re
import ast
from typing import Dict, Any

def canonicalize(obj):
    """
    Recursively normalize obj:
    1. If it's a dict, sort its key first, then recursively call on value
    2. If it's a list, recursively call on each element in the list, then sort uniformly
    3. For other types, return directly
    """
    if isinstance(obj, dict):
        # Sort dictionary keys to ensure consistent key order during comparison
        return {k: canonicalize(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        # Recursively process each element in the list, then sort uniformly
        # Use str(x) as sorting criterion to unify sorting of different types
        return sorted((canonicalize(x) for x in obj), key=lambda x: str(x))
    else:
        # For strings, numbers, booleans, etc., return directly
        return obj


def dicts_equal_ignoring_list_order(d1, d2):
    """
    Compare two dictionaries, not caring about the order of elements in any level.
    """
    return canonicalize(d1) == canonicalize(d2)


def extract_python_code(text: str, idx=None) -> str:
    """Extract Python code from triple backtick blocks."""
    matches = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if not matches:
        return ""
    return matches[idx].strip() if idx is not None and idx < len(matches) else matches[-1].strip()


def calculate_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate a weighted score from component scores and weights.
    
    Args:
        scores: Dictionary mapping component names to scores
        weights: Dictionary mapping component names to weights
    
    Returns:
        Weighted average score
    """
    if not scores or not weights:
        return 0.0
    
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for component, score in scores.items():
        weight = weights.get(component, 0.0)
        total_weighted_score += score * weight
        total_weight += weight
    
    return total_weighted_score / total_weight if total_weight > 0 else 0.0
