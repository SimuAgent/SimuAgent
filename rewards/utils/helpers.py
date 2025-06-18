"""
Helper utility functions for the reward system.

This module contains various utility functions used throughout
the reward system for data processing, validation, and calculation.
"""

import re
import ast
import signal
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def canonicalize(obj: Any) -> Any:
    """
    Recursively normalize an object for consistent comparison.
    
    Normalization rules:
    1. If it's a dict, sort its keys first, then recursively call on values
    2. If it's a list, recursively call on each element, then sort uniformly
    3. For other types, return directly
    
    Args:
        obj: The object to canonicalize
        
    Returns:
        The canonicalized object
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


def dicts_equal_ignoring_list_order(d1: Dict, d2: Dict) -> bool:
    """
    Compare two dictionaries, ignoring the order of elements at any level.
    
    Args:
        d1: First dictionary
        d2: Second dictionary
        
    Returns:
        True if dictionaries are equal (ignoring order), False otherwise
    """
    return canonicalize(d1) == canonicalize(d2)


def extract_python_code(text: str, idx: Optional[int] = None) -> str:
    """
    Extract Python code from triple backtick blocks.
    
    Args:
        text: Text containing code blocks
        idx: Index of code block to extract (None for last block)
        
    Returns:
        Extracted Python code, or empty string if no code found
    """
    matches = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if not matches:
        # Try without python specifier
        matches = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
    
    if not matches:
        return ""
    
    code = matches[idx] if idx is not None and idx < len(matches) else matches[-1]
    return code.strip()


def normalize_answer(answer: Union[str, int, float, None]) -> str:
    """
    Normalize an answer for comparison.
    
    Args:
        answer: The answer to normalize
        
    Returns:
        Normalized string representation
    """
    if answer is None:
        return ""
    
    # Convert to string and strip whitespace
    normalized = str(answer).strip()
    
    # Remove common formatting
    normalized = normalized.replace('\n', ' ').replace('\r', '')
    
    # Normalize multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized


def calculate_weighted_score(
    scores: Dict[str, float], 
    weights: Dict[str, float]
) -> float:
    """
    Calculate a weighted score from individual component scores.
    
    Args:
        scores: Dictionary of component scores
        weights: Dictionary of component weights
        
    Returns:
        Weighted total score (0.0 to 1.0)
    """
    if not scores or not weights:
        return 0.0
    
    total_weighted = 0.0
    total_weight = 0.0
    
    for component, score in scores.items():
        weight = weights.get(component, 0.0)
        if weight > 0:
            total_weighted += score * weight
            total_weight += weight
    
    return total_weighted / total_weight if total_weight > 0 else 0.0


def safe_eval(expression: str, allowed_names: Optional[Dict[str, Any]] = None) -> Any:
    """
    Safely evaluate a Python expression.
    
    Args:
        expression: Python expression to evaluate
        allowed_names: Dictionary of allowed names and their values
        
    Returns:
        Result of evaluation
        
    Raises:
        ValueError: If expression is unsafe or invalid
    """
    if allowed_names is None:
        allowed_names = {}
    
    # Parse the expression to check for safety
    try:
        parsed = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    
    # Check for disallowed nodes
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
            if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                func_name = node.func.id
                if func_name not in allowed_names:
                    raise ValueError(f"Function call not allowed: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id not in allowed_names and node.id not in ['True', 'False', 'None']:
                # Allow built-in constants
                continue
    
    # Evaluate safely
    try:
        return eval(expression, {"__builtins__": {}}, allowed_names)
    except Exception as e:
        raise ValueError(f"Evaluation error: {e}")


class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass


@contextmanager
def timeout_handler(seconds: float = 10.0):
    """
    Context manager for timing out operations.
    
    Args:
        seconds: Timeout duration in seconds
        
    Raises:
        TimeoutException: If operation times out
    """
    def timeout_func(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_func)
    signal.alarm(int(seconds))
    
    try:
        yield
    finally:
        # Restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamp a value between min and max bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format a float as a percentage string.
    
    Args:
        value: Value between 0.0 and 1.0
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def flatten_dict(d: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        separator: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    def _flatten(obj, parent_key=''):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{separator}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(d)


def get_nested_value(d: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from a nested dictionary using dot notation.
    
    Args:
        d: Dictionary to search
        key_path: Dot-separated key path (e.g., "a.b.c")
        default: Default value if key not found
        
    Returns:
        Value at key path or default
    """
    keys = key_path.split('.')
    current = d
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def filter_dict(d: Dict[str, Any], predicate: Callable[[str, Any], bool]) -> Dict[str, Any]:
    """
    Filter dictionary entries based on a predicate function.
    
    Args:
        d: Dictionary to filter
        predicate: Function that takes (key, value) and returns bool
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if predicate(k, v)}


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later dictionaries taking precedence.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result 