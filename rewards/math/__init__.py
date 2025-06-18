"""
Math Module - Mathematical Problem Evaluation

Contains specialized evaluators and graders for mathematical problems,
including symbolic evaluation, numerical comparison, and LaTeX processing.
"""

from .grader import MathGrader, grade
from .evaluators import (
    SymbolicEvaluator,
    NumericalEvaluator,
    LatexEvaluator
)
from .normalizers import (
    MathNormalizer,
    LatexNormalizer,
    AnswerNormalizer
)
from .validators import MathValidator
from .extractors import AnswerExtractor, BoxedAnswerExtractor

__all__ = [
    # Main grader interface
    'MathGrader',
    'grade',
    
    # Specialized evaluators
    'SymbolicEvaluator',
    'NumericalEvaluator', 
    'LatexEvaluator',
    
    # Normalizers
    'MathNormalizer',
    'LatexNormalizer',
    'AnswerNormalizer',
    
    # Validators and extractors
    'MathValidator',
    'AnswerExtractor',
    'BoxedAnswerExtractor',
] 