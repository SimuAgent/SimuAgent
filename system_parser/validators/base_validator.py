"""
Base validation infrastructure for power system components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.message}"


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]
    
    @property
    def infos(self) -> List[ValidationIssue]:
        """Get only info-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.INFO]
    
    def add_error(self, message: str, details: Dict[str, Any] = None, suggestions: List[str] = None) -> None:
        """Add an error to the validation result."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message=message,
            details=details or {},
            suggestions=suggestions or []
        ))
        self.is_valid = False
    
    def add_warning(self, message: str, details: Dict[str, Any] = None, suggestions: List[str] = None) -> None:
        """Add a warning to the validation result."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message=message,
            details=details or {},
            suggestions=suggestions or []
        ))
    
    def add_info(self, message: str, details: Dict[str, Any] = None, suggestions: List[str] = None) -> None:
        """Add an info message to the validation result."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            message=message,
            details=details or {},
            suggestions=suggestions or []
        ))
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.issues.extend(other.issues)
        if not other.is_valid:
            self.is_valid = False
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of issues by severity."""
        return {
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'infos': len(self.infos),
            'total': len(self.issues)
        }


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, config=None):
        self.config = config
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> ValidationResult:
        """Perform validation and return results."""
        pass
    
    def _create_result(self, is_valid: bool = True) -> ValidationResult:
        """Create a new validation result."""
        return ValidationResult(is_valid=is_valid)
    
    def _format_suggestions(self, suggestions: List[str], max_suggestions: int = 3) -> List[str]:
        """Format and limit suggestions."""
        return suggestions[:max_suggestions] if suggestions else [] 