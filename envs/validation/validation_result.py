"""
Validation result classes for the environment system.

This module provides structured data classes for representing validation
results, errors, and suggestions in a consistent way.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    
    severity: ValidationSeverity
    message: str
    location: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of the validation issue."""
        location_str = f" at {self.location}" if self.location else ""
        return f"{self.severity.value.upper()}: {self.message}{location_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "suggestions": self.suggestions,
            "metadata": self.metadata
        }


@dataclass
class ValidationResult:
    """
    Comprehensive validation result with issues and suggestions.
    
    This class provides a structured way to represent validation results,
    including multiple issues, suggestions, and metadata.
    """
    
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        # If we have errors or critical issues, mark as invalid
        if any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
               for issue in self.issues):
            self.valid = False
    
    @property
    def reason(self) -> str:
        """Get the primary reason for validation failure."""
        if self.valid:
            return ""
        
        # Return the first error or critical issue
        for issue in self.issues:
            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                return issue.message
        
        # Fallback to first issue
        return self.issues[0].message if self.issues else "Unknown validation error"
    
    def add_issue(
        self, 
        severity: ValidationSeverity, 
        message: str, 
        location: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        **metadata
    ) -> None:
        """Add a new validation issue."""
        issue = ValidationIssue(
            severity=severity,
            message=message,
            location=location,
            suggestions=suggestions or [],
            metadata=metadata
        )
        self.issues.append(issue)
        
        # Update validity based on severity
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.valid = False
    
    def add_error(self, message: str, location: Optional[str] = None, **kwargs) -> None:
        """Add an error issue."""
        self.add_issue(ValidationSeverity.ERROR, message, location, **kwargs)
    
    def add_warning(self, message: str, location: Optional[str] = None, **kwargs) -> None:
        """Add a warning issue."""
        self.add_issue(ValidationSeverity.WARNING, message, location, **kwargs)
    
    def add_info(self, message: str, location: Optional[str] = None, **kwargs) -> None:
        """Add an info issue."""
        self.add_issue(ValidationSeverity.INFO, message, location, **kwargs)
    
    def add_suggestion(self, suggestion: str) -> None:
        """Add a general suggestion."""
        if suggestion not in self.suggestions:
            self.suggestions.append(suggestion)
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get all error and critical issues."""
        return [
            issue for issue in self.issues 
            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        ]
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning issues."""
        return [
            issue for issue in self.issues 
            if issue.severity == ValidationSeverity.WARNING
        ]
    
    def has_errors(self) -> bool:
        """Check if there are any error or critical issues."""
        return len(self.get_errors()) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warning issues."""
        return len(self.get_warnings()) > 0
    
    def combine(self, other: 'ValidationResult') -> 'ValidationResult':
        """Combine this result with another validation result."""
        combined = ValidationResult(
            valid=self.valid and other.valid,
            issues=self.issues + other.issues,
            suggestions=self.suggestions + other.suggestions,
            metadata={**self.metadata, **other.metadata}
        )
        return combined
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "issues": [issue.to_dict() for issue in self.issues],
            "suggestions": self.suggestions,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of validation result."""
        if self.valid:
            return "Validation passed"
        
        status = f"Validation failed: {self.reason}"
        if self.issues:
            issue_summary = f"\nIssues ({len(self.issues)}):"
            for issue in self.issues:
                issue_summary += f"\n  - {issue}"
            status += issue_summary
        
        if self.suggestions:
            status += f"\nSuggestions: {', '.join(self.suggestions)}"
        
        return status
    
    @classmethod
    def success(cls, message: Optional[str] = None, **metadata) -> 'ValidationResult':
        """Create a successful validation result."""
        result = cls(valid=True, metadata=metadata)
        if message:
            result.add_info(message)
        return result
    
    @classmethod
    def failure(
        cls, 
        message: str, 
        suggestions: Optional[List[str]] = None,
        **metadata
    ) -> 'ValidationResult':
        """Create a failed validation result."""
        result = cls(valid=False, suggestions=suggestions or [], metadata=metadata)
        result.add_error(message)
        return result
    
    @classmethod
    def from_legacy_dict(cls, legacy_dict: Dict[str, Any]) -> 'ValidationResult':
        """Convert from legacy validation dictionary format."""
        valid = legacy_dict.get("valid", False)
        reason = legacy_dict.get("reason", "")
        suggestions = legacy_dict.get("suggestions", [])
        
        result = cls(valid=valid, suggestions=suggestions)
        if not valid and reason:
            result.add_error(reason)
        
        return result 