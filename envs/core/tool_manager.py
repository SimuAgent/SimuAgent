"""
Tool management system for environments.

This module provides a clean interface for managing tools, their schemas,
and execution within the environment system.
"""

import inspect
import json
import traceback
from typing import Any, Callable, Dict, List, Optional

from .interfaces import ToolProtocol, ToolResult, ToolManagerProtocol


class ToolSchema:
    """Represents a tool's schema information."""
    
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self._parse_function_signature()
    
    def _parse_function_signature(self) -> None:
        """Parse function signature and docstring to create schema."""
        sig = inspect.signature(self.func)
        doc = inspect.getdoc(self.func) or ""
        
        # Parse docstring sections
        doc_parts = doc.split("\n\n")
        self.description = doc_parts[0].strip()
        
        # Extract examples and return description
        self.examples = []
        self.return_description = ""
        
        for part in doc_parts:
            if part.startswith("Examples:"):
                self.examples = [
                    line.strip() 
                    for line in part.split("\n")[1:] 
                    if line.strip()
                ]
            elif part.startswith("Returns:"):
                self.return_description = part.split("\n")[1].strip()
        
        # Build arguments schema
        self.args = {}
        for name, param in sig.parameters.items():
            param_doc = self._extract_param_doc(name, doc_parts)
            
            self.args[name] = {
                "type": self._get_type_name(param.annotation),
                "description": param_doc,
            }
            
            if param.default != inspect.Parameter.empty:
                self.args[name]["default"] = param.default
        
        # Set return type
        return_type = self._get_type_name(sig.return_annotation)
        self.returns = f"{self.return_description} ({return_type})"
    
    def _extract_param_doc(self, param_name: str, doc_parts: List[str]) -> str:
        """Extract parameter documentation from docstring."""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{param_name}:"):
                        return line.strip()[len(param_name)+1:].strip()
        return ""
    
    def _get_type_name(self, annotation: Any) -> str:
        """Get string representation of type annotation."""
        if annotation == inspect.Parameter.empty:
            return "any"
        return getattr(annotation, '__name__', str(annotation))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "args": self.args,
            "returns": self.returns,
            "examples": self.examples
        }


class ToolManager:
    """
    Manages tool registration, schema generation, and execution.
    
    This class provides a centralized way to handle all tool-related
    operations in the environment system.
    """
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, ToolSchema] = {}
    
    def register_tool(self, tool: Callable) -> None:
        """
        Register a new tool.
        
        Args:
            tool: Callable function to register as a tool
        """
        if not callable(tool):
            raise ValueError(f"Tool must be callable, got {type(tool)}")
        
        name = tool.__name__
        self._tools[name] = tool
        self._schemas[name] = ToolSchema(tool)
    
    def register_tools(self, tools: List[Callable]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register_tool(tool)
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool
            
        Returns:
            ToolResult containing execution result
        """
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                content="",
                error=f"Tool '{tool_name}' not found. Available tools: {list(self._tools.keys())}"
            )
        
        try:
            tool = self._tools[tool_name]
            result = tool(**args)
            return ToolResult(
                success=True,
                content=str(result) if result is not None else "",
                metadata={"tool_name": tool_name, "args": args}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Tool execution failed: {str(e)}",
                metadata={
                    "tool_name": tool_name,
                    "args": args,
                    "traceback": traceback.format_exc()
                }
            )
    
    def execute_tool_from_json(self, tool_json: str) -> ToolResult:
        """
        Execute a tool from JSON command string.
        
        Args:
            tool_json: JSON string with tool name and arguments
            
        Returns:
            ToolResult containing execution result
        """
        try:
            command = json.loads(tool_json)
            tool_name = command.get("name")
            args = command.get("args", {})
            
            if not tool_name:
                return ToolResult(
                    success=False,
                    content="",
                    error="Tool command must include 'name' field"
                )
            
            return self.execute_tool(tool_name, args)
            
        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Invalid JSON in tool command: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Tool execution error: {str(e)}"
            )
    
    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all registered tools.
        
        Returns:
            Formatted string describing all tools
        """
        if not self._schemas:
            return "No tools available."
        
        descriptions = []
        for schema in self._schemas.values():
            desc_parts = [f"{schema.name}: {schema.description}"]
            
            # Add arguments section
            if schema.args:
                desc_parts.append("\nArguments:")
                for arg_name, arg_info in schema.args.items():
                    default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
                    desc_parts.append(f"  - {arg_name}: {arg_info['description']}{default}")
            
            # Add examples section
            if schema.examples:
                desc_parts.append("\nExamples:")
                for example in schema.examples:
                    desc_parts.append(f"  {example}")
            
            # Add returns section
            if schema.returns:
                desc_parts.append(f"\nReturns: {schema.returns}")
            
            descriptions.append("\n".join(desc_parts))
        
        return "\n\n".join(descriptions)
    
    def list_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get list of tool schemas as dictionaries."""
        return [schema.to_dict() for schema in self._schemas.values()]
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool with the given name is registered."""
        return name in self._tools
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            name: Name of the tool to unregister
            
        Returns:
            True if tool was found and removed, False otherwise
        """
        if name in self._tools:
            del self._tools[name]
            del self._schemas[name]
            return True
        return False
    
    def clear_tools(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()
        self._schemas.clear() 