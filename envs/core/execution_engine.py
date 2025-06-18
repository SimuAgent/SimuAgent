"""
Code execution engine for environments.

This module provides a clean interface for executing Python code with
validation and error handling capabilities.
"""

import ast
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import ExecutorProtocol


class ExecutionResult:
    """Result from code execution."""
    
    def __init__(
        self,
        success: bool,
        globals_dict: Dict[str, Any],
        distances: Dict[str, Any],
        execution_log: List[str],
        corrected_lines: Optional[List[str]] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.globals_dict = globals_dict
        self.distances = distances
        self.execution_log = execution_log
        self.corrected_lines = corrected_lines or []
        self.error = error


class ExecutionEngine:
    """
    Handles Python code execution with validation and error recovery.
    
    This class provides safe execution of Python code with line-by-line
    validation and the ability to recover from validation errors.
    """
    
    def __init__(self, validator: Optional[Any] = None):
        """
        Initialize the execution engine.
        
        Args:
            validator: SystemValidator instance for validating system changes
        """
        self.validator = validator
    
    def _truncate_log_message(self, message: str, max_length: Optional[int] = None) -> str:
        """
        Truncate a log message to the specified maximum length.
        
        Args:
            message: Message to truncate
            max_length: Maximum length. If None, uses default setting
            
        Returns:
            Truncated message with suffix if needed
        """
        if max_length is None:
            if self.validator and hasattr(self.validator, 'config'):
                max_length = self.validator.config.get_setting('max_execution_log_length', 300)
            else:
                max_length = 300  # Default fallback
        
        if len(message) <= max_length:
            return message
        
        if self.validator and hasattr(self.validator, 'config'):
            suffix = self.validator.config.get_setting('truncation_suffix', '...')
        else:
            suffix = '...'  # Default fallback
        
        truncated_length = max_length - len(suffix)
        
        if truncated_length <= 0:
            return suffix
        
        return message[:truncated_length] + suffix
    
    def _truncate_total_log(self, execution_log: List[str]) -> List[str]:
        """
        Truncate the entire execution log when total length exceeds limit.
        
        Args:
            execution_log: List of log messages
            
        Returns:
            Truncated log list with ellipsis if needed
        """
        if self.validator and hasattr(self.validator, 'config'):
            max_total_length = self.validator.config.get_setting('max_total_log_length', 2000)
            suffix = self.validator.config.get_setting('truncation_suffix', '...')
        else:
            max_total_length = 2000
            suffix = '...'
        
        # Calculate total length of all messages
        total_length = sum(len(msg) for msg in execution_log)
        
        if total_length <= max_total_length:
            return execution_log
        
        # Truncate by removing messages from the end until we're under the limit
        truncated_log = []
        current_length = 0
        suffix_added = False
        
        for msg in execution_log:
            if current_length + len(msg) + len(suffix) <= max_total_length:
                truncated_log.append(msg)
                current_length += len(msg)
            else:
                if not suffix_added:
                    truncated_log.append(suffix)
                    suffix_added = True
                break
        
        return truncated_log
    
    def format_execution_result(self, result: ExecutionResult) -> str:
        """
        Format the execution result with proper <result> tags.
        
        Args:
            result: ExecutionResult to format
            
        Returns:
            Formatted result string with <result> tags
        """
        status = "SUCCESS" if result.success else "FAILED"
        
        # Format execution log
        log_summary = []
        for log_entry in result.execution_log:
            log_summary.append(log_entry)
        
        # Create formatted result
        formatted_result = f"<result>\nExecution Status: {status}\n"
        
        if log_summary:
            formatted_result += "Execution Log:\n"
            for entry in log_summary:
                formatted_result += f"  {entry}\n"
        
        if result.error:
            formatted_result += f"Error: {result.error}\n"
        
        if result.corrected_lines:
            formatted_result += f"Successfully executed {len(result.corrected_lines)} lines\n"
        
        formatted_result += "</result>"
        
        return formatted_result
    
    def execute_code(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        distances: Optional[Dict[str, Any]] = None,
        validate_changes: bool = True,
        traceback_limit: int = 30
    ) -> ExecutionResult:
        """
        Execute Python code with optional validation.
        
        Args:
            code: Python code to execute
            globals_dict: Global variables dictionary
            distances: Distance constraints dictionary
            validate_changes: Whether to validate system_dict changes
            traceback_limit: Maximum traceback lines to include
            
        Returns:
            ExecutionResult containing execution status and results
        """
        if globals_dict is None:
            globals_dict = {}
        if distances is None:
            distances = {}
        
        execution_log = []
        
        try:
            if validate_changes:
                return self._execute_with_validation(
                    code, globals_dict, distances, execution_log
                )
            else:
                return self._execute_simple(
                    code, globals_dict, distances, execution_log
                )
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            if traceback_limit > 0:
                tb_lines = traceback.format_exc().split('\n')
                if len(tb_lines) > traceback_limit:
                    tb_lines = tb_lines[:traceback_limit//2] + ["..."] + tb_lines[-traceback_limit//2:]
                error_msg += f"\n\nTraceback:\n{''.join(tb_lines)}"
            
            execution_log.append(error_msg)
            # Apply total log truncation
            truncated_log = self._truncate_total_log(execution_log)
            
            return ExecutionResult(
                success=False,
                globals_dict=globals_dict,
                distances=distances,
                execution_log=truncated_log,
                error=error_msg
            )
    
    def _execute_simple(
        self,
        code: str,
        globals_dict: Dict[str, Any],
        distances: Dict[str, Any],
        execution_log: List[str]
    ) -> ExecutionResult:
        """Execute code without validation."""
        try:
            exec(code, globals_dict)
            success_msg = f"✓ Executed code successfully"
            execution_log.append(success_msg)
            # Apply total log truncation
            truncated_log = self._truncate_total_log(execution_log)
            
            return ExecutionResult(
                success=True,
                globals_dict=globals_dict,
                distances=distances,
                execution_log=truncated_log
            )
        except Exception as e:
            error_msg = f"✗ Execution failed: {str(e)}"
            execution_log.append(error_msg)
            # Apply total log truncation
            truncated_log = self._truncate_total_log(execution_log)
            
            return ExecutionResult(
                success=False,
                globals_dict=globals_dict,
                distances=distances,
                execution_log=truncated_log,
                error=error_msg
            )
    
    def _execute_with_validation(
        self,
        code: str,
        globals_dict: Dict[str, Any],
        distances: Dict[str, Any],
        execution_log: List[str]
    ) -> ExecutionResult:
        """Execute code line by line with validation."""
        corrected_lines = []
        current_system_dict = deepcopy(globals_dict.get("system_dict", {}))
        current_distances = distances.copy()
        
        # Parse the code into statements
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            error_msg = f"Syntax error in code: {e}"
            execution_log.append(error_msg)
            # Apply total log truncation
            truncated_log = self._truncate_total_log(execution_log)
            
            return ExecutionResult(
                success=False,
                globals_dict=globals_dict,
                distances=distances,
                execution_log=truncated_log,
                error=error_msg
            )
        
        for node in tree.body:
            try:
                # Convert the AST node back to source code
                line_code = ast.unparse(node)
                
                # Execute the line
                exec(line_code, globals_dict)
                
                # Get updated system_dict
                updated_system_dict = globals_dict.get("system_dict", {})
                
                # Validate the line if it modifies system_dict
                if updated_system_dict != current_system_dict:
                    validation_result = self.validator.validate_system_dict_change(
                        current_system_dict, updated_system_dict, current_distances, line_code
                    )
                    
                    if validation_result["valid"]:
                        current_system_dict = deepcopy(updated_system_dict)
                        corrected_lines.append(line_code)
                        log_msg = f"✓ Executed: {line_code}"
                        execution_log.append(log_msg)
                    else:
                        # Revert the change
                        globals_dict["system_dict"] = deepcopy(current_system_dict)
                        fail_msg = f"✗ Failed: {line_code}. {validation_result['reason']}"
                        execution_log.append(fail_msg)
                else:
                    # Line didn't change system_dict, so it's safe to execute
                    corrected_lines.append(line_code)
                    log_msg = f"✓ Executed: {line_code}"
                    execution_log.append(log_msg)
                    
            except Exception as e:
                line_code_str = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                error_msg = f"✗ Failed: {line_code_str}. Error: {str(e)}"
                execution_log.append(error_msg)
                continue
        
        # Apply total log truncation before returning
        truncated_log = self._truncate_total_log(execution_log)
        
        return ExecutionResult(
            success=True,
            globals_dict=globals_dict,
            distances=current_distances,
            execution_log=truncated_log,
            corrected_lines=corrected_lines
        )
    
    def execute_init_code(
        self,
        init_code: str,
        globals_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute initialization code and extract variables.
        
        Args:
            init_code: Python initialization code
            globals_dict: Existing globals dictionary
            
        Returns:
            Tuple of (updated_globals, distances)
        """
        if globals_dict is None:
            globals_dict = {}
        
        distances = {}
        
        try:
            # Execute init code in a temporary namespace
            temp_globals = {}
            exec(init_code, temp_globals)
            
            # Extract distances separately
            if 'distances' in temp_globals:
                distances.update(temp_globals['distances'])
                del temp_globals['distances']
            
            # Move remaining variables to globals_dict
            globals_dict.update(temp_globals)
            
        except Exception as e:
            # Log error but continue - init code is optional
            pass
        
        return globals_dict, distances
    
    def extract_python_code(self, content: str) -> str:
        """
        Extract Python code from content (e.g., from markdown code blocks).
        
        Args:
            content: Content that may contain Python code
            
        Returns:
            Extracted Python code string
        """
        # This is a placeholder - the actual implementation would depend
        # on how Python code is embedded in the content
        # For now, assuming the content IS Python code
        if "```python" in content:
            # Extract code from markdown code blocks
            start = content.find("```python") + 9
            end = content.find("```", start)
            return content[start:end].strip()
        
        return content.strip() 