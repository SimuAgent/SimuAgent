import logging
import sys
import json
import os
from typing import Optional, List, Tuple
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

# Global storage for web interface data
_web_training_data = []
_web_data_file = "web/training_data.json"

def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Setup basic logging configuration for the verifiers package.
    
    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Create a StreamHandler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    # Get the root logger for the verifiers package
    logger = logging.getLogger("verifiers")
    logger.setLevel(level.upper())
    logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False 


def print_prompt_completions_sample(
    index: int,
    prompts: list[dict],
    completions: list[dict],
    rewards: list[float],
    step: int,
) -> None:

    console = Console()
    table   = Table(show_header=True, header_style="bold white", expand=True)

    table.add_column("Prompt",     style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward",     style="bold cyan", justify="right")

    for prompt, completion, reward in zip(prompts, completions, rewards, strict=True):

        # ---------------------------------------------------------------------
        # 1. Process prompt (only modify list branch, keep others unchanged)
        # ---------------------------------------------------------------------
        formatted_prompt = Text()
        overlap_len      = 0                       # For completion branch

        if isinstance(prompt, dict):               # Single dict
            role    = prompt.get("role", "")
            content = prompt.get("content", "")
            style   = "bright_cyan" if role == "assistant" else "bright_magenta"
            formatted_prompt.append(f"{role}: ", style="bold")
            formatted_prompt.append(content, style=style)

        elif isinstance(prompt, list):             # Multiple dicts (modified here)
            completion_list = (
                completion if isinstance(completion, list) else [completion]
            )

            # ---- Calculate longest suffix / prefix overlap ----
            max_k = min(len(prompt), len(completion_list))
            for k in range(max_k, 0, -1):
                if prompt[-k:] == completion_list[:k]:
                    overlap_len = k                # Record for Completion processing
                    break

            prompt_to_show = prompt[:-overlap_len] if overlap_len else prompt

            for i, message in enumerate(prompt_to_show):
                if i: formatted_prompt.append("\n\n")
                role    = message.get("role", "")
                content = message.get("content", "")
                style   = "bright_cyan" if role == "assistant" else "bright_magenta"
                formatted_prompt.append(f"{role}: ", style="bold")
                formatted_prompt.append(content, style=style)

        else:                                      # Fallback for string
            formatted_prompt = Text(str(prompt))

        # ---------------------------------------------------------------------
        # 2. Process completion (add separator only in list branch)
        # ---------------------------------------------------------------------
        formatted_completion = Text()

        if isinstance(completion, dict):
            role    = completion.get("role", "")
            content = completion.get("content", "")
            style   = "bright_cyan" if role == "assistant" else "bright_magenta"
            formatted_completion.append(f"{role}: ", style="bold")
            formatted_completion.append(content, style=style)

        elif isinstance(completion, list):
            for i, message in enumerate(completion):
                # -------- Insert separator before first "non-repeated" message --------
                if i == overlap_len and overlap_len and i < len(completion):
                    formatted_completion.append("\n\n------ Reflection ------", style="bold")

                if i: formatted_completion.append("\n\n" if i else "")
                role    = message.get("role", "")
                content = message.get("content", "")
                style   = "bright_cyan" if role == "assistant" else "bright_magenta"
                formatted_completion.append(f"{role}: ", style="bold")
                formatted_completion.append(content, style=style)

        else:
            formatted_completion = Text(str(completion))

        # ---------------------------------------------------------------------
        # 3. Assemble table
        # ---------------------------------------------------------------------
        table.add_row(formatted_prompt, formatted_completion, Text(f"{reward:.2f}"))
        table.add_section()

    panel = Panel(table, expand=False, title=f"Step {step} - {index}", border_style="bold white")
    console.print(panel)


def _process_prompt_for_web(prompt) -> list:
    """Process prompt data for web interface, handling overlap removal logic."""
    if isinstance(prompt, dict):
        return [prompt]
    elif isinstance(prompt, list):
        return prompt
    else:
        return [{"role": "message", "content": str(prompt)}]


def _process_completion_for_web(completion, overlap_len: int = 0) -> list:
    """Process completion data for web interface, adding reflection separator if needed."""
    if isinstance(completion, dict):
        return [completion]
    elif isinstance(completion, list):
        processed = []
        for i, message in enumerate(completion):
            # Add reflection separator at the overlap point
            if i == overlap_len and overlap_len > 0:
                processed.append({"role": "separator", "content": "------ Reflection ------"})
            processed.append(message)
        return processed
    else:
        return [{"role": "message", "content": str(completion)}]


def export_training_data_for_web(
    index: int,
    prompts: list[dict],
    completions: list[dict],
    rewards: list[float],
    step: int,
    max_entries: int = 100,
    graph_visualizations: Optional[List[Optional[str]]] = None
) -> None:
    """
    Export training data in a format suitable for the web interface.
    This function should be called alongside print_prompt_completions_sample.
    
    Args:
        graph_visualizations: List of base64-encoded graph images (one per prompt/completion pair)
    """
    global _web_training_data
    
    # Process each prompt/completion/reward triplet
    for i, (prompt, completion, reward) in enumerate(zip(prompts, completions, rewards, strict=True)):
        
        # Calculate overlap for list-type prompts/completions (same logic as print function)
        overlap_len = 0
        if isinstance(prompt, list) and isinstance(completion, list):
            max_k = min(len(prompt), len(completion))
            for k in range(max_k, 0, -1):
                if prompt[-k:] == completion[:k]:
                    overlap_len = k
                    break
        
        # Process prompt (remove overlap if present)
        if isinstance(prompt, list) and overlap_len > 0:
            processed_prompt = prompt[:-overlap_len]
        else:
            processed_prompt = _process_prompt_for_web(prompt)
        
        # Process completion (add reflection separator if needed)
        processed_completion = _process_completion_for_web(completion, overlap_len)
        
        # Get graph visualization if available
        graph_viz = None
        if graph_visualizations and i < len(graph_visualizations):
            graph_viz = graph_visualizations[i]
        
        # Create entry for web interface
        web_entry = {
            "index": f"{index}-{i}" if len(prompts) > 1 else str(index),
            "step": step,
            "prompts": processed_prompt,
            "completions": processed_completion,
            "rewards": float(reward),
            "maxReward": 2.0,  # You can adjust this based on your reward scale
            "timestamp": datetime.now().isoformat(),
            "graph_visualization": graph_viz  # Add the graph visualization
        }
        
        # Add to global storage
        _web_training_data.append(web_entry)
    
    # Keep only the most recent entries
    if len(_web_training_data) > max_entries:
        _web_training_data = _web_training_data[-max_entries:]
    
    # Write to JSON file for web interface
    _write_training_data_to_file()


def print_and_export_prompt_completions_sample(
    index: int,
    prompts: list[dict],
    completions: list[dict],
    rewards: list[float],
    step: int,
    export_to_web: bool = True,
    graph_visualizations: Optional[List[Optional[str]]] = None
) -> None:
    """
    Combined function that both prints to console and exports to web interface.
    This is the recommended function to use instead of the separate functions.
    
    Args:
        graph_visualizations: List of base64-encoded graph images (one per prompt/completion pair)
    """
    # Print to console (original functionality)
    print_prompt_completions_sample(index, prompts, completions, rewards, step)
    
    # Export to web interface
    if export_to_web:
        export_training_data_for_web(index, prompts, completions, rewards, step, graph_visualizations=graph_visualizations)


def _write_training_data_to_file() -> None:
    """Write the training data to JSON file for web interface consumption."""
    try:
        # Ensure the web directory exists
        os.makedirs(os.path.dirname(_web_data_file), exist_ok=True)
        
        # Write data to file
        with open(_web_data_file, 'w', encoding='utf-8') as f:
            json.dump({
                "training_data": _web_training_data,
                "last_updated": datetime.now().isoformat(),
                "total_entries": len(_web_training_data)
            }, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Warning: Could not write training data to web file: {e}", file=sys.stderr)


def get_web_training_data() -> dict:
    """Get the current training data for web interface."""
    return {
        "training_data": _web_training_data,
        "last_updated": datetime.now().isoformat(),
        "total_entries": len(_web_training_data)
    }


def clear_web_training_data() -> None:
    """Clear all stored training data for web interface."""
    global _web_training_data
    _web_training_data = []
    _write_training_data_to_file()


def create_wandb_logs(
    all_prompts: List[List[dict]],
    all_completions: List[List[dict]],
    all_reflections: List[List[dict]] = None,
    sep: str = ")",
) -> Tuple[List[str], List[str]]:
    """
    Convert multi-turn prompt/completion sequences into numbered plain text logs for wandb.

    Returns
    -------
    all_prompts_to_log     : list[str]
        Processed prompt strings; adjacent messages separated by two newlines.
    all_completions_to_log : list[str]
        Processed completion strings; same format as above.
    """
    prompts_out, completions_out, reflections_out = [], [], []
    
    if all_reflections is None:
        all_reflections = [None] * len(all_prompts)  # Fill with None if no reflections provided

    for prompt, completion, reflection in zip(all_prompts, all_completions, all_reflections, strict=True):

        # Remove overlapping content between prompt and completion
        if len(prompt) >= len(completion) and prompt[-len(completion):] == completion:
            prompt_to_show = prompt[:-len(completion)]
        else:
            prompt_to_show = prompt

        # Add numbering to prompt messages
        prompt_lines = []
        for idx, msg in enumerate(prompt_to_show):
            role    = msg.get("role", "")
            content = msg.get("content", "")
            prompt_lines.append(f"{idx}{sep} {role}: {content}")

        # Continue numbering for completion messages
        completion_lines = []
        start_idx = len(prompt_to_show)
        for idx, msg in enumerate(completion, start=start_idx):
            role    = msg.get("role", "")
            content = msg.get("content", "")
            completion_lines.append(f"{idx}{sep} {role}: {content}")
            
        # Process reflections if provided
        if reflection is not None:
            reflection_lines = []
            start_idx = len(prompt_to_show) + len(completion)
            for idx, msg in enumerate(reflection, start=start_idx):
                role    = msg.get("role", "")
                content = msg.get("content", "")
                reflection_lines.append(f"{idx}{sep} {role}: {content}")
                
            reflections_out.append("\n\n".join(reflection_lines))

        # Combine into strings and collect results
        prompts_out.append("\n\n".join(prompt_lines))
        completions_out.append("\n\n".join(completion_lines))
    
    if all_reflections is None:
        reflections_out = None

    return prompts_out, completions_out, reflections_out
