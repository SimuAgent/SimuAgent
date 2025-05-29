import logging
import sys
from typing import Optional, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

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

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


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
