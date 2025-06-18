import warnings
from typing import Callable, Optional, Union, Any, List
import pandas as pd
import numpy as np
import copy

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from peft import PeftConfig # type: ignore
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available
)

from utils.logging_utils import print_prompt_completions_sample, create_wandb_logs
from vllm import LLM, SamplingParams

import random

from trl import GRPOTrainer, GRPOConfig
from unsloth_compiled_cache.UnslothGRPOTrainer import UnslothGRPOTrainer
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad
from trl.trainer.grpo_trainer import transformers, version

if is_wandb_available():
    import wandb


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

class UnslothGRPOEnvTrainer(UnslothGRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Any,
            reward_funcs: Any,
            scale_rewards: bool = False,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            my_eval_steps: Optional[int] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        self.vllm_client = None
        if not args.use_vllm: # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))): 
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        self.my_eval_steps = my_eval_steps
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env
        self.scale_rewards = scale_rewards
        self.sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=args.top_p,
            top_k=-1 if args.top_k is None else args.top_k,
            min_p=0.0 if args.min_p is None else args.min_p,
            repetition_penalty=args.repetition_penalty
        )
        
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Write metrics to the logger.
        Metrics are written regardless of whether we're in an evaluation step.
        If evaluation metrics are present, they are automatically prefixed with 'eval_'.
        """
        merged_metrics = {}
        for split in ("train", "eval"):
            if self._metrics[split]:                                   # Only process if buffer is not empty
                avg = {k: sum(v) / len(v) for k, v in self._metrics[split].items()}
                if split == "eval":                                    # Add prefix for evaluation metrics
                    avg = {f"eval_{k}": v for k, v in avg.items()}
                merged_metrics.update(avg)

        logs = {**logs, **merged_metrics}

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            Trainer.log(self, logs, start_time)
        else:  # transformers<=4.46
            Trainer.log(self, logs)
            
        for split in ("train", "eval"):
            self._metrics[split].clear()                           # Clear buffer after writing
        
    def _maybe_use_eval_inputs(self, inputs):
        """
        If my_eval_steps condition is met, sample from eval_dataset with the same batch size (without replacement).
        Returns (inputs, is_eval_step).
        """
        is_eval_step = (
            self.my_eval_steps is not None
            and (self.state.global_step % self.my_eval_steps == 0)
        )
        if not (is_eval_step and self.eval_dataset is not None):
            return inputs, False                      # Training step or no eval_dataset

        bs = len(inputs)
        if bs > len(self.eval_dataset):
            raise ValueError("eval_dataset size is smaller than current batch size")

        # ---------- Sample without replacement ----------
        if not hasattr(self, "_eval_remaining_indices") or len(self._eval_remaining_indices) < bs:
            self._eval_remaining_indices = list(range(len(self.eval_dataset)))
            random.shuffle(self._eval_remaining_indices)

        sample_idx = [self._eval_remaining_indices.pop() for _ in range(bs)]
        inputs = [self.eval_dataset[i] for i in sample_idx]
        # ----------------------------------------------
        return inputs, True                           # Evaluation step

    def _generate_and_score_completions(
         self, inputs: dict[str, Union[torch.Tensor, Any]]   
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Call the wrapper function to get processed inputs and evaluation step flag
        inputs, is_eval_step = self._maybe_use_eval_inputs(inputs)
        # print(f"step {self.state.global_step}, is_eval_step: {is_eval_step}")
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs] # type: ignore
        answers = [x["answer"] for x in inputs] # type: ignore
        tasks = [x["task"] for x in inputs] # type: ignore
        if "init_code" in inputs[0]:
            init_codes = [x["init_code"] for x in inputs] # type: ignore
        else:
            init_codes = None
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
        ) # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(
                prompts=all_prompts,
                llm=self.llm,
                # llm=self.vllm_client, # type: ignore
                sampling_params=self.sampling_params,
                answers=answers,
                init_codes=init_codes,
                tasks=tasks,
                lora_request=self.model.load_lora('grpo_trainer_lora_model', load_tensors=True),
                mode = "train" if not is_eval_step else "eval",
            )
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']
            reflections = env_result.get('reflections', None)
            gen_globals_list = env_result.get('gen_globals', None)

        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)
            gen_globals_list = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        gen_globals_list = broadcast_object_list(gen_globals_list, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        gen_globals_list = gen_globals_list[process_slice]

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # type: ignore

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # (B, P+C)
        
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # use message dicts for reward function inputs
        completions = completion_messages
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        gen_code_list = None
        ref_code_list = None
        correctness_reward_list = None
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
            
            # Add gen_globals_list to reward_kwargs if available
            if gen_globals_list is not None:
                reward_kwargs['gen_globals_list'] = gen_globals_list
            
            reward_outputs = reward_func(prompts=prompts, completions=completions, **reward_kwargs)  # type: ignore

            # reward_func may return a single value or a tuple (reward_list, gen_code_list, ref_code_list)
            if isinstance(reward_outputs, tuple) and len(reward_outputs) == 3:
                output_reward_func, gen_code_list, ref_code_list = reward_outputs
                correctness_reward_list = copy.deepcopy(output_reward_func)
            else:
                output_reward_func = reward_outputs
            
            # assert isinstance(output_reward_func, list), f"Reward function {reward_func.__name__} must return a list of rewards."
            
            # if reward_func.__name__ == "correctness_reward_func":
            #     correctness_reward_list, gen_code_list, ref_code_list = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # type: ignore
            #     output_reward_func = correctness_reward_list
            # else:
            #     output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # type: ignore
            
            # output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
            assert None not in output_reward_func, f"Reward function {reward_func.__name__} returned {output_reward_func}."
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()} # type: ignore
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx] # type: ignore
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )


        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        advantages = (rewards - mean_grouped_rewards)
        
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        if self.scale_rewards:
            # Scale the rewards to be between 0 and 1
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ # type: ignore  
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
            
            if is_eval_step:
                self._metrics["eval"][f"rewards/{reward_func_name}"].append(mean_rewards)
                self._metrics["eval"][f"rewards/{reward_func_name}/std"].append(std_rewards)
            
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item()) # type: ignore
        
        # Add load_satisfaction metrics if available
        for i, reward_func in enumerate(self.reward_funcs):
            if hasattr(reward_func, 'get_component_rewards_summary') or (hasattr(reward_func, '__self__') and hasattr(reward_func.__self__, 'get_component_rewards_summary')):
                if hasattr(reward_func, 'get_component_rewards_summary'):
                    comp_rewards = reward_func.get_component_rewards_summary()
                else:
                    comp_rewards = reward_func.__self__.get_component_rewards_summary()
                
                if comp_rewards and 'load_satisfaction' in comp_rewards:
                    load_satisfaction_values = torch.tensor(comp_rewards['load_satisfaction'], device=device)
                    self._metrics[mode]["load_satisfaction"].append(load_satisfaction_values.mean().item())
                    self._metrics[mode]["load_satisfaction_std"].append(nanstd(load_satisfaction_values).item())
                    
                    if is_eval_step:
                        self._metrics["eval"]["load_satisfaction"].append(load_satisfaction_values.mean().item())
                        self._metrics["eval"]["load_satisfaction_std"].append(nanstd(load_satisfaction_values).item())
                break
        
        if is_eval_step:
            self._metrics["eval"]["reward"].append(rewards.mean().item())
            self._metrics["eval"]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    for i in range(len(prompts_to_log)):
                        print_prompt_completions_sample(
                            i,
                            [prompts_to_log[i]],
                            [completions_to_log[i] + reflections[i] if reflections is not None else completions_to_log[i]],
                            [rewards_to_log[i]],
                            self.state.global_step,
                        )
                    # print_prompt_completions_sample(
                    #         [prompts_to_log[1]],
                    #         [completions_to_log[1] + reflections[1] if reflections is not None else completions_to_log[1]],
                    #         [rewards_to_log[1]],
                    #         self.state.global_step,
                    #     )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None: # type: ignore
                    prompts_to_log_strs, completions_to_log_strs, reflections_to_log_strs = create_wandb_logs(prompts_to_log, completions_to_log, reflections)

                    # Get graph visualizations, network plots, and reward components from reward functions if available
                    graph_visualizations = []
                    network_plots = []
                    component_rewards = {}
                    formatted_reward_strings = []
                    for reward_func in self.reward_funcs:
                        func_name = getattr(reward_func, '__name__', str(reward_func))
                        print(f"Checking reward function: {func_name}")
                        
                        # Check if the reward function itself has the method
                        if hasattr(reward_func, 'get_last_graph_visualizations'):
                            viz = reward_func.get_last_graph_visualizations()
                            if viz and any(v is not None for v in viz):
                                graph_visualizations = viz
                        # Check if it's a bound method and its instance has the method
                        elif hasattr(reward_func, '__self__') and hasattr(reward_func.__self__, 'get_last_graph_visualizations'):
                            viz = reward_func.__self__.get_last_graph_visualizations()
                            if viz and any(v is not None for v in viz):
                                graph_visualizations = viz
                        else:
                            print(f"  No get_last_graph_visualizations method found")
                            
                        # Get network plots from reward functions if available
                        if hasattr(reward_func, 'get_last_network_plots'):
                            plots = reward_func.get_last_network_plots()
                            if plots and any(p is not None for p in plots):
                                network_plots = plots
                        elif hasattr(reward_func, '__self__') and hasattr(reward_func.__self__, 'get_last_network_plots'):
                            plots = reward_func.__self__.get_last_network_plots()
                            if plots and any(p is not None for p in plots):
                                network_plots = plots
                        else:
                            print(f"  No get_last_network_plots method found")
                        
                        # Get component rewards from reward functions if available
                        if hasattr(reward_func, 'get_component_rewards_summary'):
                            comp_rewards = reward_func.get_component_rewards_summary()
                            if comp_rewards:
                                component_rewards.update(comp_rewards)
                        elif hasattr(reward_func, '__self__') and hasattr(reward_func.__self__, 'get_component_rewards_summary'):
                            comp_rewards = reward_func.__self__.get_component_rewards_summary()
                            if comp_rewards:
                                component_rewards.update(comp_rewards)
                        else:
                            print(f"  No get_component_rewards_summary method found")
                        
                        # Get formatted reward strings if available
                        if hasattr(reward_func, 'get_formatted_reward_strings'):
                            reward_strings = reward_func.get_formatted_reward_strings()
                            if reward_strings:
                                formatted_reward_strings = reward_strings
                        elif hasattr(reward_func, '__self__') and hasattr(reward_func.__self__, 'get_formatted_reward_strings'):
                            reward_strings = reward_func.__self__.get_formatted_reward_strings()
                            if reward_strings:
                                formatted_reward_strings = reward_strings
                    
                    print(f"Final graph_visualizations: {len(graph_visualizations)} items")
                    print(f"Final network_plots: {len(network_plots)} items")
                    print(f"Component rewards: {list(component_rewards.keys())}")
                    
                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(prompts_to_log),
                        "prompt": prompts_to_log_strs,
                        "completion": completions_to_log_strs,
                        # "gen_code": gen_code_list,
                        # "ref_code": ref_code_list,
                        "correctness_reward": correctness_reward_list,
                        # "reward": rewards.tolist(),
                    }
                    
                    if reflections is not None:
                        table["reflections"] = reflections_to_log_strs
                    
                    if correctness_reward_list is not None:
                        table["gen_code"] = gen_code_list
                        table["ref_code"] = ref_code_list
                        table["correctness_reward"] = correctness_reward_list
                    else:
                        # Use pre-formatted reward strings if available, otherwise format on the fly
                        if formatted_reward_strings:
                            table["reward"] = formatted_reward_strings
                        elif component_rewards:
                            # Create concise formatted reward strings
                            reward_strings = []
                            
                            # Get power system weights to check which components to include
                            power_system_weights = {}
                            for reward_func in self.reward_funcs:
                                if hasattr(reward_func, 'get_power_system_weights'):
                                    power_system_weights = reward_func.get_power_system_weights()
                                    break
                                elif hasattr(reward_func, '__self__') and hasattr(reward_func.__self__, 'get_power_system_weights'):
                                    power_system_weights = reward_func.__self__.get_power_system_weights()
                                    break
                            
                            for i in range(len(prompts_to_log)):
                                reward_parts = []
                                
                                # Add total first
                                if 'total' in component_rewards:
                                    reward_parts.append(f"Total: {component_rewards['total'][i]:.3f}")
                                
                                # Add tool execution rewards (only if weight > 0)
                                if 'tool_execution' in component_rewards and power_system_weights.get('tool_execution', 0.0) > 0:
                                    reward_parts.append(f"Tool Exec: {component_rewards['tool_execution'][i]:.3f}")
                                if 'format' in component_rewards and power_system_weights.get('format', 0.0) > 0:
                                    reward_parts.append(f"Format: {component_rewards['format'][i]:.3f}")
                                if 'xml' in component_rewards and power_system_weights.get('xml', 0.0) > 0:
                                    reward_parts.append(f"XML: {component_rewards['xml'][i]:.3f}")
                                
                                # Add power system rewards (only if weight > 0)
                                if 'connectivity' in component_rewards and power_system_weights.get('connectivity', 0.0) > 0:
                                    reward_parts.append(f"Connectivity: {component_rewards['connectivity'][i]:.3f}")
                                if 'validation' in component_rewards and power_system_weights.get('validation', 0.0) > 0:
                                    reward_parts.append(f"Validation: {component_rewards['validation'][i]:.3f}")
                                if 'parameter' in component_rewards and power_system_weights.get('parameter', 0.0) > 0:
                                    reward_parts.append(f"Parameter: {component_rewards['parameter'][i]:.3f}")
                                if 'conversion' in component_rewards and power_system_weights.get('conversion', 0.0) > 0:
                                    reward_parts.append(f"Conversion: {component_rewards['conversion'][i]:.3f}")
                                if 'diagnostic' in component_rewards and power_system_weights.get('diagnostic', 0.0) > 0:
                                    reward_parts.append(f"Diagnostic: {component_rewards['diagnostic'][i]:.3f}")
                                if 'load_satisfaction' in component_rewards and power_system_weights.get('load_satisfaction', 0.0) > 0:
                                    reward_parts.append(f"Load Satisfaction: {component_rewards['load_satisfaction'][i]:.3f}")
                                if 'structure' in component_rewards and power_system_weights.get('structure', 0.0) > 0:
                                    reward_parts.append(f"Structure: {component_rewards['structure'][i]:.3f}")
                                
                                reward_strings.append("\n".join(reward_parts))
                            
                            table["reward"] = reward_strings
                        else:
                            table["reward"] = [f"Total: {r:.3f}" for r in rewards_to_log]
                    
                    # Add graph visualizations if available
                    if graph_visualizations:
                        print(f"Processing {len(graph_visualizations)} graph visualizations")
                        # Convert base64 images to wandb.Image objects
                        wandb_images = []
                        for i, viz in enumerate(graph_visualizations):
                            if viz:
                                try:
                                    import io
                                    import base64
                                    from PIL import Image
                                    # Decode base64 to image
                                    image_data = base64.b64decode(viz)
                                    image = Image.open(io.BytesIO(image_data))
                                    wandb_images.append(wandb.Image(image))
                                except Exception as e:
                                    print(f"Error converting graph visualization {i}: {e}")
                                    wandb_images.append(None)
                            else:
                                print(f"  Visualization {i} is None")
                                wandb_images.append(None)
                        
                        # Ensure the length matches the table
                        expected_length = len(prompts_to_log)
                        if len(wandb_images) != expected_length:
                            print(f"Warning: Image count ({len(wandb_images)}) doesn't match table length ({expected_length})")
                            # Pad or truncate to match
                            if len(wandb_images) < expected_length:
                                wandb_images.extend([None] * (expected_length - len(wandb_images)))
                            else:
                                wandb_images = wandb_images[:expected_length]
                        
                        table["graph_visualization"] = wandb_images
                    
                    # Add network plots if available
                    if network_plots:
                        print(f"Processing {len(network_plots)} network plots")
                        # Convert base64 images to wandb.Image objects
                        wandb_network_images = []
                        for i, plot in enumerate(network_plots):
                            if plot:
                                try:
                                    import io
                                    import base64
                                    from PIL import Image
                                    # Decode base64 to image
                                    image_data = base64.b64decode(plot)
                                    image = Image.open(io.BytesIO(image_data))
                                    wandb_network_images.append(wandb.Image(image))
                                except Exception as e:
                                    print(f"Error converting network plot {i}: {e}")
                                    wandb_network_images.append(None)
                            else:
                                print(f"  Network plot {i} is None")
                                wandb_network_images.append(None)
                        
                        # Ensure the length matches the table
                        expected_length = len(prompts_to_log)
                        if len(wandb_network_images) != expected_length:
                            print(f"Warning: Network image count ({len(wandb_network_images)}) doesn't match table length ({expected_length})")
                            # Pad or truncate to match
                            if len(wandb_network_images) < expected_length:
                                wandb_network_images.extend([None] * (expected_length - len(wandb_network_images)))
                            else:
                                wandb_network_images = wandb_network_images[:expected_length]
                        
                        table["network_plot"] = wandb_network_images
                    
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)}) # type: ignore
                    
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }