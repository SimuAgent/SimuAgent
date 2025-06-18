# Set global random seed before imports
import unsloth
import os, random, numpy as np, torch
import logging

# Configuration for log suppression
SUPPRESS_LOGS = True  # Set to False to see all logs

if SUPPRESS_LOGS:
    # Suppress all logging messages from system_parser and related modules
    logging.getLogger('system_parser').setLevel(logging.CRITICAL)
    logging.getLogger('system_parser.system_graph').setLevel(logging.CRITICAL)
    logging.getLogger('system_parser.pandapower').setLevel(logging.CRITICAL)
    logging.getLogger('system_parser.pandapower.pandapower_converter').setLevel(logging.CRITICAL)
    
    # Suppress pandapower library logs
    logging.getLogger('pandapower').setLevel(logging.CRITICAL)
    logging.getLogger('pandapower.diagnostic_reports').setLevel(logging.CRITICAL)
    
    # Also suppress other potential noisy loggers
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('unsloth').setLevel(logging.ERROR)
    
    # Set root logger to only show critical messages
    logging.getLogger().setLevel(logging.CRITICAL)
else:
    # Suppress system_parser module logs when not fully suppressing
    logging.getLogger('system_parser.system_graph').setLevel(logging.CRITICAL)

# Import the refactored ToolEnvironment class
from envs.environments import ToolEnvironment
from envs.validation_config import ValidationConfig
from trainers.grpo_env_trainer_unsloth import UnslothGRPOEnvTrainer
from rewards.power_system_reward import PowerSystemReward

# Environment variables and settings
SEED = 1000
use_reflection = False

# Set random seeds for reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import dependencies
from trl import GRPOConfig
from unsloth import FastLanguageModel
from utils.data_utils import preprocess_dataset
from tools.search_blocks import search_blocks

# Project configuration
dataset_name = "SimuAgent/CompleteSystem"
wandb_project = "CompleteSystem"
os.environ["WANDB_PROJECT"] = wandb_project

# Model and training parameters
max_seq_length       = 512 * 8
max_prompt_length    = 512 * 1
num_generations      = 4
lora_rank            = 32
lora_alpha           = 32
learning_rate        = 5e-6

model_name         = "Qwen/Qwen2.5-7B-Instruct"

run_name = "grpo" + f"-seed{SEED}-" + os.environ.get("SSH_ALIAS", "")

# Tool prompt for reasoning
SYSTEM_PROMPT = """
Think step-by-step inside <think>...</think> tags. Provide your final answer inside <answer>...</answer> tags.

You have access to tools to help solve problems:
{tool_descriptions}

Call tools using a JSON command within <tool> tags, including:

"name": tool name
"args": tool arguments
Tool output will appear in <result> tags. Multiple tool calls are allowed if needed.
<answer>...</answer> tags must contain only the final answer.</answer>
"""

# Setup environment
dataset = preprocess_dataset(dataset_name, "train")

# Create validation configuration
validation_config = ValidationConfig()

# Create the refactored ToolEnvironment
env = ToolEnvironment(
    dataset=dataset,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_blocks],
    max_steps=3,
    reward=PowerSystemReward(),
    validation_config=validation_config,
)

print(env.system_prompt)

# Load and configure model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_alpha,
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

# Training configuration
training_args = GRPOConfig(
    seed=SEED,
    output_dir=f"outputs/{wandb_project}/{run_name}",
    run_name=run_name,
    learning_rate=learning_rate,
    lr_scheduler_type="constant_with_warmup",  # Use constant lr for easier testing (can be changed to cosine)
    warmup_steps=10,
    num_train_epochs=1,
    temperature=1.0,
    max_steps=1000,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=2,
    beta=0.002,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    per_device_train_batch_size=num_generations,
    num_generations=num_generations,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.9,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
    reward_weights=env.get_reward_weights(),
)

# Initialize trainer
trainer = UnslothGRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=env.get_reward_funcs(),
    env=env,
    args=training_args,
    train_dataset=env.get_dataset(),
    eval_dataset=env.get_eval_dataset(),
    my_eval_steps=100,
)

if __name__ == "__main__":
    trainer.train() 