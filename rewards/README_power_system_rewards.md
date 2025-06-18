# Power System Reward Framework

## Overview

The `PowerSystemReward` class provides a comprehensive, modular reward system that evaluates power system designs across multiple dimensions. It extends the `ToolReward` class to include all standard tool execution and formatting rewards, while replacing the `correct_answer_reward_func` with a sophisticated component-based evaluation system.

**Key Features:**
- **Component-based evaluation**: Seven distinct reward components focusing on different aspects of system quality
- **ToolReward integration**: Inherits all ToolReward functionality (tool execution, XML formatting, parser validation)
- **Flexible weighting**: Customizable weights for different training objectives
- **Real-time feedback**: Detailed breakdown of each reward component

## Reward Components

### 1. Connectivity Reward (`connectivity_reward`)
**Purpose**: Evaluates how well generators can reach loads through the transmission network.

**Calculation**: 
- Uses graph traversal to find paths from generators to loads
- Returns the ratio of connected loads to total loads (0.0 to 1.0)

**When to emphasize**: 
- Early training phases focusing on basic network topology
- When transmission network connectivity is critical

### 2. Validation Reward (`validation_reward`)
**Purpose**: Assesses basic graph validation issues.

**Calculation**: 
- Penalizes validation errors (weight: 1.0), warnings (weight: 0.5), and unconnected ports (weight: 0.2)
- Formula: `max(0.0, min(1.0, 1.0 / (1.0 + penalty)))`

**When to emphasize**: 
- When ensuring basic structural correctness
- For debugging malformed system configurations

### 3. Parameter Reward (`parameter_reward`)
**Purpose**: Evaluates the correctness of block parameters.

**Calculation**: 
- Counts parameter validation issues and converts to 0-1 reward
- Formula: `max(0.0, min(1.0, 1.0 / (1.0 + param_issues)))`

**When to emphasize**: 
- When parameter accuracy is critical
- During fine-tuning phases focusing on component specifications

### 4. Conversion Reward (`conversion_reward`)
**Purpose**: Measures success of converting the system to Pandapower format.

**Calculation**: 
- Considers connection usage ratio, element creation success, and network structure
- Penalizes unused connections as they indicate incorrect specifications

**When to emphasize**: 
- When ensuring compatibility with power system analysis tools
- For validating that the designed system can be simulated

### 5. Diagnostic Reward (`diagnostic_reward`)
**Purpose**: Evaluates electrical validity through Pandapower diagnostics.

**Calculation**: 
- Runs power flow analysis and checks for electrical issues
- Heavily penalizes convergence failures, overloads, and electrical violations
- Rewards systems with no diagnostic issues

**When to emphasize**: 
- When electrical correctness is paramount
- For ensuring power flow feasibility

### 6. Load Satisfaction Reward (`load_satisfaction_reward`)
**Purpose**: Assesses whether loads are adequately satisfied.

**Calculation**: 
- Base satisfaction from connectivity ratio
- Penalizes power flow convergence failures (penalty: 0.5)
- Penalizes overload conditions (penalty: 0.1 per overloaded element, max 0.3)

**When to emphasize**: 
- When load serving capability is the primary concern
- For distribution system design

### 7. Structure Reward (`structure_reward`)
**Purpose**: Evaluates overall network structure quality.

**Calculation**: 
- Checks for presence of essential element types (generators, loads, transmission)
- Verifies successful creation of corresponding Pandapower elements
- Scores: generators (0.4), loads (0.4), transmission (0.2), buses (0.2)

**When to emphasize**: 
- During initial training phases
- When ensuring basic system architecture

## Usage

### Basic Usage

```python
from rewards.power_system_reward import PowerSystemReward

# Initialize with default weights (includes all ToolReward functionality)
reward_system = PowerSystemReward()

# Use in your training loop - includes tool execution, formatting, and power system evaluation
rewards = reward_system.correct_answer_reward_func(completions, init_code, gen_globals_list)
```

### Custom Weight Configuration

```python
# Define custom weights for power system components
custom_power_weights = {
    'connectivity': 0.30,      # Higher emphasis on connectivity
    'validation': 0.10,        # Lower emphasis on basic validation
    'parameter': 0.15,         # Moderate parameter importance
    'conversion': 0.20,        # Standard conversion weight
    'diagnostic': 0.20,        # Standard diagnostic weight
    'load_satisfaction': 0.05, # Basic load satisfaction
    'structure': 0.00          # Disable structure component
}

# ToolReward parameters can also be customized
reward_system = PowerSystemReward(
    power_system_weights=custom_power_weights,
    tools=[some_tool_function],  # Add custom tools
    parser=custom_parser         # Custom XML parser
)
```

### Accessing Individual Components

```python
# After running reward calculation
components = reward_system.get_last_reward_components()
summary = reward_system.get_component_rewards_summary()

# Get detailed breakdown for each sample
for i, comp in enumerate(components):
    print(f"Sample {i}:")
    print(f"  Connectivity: {comp.connectivity_reward:.3f}")
    print(f"  Validation: {comp.validation_reward:.3f}")
    print(f"  Parameters: {comp.parameter_reward:.3f}")
    # ... etc
```

### Dynamic Weight Updates

```python
# Update power system weights during training for different phases
reward_system.update_power_system_weights({
    'diagnostic': 0.40,        # Focus on electrical correctness
    'conversion': 0.30,        # Ensure convertibility
    'connectivity': 0.30       # Maintain connectivity
})

# Get current weights
current_weights = reward_system.get_power_system_weights()
```

### ToolReward Integration

```python
# PowerSystemReward automatically includes all ToolReward functionality
print(f"Number of reward functions: {len(reward_system.reward_funcs)}")
print(f"Reward function names: {[func.__name__ for func in reward_system.reward_funcs]}")

# Inherited reward functions include:
# - tool_execution_reward_func
# - get_format_reward_func (from parser)
# - get_xml_reward_func (from parser)  
# - Tool-specific reward functions (if tools are provided)
```

## Training Strategies

### Progressive Training Phases

1. **Phase 1 - Basic Structure** (Weights: structure=0.4, validation=0.3, connectivity=0.3)
   - Focus on creating valid system architectures
   - Ensure basic component types are present

2. **Phase 2 - Connectivity Focus** (Weights: connectivity=0.5, structure=0.2, conversion=0.2, validation=0.1)
   - Emphasize generator-to-load connectivity
   - Build proper transmission networks

3. **Phase 3 - Electrical Correctness** (Weights: diagnostic=0.35, conversion=0.25, load_satisfaction=0.2, connectivity=0.2)
   - Focus on electrical validity and power flow
   - Ensure systems can be analyzed

4. **Phase 4 - Parameter Precision** (Weights: parameter=0.3, diagnostic=0.25, conversion=0.2, connectivity=0.15, validation=0.1)
   - Fine-tune component parameters
   - Optimize electrical characteristics

### Specialized Applications

**Distribution System Design**: Emphasize connectivity and load satisfaction
```python
weights = {'connectivity': 0.4, 'load_satisfaction': 0.3, 'structure': 0.3}
```

**Transmission System Design**: Focus on electrical validity and conversion
```python
weights = {'diagnostic': 0.4, 'conversion': 0.3, 'connectivity': 0.3}
```

**Research Applications**: Balance all components equally
```python
weights = {comp: 1.0/7 for comp in ['connectivity', 'validation', 'parameter', 'conversion', 'diagnostic', 'load_satisfaction', 'structure']}
```

## API Reference

### PowerSystemReward Class

#### Constructor
```python
PowerSystemReward(parser=None, env_parser=None, tools=[], power_system_weights=None)
```

#### Inherited from ToolReward
- All `ToolReward` methods and properties
- `reward_funcs`: List of all reward functions
- `reward_weights`: Weights for each reward function
- `tools`: Dictionary of available tools
- Tool execution and formatting validation

#### Power System Specific Methods

**`correct_answer_reward_func(completions, init_code, gen_globals_list=None, **kwargs)`**
- Overrides ToolReward's method with sophisticated power system evaluation
- Returns list of total rewards for each sample

**`get_last_reward_components()`**
- Returns list of `RewardComponents` objects from last calculation

**`get_component_rewards_summary()`**
- Returns dictionary with component name -> list of values mapping

**`update_power_system_weights(new_weights)`**
- Updates power system reward weights with new values

**`get_power_system_weights()`**
- Returns current power system reward weights

### RewardComponents Class

#### Attributes
- `connectivity_reward`: float (0.0 to 1.0)
- `validation_reward`: float (0.0 to 1.0)
- `parameter_reward`: float (0.0 to 1.0)
- `conversion_reward`: float (0.0 to 1.0)
- `diagnostic_reward`: float (0.0 to 1.0)
- `load_satisfaction_reward`: float (0.0 to 1.0)
- `structure_reward`: float (0.0 to 1.0)

#### Methods

**`get_total_reward(weights=None)`**
- Calculates weighted sum of all components
- Uses provided weights or default if None

## Default Configurations

### Power System Weights
```python
default_power_weights = {
    'connectivity': 0.25,      # Primary importance
    'validation': 0.15,        # Moderate importance
    'parameter': 0.10,         # Basic importance
    'conversion': 0.20,        # High importance
    'diagnostic': 0.20,        # High importance
    'load_satisfaction': 0.05, # Basic importance
    'structure': 0.05          # Basic importance
}
```

### ToolReward Integration
- `correct_answer_reward_func`: Weight increased to 2.0 (contains power system evaluation)
- `tool_execution_reward_func`: Weight 0.5 (standard)
- `get_format_reward_func`: Weight 0.25 (standard)
- `get_xml_reward_func`: Weight 0.25 (standard)

## Migration Guide

### From CompleteReward

```python
# Old
from rewards.complete_reward import CompleteReward
reward_system = CompleteReward(reward_weights=weights)

# New  
from rewards.power_system_reward import PowerSystemReward
reward_system = PowerSystemReward(power_system_weights=weights)
```

### Method Name Changes
- `update_reward_weights()` → `update_power_system_weights()`
- `get_reward_weights()` → `get_power_system_weights()`

All other methods remain the same for backward compatibility.

## Examples

See `rewards/power_system_usage_example.py` for comprehensive usage examples and training scenarios demonstrating both ToolReward integration and power system specific functionality. 