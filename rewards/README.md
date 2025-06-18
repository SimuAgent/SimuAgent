# Rewards Package

**A comprehensive, modular reward system for evaluating AI agent performance across multiple domains.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Overview

The Rewards package provides a flexible, extensible framework for evaluating AI agent performance across diverse domains including mathematics, code execution, tool usage, and power system analysis. Built with modern Python practices, it offers a clean, object-oriented architecture that's easy to extend and maintain.

### Key Features

- 🧩 **Modular Architecture**: Pluggable components for different evaluation types
- 🔧 **Extensible Design**: Easy to add new reward types and evaluation methods
- 🎯 **Multi-Domain Support**: Mathematics, code, tools, and power systems
- 📊 **Comprehensive Metrics**: Detailed scoring with component breakdown
- 🔄 **Backward Compatible**: Smooth migration from legacy implementations
- 🚀 **High Performance**: Optimized algorithms with intelligent caching
- 📚 **Well Documented**: Comprehensive guides and examples

## 🚀 Quick Start

### Installation

```bash
# Install the package (assuming it's part of a larger project)
pip install -e .
```

### Basic Usage

```python
from rewards import PowerSystemReward, ToolReward, MathReward

# Power system evaluation
ps_reward = PowerSystemReward(
    weights={
        'load_satisfaction': 1.0,
        'connectivity': 0.5,
        'validation': 0.3
    }
)

# Evaluate power system designs
scores = ps_reward.evaluate(completions, init_code)
print(f"Power system score: {scores[0]:.3f}")

# Tool-based evaluation
tool_reward = ToolReward(
    tools=[calculator_tool, plotter_tool],
    weights={'execution': 0.8, 'format': 0.2}
)

# Math problem evaluation
math_reward = MathReward(
    grading_mode='symbolic',
    timeout=10.0
)
```

## 📁 Architecture

The package is organized into several focused modules:

```
rewards/
├── core/                   # Base classes and interfaces
│   ├── base_reward.py     # Abstract base reward class
│   ├── interfaces.py      # Protocols and abstract classes
│   └── exceptions.py      # Custom exception hierarchy
├── components/            # Individual reward components
│   ├── reward_components.py  # Main components container
│   ├── connectivity_component.py
│   ├── validation_component.py
│   └── tool_execution_component.py
├── evaluators/           # Specialized evaluation logic
│   ├── math_evaluator.py
│   ├── code_evaluator.py
│   ├── tool_evaluator.py
│   └── power_system_evaluator.py
├── implementations/      # Main reward implementations
│   ├── power_system_reward.py
│   ├── tool_reward.py
│   ├── math_reward.py
│   └── complete_reward.py
├── math/                # Mathematical evaluation
│   ├── grader.py        # Core grading logic
│   ├── evaluators.py    # Symbolic/numerical evaluators
│   └── normalizers.py   # Answer normalization
└── utils/               # Utility functions
    ├── helpers.py       # Helper functions
    ├── formatters.py    # Result formatting
    └── validators.py    # Input validation
```

## 💡 Usage Examples

### Power System Evaluation

```python
from rewards import PowerSystemReward
from rewards.components import RewardComponents

# Create evaluator with custom weights
evaluator = PowerSystemReward(
    weights={
        'load_satisfaction': 1.0,      # Primary objective
        'connectivity': 0.5,           # Network connectivity
        'validation': 0.3,             # Basic validation
        'frequency_coherence': 0.2,    # Frequency consistency
        'voltage_coherence': 0.2,      # Voltage compatibility
    }
)

# Evaluate system designs
results = evaluator.evaluate(
    completions=agent_responses,
    init_code=initialization_code
)

# Get detailed component breakdown
components = evaluator.get_last_reward_components()
for i, comp in enumerate(components):
    print(f"Sample {i}:")
    print(f"  Total Score: {comp.get_total_reward():.3f}")
    print(f"  Load Satisfaction: {comp.load_satisfaction_reward:.3f}")
    print(f"  Connectivity: {comp.connectivity_reward:.3f}")
    print(f"  Validation: {comp.validation_reward:.3f}")
```

### Tool Usage Evaluation

```python
from rewards import ToolReward

# Define your tools
def calculator_tool(expression):
    """Calculate mathematical expressions."""
    try:
        return eval(expression)  # Use safe_eval in production!
    except:
        return "Error: Invalid expression"

def file_tool(operation, filename, content=None):
    """File operations tool."""
    if operation == "write":
        with open(filename, 'w') as f:
            f.write(content)
        return f"File {filename} written successfully"
    elif operation == "read":
        with open(filename, 'r') as f:
            return f.read()

# Create evaluator
tool_evaluator = ToolReward(
    tools=[calculator_tool, file_tool],
    weights={
        'execution': 0.7,     # Tool execution success
        'format': 0.2,        # Proper formatting
        'xml': 0.1           # XML structure
    }
)

# Evaluate agent tool usage
scores = tool_evaluator.evaluate(conversation_trajectories)
```

### Mathematical Problem Evaluation

```python
from rewards import MathReward

# Create math evaluator
math_evaluator = MathReward(
    grading_mode='symbolic',  # or 'numerical'
    timeout=10.0,
    normalize_answers=True
)

# Evaluate mathematical reasoning
math_scores = math_evaluator.evaluate(
    completions=math_responses,
    answers=ground_truth_answers,
    task=['math'] * len(math_responses)
)

# Check specific math components
for i, score in enumerate(math_scores):
    print(f"Problem {i}: {score:.3f}")
```

### Custom Reward Components

```python
from rewards.core.interfaces import RewardComponent, RewardType, RewardResult
from rewards.core.base_reward import BaseReward

class CustomReward(RewardComponent):
    """Custom reward component example."""
    
    @property
    def reward_type(self) -> RewardType:
        return RewardType.CUSTOM  # Define your own type
    
    def calculate(self, context: Dict[str, Any]) -> RewardResult:
        # Your custom evaluation logic
        score = self._evaluate_custom_criteria(context)
        
        return RewardResult(
            score=score,
            max_score=1.0,
            details={'custom_metric': score}
        )
    
    def _evaluate_custom_criteria(self, context):
        # Implement your evaluation logic
        return 0.85

# Use in a larger reward system
class MyRewardSystem(BaseReward):
    def _initialize_reward_system(self):
        self.add_reward_function(
            CustomReward().calculate, 
            weight=0.5
        )
```

## 🔧 Configuration

### Weight Configuration

```python
# Define custom weights for different scenarios
research_weights = {
    'load_satisfaction': 1.0,
    'connectivity': 0.8,
    'validation': 0.5,
    'structure': 0.3
}

industrial_weights = {
    'load_satisfaction': 1.0,
    'validation': 0.9,
    'frequency_coherence': 0.7,
    'voltage_coherence': 0.7,
    'connectivity': 0.5
}

# Apply weights
evaluator = PowerSystemReward(weights=research_weights)
# or update at runtime
evaluator.update_weights(industrial_weights)
```

### Component Configuration

```python
from rewards.components import RewardComponents

# Configure default behavior
components = RewardComponents()
components.update_weights({
    'tool_execution': 0.8,
    'format': 0.15,
    'xml': 0.05
})

# Access individual components
print(f"Tool execution weight: {components.get_component('tool_execution')}")
```

## 🔄 Migration from Legacy Code

### Old API → New API

```python
# OLD (deprecated but still works)
from rewards.power_system_reward import PowerSystemReward
from rewards.complete_reward import CompleteReward

# NEW (recommended)
from rewards import PowerSystemReward, ToolReward

# OLD method names
reward.update_reward_weights(new_weights)
reward.get_reward_weights()

# NEW method names
reward.update_weights(new_weights)
reward.get_weights()
```

### Automatic Migration Warnings

The package provides helpful deprecation warnings:

```python
# This will work but show a warning
from rewards.complete_reward import CompleteReward
# DeprecationWarning: CompleteReward is deprecated. 
# Use PowerSystemReward instead.
```

### Migration Script

```python
# Use this helper to migrate existing code
from rewards.legacy_compatibility import print_migration_guide

print_migration_guide()  # Shows detailed migration instructions
```

## 📊 Performance

### Benchmarks

| Component | Evaluation Time | Memory Usage |
|-----------|----------------|--------------|
| Power System | ~50ms | ~10MB |
| Tool Execution | ~20ms | ~5MB |
| Math Grading | ~100ms | ~15MB |

### Optimization Features

- **Lazy Loading**: Components loaded only when needed
- **Intelligent Caching**: Expensive operations cached automatically
- **Parallel Evaluation**: Multiple completions evaluated concurrently
- **Early Termination**: Skip expensive calculations for zero-weight components

```python
# Enable performance monitoring
import logging
logging.getLogger('rewards').setLevel(logging.DEBUG)

# Use caching for repeated evaluations
evaluator = PowerSystemReward(enable_caching=True)
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest rewards/tests/

# Run specific test modules
python -m pytest rewards/tests/test_power_system.py
python -m pytest rewards/tests/test_tool_evaluation.py

# Run with coverage
python -m pytest rewards/tests/ --cov=rewards --cov-report=html
```

### Writing Tests

```python
import pytest
from rewards import PowerSystemReward

def test_power_system_evaluation():
    evaluator = PowerSystemReward()
    
    # Mock data
    completions = [[{'role': 'assistant', 'content': 'test'}]]
    init_code = ["system_dict = {}"]
    
    scores = evaluator.evaluate(completions, init_code)
    
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup development environment
git clone <repository-url>
cd rewards
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest
```

### Code Style

- Follow [PEP 8](https://pep8.org/) guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Add type hints for all public APIs
- Write comprehensive docstrings

### Adding New Components

1. Create new component in `rewards/components/`
2. Implement the `RewardComponent` interface
3. Add appropriate tests
4. Update documentation

Example:
```python
# rewards/components/my_component.py
from rewards.core.interfaces import RewardComponent, RewardType, RewardResult

class MyComponent(RewardComponent):
    @property
    def reward_type(self) -> RewardType:
        return RewardType.CUSTOM
    
    def calculate(self, context: Dict[str, Any]) -> RewardResult:
        # Implementation here
        pass
```

## 📚 API Reference

### Core Classes

- **`BaseReward`**: Abstract base class for all reward implementations
- **`RewardComponent`**: Interface for individual reward components
- **`RewardResult`**: Container for evaluation results
- **`RewardConfiguration`**: Configuration management protocol

### Main Implementations

- **`PowerSystemReward`**: Comprehensive power system evaluation
- **`ToolReward`**: Tool usage and execution evaluation
- **`MathReward`**: Mathematical problem solving evaluation

### Utility Classes

- **`RewardComponents`**: Container for component scores
- **`MathGrader`**: Mathematical answer grading
- **`RewardFormatter`**: Result formatting utilities

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ImportError` when importing old modules
```python
# Solution: Use new import structure
from rewards import PowerSystemReward  # ✅ Correct
# not: from rewards.power_system_reward import PowerSystemReward  # ❌ Old
```

**Issue**: Deprecation warnings in console
```python
# Solution: Update to new API as shown in warnings
# Warnings provide specific guidance for migration
```

**Issue**: Performance degradation
```python
# Solution: Enable caching and check weights
evaluator = PowerSystemReward(
    weights={k: v for k, v in weights.items() if v > 0},  # Only non-zero
    enable_caching=True
)
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed evaluation steps
evaluator = PowerSystemReward()
scores = evaluator.evaluate(completions)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on concepts from the verifiers project
- Mathematical evaluation powered by SymPy and math_verify
- Power system analysis inspired by Pandapower
- Community contributions and feedback

---

For more detailed documentation, see the individual module README files and API documentation. 