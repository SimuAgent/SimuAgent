# Environments Package

**A robust, flexible environment management system for AI agent simulations and experiments.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pydantic](https://img.shields.io/badge/powered%20by-Pydantic-blue)](https://pydantic-docs.helpmanual.io/)

## 🌟 Overview

The Environments package provides a comprehensive framework for managing AI agent simulation environments. It offers standardized interfaces for environment creation, configuration, execution, and monitoring with support for multiple environment types and execution modes.

### Key Features

- 🏗️ **Modular Architecture**: Clean separation between environment types and execution logic
- 🔧 **Flexible Configuration**: YAML/JSON configuration with schema validation
- 🎯 **Multi-Environment Support**: Simultaneous management of different environment types
- 📊 **Comprehensive Monitoring**: Built-in logging, metrics, and performance tracking
- 🚀 **High Performance**: Optimized for both single and multi-agent scenarios
- 🔄 **Plugin Architecture**: Easy extension with custom environment implementations
- 📚 **Well Documented**: Complete API documentation and usage examples
- ✅ **Type Safety**: Full type annotations with runtime validation

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install pydantic pyyaml numpy pandas
pip install -e .  # Install the package
```

### Basic Usage

```python
from envs import EnvironmentManager, EnvironmentConfig

# Create environment configuration
config = EnvironmentConfig(
    name="power_system_sim",
    type="power_system",
    parameters={
        "max_steps": 1000,
        "reward_weights": {
            "efficiency": 0.7,
            "stability": 0.3
        }
    }
)

# Initialize environment manager
manager = EnvironmentManager()
env = manager.create_environment(config)

# Run simulation
initial_state = env.reset()
for step in range(100):
    action = agent.get_action(initial_state)
    state, reward, done, info = env.step(action)
    
    if done:
        break

# Get performance metrics
metrics = env.get_metrics()
print(f"Total reward: {metrics.total_reward:.3f}")
print(f"Success rate: {metrics.success_rate:.2%}")
```

## 📁 Architecture

The package is organized into focused modules:

```
envs/
├── core/                      # Core environment abstractions
│   ├── base_env.py           # Base environment interface
│   ├── manager.py            # Environment manager
│   ├── config.py             # Configuration schemas
│   └── exceptions.py         # Custom exceptions
├── types/                    # Specific environment implementations
│   ├── power_system/         # Power system environments
│   ├── code_execution/       # Code execution environments
│   └── multi_agent/          # Multi-agent environments
├── utils/                    # Utility functions
│   ├── logging.py            # Environment logging
│   ├── metrics.py            # Performance metrics
│   └── visualization.py      # Environment visualization
├── monitoring/               # Monitoring and debugging
│   ├── profiler.py           # Performance profiling
│   └── recorder.py           # Interaction recording
└── plugins/                  # Plugin system
    ├── registry.py           # Plugin registry
    └── loader.py             # Dynamic plugin loading
```

## 💡 Usage Examples

### Power System Environment

```python
from envs.types.power_system import PowerSystemEnvironment

# Configure power system environment
config = {
    "system_size": "medium",
    "max_load": 1000,
    "renewable_ratio": 0.3,
    "evaluation_metrics": ["stability", "efficiency"]
}

env = PowerSystemEnvironment(config)
state = env.reset()

# Take action
action = {
    "generator_dispatch": [0.8, 0.6, 0.9],
    "load_shedding": [0.0, 0.0, 0.1]
}

next_state, reward, done, info = env.step(action)
print(f"System stable: {info['system_stable']}")
```

### Multi-Agent Environment

```python
from envs.types.multi_agent import MultiAgentEnvironment

# Configure multi-agent environment
env = MultiAgentEnvironment(
    num_agents=3,
    communication_enabled=True,
    coordination_protocol="hierarchical"
)

# Multi-agent simulation
states = env.reset()
for step in range(100):
    actions = {agent_id: agent.get_action(states[agent_id]) 
               for agent_id in env.agent_ids}
    
    next_states, rewards, dones, infos = env.step(actions)
    
    if all(dones.values()):
        break
```

## 🔧 Configuration

### YAML Configuration

```yaml
environments:
  power_system:
    type: "power_system"
    parameters:
      system_size: "large"
      max_load: 2000
      renewable_ratio: 0.4
  
  code_execution:
    type: "code_execution"
    parameters:
      security_level: "moderate"
      timeout: 30.0
      memory_limit: "256MB"
```

## 📊 Monitoring

```python
from envs.monitoring import EnvironmentMonitor

monitor = EnvironmentMonitor(env)
monitor.start()

# Run simulation with monitoring
for episode in range(100):
    state = env.reset()
    episode_reward = 0
    
    for step in range(1000):
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break

# Get performance report
report = monitor.get_report()
print(f"Average reward: {report.avg_reward:.3f}")
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest envs/tests/

# Run with coverage
python -m pytest envs/tests/ --cov=envs --cov-report=html
```

## 📚 API Reference

### Core Classes

- **`BaseEnvironment`**: Abstract base class for all environments
- **`EnvironmentManager`**: Factory for creating and managing environments
- **`EnvironmentConfig`**: Configuration schema for environments

### Environment Types

- **`PowerSystemEnvironment`**: Power system simulation
- **`CodeExecutionEnvironment`**: Secure code execution
- **`MultiAgentEnvironment`**: Multi-agent coordination

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

For more detailed documentation, see the API documentation and individual environment guides. 