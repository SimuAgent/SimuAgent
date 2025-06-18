# SPS (Specialized Power Systems) - Pandapower Integration

This folder contains components for integrating with pandapower, a Python library for power system analysis.

## Contents

- `pandapower_converter.py` - Main converter class that transforms SystemGraph objects into pandapower network representations
- `simulink_to_pandapower_map.json` - Configuration file that maps Simulink block types to pandapower elements
- `__init__.py` - Module initialization file that exports the PandapowerConverter class

## Purpose

The SPS (Specialized Power Systems) integration allows the system_parser to:

1. Convert parsed power system models into pandapower format
2. Run power system analysis and diagnostics
3. Generate network plots and visualizations
4. Calculate diagnostic and conversion rewards for system validation

## Usage

```python
from system_parser.pandapower import PandapowerConverter

# Create converter with a SystemGraph object
converter = PandapowerConverter(system_graph)

# Convert to pandapower network
pandapower_net, diagnostic_reward, conversion_reward = converter.convert_to_pandapower_net()

# Generate network plot
converter.create_network_plot("output_path.png")
```

## Dependencies

- pandapower
- networkx
- matplotlib (for plotting)
- json
- logging 