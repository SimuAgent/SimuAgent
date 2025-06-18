# System Parser Package

**A comprehensive, object-oriented library for parsing, validating, and analyzing power system configurations.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NetworkX](https://img.shields.io/badge/powered%20by-NetworkX-blue)](https://networkx.org/)

## 🌟 Overview

The System Parser package provides a powerful, modular framework for working with power system configurations. It transforms complex system definitions into validated, analyzable graph representations with comprehensive error checking and performance optimization.

### Key Features

- 🔧 **Modular Architecture**: Clean separation between parsing, validation, and analysis
- 📊 **Comprehensive Analysis**: Connectivity, electrical parameters, and topology analysis
- ✅ **Robust Validation**: Multi-layer validation with detailed error reporting
- 🎯 **Extensible Design**: Plugin architecture for custom validators and analyzers
- 🚀 **High Performance**: Optimized algorithms with intelligent caching
- 📈 **Rich Visualization**: Advanced graph rendering and layout management
- 🔄 **Pandapower Integration**: Seamless conversion to Pandapower format
- 📚 **Well Documented**: Comprehensive guides and examples

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install networkx graphviz pandas numpy
pip install -e .  # Install the package
```

### Basic Usage

```python
from system_parser import SystemGraph, GraphBuilder, SystemValidator

# Load system configuration
system_dict = {
    "blocks": {
        "gen1": {"type": "Three-Phase Source", "params": {...}},
        "load1": {"type": "Three-Phase Parallel RLC Load", "params": {...}},
        "line1": {"type": "Three-Phase PI Section Line", "params": {...}}
    },
    "connections": [
        {"from": "gen1/P1", "to": "line1/P1"},
        {"from": "line1/P2", "to": "load1/P1"}
    ]
}

# Create and validate system
builder = GraphBuilder(config_file="config/block_config.json")
system = builder.build_from_dict(system_dict)

# Validate system
validator = SystemValidator()
result = validator.validate(system)

if result.is_valid:
    print("✅ System is valid!")
    
    # Analyze connectivity
    connectivity = system.analyze_connectivity()
    print(f"Load satisfaction: {connectivity.connectivity_ratio:.2%}")
    
    # Get validation report
    print(system.get_validation_report())
else:
    print(f"❌ Validation failed: {result.summary}")
```

## 📁 Architecture

The package is organized into focused, single-responsibility modules:

```
system_parser/
├── core/                      # Base models and interfaces
│   ├── models.py             # Block, Port, Connection models
│   ├── interfaces.py         # Abstract interfaces
│   └── exceptions.py         # Custom exception hierarchy
├── graph/                    # Graph construction and analysis
│   ├── system_graph.py       # Main SystemGraph class
│   ├── builder.py            # Graph construction logic
│   ├── analyzer.py           # Graph analysis tools
│   └── connectivity.py       # Connectivity analysis
├── validation/               # Validation framework
│   ├── validator.py          # Main validation orchestrator
│   ├── rules.py              # Validation rule definitions
│   └── results.py            # Validation result structures
├── analysis/                 # Analysis components
│   ├── connectivity.py       # Generator-to-load analysis
│   ├── electrical.py         # Electrical parameter analysis
│   └── topology.py           # Network topology analysis
├── config/                   # Configuration management
│   ├── block_config.py       # Block type configurations
│   ├── validation_config.py  # Validation parameters
│   └── loader.py             # Configuration loading
├── validators/               # Specialized validators
│   ├── connection_validator.py
│   ├── parameter_validator.py
│   ├── distance_validator.py
│   └── electrical_validator.py
├── visualization/            # Graph rendering
│   ├── graph_renderer.py     # Graph visualization
│   └── layout_manager.py     # Layout algorithms
└── pandapower/              # Pandapower integration
    ├── converter.py          # Format conversion
    └── exporter.py           # Data export utilities
```

## 💡 Usage Examples

### System Construction

```python
from system_parser import GraphBuilder
from system_parser.config import BlockConfiguration

# Load configuration
config = BlockConfiguration("config/block_config.json")

# Create builder with custom validation
builder = GraphBuilder(
    block_config=config,
    validate_parameters=True,
    validate_distances=True
)

# Build system from dictionary
system = builder.build_from_dict(system_dict)

# Or build incrementally
builder.add_block("generator", "Three-Phase Source", {"voltage": 11000})
builder.add_block("load", "Three-Phase Parallel RLC Load", {"power": 1000})
builder.add_connection("generator/P1", "load/P1")
system = builder.build()
```

### Comprehensive Validation

```python
from system_parser.validation import SystemValidator, ValidationRuleSet

# Create custom validation rules
rules = ValidationRuleSet()
rules.add_rule("frequency_coherence", weight=0.8)
rules.add_rule("voltage_compatibility", weight=0.9)
rules.add_rule("parameter_validation", weight=1.0)

# Validate with custom rules
validator = SystemValidator(rules=rules)
result = validator.validate(system)

# Get detailed validation report
if not result.is_valid:
    print("Validation Issues:")
    for issue in result.issues:
        print(f"  {issue.severity}: {issue.message}")
        print(f"    Component: {issue.component}")
        print(f"    Suggestion: {issue.suggestion}")
```

### Connectivity Analysis

```python
from system_parser.analysis import ConnectivityAnalyzer

# Analyze generator-to-load connectivity
analyzer = ConnectivityAnalyzer(system)
connectivity = analyzer.analyze()

print(f"Total generators: {connectivity.total_generators}")
print(f"Total loads: {connectivity.total_loads}")
print(f"Connected loads: {connectivity.connected_loads}")
print(f"Connectivity ratio: {connectivity.connectivity_ratio:.2%}")

# Get paths from generators to loads
for load, generators in connectivity.paths_found.items():
    print(f"Load {load} can be reached from: {', '.join(generators)}")

# Identify isolated components
for load in connectivity.isolated_load_names:
    print(f"⚠️  Load {load} is isolated!")
```

### Electrical Analysis

```python
from system_parser.analysis import ElectricalAnalyzer

# Analyze electrical parameters
electrical = ElectricalAnalyzer(system)

# Check frequency coherence
freq_result = electrical.analyze_frequency_coherence()
if freq_result.is_coherent:
    print(f"✅ All components operate at {freq_result.common_frequency} Hz")
else:
    print("❌ Frequency mismatch detected:")
    for component, freq in freq_result.component_frequencies.items():
        print(f"  {component}: {freq} Hz")

# Check voltage compatibility
voltage_result = electrical.analyze_voltage_compatibility()
for issue in voltage_result.compatibility_issues:
    print(f"⚠️  Voltage mismatch: {issue.component1} ({issue.voltage1}V) "
          f"↔️ {issue.component2} ({issue.voltage2}V)")
```

### Graph Visualization

```python
from system_parser.visualization import GraphRenderer, LayoutManager

# Create visualization
renderer = GraphRenderer(system)

# Render with different layouts
layout_manager = LayoutManager()

# Hierarchical layout for power systems
svg_content = renderer.render(
    layout=layout_manager.hierarchical_layout(),
    highlight_issues=True,
    show_parameters=True
)

# Save visualization
with open("system_graph.svg", "w") as f:
    f.write(svg_content)

# Interactive visualization
renderer.render_interactive(
    output_file="system_interactive.html",
    physics_enabled=True
)
```

### Pandapower Integration

```python
from system_parser.pandapower import PandapowerConverter

# Convert to Pandapower format
converter = PandapowerConverter()
pp_net = converter.convert(system)

if converter.is_valid():
    print("✅ Conversion successful!")
    
    # Run power flow analysis
    import pandapower as pp
    pp.runpp(pp_net)
    
    # Access results
    print("Bus voltages:")
    print(pp_net.res_bus.vm_pu)
    
    print("Line loadings:")
    print(pp_net.res_line.loading_percent)
else:
    print("❌ Conversion failed:")
    for error in converter.get_errors():
        print(f"  {error}")
```

### Custom Validators

```python
from system_parser.validators import BaseValidator
from system_parser.core.interfaces import ValidationResult

class CustomValidator(BaseValidator):
    """Custom validation logic example."""
    
    def validate_block(self, block):
        """Validate individual block."""
        issues = []
        
        # Custom validation logic
        if block.type == "Three-Phase Source":
            voltage = block.parameters.get("voltage", 0)
            if voltage < 1000:
                issues.append(f"Low voltage source: {voltage}V")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            component=block.name
        )
    
    def validate_connection(self, connection):
        """Validate connections."""
        # Custom connection validation
        return ValidationResult(is_valid=True)

# Use custom validator
from system_parser.validation import SystemValidator

validator = SystemValidator()
validator.add_validator(CustomValidator())
result = validator.validate(system)
```

## 🔧 Configuration

### Block Configuration

```python
from system_parser.config import BlockConfiguration

# Load block type definitions
config = BlockConfiguration("config/block_config.json")

# Query block information
block_info = config.get_block_info("Three-Phase Source")
print(f"Required ports: {block_info.required_ports}")
print(f"Optional parameters: {block_info.optional_parameters}")

# Validate parameters
is_valid = config.validate_parameters("Three-Phase Source", {
    "voltage": 11000,
    "frequency": 50
})
```

### Validation Configuration

```python
from system_parser.config import ValidationConfiguration

# Configure validation parameters
val_config = ValidationConfiguration({
    "frequency_tolerance": 0.1,  # ±0.1 Hz
    "voltage_tolerance": 0.05,   # ±5%
    "distance_limits": {
        "transmission_line": 100000,  # 100 km max
        "distribution_line": 50000    # 50 km max
    }
})

# Use in validation
validator = SystemValidator(config=val_config)
```

## 📊 Performance

### Benchmarks

| Operation | Small System (10 blocks) | Large System (1000 blocks) |
|-----------|-------------------------|----------------------------|
| Graph Construction | ~5ms | ~200ms |
| Validation | ~10ms | ~500ms |
| Connectivity Analysis | ~2ms | ~100ms |
| Visualization | ~50ms | ~2s |

### Optimization Features

- **Intelligent Caching**: Expensive operations cached automatically
- **Lazy Evaluation**: Components loaded only when needed
- **Parallel Processing**: Multiple validation rules run concurrently
- **Memory Optimization**: Efficient graph representation

```python
# Enable performance monitoring
import logging
logging.getLogger('system_parser').setLevel(logging.DEBUG)

# Use caching for repeated operations
system = SystemGraph(
    system_dict,
    enable_caching=True,
    cache_size=1000
)
```

## 🔄 Migration from Legacy Code

### Old API → New API

```python
# OLD (monolithic approach)
from system_parser.system_graph import SystemGraph

system = SystemGraph(system_dict, block_config_instance=config)
if system.validation_errors:
    print("Errors found")

# NEW (modular approach)
from system_parser import GraphBuilder, SystemValidator

builder = GraphBuilder(config)
system = builder.build_from_dict(system_dict)

validator = SystemValidator()
result = validator.validate(system)
if not result.is_valid:
    print("Validation failed")
```

### Automatic Migration

The package provides backward compatibility:

```python
# This still works but shows deprecation warnings
from system_parser.system_graph import SystemGraph as PowerSystemGraph
# DeprecationWarning: Use system_parser.SystemGraph instead
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest system_parser/tests/

# Run specific test categories
python -m pytest system_parser/tests/test_validation.py
python -m pytest system_parser/tests/test_connectivity.py
python -m pytest system_parser/tests/test_pandapower.py

# Run with coverage
python -m pytest system_parser/tests/ --cov=system_parser --cov-report=html
```

### Writing Tests

```python
import pytest
from system_parser import GraphBuilder, SystemValidator

def test_system_validation():
    # Create test system
    system_dict = {
        "blocks": {
            "gen": {"type": "Three-Phase Source", "params": {"voltage": 11000}},
            "load": {"type": "Three-Phase Parallel RLC Load", "params": {"power": 1000}}
        },
        "connections": [{"from": "gen/P1", "to": "load/P1"}]
    }
    
    # Build and validate
    builder = GraphBuilder()
    system = builder.build_from_dict(system_dict)
    
    validator = SystemValidator()
    result = validator.validate(system)
    
    assert result.is_valid
    assert len(system.blocks) == 2
    assert len(system.connections) == 1
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd system_parser
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest
```

### Adding New Validators

1. Create validator in `system_parser/validators/`
2. Inherit from `BaseValidator`
3. Implement validation methods
4. Add tests and documentation

Example:
```python
# system_parser/validators/my_validator.py
from .base_validator import BaseValidator
from ..core.interfaces import ValidationResult

class MyValidator(BaseValidator):
    def validate_block(self, block):
        # Validation logic here
        return ValidationResult(is_valid=True)
```

### Adding New Analyzers

```python
# system_parser/analysis/my_analyzer.py
from ..core.interfaces import SystemAnalyzer

class MyAnalyzer(SystemAnalyzer):
    def analyze(self, system):
        # Analysis logic here
        return analysis_result
```

## 📚 API Reference

### Core Classes

- **`SystemGraph`**: Main graph representation of the power system
- **`GraphBuilder`**: Constructs SystemGraph from various input formats
- **`SystemValidator`**: Orchestrates validation across multiple validators
- **`Block`**: Represents individual system components
- **`Connection`**: Represents connections between blocks

### Analysis Classes

- **`ConnectivityAnalyzer`**: Analyzes generator-to-load connectivity
- **`ElectricalAnalyzer`**: Validates electrical parameters
- **`TopologyAnalyzer`**: Analyzes network topology

### Configuration Classes

- **`BlockConfiguration`**: Manages block type definitions
- **`ValidationConfiguration`**: Manages validation parameters

## 🐛 Troubleshooting

### Common Issues

**Issue**: Block type not found
```python
# Solution: Check block configuration file
config = BlockConfiguration("config/block_config.json")
available_types = config.get_available_types()
print(f"Available types: {available_types}")
```

**Issue**: Validation fails with unclear errors
```python
# Solution: Enable detailed logging
import logging
logging.getLogger('system_parser.validation').setLevel(logging.DEBUG)

validator = SystemValidator()
result = validator.validate(system)
```

**Issue**: Poor performance on large systems
```python
# Solution: Enable optimizations
system = SystemGraph(
    system_dict,
    enable_caching=True,
    lazy_validation=True
)
```

### Debug Mode

```python
# Enable comprehensive debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# This shows detailed construction and validation steps
builder = GraphBuilder(debug=True)
system = builder.build_from_dict(system_dict)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Powered by NetworkX for graph operations
- Visualization support from Graphviz
- Integration with Pandapower for power flow analysis
- Community contributions and feedback

---

For more detailed documentation, see the API documentation and individual module guides. 