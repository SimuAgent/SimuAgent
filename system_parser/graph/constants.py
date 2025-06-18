"""
Constants for power system component categorization.
"""

from typing import Set

# Block type categorization for connectivity analysis
GENERATOR_TYPES: Set[str] = {
    "Three-Phase Source",  # External grid or generator
    "Synchronous Machine",  # Sync generator/motor
    "AC Voltage Source",   # AC source
}

LOAD_TYPES: Set[str] = {
    "Three-Phase Parallel RLC Load",
    "Three-Phase Series RLC Load",
}

TRANSMISSION_TYPES: Set[str] = {
    "Three-Phase PI Section Line",
    "Single-Phase Transmission Line",
    "Three-Phase Transformer (Two Windings)",
    "Three-Phase Transformer (Three Windings)",
    "Single-Phase Transformer",
}

BUS_TYPES: Set[str] = {
    "Three-Phase V-I Measurement",  # Acts as a bus/node
}

# Edge types for graph construction
EDGE_TYPE_HAS_PORT = 'has_port'
EDGE_TYPE_CONNECTS_TO = 'connects_to'

# Validation constants
MAX_ERROR_ITEMS = 5
MAX_WARNING_ITEMS = 5 