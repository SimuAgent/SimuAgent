"""
Example usage of the new PowerSystemReward class.

This script demonstrates how to:
1. Initialize the PowerSystemReward with custom weights (extends ToolReward)
2. Process system evaluations with all ToolReward functionality
3. Access individual reward components
4. Modify reward weights dynamically
"""

from rewards.power_system_reward import PowerSystemReward, RewardComponents

def example_usage():
    """Demonstrate the usage of the PowerSystemReward class."""
    
    # 1. Initialize with custom power system weights (optional)
    # This still includes all ToolReward functionality automatically
    custom_power_weights = {
        'connectivity': 0.30,     # Emphasize connectivity more
        'validation': 0.10,       # Less emphasis on basic validation
        'parameter': 0.15,        # More emphasis on parameters
        'conversion': 0.20,       # Standard conversion weight
        'diagnostic': 0.20,       # Standard diagnostic weight
        'load_satisfaction': 0.05, # Keep load satisfaction weight
        'structure': 0.00         # Disable structure component
    }
    
    reward_system = PowerSystemReward(power_system_weights=custom_power_weights)
    
    # 2. Show that this includes all ToolReward functionality
    print("ToolReward functionality included:")
    print(f"  Number of reward functions: {len(reward_system.reward_funcs)}")
    print(f"  Reward function names: {[func.__name__ for func in reward_system.reward_funcs]}")
    print(f"  Reward weights: {reward_system.reward_weights}")
    
    print("\nPower system specific weights:")
    weights = reward_system.get_power_system_weights()
    for component, weight in weights.items():
        print(f"  {component}: {weight:.2f}")
    
    # 3. After running reward calculation, you can access detailed components
    print("\n" + "="*50)
    print("After processing samples (example output):")
    print("="*50)
    
    # Simulated reward components for demonstration
    example_components = [
        RewardComponents(
            connectivity_reward=0.85,
            validation_reward=0.90,
            parameter_reward=0.75,
            conversion_reward=0.80,
            diagnostic_reward=0.70,
            load_satisfaction_reward=0.85,
            structure_reward=0.90
        ),
        RewardComponents(
            connectivity_reward=0.60,
            validation_reward=0.80,
            parameter_reward=0.50,
            conversion_reward=0.65,
            diagnostic_reward=0.40,
            load_satisfaction_reward=0.60,
            structure_reward=0.70
        )
    ]
    
    # Simulate storing the components
    reward_system._last_reward_components = example_components
    
    # 4. Access detailed reward breakdown
    components_summary = reward_system.get_component_rewards_summary()
    if components_summary:
        print("Power system reward components summary:")
        for component, values in components_summary.items():
            avg_value = sum(values) / len(values)
            print(f"  {component}: {values} (avg: {avg_value:.3f})")
    
    # 5. Dynamically update weights based on requirements
    print("\n" + "="*50)
    print("Updating weights for different scenarios:")
    print("="*50)
    
    # Scenario 1: Focus on electrical correctness
    electrical_focus_weights = {
        'diagnostic': 0.40,      # High emphasis on electrical validity
        'conversion': 0.25,      # High emphasis on conversion success
        'connectivity': 0.20,    # Moderate connectivity importance
        'load_satisfaction': 0.15  # Ensure loads are satisfied
    }
    reward_system.update_power_system_weights(electrical_focus_weights)
    print("Electrical focus weights:")
    for component, weight in reward_system.get_power_system_weights().items():
        if weight > 0:
            print(f"  {component}: {weight:.2f}")
    
    # Calculate total rewards with new weights
    total_rewards = [comp.get_total_reward(reward_system.get_power_system_weights()) 
                    for comp in example_components]
    print(f"Total rewards with electrical focus: {total_rewards}")
    
    # Scenario 2: Focus on system topology
    topology_focus_weights = {
        'connectivity': 0.35,    # High emphasis on connectivity
        'structure': 0.25,       # High emphasis on structure
        'validation': 0.20,      # Moderate validation importance
        'conversion': 0.20       # Moderate conversion importance
    }
    reward_system.update_power_system_weights(topology_focus_weights)
    print(f"\nTopology focus weights:")
    for component, weight in reward_system.get_power_system_weights().items():
        if weight > 0:
            print(f"  {component}: {weight:.2f}")
    
    total_rewards = [comp.get_total_reward(reward_system.get_power_system_weights()) 
                    for comp in example_components]
    print(f"Total rewards with topology focus: {total_rewards}")
    
    # 6. Show individual component analysis
    print("\n" + "="*50)
    print("Individual component analysis:")
    print("="*50)
    
    detailed_components = reward_system.get_last_reward_components()
    if detailed_components:
        for i, comp in enumerate(detailed_components):
            print(f"\nSample {i}:")
            print(f"  Connectivity: {comp.connectivity_reward:.3f}")
            print(f"  Validation: {comp.validation_reward:.3f}")
            print(f"  Parameters: {comp.parameter_reward:.3f}")
            print(f"  Conversion: {comp.conversion_reward:.3f}")
            print(f"  Diagnostics: {comp.diagnostic_reward:.3f}")
            print(f"  Load Satisfaction: {comp.load_satisfaction_reward:.3f}")
            print(f"  Structure: {comp.structure_reward:.3f}")
            print(f"  Total (current weights): {comp.get_total_reward(reward_system.get_power_system_weights()):.3f}")


def tool_reward_integration_example():
    """Demonstrate how PowerSystemReward integrates with ToolReward functionality."""
    
    print("\n" + "="*60)
    print("TOOL REWARD INTEGRATION EXAMPLE")
    print("="*60)
    
    # PowerSystemReward automatically includes all ToolReward functionality
    reward_system = PowerSystemReward()
    
    print("Inherited ToolReward functionality:")
    print(f"  Parser: {type(reward_system.parser).__name__}")
    print(f"  Environment Parser: {type(reward_system.env_parser).__name__}")
    print(f"  Tools: {list(reward_system.tools.keys())}")
    print(f"  Total reward functions: {len(reward_system.reward_funcs)}")
    
    # Show the reward functions inherited from ToolReward
    print("\nReward functions from ToolReward:")
    for i, func in enumerate(reward_system.reward_funcs):
        weight = reward_system.reward_weights[i] if i < len(reward_system.reward_weights) else 0.0
        print(f"  {i}: {func.__name__} (weight: {weight:.2f})")
    
    # The correct_answer_reward_func is now our sophisticated power system evaluation
    print(f"\nThe correct_answer_reward_func has been replaced with sophisticated power system evaluation")
    print(f"All other ToolReward functions (tool_execution, formatting, XML validation) are preserved")


def component_based_training_example():
    """Example of how to use component-based rewards for focused training."""
    
    print("\n" + "="*60)
    print("COMPONENT-BASED TRAINING EXAMPLE")
    print("="*60)
    
    # Different training phases with different reward emphasis
    training_phases = {
        "Phase 1 - Basic Structure": {
            'structure': 0.40,
            'validation': 0.30,
            'connectivity': 0.30
        },
        "Phase 2 - Connectivity Focus": {
            'connectivity': 0.50,
            'structure': 0.20,
            'conversion': 0.20,
            'validation': 0.10
        },
        "Phase 3 - Electrical Correctness": {
            'diagnostic': 0.35,
            'conversion': 0.25,
            'load_satisfaction': 0.20,
            'connectivity': 0.20
        },
        "Phase 4 - Parameter Precision": {
            'parameter': 0.30,
            'diagnostic': 0.25,
            'conversion': 0.20,
            'connectivity': 0.15,
            'validation': 0.10
        }
    }
    
    reward_system = PowerSystemReward()
    
    for phase_name, weights in training_phases.items():
        print(f"\n{phase_name}:")
        reward_system.update_power_system_weights(weights)
        active_weights = reward_system.get_power_system_weights()
        
        for component, weight in active_weights.items():
            if weight > 0:
                print(f"  {component}: {weight:.2f}")
        
        print(f"  Focus: {', '.join([k for k, v in weights.items() if v >= 0.25])}")


if __name__ == "__main__":
    example_usage()
    tool_reward_integration_example()
    component_based_training_example() 