from typing import List, Dict, Any, Callable
import wandb

from utils.xml_parser import XMLParser
from rewards.reward_helpers import dicts_equal_ignoring_list_order, extract_python_code
from rewards.base_reward import BaseReward
from rewards.math_grader import grade


class SimuAgentReward(BaseReward):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"]),
                 tools: List[Callable] = []):
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {tool.__name__: tool for tool in tools}
        self.reward_funcs = [
            self.correctness_reward_func,
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ]
        self.reward_weights = [
            1.0,
            0.5,
            0.25,
            0.25,
        ]
        for tool_name in self.tools.keys():
            self.reward_funcs.append(self.get_named_tool_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_count_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_attempt_reward_func(tool_name))
            self.reward_weights.append(0.0)
    
    def correctness_reward_func(self, completions, answer, init_code, **kwargs) -> list[float]:
        """Check if executing code produces correct result."""
        rewards = []
        # print('completion[0]:', completions[0])
        responses = [completion[-1]['content'] for completion in completions]
        gen_code_list = []
        ref_code_list = []
        
        for i, (response, correct_answer, each_init_code) in enumerate(zip(responses, answer, init_code)):
            try:
                # Extract codes to test
                generated_code = extract_python_code(response)
                gen_code_list.append(generated_code)
                
                reference_code = extract_python_code(correct_answer)
                ref_code_list.append(reference_code)
                
                if not generated_code:
                    rewards.append(0.0)
                    continue
                
                each_init_code = extract_python_code(each_init_code)
                
                # Create execution environments
                gen_globals = {}
                ref_globals = {}
                
                # First execute prompt code to set up system_dict
                if each_init_code:
                    try:
                        exec(each_init_code, gen_globals)
                        exec(each_init_code, ref_globals)
                    except Exception:
                        raise RuntimeError("Failed to execute initial code.")
                
                # Execute test codes
                exec(generated_code, gen_globals)
                exec(reference_code, ref_globals)
                
                # Get the result dictionaries
                gen_dict = gen_globals.get('system_dict', {})
                ref_dict = ref_globals.get('system_dict', {})
                
                # Compare results
                # is_correct = gen_dict == ref_dict
                is_correct = dicts_equal_ignoring_list_order(gen_dict, ref_dict)
                reward = 2.0 if is_correct else 0.6
                
                # Print result only for the first case
                if i == 0:
                    print(f"Result: {reward}")
                    
                rewards.append(reward)
                    
            except Exception:
                rewards.append(0.0)
        
        return rewards, gen_code_list, ref_code_list
    
    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                            tool_attempts += 1
                            # Check response with env_parser
                            parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                            if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                successful_executions += 1
            
            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return 0.2 * (successful_executions / tool_attempts)
        
        return [check_execution(c) for c in completions]

