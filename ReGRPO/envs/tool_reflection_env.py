from typing import List, Dict, Any, Callable, Optional, Sequence

from datasets import Dataset

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random
import time
import re

from ReGRPO.envs.tool_env import ToolEnv

from ReGRPO.rewards.reward_helpers import dicts_equal_ignoring_list_order, extract_python_code
from ReGRPO.rewards.math_grader import grade
import textwrap


def truncate_answer_text(
    answer_text: str,
    max_chars: int = 500,
    head_keep: int = 1,
    tail_keep: int = 3,
    ellipsis: str = "..."
) -> str:
    """
    Truncate long text while ensuring the most critical content at the end is preserved.

    Parameters
    ----------
    answer_text : str
        The original text to be truncated
    max_chars : int
        Maximum number of characters allowed
    head_keep : int
        Number of segments to keep from the beginning
    tail_keep : int
        Number of segments to keep from the end
    ellipsis : str
        Text to use as ellipsis (customizable)

    Returns
    -------
    str
        Truncated string. Returns the original text if it's already within the limit.
    """
    # ----- 1. Initial check -----
    if len(answer_text) <= max_chars:
        return answer_text

    # ----- 2. Split paragraphs by "empty lines" -----
    paragraphs: List[str] = re.split(r'\n\s*\n', answer_text.strip())

    # If there are very few paragraphs, directly use the "tail-first" truncation
    if len(paragraphs) <= head_keep + tail_keep:
        return _truncate_keep_tail(answer_text, max_chars, ellipsis)

    # ----- 3. First concatenate by paragraphs: head + ... + tail -----
    head_part = "\n\n".join(paragraphs[:head_keep])
    tail_part = "\n\n".join(paragraphs[-tail_keep:])
    candidate = f"{head_part}\n\n{ellipsis}\n\n{tail_part}"

    # ----- 4. Second fallback: still too long → compress head and if necessary compress tail start -----
    if len(candidate) > max_chars:
        return _truncate_keep_tail(candidate, max_chars, ellipsis)
    return candidate


def _truncate_keep_tail(text: str, max_chars: int, ellipsis: str) -> str:
    """
    Truncate text while prioritizing keeping the tail intact: first truncate the head,
    and only truncate the beginning of the tail if absolutely necessary.
    """
    if len(text) <= max_chars:
        return text

    # Tail without leading ellipsis
    # First try to keep the entire tail
    ellip_section = f"{ellipsis}\n\n"
    tail_max_len = max_chars - len(ellip_section)
    if tail_max_len <= 0:
        # max_chars is too small, can only hard truncate the end
        return text[-max_chars:]

    tail_part = text[-tail_max_len:]
    return ellip_section + tail_part


class ToolReflectionEnv(ToolEnv):
    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 tools: List[Callable] = [],
                 system_prompt: str | None = None,
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args={
                     "stop": ["</tool>\n", "</answer>\n"],
                     #"stop": [],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True,
                 max_steps: int = 10,
                 reward: Any = None, **kwargs):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            tools=tools,
            system_prompt=system_prompt,
            few_shot=few_shot,
            sampling_args=sampling_args,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            reward=reward,
            **kwargs
        )
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count > self.max_steps:
                return True
            
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    return {"role": "user", "content": self.env_parser.format(result=result)}
                else:
                    return {"role": "user", "content": "Error: Tool execution returned empty output."}
        except Exception:
            pass
        return {"role": "user", "content": "Error: Tool command not found or invalid XML format. Please ensure correct formatting."}
    
    def step(self,
             state: Dict[str, Any],
             llm: Any,
             sampling_params: Any,
             lora_request: Any) -> List[Dict[str, Any]]:
        """
        Update a single state for one step, following similar logic to the original method
        but processing only one state.
        """
        messages_to_step = state["messages"]
        # Since we only process one state, put messages in a list and take the first response from llm.chat
        llm_response = llm.chat([messages_to_step], sampling_params=sampling_params, use_tqdm=False, lora_request=lora_request)[0]

        # Randomly wait for a while to avoid rate limiting
        # time.sleep(self.sleep_time * random.random())  # TODO: Is this necessary?

        # Update state: if prompt token ids haven't been saved yet, record them
        if len(state["prompt_ids"]) == 0:
            state["prompt_ids"] = llm_response.prompt_token_ids
        # Append current llm output to messages
        state["messages"].append({
            "role": "assistant",
            "content": llm_response.outputs[0].text
        })

        # Calculate token count for environment response and newly completed parts in this call
        total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
        env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len  # type: ignore
        new_completion_len = len(llm_response.outputs[0].token_ids)

        # Update completion mask: use self.env_mask to mark environment response parts, mark others as 1
        state["completion_mask"].extend([self.env_mask] * env_response_len)
        state["completion_mask"].extend([1] * new_completion_len)

        # Update token id list: first put in llm's prompt tokens, then append newly generated tokens, then remove prompt part
        state["completion_ids"] = list(llm_response.prompt_token_ids)  # type: ignore
        state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
        state["completion_ids"] = state["completion_ids"][len(state["prompt_ids"]):]

        # Check if end conditions are met: e.g., state messages satisfy completion criteria or token count exceeds limit
        if self.is_completed(state["messages"]) or len(state["completion_ids"]) > sampling_params.max_tokens:
            state["completed"] = True
            state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
            state["completion_mask"] = state["completion_mask"][:len(state["completion_ids"])]
        else:
            # If not completed, automatically append environment response prompt (e.g., reconstruct next round input)
            state["messages"].append(self.env_response(state["messages"]))

        # Verify token length consistency
        if not len(state["completion_mask"]) == len(state["completion_ids"]):
            print(state["messages"])
            print(state["completion_mask"])
            print(state["completion_ids"])
            raise ValueError("Completion mask and completion ids are not the same length")
        return state
    
    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: Any,
                 sampling_params: Any,
                 answers: List[str],
                 init_codes: List[str],
                 tasks: List[str],
                 lora_request: Any,
                 mode: str,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        """
        Process each branch (prompt) sequentially. If a previous branch has reflection output,
        add that reflection result to the initial state of the next branch.
        """
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        breakpoint()

        final_states = []
        
        # if not is_eval_step:
        reflection_messages = [[] for _ in prompts]
        
        # Process each branch sequentially
        for idx, (prompt, ref_answer, task) in enumerate(zip(prompts, answers, tasks)):
            breakpoint()
            # If previous branch has reflection results, put them in current branch's initial messages
            if mode == "train" and idx > 0 and (reflection_messages[idx-1] and 'None' not in reflection_messages[idx-1][-1]["content"]):  # TODO: use better method to check if the last message has none
                # Here you can choose message role, like system or assistant, depending on actual situation
                
                assert reflection_messages[idx-1][-1]["role"] == "assistant"

                # reflection_initial_prompt = "Your previous answer to the following question conveyed the following lesson learned—please ensure you do not repeat the same mistake: "
                # reflection_initial_prompt = "Your previous response to the following question revealed an important lesson—apply it consistently to avoid repeat errors: "

                # prompt.insert(-1, {"content": reflection_initial_prompt + reflection_messages[idx-1][-1]["content"], "role": "user"})
                
                # reflection_initial_prompt = "Your previous response to the above question revealed an important lesson—apply it consistently to avoid repeat errors: "
                reflection_initial_prompt = "Your earlier answer underscored an important lesson—learn from it to avoid repeating mistakes: "

                prompt.append({"content": reflection_initial_prompt + reflection_messages[idx-1][-1]["content"], "role": "user"})
                
            # Initialize single branch state
            state = {
                "messages": prompt,
                "prompt_messages": len(prompt),
                "prompt_ids": [],
                "completed": False,
                "completion_ids": [],
                "completion_mask": []
            }

            # Serial generation: repeatedly call step until current state is complete
            while not state["completed"]:
                state = self.step(state, llm, custom_sp, lora_request)

            # Reflection phase: check current branch's answer. If reflect method returns reflection prompt,
            # call llm.chat to get reflection result, add it to state's messages, then continue reflection check,
            # until no more reflection is needed.
            
            # Concatenate generated parts into string for reflection judgment
            full_answer_text = next(
                (msg["content"] for msg in reversed(state["messages"][state["prompt_messages"]:]) 
                if msg.get("role") == "assistant"),
                ""
            )
            
            answer_text = self.reward.get_last_answer(state["messages"][state["prompt_messages"]:])
            answer_text = str(answer_text) if answer_text is not None else ""
            
            if task == "math":
                reflection_prompt = self.reflect_math_qa(answer_text, ref_answer, full_answer_text)
            elif task == "simuagent":
                assert init_code is not None
                init_code = init_codes[idx]
                reflection_prompt = self.reflect_simuagent_qa(answer_text, ref_answer, init_code)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            if mode == "train" and reflection_prompt is not None and idx < len(prompts) - 1:
                # Call llm to get reflection result (calls are serial here, only processing current state)
                reflection_messages[idx].append({"role": "user", "content": reflection_prompt})
                reflection_llm_response = llm.chat(
                    # state["messages"][-2:] + reflection_messages[idx],  # Skip system prompt and last reflection, start directly from question
                    state["messages"][1:] + reflection_messages[idx],  # Skip system prompt
                    sampling_params=custom_sp,
                    use_tqdm=False
                )
                # Extract llm's reflection output (assuming only one message is returned)
                reflection_result = reflection_llm_response[0].outputs[0].text
                reflection_result = reflection_result.replace("<|im_end|>", "").strip()  # TODO: Why does my vllm always output <|im_end|>
                reflection_messages[idx].append({"role": "assistant", "content": reflection_result})

            # Save processed state
            final_states.append(state)
            
        if mode == "eval":
            assert len(reflection_messages) == len(prompts)
            assert all(not sub for sub in reflection_messages)  # Each sublist should be empty
                
            reflection_messages = [[{"role": "EVAL", "content": "N/A"}] for _ in prompts]

        # Organize output: extract each branch's generated parts, token ids, and mask
        completion_messages = [s["messages"][s["prompt_messages"]:] for s in final_states]
        completion_ids = [s["completion_ids"] for s in final_states]
        completion_mask = [s["completion_mask"] for s in final_states]
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
            "reflections": reflection_messages
        }
        return output
    
    def reflect_math_qa(self, gen_answer: str, ref_answer: str, full_gen_answer) -> Optional[str]:
        m = re.search(r"<answer>(.*?)</answer>", ref_answer, re.DOTALL)
        ref_answer = m.group(1).strip() if m else ref_answer
        
        m = re.search(r"<answer>(.*?)</answer>", full_gen_answer, re.DOTALL)
        
        if m:
            full_gen_answer = f"<answer>{m.group(1)}</answer>"
        else:
            full_gen_answer = truncate_answer_text(full_gen_answer, 150)
        
        # previous_response = f": {gen_answer}" if gen_answer else ""
        
        # format_suggestion = ""
        
#         if self.reward.reward_funcs[-1](completions) < 0.8 and self.reward.reward_funcs[-2](completions) < 0.8:
#             format_suggestion = """Format incorrect. Expected:
# <think>
# ...
# </think>
# <answer>
# ...
# </answer>

# """
#         else:
#             format_suggestion = ""
        
        if grade(gen_answer, ref_answer):
            return None
        else:
            return f"""\
The reference answer is:
<answer>
{ref_answer}
</answer>

However, your answer is:
{full_gen_answer}

Your answer differs from the reference. Only the content within the <answer> tags is compared, so incorrect answers or any extra, irrelevant text inside the tags can lead to a mismatch.

Task  
In no more than 100 words, try to identify which part of your reasoning or output most likely led to the discrepancy, so you can better monitor it in future work.
If you truly have no reasonable idea, output exactly:

None

and nothing else.

Guidelines  
* Give concrete, actionable feedback (e.g., "I forgot to convert units before adding rates," not "I should read the question more carefully").  
* Do not quote or mention the reference answer."""

#             return f"""\
# The reference answer is:
# <answer>
# {ref_answer}
# </answer>

# However, your answer is:
# {full_gen_answer}

# Literally, your answer differs from the reference answer.

# Task  
# In no more than 100 words, try to identify which part of your reasoning or output most likely led to the discrepancy, so you can better monitor it in future work.
# If you truly have no reasonable idea, output exactly:

# None

# and nothing else.

# Guidelines  
# * Give concrete, actionable feedback (e.g., "I forgot to convert units before adding rates," not "I should be more careful").  
# * Do not quote or mention the reference answer.  
# * Avoid unfounded speculation—return **None** if the faulty step cannot be reasonably guessed."""
        
        if self.tools:
            return f"""\
The reference answer is:
<answer>
{ref_answer}
</answer>
which differs from your previous response{previous_response}.

Please summarize what you've learned from your reasoning process or tool usage. The purpose is to help you avoid making similar mistakes in the future.

Do not write vague statements like "I should ensure that I correctly interpret and calculate the problem statement."

Instead, provide concrete and specific insights. For example:

I should use the print function to get output from the Python tool; otherwise, it just returns empty output.

I should check if the denominator is zero before performing division; otherwise, it will cause a runtime error.

Your summary MUST be under 100 words and MUST not include or refer to the reference answer."""

        else:
            return f"""\
The reference answer is:
<answer>
{ref_answer}
</answer>
This differs from your previous response{previous_response}.

Task  
In no more than 100 words, try to guess which single step in your reasoning most likely caused the error, so you can monitor it in future work.
If you truly have no reasonable idea, output exactly:

None

and nothing else.

Guidelines  
* Give concrete, actionable feedback (e.g., "I forgot to convert units before adding rates," not "I should be more careful").  
* Do not quote or mention the reference answer.  
* Avoid unfounded speculation—return **None** if the faulty step cannot be reasonably guessed."""

#         else:
#             return f"""\
# {format_suggestion}
# {answer_suggesion}

# Your task:
# In 100 words or fewer, summarize the specific lessons you've learned from your reasoning process so you can avoid similar mistakes in the future.

# Guidelines:
# * Focus on actionable insights. Avoid vague statements like "I should ensure that I correctly interpret and calculate the problem statement."
# * Do not quote or reference the reference answer above.

# Examples of useful insights:
# * I should convert all units (e.g., liters, kilometers, and liters per 100 km) to a consistent format before performing calculations to avoid mixing raw values with rates.
# * When a total is fixed (e.g., 52 stamps), I must ensure that the sum of individual components precisely matches the given total."""


    def simuagent_reflect(self, gen_answer: str, ref_answer: str, init_code: str=None) -> Optional[str]:
        """
        If the answer contains a code block starting with ```python, this method extracts the Python code 
        from both the answer and the reference answer, executes them to obtain the variable system_dict,
        and compares the results. If they match, it returns None; otherwise, it returns a reflection prompt 
        in English that specifies the differences and includes instructions for reflection.
        """
        
        # If contains <answer>...</answer> tags, only return content inside tags; otherwise return entire text
        m = re.search(r"<answer>(.*?)</answer>", gen_answer, re.DOTALL)
        gen_answer = m.group(1).strip() if m else gen_answer
        
        REFL_PROMPT = textwrap.dedent("""\
            The reference answer is:
            <answer>
            {ref_answer}
            </answer>
            
            However, Your answer is:
            {gen_answer}

            {error_msg}

            Please summarize the main cause of the error. For example:
            In this three-phase power system, I mistakenly used a single-phase transformer instead of a three-phase one, which caused connection and simulation errors.
            You MUST keep your summary under 100 words.""")
        
# Summarize the main cause of the error into a general, practical, and specific lesson learned, without including any information about the reference answer. For example:
    
# Please summarize the main cause of the error. For example:
#             In a three-phase power system, I mistakenly used a single-phase transformer instead of a three-phase one, which caused connection and simulation errors.
        
        if "```python" not in gen_answer:
            return REFL_PROMPT.format(
                gen_answer=gen_answer,
                ref_answer=ref_answer,
                error_msg="Your answer does not contain a code block, e.g. ```python\n...\n```."
            )

        # Extract the answer code block.
        gen_code = extract_python_code(gen_answer)
        # Extract the reference answer code block.
        if "```python" in ref_answer:
            ref_code = extract_python_code(ref_answer)
        else:
            raise ValueError("Reference answer does not contain a code block.")

        # Execute the code in isolated environments.
        gen_globals = {}
        ref_globals = {}
        
        if init_code is not None:
            init_code = extract_python_code(init_code)
            exec(init_code, gen_globals)
            exec(init_code, ref_globals)
            
        try:
            exec(gen_code, {}, gen_globals)
        except Exception as e:
            return REFL_PROMPT.format(
                gen_answer=gen_answer,
                ref_answer=ref_answer,
                error_msg=f"Your answer code execution failed. Error: {e}"
            )
        
        try:
            exec(ref_code, {}, ref_globals)
        except Exception as e:
            return REFL_PROMPT.format(
                gen_answer=gen_answer,
                ref_answer=ref_answer,
                error_msg=f"Reference answer code execution failed. Error: {e}"
            )

        # Retrieve the system_dict results.
        answer_result = gen_globals.get("system_dict", None)
        ref_result = ref_globals.get("system_dict", None)

        if answer_result is None:
            return REFL_PROMPT.format(
                gen_answer=gen_answer,
                ref_answer=ref_answer,
                error_msg="Your answer code did not produce the 'system_dict' variable."
            )
        if ref_result is None:
            raise ValueError("Reference answer code did not produce the 'system_dict' variable.")

        # If overall the dictionaries match (ignoring list order), no reflection is needed.
        if dicts_equal_ignoring_list_order(answer_result, ref_result):
            return None
        
        breakpoint()

        # Define a recursive function to compare nested data structures.
        def compare_structures(a, b, path=""):
            differences = []
            if isinstance(a, dict) and isinstance(b, dict):
                for key in b:
                    current_path = f"{path}.{key}" if path else key
                    if key not in a:
                        differences.append(f"Missing key: {current_path} was not present in your answer.")
                    else:
                        differences.extend(compare_structures(a[key], b[key], current_path))
                for key in a:
                    current_path = f"{path}.{key}" if path else key
                    if key not in b:
                        differences.append(f"Extra key: {current_path} was found in your answer but should not be there.")
            elif isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    differences.append(f"Value mismatch: Different number of items at {path}: got {len(a)}, expected {len(b)}.")
                else:
                    for index, (item_a, item_b) in enumerate(zip(a, b)):
                        differences.extend(compare_structures(item_a, item_b, f"{path}[{index}]"))
            else:
                if a != b:
                    differences.append(f"Value mismatch: At {path}: got {a}, expected {b}.")
            return differences

        # Generate a list of difference messages.
        differences = compare_structures(answer_result, ref_result)
        diff_message = "\n".join(differences[:6])
        
        return REFL_PROMPT.format(
            gen_answer=gen_answer,
            ref_answer=ref_answer,
            error_msg=f"The results of your answer do not match the reference answer. The differences are as follows:\n{diff_message}"
        )
        