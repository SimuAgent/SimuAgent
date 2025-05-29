import random
import json
from typing import List, Dict, Callable, Any

from datasets import Dataset, load_dataset, concatenate_datasets # type: ignore

def extract_boxed_answer(text: str) -> str | None:
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find \boxed{
    boxed_start = text.find('\\boxed{')
    if boxed_start == -1:
        return text
    # Find the content between the braces
    content_start = boxed_start + 7  # len('\\boxed{')
    closing_brace = find_matching_brace(text, content_start)
    
    if closing_brace == -1:
        return text
    
    return text[content_start:closing_brace]

def strip_non_numeric(text: str) -> str:
    return "".join(c for c in text if c.isdigit() or c == '.')

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_preprocess_fn(name: str) -> Callable[[Dict], Dict]:
    if name == "gsm8k":
        def preprocess_gsm8k(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["question"],
                "answer": extract_hash_answer(x["answer"]),
                "task": "math"
            }
        return preprocess_gsm8k
    elif name == "math":
        def preprocess_math(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": extract_boxed_answer(x["solution"]),
                "task": "math"
            }
        return preprocess_math
    elif name == "math500":
        def preprocess_math500(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": x["answer"],
                "task": "math"
            }
        return preprocess_math500
    elif name == "mmlu":
        mmlu_map = ["A", "B", "C", "D"]
        def preprocess_mmlu(x: Dict[str, Any]) -> Dict[str, Any]:
            options = x["choices"]
            answer = x["answer"]
            question = f"Question: {x['question']}\n"
            for i, option in enumerate(options):
                question += f"\n{mmlu_map[i]}: {option}"
            return {
                "question": question,
                "temp_answer": mmlu_map[answer],
                "task": "mc"
            }
        return preprocess_mmlu
    elif name == "mmlu_pro":
        mmlu_map = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        def preprocess_mmlu(x: Dict[str, Any]) -> Dict[str, Any]:
            options = x["options"]
            answer = x["answer"]
            question = f"Question: {x['question']}\n"
            for i, option in enumerate(options):
                question += f"\n{mmlu_map[i]}: {option}"
            return {
                "question": question,
                "answer": answer,
                "task": "mc"
            }
        return preprocess_mmlu
    elif name == "openbookqa":
        def preprocess_openbookqa(x: Dict[str, Any]) -> Dict[str, Any]:
            choices_texts = x['choices']['text']
            choices_labels = x['choices']['label']
            
            formatted_choices = []
            for i in range(len(choices_labels)):
                formatted_choices.append(f"{choices_labels[i]}. {choices_texts[i]}")
            
            question = f"Question: {x['question_stem']}\n\nChoices:\n" + "\n".join(formatted_choices)
            return {
                "question": question,
                "answer": x["answerKey"],
                "task": "mc"
            }
        return preprocess_openbookqa
    elif name in ["openrs", "openrs_easy", "openrs_hard"]:
        def preprocess_openrs(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["problem"],
                "answer": x["answer"],
                "task": "math"
            }
        return preprocess_openrs
    elif name == "prime_code":
        def preprocess_prime_code(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["prompt"],
                "answer": x["verification_info"],
                "task": "code"
            }
        return preprocess_prime_code
    # My Datasets
    elif 'create-system' in name:
        def preprocess_create_system(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["question"],
                "answer": x["answer"],
                # "task": "create-system"  # TODO: what does the task mean here?
                "task": "simuagent"
            }
        return preprocess_create_system
    
    elif 'system-creation' in name:
        def preprocess_create_system(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["question"],
                "answer": x["answer"],
                "init_code": x["init_code"],
                "task": "simuagent"
            }
        return preprocess_create_system
    
    elif 'correction' in name:
        def preprocess_correction(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "question": x["question"],
                "answer": x["answer"],
                "init_code": x["init_code"]
            }
        return preprocess_correction
        
    else:
        raise ValueError(f"Dataset {name} not supported for preprocess_dataset.")

def preprocess_dataset(name: str = "gsm8k",
                       split: str | None = None,
                       n: int | None = None,
                       seed: int = 0) -> Dataset:
    if name == "gsm8k":
        if split is None:
            split = "test"  # TODO: why not train?
        dataset: Dataset = load_dataset("openai/gsm8k", "main")[split].filter(lambda ex: len(ex["question"]) < 500)  # type: ignore
    elif name == "math":
        if split is None:
            split = "train"
        dataset: Dataset = load_dataset("chiayewken/competition_math")[split].filter(lambda ex: len(ex["problem"]) < 500) # type: ignore
    elif name == "math500":
        if split is None:
            split = "test"
        dataset: Dataset = load_dataset("HuggingFaceH4/MATH-500")[split] # type: ignore
    elif 'SimuGPT' in name:
        if split is None:
            split = "train"
        dataset: Dataset = load_dataset(name)[split] # type: ignore
    else:
        raise ValueError(f"Dataset {name} not supported for preprocess_dataset. \
Please ensure that the dataset is formatted with 'prompt' (str) and 'answer' (str) keys.")
    
    preprocess_fn = get_preprocess_fn(name)
    if n is not None and n > 0:
        dataset = dataset.shuffle(seed=seed).select(range(n)) # type: ignore
    dataset = dataset.map(preprocess_fn, num_proc=10, remove_columns=dataset.column_names) # type: ignore
    if "temp_answer" in dataset.column_names:
        dataset = dataset.rename_column("temp_answer", "answer")
    return dataset

def format_prompt(prompt: str,
                  system_prompt: str | None = None,
                  few_shot: List[Dict[str, str]] | None = None,
                  fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot and random.random() < fewshot_prob:
        messages.extend(few_shot)
    messages.append({"role": "user", "content": prompt})
    return messages

def format_dataset(dataset: Dataset,
                   system_prompt: str | None = None,
                   few_shot: List[Dict[str, str]] | None = None,
                   fewshot_prob: float = 1.0,
                   question_key: str = "question",
                   answer_key: str = "answer",
                   ) -> Dataset:
    return dataset.map(lambda x: {
        "prompt": format_prompt(x[question_key], system_prompt, few_shot, fewshot_prob),
        "answer": x[answer_key]
    }, num_proc=10)