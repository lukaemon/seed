from functools import lru_cache, partial
from pathlib import Path
import re
from typing import List

import numpy as np
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

FEW_SHOT_N = 3
COT_PATH = Path("./bbh-cot")

# set numpy random seed to 42
rng = np.random.default_rng(42)

task_list = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

# task_list = ["boolean_expressions"]  # for testing


def load_bbh(task_name, cutoff=32):
    ds = load_dataset("lukaemon/bbh", task_name, split="test")

    if cutoff:
        ds = ds.select(range(cutoff))

    return ds


@lru_cache
def task_instruction(task_name: str) -> str:
    cot_txt_path = COT_PATH / f"{task_name}.txt"
    with cot_txt_path.open() as f:
        lines = f.readlines()

    return lines[2].strip()  # instruction is the 3rd line


@lru_cache
def task_cot(task_name) -> str:
    cot_txt_path = COT_PATH / f"{task_name}.txt"
    with cot_txt_path.open() as f:
        lines = f.readlines()

    return "".join(lines[4:])  # cot prompt is the 5th line and beyond


def few_shot(ds, n=FEW_SHOT_N) -> List[dict]:
    """Random sample 3 instances from the dataset as few shot examples
    return: [{input: str, target: str}]
    """
    random_list = rng.choice(np.arange(len(ds)), size=n, replace=False)
    res = ds.select(random_list).to_pandas().to_dict(orient="records")

    return res


def answer_only_prompt(instruction: str, few_shot: List[dict], instance: dict) -> str:
    """
    few_shot: [{input: str, target: str}]
    instance: {input: str, target: str}
    """
    prompt = instruction + "\n\n"

    for shot in few_shot:
        prompt += f"Q: {shot['input']}\nA: {shot['target']}\n\n"

    prompt += f"Q: {instance['input']}\nA:"

    return prompt


def cot_prompt(instruction: str, cot: str, instance: dict) -> str:
    """
    few_shot: [{input: str, target: str}]
    instance: {input: str, target: str}
    """
    prompt = instruction + "\n\n"
    prompt += cot + "\n\n"
    prompt += f"Q: {instance['input']}\nA: Let's think step by step.\n"

    return prompt


def build_dataloader(task_name, tokenizer, cot, cutoff, n=FEW_SHOT_N, batch_size=8):
    ds = load_bbh(task_name, cutoff)

    if cot:
        prompt_fn = partial(
            cot_prompt, task_instruction(task_name), task_cot(task_name)
        )
    else:
        prompt_fn = partial(
            answer_only_prompt,
            task_instruction(task_name),
            few_shot(ds, n=n),
        )

    # apply prompt_fn to each instance
    ds = ds.map(lambda instance: {"prompt": prompt_fn(instance)})

    # tokenize
    ds = ds.map(lambda instance: tokenizer(instance["prompt"]), batched=True)

    target = ds["target"]  # save target for eval

    # remove columns for dataloader collator, leaves only input_ids, attention_mask
    ds = ds.remove_columns(["prompt", "input", "target"])

    # setup collator
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        max_length=2048,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # setup dataloader
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator,
    )

    return dl, target


def response_regex(response: str, cot=False) -> str:
    if cot:
        match = re.search(r"the answer is (.*).", response)
    else:
        match = re.search(r"(.*)", response)

    if match:
        return match.group(1)
    return "Parsing Error"
