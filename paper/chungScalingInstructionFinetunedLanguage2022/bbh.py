from functools import lru_cache
from pathlib import Path
import re

COT_PATH = Path("./bbh-cot")


# task_list = [
#     "boolean_expressions",
#     "causal_judgement",
#     "date_understanding",
#     "disambiguation_qa",
#     "dyck_languages",
#     "formal_fallacies",
#     "geometric_shapes",
#     "hyperbaton",
#     "logical_deduction_five_objects",
#     "logical_deduction_seven_objects",
#     "logical_deduction_three_objects",
#     "movie_recommendation",
#     "multistep_arithmetic_two",
#     "navigate",
#     "object_counting",
#     "penguins_in_a_table",
#     "reasoning_about_colored_objects",
#     "ruin_names",
#     "salient_translation_error_detection",
#     "snarks",
#     "sports_understanding",
#     "temporal_sequences",
#     "tracking_shuffled_objects_five_objects",
#     "tracking_shuffled_objects_seven_objects",
#     "tracking_shuffled_objects_three_objects",
#     "web_of_lies",
#     "word_sorting",
# ]

task_list = ["boolean_expressions"]  # for testing


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


def extract_answer(response: str, cot: bool) -> str:
    if cot:
        match = re.search(r"answer is (.*).", response)
    else:
        match = re.search(r"(.*)", response)

    return match.group(1) if match else "Parsing Failed"
