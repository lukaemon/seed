from functools import lru_cache
from pathlib import Path
import json
import re

task_list = [
    "high_school_european_history",
    "business_ethics",
    "clinical_knowledge",
    "medical_genetics",
    "high_school_us_history",
    "high_school_physics",
    "high_school_world_history",
    "virology",
    "high_school_microeconomics",
    "econometrics",
    "college_computer_science",
    "high_school_biology",
    "abstract_algebra",
    "professional_accounting",
    "philosophy",
    "professional_medicine",
    "nutrition",
    "global_facts",
    "machine_learning",
    "security_studies",
    "public_relations",
    "professional_psychology",
    "prehistory",
    "anatomy",
    "human_sexuality",
    "college_medicine",
    "high_school_government_and_politics",
    "college_chemistry",
    "logical_fallacies",
    "high_school_geography",
    "elementary_mathematics",
    "human_aging",
    "college_mathematics",
    "high_school_psychology",
    "formal_logic",
    "high_school_statistics",
    "international_law",
    "high_school_mathematics",
    "high_school_computer_science",
    "conceptual_physics",
    "miscellaneous",
    "high_school_chemistry",
    "marketing",
    "professional_law",
    "management",
    "college_physics",
    "jurisprudence",
    "world_religions",
    "sociology",
    "us_foreign_policy",
    "high_school_macroeconomics",
    "computer_security",
    "moral_scenarios",
    "moral_disputes",
    "electrical_engineering",
    "astronomy",
    "college_biology",
]

# task_list = ["high_school_european_history"]  # for testing


# train set is used for few-shot prompts
# val set could be used for hyperparameter tuning
# test set is used to compute the final accuracy
split = ["train", "val", "test"]

COT_PATH = Path("./mmlu-cot.json")


@lru_cache
def task_instruction(task_name):
    cot = json.load(COT_PATH.open())
    cot_in_lines = cot[task_name].split("\n")

    return cot_in_lines[0].strip()  # instruction is the 1st line


@lru_cache
def task_cot(task_name):
    cot = json.load(COT_PATH.open())
    cot_in_lines = cot[task_name].split("\n")

    return "\n".join(cot_in_lines[2:])  # cot prompt is the 3nd line and beyond


def extract_answer(response: str, cot: bool) -> str:
    if cot:
        match = re.search(r"answer is \((.*)\).", response)
    else:
        match = re.search(r"(.*)", response)

    return match.group(1) if match else "Parsing Failed"
