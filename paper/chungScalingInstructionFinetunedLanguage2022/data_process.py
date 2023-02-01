from functools import lru_cache, cached_property
from pathlib import Path
import json
import re
from typing import List

from transformers import DataCollatorForSeq2Seq

BBH_DATA_DIR = "/workspaces/seed/cache/data/bbh/data"  # json
BBH_COT_DIR = "/workspaces/seed/cache/data/bbh/cot-prompts"  # txt
BBH_FEW_SHOT_N = 3  # I'll just take the first 3 instances as few shot examples


class BBH:
    def __init__(self):
        self.data_dir = Path(BBH_DATA_DIR)
        self.cot_dir = Path(BBH_COT_DIR)

    @cached_property
    def task_list(self) -> List[str]:
        res = []
        for path in self.data_dir.glob("*.json"):
            file_name = path.name
            task_name = file_name.split(".")[0]
            res.append(task_name)
        return sorted(res)

    @cached_property
    def n_task(self):
        return len(self.task_list)

    @lru_cache
    def instruction(self, task_name: str) -> str:
        cot_txt_path = self.cot_dir / f"{task_name}.txt"
        with cot_txt_path.open() as f:
            lines = f.readlines()

        return lines[2].strip()  # instruction is the 3rd line

    @lru_cache
    def cot(self, task_name: str) -> str:
        cot_txt_path = self.cot_dir / f"{task_name}.txt"
        with cot_txt_path.open() as f:
            lines = f.readlines()

        return "".join(lines[4:])  # cot prompt is the 5th line and beyond

    @lru_cache
    def few_shot(self, task_name: str) -> List[dict]:
        """
        return: [{input: str, target: str}]
        """
        json_path = self.data_dir / f"{task_name}.json"
        with json_path.open() as f:
            data = json.load(f)

        return data["examples"][:BBH_FEW_SHOT_N]

    def instances(self, task_name: str) -> List[dict]:
        """
        return: [{input: str, target: str}]
        """
        json_path = self.data_dir / f"{task_name}.json"
        with json_path.open() as f:
            data = json.load(f)

        return data["examples"][BBH_FEW_SHOT_N:]

    def prompted_instances(self, task_name: str, cot=False) -> List[str]:
        res = []

        for instance in self.instances(task_name):
            instruction = self.instruction(task_name)

            if cot:
                prompt = self.cot_prompt(instruction, self.cot(task_name), instance)
            else:
                prompt = self.answer_only_prompt(
                    instruction, self.few_shot(task_name), instance
                )

            res.append(prompt)
        return res

    def target(self, task_name: str) -> List[str]:
        return [instance["target"] for instance in self.instances(task_name)]

    @staticmethod
    def answer_only_prompt(
        instruction: str, few_shot: List[dict], instance: dict
    ) -> str:
        """
        few_shot: [{input: str, target: str}]
        instance: {input: str, target: str}
        """
        prompt = instruction + "\n\n"

        for shot in few_shot:
            prompt += f"Q: {shot['input']}\nA: {shot['target']}\n\n"

        prompt += f"Q: {instance['input']}\nA:"

        return prompt

    @staticmethod
    def cot_prompt(instruction: str, cot: str, instance: dict) -> str:
        """
        few_shot: [{input: str, target: str}]
        instance: {input: str, target: str}
        """
        prompt = instruction + "\n\n"
        prompt += cot + "\n\n"
        prompt += f"Q: {instance['input']}\nA: Let's think step by step.\n"

        return prompt

    @staticmethod
    def bbh_response_regex(response: str, cot=False) -> str:
        if cot:
            match = re.search(r"the answer is (.*).", response)
        else:
            match = re.search(r"A: (.*)", response)

        if match:
            return match.group(1)
        return None
