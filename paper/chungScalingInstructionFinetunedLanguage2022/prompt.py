from typing import List


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
    prompt += f"Q: {instance['input']}\nA: Let's think step by step."

    return prompt
