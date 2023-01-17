import re

from datasets import load_dataset

from utils import logger
from prompt import math_word_problem_template


class GSM8K:
    def __init__(self):
        pass

    def build_dataloader(self):
        pass

    def eval(self, checkpoint, dataset_cutoff=10):
        pass

    @staticmethod
    def regex_answer(source_text):
        """extract answer from source dataset answer text

        ex: 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
        She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
        #### 18'
        """
        match = re.search(r"####\s(\d+)", source_text)
        answer = int(match.group(1)) if match else -1

        return {"label": answer}

    @staticmethod
    def regex_predict(target_text):
        """extract answer from generated text

        ex: '"She eats 3 + 4 = 7 eggs every day.
        She has 16 - 7 = 9 eggs left.
        She sells 9 * 2 = $18 at the farmers' market.
        The answer is 18."'
        """
        match = re.search(r"The answer is (\d+)", target_text)
        predict = int(match.group(1)) if match else -1
        return predict
