import re

from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

import prompt


def load_gsm8k(cutoff=None):
    ds = load_dataset("gsm8k", "main", split="test")

    if cutoff:
        ds = ds.select(range(cutoff))

    # apply prompt template
    ds = ds.map(lambda x: {"prompt": prompt.math_word_problem_template(x)}).map(
        lambda x: {"target": extract_target_gsm8k(x["answer"])}
    )

    return ds


def extract_target_gsm8k(source_text: str) -> int:
    """extract answer from source dataset answer text

    ex: 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
    She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
    #### 18'
    """
    match = re.search(r"####\s(\d+)", source_text)
    answer = int(match.group(1)) if match else -1

    return answer


def extract_prediction_gsm8k(target_text: str) -> int:
    """extract answer from generated text

    ex: '"She eats 3 + 4 = 7 eggs every day.
    She has 16 - 7 = 9 eggs left.
    She sells 9 * 2 = $18 at the farmers' market.
    The answer is 18."'
    """
    match = re.search(r"The answer is (\d+)", target_text)
    predict = int(match.group(1)) if match else -1
    return predict


def build_gsm8k_dataloader(dataset, tokenizer, batch_size=8):
    ds = dataset.map(lambda x: tokenizer(x["prompt"]), batched=True).remove_columns(
        ["question", "answer", "prompt", "target"]
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        max_length=1024,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    return DataLoader(ds, batch_size=batch_size, collate_fn=collator)
