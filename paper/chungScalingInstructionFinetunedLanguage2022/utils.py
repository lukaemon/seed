import logging
import re
from typing import List, Optional
from functools import partial

import numpy as np
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

import prompt


logging.basicConfig(
    format="[%(levelname)s] [%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
)

logger = logging.getLogger("flan-t5")
logger.setLevel(logging.INFO)

# set numpy random seed to 42
rng = np.random.default_rng(42)


def load(dataset_name, task_name, split, cutoff: Optional[int]):
    """load dataset
    split: str, train, validation, test
    cutoff: Optional[int], number of instances to load
    """
    ds = load_dataset(dataset_name, task_name, split=split)

    if cutoff:
        ds = ds.select(range(min(cutoff, len(ds))))

    return ds


def few_shot(ds: Dataset, n) -> List[dict]:
    """Random sample n instances from the dataset as few shot examples
    return: [{input: str, target: str}]
    """
    random_list = rng.choice(np.arange(len(ds)), size=n, replace=False)
    res = ds.select(random_list).to_pandas().to_dict(orient="records")

    return res


def build_dataloader(
    dataset_name,
    task_name,
    split,
    tokenizer,
    cot: bool,
    cutoff,
    n,
    batch_size,
    inst_fn,
    cot_fn,
):
    """
    cot: bool, whether to use cot prompt
    n: int, number of few shot examples
    inst_fn: fn(task_name) -> instruction
    cot_fn: fn(task_name) -> cot few shot examples
    """
    raw_ds = load(dataset_name, task_name, split, cutoff)

    if cot:
        prompt_fn = partial(prompt.cot_prompt, inst_fn(task_name), cot_fn(task_name))
    else:
        prompt_fn = partial(
            prompt.answer_only_prompt,
            inst_fn(task_name),
            few_shot(raw_ds, n=n),
        )

    # apply prompt_fn to each instance
    ds = raw_ds.map(lambda instance: {"prompt": prompt_fn(instance)})

    # tokenize
    ds = ds.with_transform(lambda instance: tokenizer(instance["prompt"]))

    # remove columns for dataloader collator, leaves only input_ids, attention_mask
    # ds = ds.remove_columns(["prompt", "input", "target"])

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

    return dl, raw_ds
