import re
from typing import List
import os

import torch
import numpy as np
from datasets import load_dataset
from prompt import csqa_template, csqa_example2text
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import logger


OUTPUT_DIR = "./output/csqa"


def load_csqa(cutoff=None):
    """
    ['id','question','question_concept','choices','answerKey','prompt','target']
    """
    csqa = load_dataset("commonsense_qa", "main", split="validation")
    if cutoff:
        csqa = csqa.select(range(cutoff))

    csqa = csqa.map(csqa_template).map(
        lambda example: {"target": ord(example["answerKey"].lower())}
    )

    logger.info(f"csqa loaded: {len(csqa)} examples. {csqa.column_names=}")
    return csqa


def build_dataloader(csqa, tokenizer, batch_size=8):
    """
    caqa: ['id','question','question_concept','choices','answerKey','prompt','target']
    """
    ds = csqa.map(
        lambda examples: tokenizer(examples["prompt"]), batched=True
    ).remove_columns(
        [
            "id",
            "question",
            "question_concept",
            "choices",
            "answerKey",
            "prompt",
            "target",
        ]
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        max_length=1024,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    return DataLoader(ds, collate_fn=collator, batch_size=batch_size)


def load_model(checkpoint):
    model = T5ForConditionalGeneration.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    model.parallelize()

    tk = AutoTokenizer.from_pretrained(checkpoint)

    logger.info(f"model loaded: {checkpoint=}")
    return model, tk


def extract_answer(string) -> int:
    match = re.search(r"the answer is \(([a-zA-Z])\)", string)

    return ord(match.group(1).lower()) if match else -1


def eval_loop(model, tokenizer, eval_loader):
    progress_bar = tqdm(range(len(eval_loader)))
    progress_bar.set_description("eval in progress")

    pred: List[int] = []
    output_seq: List[str] = []

    for batch in eval_loader:
        batch_output = model.generate(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            max_length=256,
            temperature=0,
        )

        batch_output_seq = tokenizer.batch_decode(
            batch_output, skip_special_tokens=True
        )
        batch_pred = list(map(extract_answer, batch_output_seq))
        pred += batch_pred
        output_seq += batch_output_seq

        progress_bar.update(1)

    del model
    torch.cuda.empty_cache()
    logger.info("model released from cuda memory")

    return pred, output_seq


def compute_accuracy(target: List[int], pred: List[int]):
    return (np.array(target) == np.array(pred)).mean()


def report(csqa, pred: List[int], output_seq: List[str]):
    def stich(example):
        return f"{example['full_question']} {example['generated_answer']}\nRight answer: ({example['answerKey'].lower()})\n\n"

    csqa = (
        csqa.add_column("generated_answer", output_seq)
        .add_column("prediction", pred)
        .map(lambda example: {"full_question": csqa_example2text(example)})
        .map(lambda example: {"full_qa": stich(example)})
    )

    failed_idx = np.where(np.array(csqa["target"]) != np.array(pred))[0]
    failed_cases = csqa.select(failed_idx)["full_qa"]

    # save failed cases to OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/failed_cases.txt", "w") as f:
        f.writelines(failed_cases)

    logger.info(f"failed cases saved to {OUTPUT_DIR}/failed_cases.txt")
