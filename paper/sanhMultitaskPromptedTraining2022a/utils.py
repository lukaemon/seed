import logging
import time
import os
import json
from dataclasses import dataclass, asdict
from typing import List

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration
from promptsource.templates import DatasetTemplates
from datasets import load_dataset


logging.basicConfig(
    format="[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
)

logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)


def build_t2t(checkpoint: str) -> callable:
    logger.info(f"loading model from {checkpoint}...")

    model = T5ForConditionalGeneration.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    model.parallelize()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")

    def t2t(batch):
        inputs = tokenizer(
            batch,
            padding="longest",  # not on TPU so pad differently per batch is fine. Pad to max is waste of compute
            max_length=1024,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

        outputs = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_length=256,
            temperature=0,
        )

        output_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # post process to lower case
        # TODO: definitely problematic for multilingual usecase and more challenginge test cases
        output_sequences = [s.strip().lower() for s in output_sequences]

        return output_sequences

    return t2t


def load_raw_dataset(dataset_name, subset_name):
    raw_dataset = load_dataset(
        dataset_name,
        subset_name,
        split="dev_r1" if dataset_name == "anli" else "validation",
    )

    return raw_dataset


def get_prompt(dataset_name, subset_name, prompt_name):
    template = DatasetTemplates(
        f"{dataset_name}/{subset_name}" if subset_name else dataset_name
    )
    prompt = template[prompt_name]

    return prompt


def preprocess_dataset(raw_dataset, prompt, cutoff=None):
    """
    raw_dataset: dataset object from hf datasets.load_dataset
    prompt: prompt object from promptsource
    """
    if cutoff:
        cutoff = min(cutoff, len(raw_dataset))
        raw_dataset = raw_dataset.shuffle(seed=42).select(range(cutoff))

    input_text = []
    target_text = []
    for i in raw_dataset:
        try:
            i, o = prompt.apply(i)
            input_text.append(i)
            target_text.append(
                o.lower()
            )  # lower case for all target output TODO: rebuild in vector space

        # log the error and continue
        except Exception as e:
            logger.error(f"Error when applying {prompt.name} on {i}")
            logger.error(e)
            continue

    return input_text, target_text


# build a dataclass for results
@dataclass
class Result:
    checkpoint: str
    dataset_name: str
    subset_name: str
    test_size: int
    time: float  # in seconds
    prompt_name: str
    accuracy: float


def eval(t2t, input_text, target_text, batch_size=32):
    t_start = time.time()

    data_size = len(input_text)

    correct = 0
    failed_cases = []

    for i in range(0, data_size, batch_size):
        batch = input_text[i : i + batch_size]

        pred = np.array(t2t(batch))
        target = np.array(target_text[i : i + batch_size])

        batch_correct = np.array(pred) == np.array(target)
        correct += batch_correct.sum()

        idx = np.where(batch_correct == False)[0]
        batch_failed_input = np.take(batch, idx)
        batch_failed_target = np.take(target, idx)
        batch_failed_pred = np.take(pred, idx)

        for i, t, pred in zip(
            batch_failed_input, batch_failed_target, batch_failed_pred
        ):
            failed_cases.append({"input": i, "target": t, "pred": pred})

    accuracy = correct / len(input_text)

    t_end = time.time()
    t_lapse = t_end - t_start

    return accuracy, t_lapse, failed_cases


def dump_result_as_csv(results: List[Result], checkpoint: str, output_dir: str):
    result_df = pd.DataFrame([asdict(r) for r in results])

    model_name = checkpoint.split("/")[-1]
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{model_name}.csv")

    logger.info(f"dumping results to {file_path}...")
    result_df.to_csv(file_path, index=False)


def dump_failed_cases_as_json(
    failed_cases: List[dict],
    checkpoint,
    dataset_name,
    subset_name,
    prompt_name,
    output_dir,
):
    model_name = checkpoint.split("/")[-1]
    prompt_name = prompt_name.replace("/", "_")

    file_parent_path = os.path.join(
        output_dir, model_name, f"{dataset_name}/{subset_name}"
    )
    os.makedirs(file_parent_path, exist_ok=True)

    file_path = os.path.join(file_parent_path, f"{prompt_name}.json")

    logger.info(f"dumping failed cases to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(failed_cases, f, indent=4)
