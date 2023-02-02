from functools import partial
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset


import t5
import utils
from utils import logger
import bbh
import mmlu

OUTPUT_DIR = "output"


@dataclass
class EvalConfig:
    dataset_name: str
    task_name: str
    split: str
    cutoff: Optional[int]
    cot: bool
    n: int  # number of few shot examples
    batch_size: int
    inst_fn: callable
    cot_fn: callable
    checkpoint: str
    temp: float
    k: int


@dataclass
class EvalResult:
    dataset_name: str
    checkpoint: str
    cot: bool
    k: int
    task_name: str
    accuracy: float
    failed_case: Dataset  # [{input: str, target: str, generated_text: str, regexed_pred: str}]


def analyze(full_text_pred: List[str], target: List[str], config: EvalConfig):
    if "bbh" in config.dataset_name:
        regex_fn = partial(bbh.extract_answer, cot=config.cot)
    elif "mmlu" in config.dataset_name:
        regex_fn = partial(mmlu.extract_answer, cot=config.cot)

    pred = np.array(list(map(regex_fn, full_text_pred)))
    target = np.array(target)

    right = pred == target
    accuracy = np.mean(right)

    failed_idx = np.argwhere(right == False).flatten()
    full_text_pred = np.array(full_text_pred)

    failed_output = list(full_text_pred[failed_idx])
    failed_pred = list(pred[failed_idx])

    return accuracy, failed_idx, failed_output, failed_pred


def eval_task(config: EvalConfig) -> EvalResult:
    logger.info(f"Loading model from {config.checkpoint}.")
    m, tk = t5.load_model(config.checkpoint)

    dl, ds = utils.build_dataloader(
        config.dataset_name,
        config.task_name,
        config.split,
        tk,
        config.cot,
        config.cutoff,
        config.n,
        config.batch_size,
        config.inst_fn,
        config.cot_fn,
    )

    full_text_pred = []

    for batch in dl:
        full_text_pred.extend(t5.predict(m, tk, batch, temp=config.temp, k=config.k))

    # analyze
    accuracy, failed_idx, failed_output, failed_pred = analyze(
        full_text_pred, ds["target"], config
    )

    failed_cases = (
        ds.select(failed_idx)
        .add_column("generated_text", failed_output)
        .add_column("regexed_pred", failed_pred)
    )

    return EvalResult(
        config.dataset_name,
        config.checkpoint,
        config.cot,
        config.k,
        config.task_name,
        accuracy,
        failed_cases,
    )


def save_result(results: List[EvalResult]):
    """Save results of all tasks per dataset
    results: List[EvalResult], each result is for a task
    """
    # get metadata out of one result
    a_result = results[0]
    dataset_name = a_result.dataset_name.split("/")[-1]
    model_name = a_result.checkpoint.split("/")[-1]
    cot = a_result.cot
    k = a_result.k

    # saving accuracy
    acc_file_dir = os.path.join(OUTPUT_DIR, dataset_name, "accuracy")
    os.makedirs(acc_file_dir, exist_ok=True)

    acc_file_name = f"{model_name}_cot_{cot}_k_{k}.csv"
    acc_file_path = os.path.join(acc_file_dir, acc_file_name)

    acc = [
        {
            "model_name": model_name,
            "task_name": r.task_name,
            "accuracy": r.accuracy,
            "cot": cot,
            "k": k,
        }
        for r in results
    ]

    logger.info(f"Saving accuracy result to {acc_file_path}.")
    pd.DataFrame(acc).to_csv(
        acc_file_path,
        index=False,
    )

    # saving failed output
    fail_file_dir = os.path.join(OUTPUT_DIR, dataset_name, "failed", model_name)
    os.makedirs(fail_file_dir, exist_ok=True)

    logger.info(f"Saving failed output to {fail_file_dir}.")
    for r in results:
        fail_file_name = f"{r.task_name}_cot_{r.cot}_k_{r.k}.csv"
        fail_file_path = os.path.join(fail_file_dir, fail_file_name)

        r.failed_case.to_csv(fail_file_path, index=False)
