import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


import bbh
import t5
from utils import logger

OUTPUT_DIR = "output"


@dataclass
class BBHEvalConfig:
    checkpoint: str
    cutoff: Optional[int] = None
    batch_size: int = 8
    cot: bool = False
    temp: float = 0.0
    k: int = 1


@dataclass
class BBHEvalResult:
    task_name: str
    accuracy: float
    failed_output: List[str]
    failed_pred: List[str]


def bbh_analyze(full_text_pred: List[str], target: List[str], cot: bool):
    pred = np.array(list(map(lambda r: bbh.response_regex(r, cot), full_text_pred)))
    target = np.array(target)

    right = pred == target
    accuracy = np.mean(right)

    failed_idx = np.argwhere(right == False).flatten()
    full_text_pred = np.array(full_text_pred)

    failed_output = list(full_text_pred[failed_idx])
    failed_pred = list(pred[failed_idx])

    return accuracy, failed_output, failed_pred


def bbh_eval_loop(config: BBHEvalConfig) -> List[BBHEvalResult]:
    logger.info(f"Loading model from {config.checkpoint}...")
    m, tk = t5.load_model(config.checkpoint)

    eval_result = []

    for task_name in tqdm(bbh.task_list):
        dl, target = bbh.build_dataloader(
            task_name, tk, cutoff=config.cutoff, cot=config.cot
        )

        full_text_pred = []

        for batch in dl:
            full_text_pred.extend(
                t5.predict(m, tk, batch, temp=config.temp, k=config.k)
            )

        eval_result.append(
            BBHEvalResult(task_name, *bbh_analyze(full_text_pred, target, config.cot))
        )

    del m
    torch.cuda.empty_cache()

    return eval_result


def bbh_save_result(eval_result: List[BBHEvalResult], config: BBHEvalConfig):
    # saving accuracy
    acc_file_dir = os.path.join(OUTPUT_DIR, "bbh", "accuracy")
    model_name = config.checkpoint.split("/")[-1]

    os.makedirs(acc_file_dir, exist_ok=True)
    acc_file_name = f"bbh_{model_name}_cot_{config.cot}_k_{config.k}.csv"
    acc_file_path = os.path.join(acc_file_dir, acc_file_name)

    acc = [
        {
            "model_name": model_name,
            "task_name": r.task_name,
            "accuracy": r.accuracy,
            "cot": config.cot,
            "k": config.k,
        }
        for r in eval_result
    ]

    logger.info(f"Saving accuracy result to {acc_file_path}.")
    pd.DataFrame(acc).to_csv(
        acc_file_path,
        index=False,
    )

    # saving failed output
    fail_file_dir = os.path.join(OUTPUT_DIR, "bbh", "failed", model_name)
    os.makedirs(fail_file_dir, exist_ok=True)

    logger.info(f"Saving failed output to {fail_file_dir}.")
    for r in eval_result:
        fail_file_name = f"{r.task_name}_cot_{config.cot}_k_{config.k}.txt"
        fail_file_path = os.path.join(fail_file_dir, fail_file_name)

        with open(fail_file_path, "w") as f:
            f.write("\n\n".join(r.failed_output))
