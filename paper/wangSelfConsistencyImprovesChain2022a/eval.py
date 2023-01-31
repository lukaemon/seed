import json
import os
from typing import List, Union
from dataclasses import asdict, dataclass

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import data
import t5
import gpt


def inference_loop(m, tk, dataloader, k, temp=0.7):
    """inference on full dataset
    m: T5ForConditionalGeneration
    tk: T5Tokenizer
    dataloader: DataLoader
    k: number of samples to generate per each input
    """
    generated_text = []

    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description("generating text")

    for batch in dataloader:
        batch_preds = t5.predict(m, tk, batch, k, temp=temp)
        generated_text.extend(batch_preds)

        progress_bar.update()

    return generated_text


def voting(generated_text: List[List[str]], n):
    """
    generated_text: [batch_size, k], list of lists of strings
    n: number of samples to take into consideration. n <= k(sample size)
    """
    pred = []
    vote = []
    generated_text_n_cut = [texts[:n] for texts in generated_text]

    for first_n in generated_text_n_cut:

        extracted_pred = list(map(data.extract_prediction_gsm8k, first_n))
        answer_candidate, cnt = np.unique(extracted_pred, return_counts=True)
        voted_answer = answer_candidate[np.argmax(cnt)]  # take the most voted answer

        pred.append(voted_answer)
        vote.append(cnt)

    return pred, vote, generated_text_n_cut


@dataclass
class EvalResultRaw:
    k: int
    n: int
    generated_text_n_cut: List[List[str]]  # [dataset_size, n]
    vote: List[List[int]]  # [dataset_size, unique_answer]: votes given n
    pred: List[Union[int, str]]  # [dataset_size]: predicted answer
    target: List[Union[int, str]]


@dataclass
class EvalResultProcessed:
    n: int
    accuracy: float
    indi_certainty: List[float]  # certainty of each prediction
    avg_certainty: float


def compute_accuracy(preds: List[int], targets: List[int]) -> float:
    return np.mean(np.array(preds) == np.array(targets))


def gini(x):
    """Gini coefficient, measurement of inequality
    - used as proxy to certainty of LLM prediction
    - https://en.wikipedia.org/wiki/Mean_absolute_difference#Relative_mean_absolute_difference
    - https://stackoverflow.com/a/39513799
    """
    if len(x) == 1:
        return 1.0

    md = np.abs(np.subtract.outer(x, x)).mean()
    rmd = md / np.mean(x)
    g = 0.5 * rmd

    return g


def compute_avg_certainty(vote: List[List[int]]) -> float:
    ginis = [gini(v) for v in vote]

    return np.mean(ginis), ginis


def t5_eval_loop(dataset, m, tk, k, batch_size=1, temp=0.7):
    """
    m: T5ForConditionalGeneration
    tk: T5Tokenizer
    k: number of samples to generate per each input,
    temp: temperature for sampling

    top-k truncation = 40 for UL2, LaMDA, PaLM, GPT-3 without top-k truncation
    temp settings:
        PaLM, GPT-3 = 0.7
        UL2, LaMBDA = 0.5
    """
    dataloader = data.build_gsm8k_dataloader(dataset, tk, batch_size=batch_size)
    generated_text = inference_loop(m, tk, dataloader, k, temp=temp)

    result_raw = []
    for n in range(2, k + 1):  # observe the effect of increasing sampling size
        pred, vote, generated_text_n_cut = voting(generated_text, n)
        result_raw.append(
            EvalResultRaw(k, n, generated_text_n_cut, vote, pred, dataset["target"])
        )

    result_processed = []
    for r in result_raw:
        accuracy = compute_accuracy(r.pred, r.target)
        avg_certainty, indi_certainty = compute_avg_certainty(r.vote)
        result_processed.append(
            EvalResultProcessed(r.n, accuracy, indi_certainty, avg_certainty)
        )

    # release cuda memory
    del m
    torch.cuda.empty_cache()

    return result_raw, result_processed


def plot_sample_vs_accuracy(result_processed: List[EvalResultProcessed]):
    fig, ax = plt.subplots()
    ax.plot([r.n for r in result_processed], [r.accuracy for r in result_processed])
    ax.set_ylabel("accuracy", color="b")
    ax.set(xlabel="sample size", title="sample size vs accuracy")

    # add certainty on the same graph
    ax2 = ax.twinx()
    ax2.plot(
        [r.n for r in result_processed],
        [r.avg_certainty for r in result_processed],
        "ro",
    )
    ax2.set_ylabel("certainty", color="r")

    ax.grid()
    plt.show()


def gpt_inference_loop(model, dataset, k, temp=0.7):
    generated_text = []

    for prompt in tqdm(dataset["prompt"]):
        generated_text.append(gpt.complete(model, prompt, k=k, temp=temp))

    return generated_text


def gpt_eval_loop(model, dataset, k, temp=0.7):
    generated_text = gpt_inference_loop(model, dataset, k, temp=temp)

    result_raw = []
    for n in range(2, k + 1):  # observe the effect of increasing sampling size
        pred, vote, generated_text_n_cut = voting(generated_text, n)
        result_raw.append(
            EvalResultRaw(k, n, generated_text_n_cut, vote, pred, dataset["target"])
        )

    result_processed = []
    for r in result_raw:
        accuracy = compute_accuracy(r.pred, r.target)
        avg_certainty, indi_certainty = compute_avg_certainty(r.vote)
        result_processed.append(
            EvalResultProcessed(r.n, accuracy, indi_certainty, avg_certainty)
        )

    return result_raw, result_processed


def save_eval_result(results, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        df = pd.DataFrame(results)
        df.to_json(f, orient="records")
