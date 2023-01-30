import numpy as np
from tqdm.auto import tqdm

import data
import t5


def vote(preds):
    """
    preds: [batch_size, k], list of lists of strings
    """
    voted_preds = []
    cnts = []

    for instances in preds:
        extracted_preds = list(map(data.extract_prediction_gsm8k, instances))
        answer, cnt = np.unique(extracted_preds, return_counts=True)
        voted_preds.append(answer[np.argmax(cnt)])

    return voted_preds, cnts


def eval_loop(m, tk, dataloader, k):
    """
    m: T5ForConditionalGeneration
    tk: T5Tokenizer
    dataloader: DataLoader
    k: number of samples to generate per each input
    """
    m.eval()

    preds = []
    votes = []

    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description("eval in progress")

    for batch in dataloader:
        batch_preds = t5.predict(m, tk, batch, k)
        batch_voted_preds, batch_vote = vote(batch_preds)

        preds.extend(batch_voted_preds)
        votes.extend(batch_vote)

        progress_bar.update()

    return preds, votes


def compute_accuracy(preds, targets):
    """
    preds: list of ints
    targets: list of ints
    """
    return np.mean(np.array(preds) == np.array(targets))
